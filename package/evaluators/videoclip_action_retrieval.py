from genericpath import exists
import time
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn
import torchvision
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import re
import editdistance
from termcolor import colored
import pytorch_lightning as pl
from collections import defaultdict
from joblib import Parallel, delayed

from package.models import VideoCLIP
# from posttraining.ssv2 import SomethingSomething
# from midtraining.pl_contrastive import VideoCLIP, compute_metrics
# from midtraining.processors import load_videoclip_model
from package.utils.log import print_retrieval_metrics_for_csv

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def average_precision(scores, correct_indices):
    ranks = np.argsort(-scores)
    rank_flags = np.isin(ranks, correct_indices).astype(int)
    pred_ranks = np.where(rank_flags == 1)[0] + 1
    numerator = np.arange(len(pred_ranks)) + 1
    ap = np.sum(numerator / pred_ranks) / len(pred_ranks)
    return ap


def mean_average_precision(sim_mat, labels):
    iterator = tqdm(
        range(len(labels)),
        desc="Computing AP",
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
    )

    # ap_computer = lambda i: average_precision(sim_mat[i], np.where(labels == labels[i])[0])
    # aps = Parallel(n_jobs=8)(
    #     delayed(ap_computer)(i) for i in iterator
    # )
    # mean_ap = np.mean(aps)

    mean_ap = np.mean(
        [
            average_precision(sim_mat[i], np.where(labels == labels[i])[0]) \
            for i in iterator
        ]
    )
    
    return np.round(mean_ap * 100, 3)


class VideoActionRetrieval(VideoCLIP):
    def __init__(self, config, model):
        super().__init__(config, model) # this will load the model

    def get_batch_outputs(self, batch):

        z_video_forward = self.model.forward_video(
            vfeats=batch["vfeats"],
            vmasks=batch["vmasks"],
            caps=batch["caps"],
            cmasks=batch["cmasks"],
        )
        z_text_forward = self.model.forward_text(
            caps=batch["caps"],
            cmasks=batch["cmasks"],
        )

        outputs = {
            "z_video_forward": z_video_forward,
            "z_text_forward": z_text_forward,
            "labels": batch["raw_label_idx"],
        }
        
        return outputs

    def validation_step(self, batch, batch_idx):
        # get video and text representations
        outputs = self.get_batch_outputs(batch)
        return outputs

    @staticmethod
    def get_metrics(ranks):
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        metrics = {"R1": ir1, "R5": ir5, "R10": ir10}
        return metrics

    def validation_epoch_end(self, outputs):
        
        # gather outputs from all GPUs, from all batches
        outputs = self.all_gather(outputs)

        # accumulate all batches across all GPUs
        gathered_outputs = defaultdict(dict)
        keys = outputs[0].keys()
        for key in keys:
            vout = [outputs[i][key] for i in range(len(outputs))]
            
            if len(vout[0].shape) == 3:
                vout = [x.view(x.shape[0] * x.shape[1], x.shape[2]) for x in vout]

            vout = torch.cat(vout, dim=0)
            # vout: (M * G * B) x D
            gathered_outputs[key] = vout

        if self.trainer.is_global_zero:

            ### Metric computation ###

            # normalize representations
            video_embeds = gathered_outputs["z_video_forward"]
            video_embeds = self.normalize(video_embeds)

            text_forward = gathered_outputs["z_text_forward"]
            text_forward = self.normalize(text_forward)
            
            labels = gathered_outputs["labels"].squeeze(1)
            
            print(">>> Computing retrieval metrics...")
            start = time.time()
            metrics = dict()

            # Video to text
            sim_matrix = torch.matmul(video_embeds, text_forward.T)
            classes = labels.unique()
            video_to_class_prob = torch.zeros((video_embeds.shape[0], 174))
            for cls in tqdm(classes, desc="Computing video to class prob"):
                indices = torch.where(labels == cls)[0]
                video_to_class_prob[:, cls] = sim_matrix[:, indices].mean(dim=1)
            video_to_class_prob = video_to_class_prob.softmax(dim=1)
            video_to_pred_class = (-video_to_class_prob).argsort(dim=1)
            ranks = (video_to_pred_class == labels.cpu().unsqueeze(1)).int().argmax(dim=1).cpu().numpy()
            v2t_metrics = self.get_metrics(ranks)
            print("\n >>> V2T metrics ...")
            print_retrieval_metrics_for_csv(v2t_metrics, scale=1.)
            print()
            v2t_metrics = {f"v2t_{k}": v for k, v in v2t_metrics.items()}
            metrics.update(v2t_metrics)

            # Text to video
            sim_matrix = torch.matmul(text_forward, video_embeds.T)    
            # Retrieval metrics
            sim_t2v = sim_matrix.cpu().softmax(dim=1)
            indices = (-sim_t2v).argsort(dim=1).numpy()
            iterator = tqdm(range(len(indices)), desc="Computing ranks")
            labels = labels.cpu().numpy()
            # parallelize rank computation
            from joblib import Parallel, delayed
            rank_computer = lambda i: np.where(np.isin(indices[i], np.where(labels == labels[i])[0]) == True)[0][0]
            ranks = Parallel(n_jobs=8)(delayed(rank_computer)(i) for i in iterator)
            ranks = np.array(ranks)
            t2v_metrics = self.get_metrics(ranks)
            print("\n >>> T2V metrics ...")
            print_retrieval_metrics_for_csv(t2v_metrics, scale=1.)
            print()
            t2v_metrics = {f"t2v_{k}": v for k, v in t2v_metrics.items()}
            metrics.update(t2v_metrics)
            
            # add mean average precision
            metrics.update({"t2v_mAP": mean_average_precision(sim_matrix.cpu().numpy(), labels)})

            end = time.time()
            print(">>> Time to compute retrieval metrics: {:.4f} s".format(end - start))
            for metric in metrics:
                self.log(
                    f"metric_{metric}",
                    metrics[metric],
                    sync_dist=True,
                    rank_zero_only=True,
                )

            # save outputs for analysis
            ckpt_id = "none"
            if self.config.ckpt_path is not None:
                ckpt_id = self.config.ckpt_path.split("/")[1]
            if not self.config.no_save:
                from package.utils.log import repo_path
                print(">>> Saving outputs in ./cache/ for analysis...")
                dataset = self.config.dataset
                labels = np.concatenate([o["labels"].squeeze(1).cpu().numpy() for o in outputs])
                save_path = os.path.join(repo_path, "cache", f"{ckpt_id}_on_{dataset}_t2v_labels.npy")
                np.save(save_path, labels)
                save_path = os.path.join(repo_path, "cache", f"{ckpt_id}_on_{dataset}_t2v_ranks.npy")
                np.save(save_path, ranks)
                # save sim matrix
                np.save(
                    os.path.join(repo_path, "cache", f"{ckpt_id}_on_{dataset}_t2v_sim_matrix.npy"),
                    sim_matrix.cpu().numpy()
                )


if __name__ == "__main__":
    # TODO: add tests with dummy data
    pass