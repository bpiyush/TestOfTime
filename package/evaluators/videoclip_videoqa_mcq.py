"""Zero-shot evaluation of VideoClip on NextQA MCQ."""
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

# from posttraining.nextqa import NextQA
# from posttraining.nextqa_mcq import NextQA
from package.models import VideoCLIP
# from midtraining.processors import load_videoclip_model

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VideoQAMCQ(VideoCLIP):
    def __init__(self, config, model, num_answer_candidates=5, log_csv=False):
        super().__init__(config, model) # this will load the model
        self.num_answer_candidates = num_answer_candidates
        self.log_csv = log_csv

    def get_batch_outputs(self, batch):

        v_len = batch["vfeats"].size(1)
        hidden_size = batch["vfeats"].size(2)
        batch["vfeats"] = batch["vfeats"].unsqueeze(1).repeat(
            1, self.num_answer_candidates, 1, 1
        ).view(-1, v_len, hidden_size)
        batch["vmasks"] = batch["vmasks"].unsqueeze(1).repeat(
            1, self.num_answer_candidates, 1
        ).view(-1, v_len)
        
        t_len = batch["caps"].size(-1)
        batch["caps"] = batch["caps"].view(-1, t_len)
        batch["cmasks"] = batch["cmasks"].view(-1, t_len)
        
        outputs = self.model(**batch)
        # outputs.update(batch)
        
        hidden_size = outputs["pooled_video"].size(-1)
        pooled_video = outputs["pooled_video"].view(
            -1, self.num_answer_candidates, hidden_size,
        )
        pooled_text = outputs["pooled_text"].view(
            -1, self.num_answer_candidates, hidden_size,
        )
        scores = torch.bmm(pooled_video, pooled_text.transpose(2, 1))
        scores = scores.argmax(-1)

        return {
            "predictions": scores[:, 0],
            "answers": batch["answers"].flatten(),
            "category": batch["category"],
        }


    def validation_step(self, batch, batch_idx):
        # get video and text representations
        outputs = self.get_batch_outputs(batch)
        return outputs

    def validation_epoch_end(self, outputs):
        
        # gather outputs from all GPUs, from all batches
        outputs = self.all_gather(outputs)

        # accumulate all batches across all GPUs
        gathered_outputs = defaultdict(dict)
        keys = outputs[0].keys()
        for key in keys:
            if key not in ["category"]:
                vout = [outputs[i][key].flatten() for i in range(len(outputs))]
                vout = torch.cat(vout, dim=0)
            else:
                X = []
                for i in range(len(outputs)):
                    X.extend(outputs[i][key])
                vout = X
            
            # vout: (M * G * B) x D
            gathered_outputs[key] = vout

        if self.trainer.is_global_zero:

            ### Metric computation ###
            metrics = dict()
            categories = np.array(gathered_outputs["category"])
            uniq_categories = np.unique(categories)
            for cat in uniq_categories:
                indices = np.where(categories == cat)[0]
                preds = gathered_outputs["predictions"][indices]
                labels = gathered_outputs["answers"][indices]
                metrics.update({cat: (preds.float() == labels.float()).sum() / len(preds)})
            
            correct = (
                gathered_outputs["predictions"] == gathered_outputs["answers"]
            ).sum().item()
            accuracy = correct / gathered_outputs["predictions"].shape[0]
            # print("Accuracy: ", accuracy)
            metrics.update({"total": torch.tensor(accuracy)})
            scale = 100.
            metrics = {k: np.round(v.cpu().numpy() * scale, decimals=3) for k, v in metrics.items()}
            
            if self.log_csv:
                order = ["desc"] * ("desc" in metrics) + ["causal", "temporal", "total"]
                csv_print = ",".join([str(metrics[k]) for k in order])
                print("CSV: ", csv_print)

            for metric in metrics:
                self.log(f"metric_{metric}", metrics[metric], sync_dist=True, rank_zero_only=True)


if __name__ == "__main__":
    # TODO: add tests with dummy data
    pass