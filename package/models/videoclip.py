"""Defines PL Module for VideoClip."""

from collections import defaultdict

import numpy as np
import torch
import pytorch_lightning as pl
from termcolor import colored

from package.losses.weighted_contrastive import (
    LossAddition,
    T2VContraLoss,
    V2TContraLoss,
)
from package.metrics.retrieval import (
    compute_metrics,
)
from package.utils.log import (
    print_retrieval_metrics_for_csv,
)


class VideoCLIP(pl.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        self.no_reverse = config.get("no_reverse", False)
        
        self.alpha_same = config.get("alpha_same", 1.0)
        self.alpha_cross = config.get("alpha_cross", 1.0)
        self.beta = config.get("beta", 1.0)
        
        batch_size = config.get("batch_size", 32)
        sample_weights = torch.ones(2 * batch_size, device=self.device)
        sample_weights[batch_size // 2:] *= self.beta

        # pass config to logger
        self.save_hyperparameters(config)
        
        # define the losses
        self.contrastive_lambda = config.contrastive_lambda
        # self.temporal_lambda = config.temporal_lambda
        # self.temporal = nn.BCEWithLogitsLoss()
        alpha_matrix = self.alpha_cross * np.ones((batch_size, batch_size))
        alpha_matrix[np.arange(batch_size), np.arange(batch_size)] = self.alpha_same
        self.contrastive = LossAddition(
            [
                T2VContraLoss(sample_weights, alpha_matrix),
                V2TContraLoss(sample_weights, alpha_matrix),
            ],
        )
    
    def normalize(self, x):
        return x / x.norm(dim=-1, keepdim=True)
    
    def compute_batch_losses(self, outputs):
        # normalize representations
        if self.no_reverse:
            video = outputs["z_video_forward"]
        else:    
            video_forward = outputs["z_video_forward"]
            video_reverse = outputs["z_video_reverse"]
            video = torch.cat([video_forward, video_reverse], dim=0)
        
        if self.no_reverse:
            text = outputs["z_text_forward"]
        else:
            text_forward = outputs["z_text_forward"]
            text_reverse = outputs["z_text_reverse"]
            text = torch.cat([text_forward, text_reverse], dim=0)

        # contrastive loss
        loss_contrastive = self.contrastive(pooled_video=video, pooled_text=text)

        # total loss
        total_loss = self.contrastive_lambda * loss_contrastive

        return {"total": total_loss, "contrastive": loss_contrastive}

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.

        # get video and text representations
        outputs = self.get_batch_outputs(batch)
        
        # compute losses
        losses = self.compute_batch_losses(outputs)

        # log
        for key in losses:
            self.log(f"batch/train/loss_{key}", losses[key], sync_dist=True)
        
        return losses["total"]

    def validation_step(self, batch, batch_idx):
        
        # get video and text representations
        outputs = self.get_batch_outputs(batch)
        
        # compute losses
        losses = self.compute_batch_losses(outputs)

        # log
        for key in losses:
            self.log(f"batch/val/loss_{key}", losses[key], sync_dist=True)

        return outputs

    def validation_epoch_end(self, outputs):
        
        # gather outputs from all GPUs, from all batches
        outputs = self.all_gather(outputs)
        
        # structure of outputs.
        # Let G be number of GPUs, B batch size, M number of batches, D dimension of representation
        # outputs = [o_1,.., o_M]
        # o_i: {{"z_video": {G x B x D}, .., ..}
        
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
            
            # Retrieval metrics
            sim_matrix = torch.matmul(text_forward, video_embeds.T)
            retrieval_metrics = compute_metrics(sim_matrix.cpu().numpy(), scale=100.)
            print(colored("[Retrieval metrics]", "yellow"), retrieval_metrics)
            print_retrieval_metrics_for_csv(retrieval_metrics, scale=1.)
            for metric in retrieval_metrics:
                self.log(
                    f"epoch/val/metric_{metric}",
                    retrieval_metrics[metric],
                    sync_dist=True,
                    rank_zero_only=True,
                )

            # Temporality metrics
            if not self.no_reverse:
                text_reverse = gathered_outputs["z_text_reverse"]
                text_reverse = self.normalize(text_reverse)

                sim_regular = torch.bmm(video_embeds[..., None, :], text_forward[..., None])
                sim_temporal = torch.bmm(video_embeds[..., None, :], text_reverse[..., None])
                accuracy = np.sum((sim_regular > sim_temporal).cpu().numpy()) / sim_regular.shape[0]
                temporality_metrics = {"accuracy": np.round(100 * accuracy, 2)}
                print(colored("[Temporality metrics]", "magenta"), temporality_metrics)
                for metric in temporality_metrics:
                    self.log(
                        f"epoch/val/metric_{metric}",
                        temporality_metrics[metric],
                        sync_dist=True,
                        rank_zero_only=True,
                    )
                
                # add GeometricMean of R@1 and Temporal Accuracy
                key_1 = "R1"
                key_2 = "accuracy"
                metric = np.sqrt(
                    retrieval_metrics[key_1] * max(0.0, temporality_metrics[key_2] - 50.)
                )
                print("[GeometricMean] of R@1 and Temporal Accuracy", metric)
                self.log(
                    f"epoch/val/metric_geometric_mean",
                    metric,
                    sync_dist=True,
                    rank_zero_only=True,
                )

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
        }

        if not self.no_reverse:
            z_video_reverse = self.model.forward_video(
                vfeats=batch["vfeats_swapped"],
                vmasks=batch["vmasks_swapped"],
                caps=batch["caps"],
                cmasks=batch["cmasks"],
            )
            z_text_reverse = self.model.forward_text(
                caps=batch["caps_swapped"],
                cmasks=batch["cmasks_swapped"],
            )
            outputs.update({
                "z_video_reverse": z_video_reverse,
                "z_text_reverse": z_text_reverse,
            })
        
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.lr, betas=(0.9, 0.98), eps=1e-09
        )
        return optimizer

    def configure_scheduler(self):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[20, 30, 70], gamma=0.5
        )
        return scheduler