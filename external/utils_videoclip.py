"""Helper functions to access VideoCLIP model and its data processors."""

import numpy as np
import torch

from external.fairseq.examples.MMPT.mmpt.tasks import Task
from external.fairseq.examples.MMPT.mmpt.utils import load_config


def load_videoclip_model(
    cfg_path="external/fairseq/examples/MMPT/projects/retri/videoclip/how2.yaml",
    checkpoint_path="external/requirements/fairseq/runs/retri/videoclip/checkpoint_best.pt",
):

    import argparse
    # create a dummy argument parser
    parser = argparse.ArgumentParser()
    # parser.add_argument("-taskconfig", type=str, default=cfg_path)
    args = parser.parse_args("")
    args.taskconfig = cfg_path

    # load config
    config = load_config(args)

    # load model
    mmtask = Task.config_task(config)
    mmtask.build_model()

    # load checkpoint
    model = mmtask.load_checkpoint(checkpoint_path)

    return config, model


class Aligner(object):
    """
    An alignprocessor align video and text and output a dict of tensors (for a model).
    """
    def __init__(self, max_video_len, max_len, bert_name):
        """__init__ needs to be light weight for more workers/threads."""
        # self.split = config.split
        # self.max_video_len = config.max_video_len
        # self.max_len = config.max_len
        self.max_video_len = max_video_len
        self.max_len = max_len
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(bert_name), use_fast=False,
        )
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id

    def __call__(self, video_id, video_feature, text_feature):
        raise NotImplementedError

    def _build_video_seq(self, video_feature, video_clips=None):
        """
        `video_feature`: available video tokens.
        `video_clips`: video clip sequence to build.
        """
        if not isinstance(video_feature, np.ndarray):
            raise ValueError(
                "unsupported type of video_feature", type(video_feature)
            )

        if video_clips is None:
            # this is borrowed from DSAligner
            video_start = 0
            video_end = min(len(video_feature), self.max_video_len)
            # the whole sequence is a single clip.
            video_clips = {"start": [video_start], "end": [video_end]}

        vfeats = np.zeros(
            (self.max_video_len, video_feature.shape[1]), dtype=np.float32
        )
        vmasks = torch.zeros((self.max_video_len,), dtype=torch.bool)
        video_len = 0
        for start, end in zip(video_clips["start"], video_clips["end"]):
            clip_len = min(self.max_video_len - video_len, (end - start))
            if clip_len > 0:
                vfeats[video_len: video_len + clip_len] = video_feature[
                    start: start + clip_len
                ]
                vmasks[video_len: video_len + clip_len] = 1
                video_len += clip_len
        vfeats = torch.from_numpy(vfeats)

        return vfeats, vmasks

    def _build_text_seq(self, text_feature, text_clip_indexs=None):
        """
        `text_feature`: all available clips.
        `text_clip_indexes`: clip sequence to build.
        """
        if text_clip_indexs is None:
            text_clip_indexs = [0]

        full_caps = []
        if isinstance(text_feature, dict):
            for clip_idx in text_clip_indexs:
                full_caps.extend(text_feature["cap"][clip_idx])
        else:
            full_caps = text_feature
        max_text_len = self.max_len - self.max_video_len - 3
        full_caps = full_caps[:max_text_len]
        full_caps = (
            [self.cls_token_id, self.sep_token_id] + full_caps + [self.sep_token_id]
        )
        text_pad_len = self.max_len - len(full_caps) - self.max_video_len
        padded_full_caps = full_caps + [self.pad_token_id] * text_pad_len
        caps = torch.LongTensor(padded_full_caps)
        cmasks = torch.zeros((len(padded_full_caps),), dtype=torch.bool)
        cmasks[: len(full_caps)] = 1

        return caps, cmasks

    def batch_post_processing(self, batch, video_feature):
        return batch


class DSAligner(Aligner):
    """
    Downstream (DS) aligner shared by all datasets.
    """
    def __init__(self, max_video_len, max_len, bert_name):
        super().__init__(max_video_len, max_len, bert_name)

    def __call__(self, video_id, video_feature, text_feature, wps=0.7):
        # random sample a starting sec for video.
        video_start = 0
        video_end = min(len(video_feature), self.max_video_len)
        # the whole sequence is a single clip.
        video_clips = {"start": [video_start], "end": [video_end]}

        text_feature = {
            "cap": [text_feature],
            "start": [video_start],
            "end": [len(text_feature) / wps],
        }
        text_clip_indexs = [0]

        vfeats, vmasks = self._build_video_seq(
            video_feature, video_clips
        )
        caps, cmasks = self._build_text_seq(
            text_feature, text_clip_indexs
        )

        return {
            "caps": caps,
            "cmasks": cmasks,
            "vfeats": vfeats,
            "vmasks": vmasks,
            "video_id": video_id,
        }


if __name__ == "__main__":
    import os

    from transformers import AutoTokenizer

    from package.utils.io import load_txt
    from package.utils.log import print_update

    print_update("TEST 1: Loading VideoCLIP model")
    cfg_path = "external/fairseq/examples/MMPT/projects/retri/videoclip/test_vtt_zs.yaml"
    checkpoint_path="external/requirements/fairseq/runs/retri/videoclip/checkpoint_best.pt"
    config, model = load_videoclip_model(
        cfg_path=cfg_path,
        checkpoint_path=checkpoint_path,
    )
    print_update("TEST 1 FINISHED: Successful!\n")
    print()

    
    print_update("TEST 2: Loading data processing pipeline for VideoCLIP")
    # Load sample video tokens and caption
    print("Loading data for a sample dummy video.")
    video_id = "video_000"
    sample_video_folder = f"sample_data/{video_id}/"
    caption = load_txt(os.path.join(sample_video_folder, "caption.txt"))[0]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    aligner = DSAligner(max_video_len=32, max_len=96, bert_name="bert-base-uncased")
    text_tokens = tokenizer(caption, add_special_tokens=False)["input_ids"]
    assert isinstance(text_tokens, list)
    video_tokens = np.load(os.path.join(sample_video_folder, "how2_s3d_feat.npy"))
    assert isinstance(video_tokens, np.ndarray)
    assert len(video_tokens.shape) == 2 and video_tokens.shape[1] == 512
    inputs = aligner(video_id, video_tokens, text_tokens)
    assert inputs["caps"].shape == torch.Size([64])
    assert inputs["vfeats"].shape == torch.Size([32, 512])
    print_update("TEST 2 FINISHED: Successful!")
    print()

