"""All processors."""
from genericpath import exists
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchvision
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import re
import editdistance
import pandas as pd

from package.utils.io import load_json
from external.fairseq.examples.MMPT.mmpt.losses.loss import Loss
from external.utils_videoclip import DSAligner

import warnings
warnings.filterwarnings("ignore")


def active_event_description_combiner(X_desc, Y_desc, tau):
    """Combines textual descriptions of event X, Y related by temporal relation tau."""
    return X_desc + " " + tau + " " + Y_desc


def passive_event_description_combiner(X_desc, Y_desc, tau):
    """Combines textual descriptions of event X, Y related by temporal relation tau."""
    return tau + " " + X_desc + " " + Y_desc


def remove_multiple_spaces(string):
    return re.sub(' +', ' ', string)


def check_close_enough(desc, gen_desc, threshold=10):
    return editdistance.eval(desc, gen_desc) < threshold


def swap_event_descriptions(description, X_desc, Y_desc, tau):
    assert tau in description, "[{}] not in [{}]".format(tau, description)
    assert X_desc in description, "[{}] not in [{}]".format(X_desc, description)
    assert Y_desc in description, "[{}] not in [{}]".format(Y_desc, description)
    
    if check_close_enough(description, active_event_description_combiner(X_desc, Y_desc, tau)):
        return active_event_description_combiner(Y_desc, X_desc, tau)
    elif check_close_enough(description, passive_event_description_combiner(X_desc, Y_desc, tau)):
        return passive_event_description_combiner(Y_desc, X_desc, tau)
    else:
        # import ipdb; ipdb.set_trace()
        print(description)
        print(X_desc, " | " ,Y_desc, " | ", tau)
        print("-" * 80)
        raise ValueError("Description does not match any combiner.")


def process_time_annotation(annotation, video_duration, truncate_video_at=30., clip_duration=5.):
    """Returns actual timestamps from annotations."""

    t_start, t_end = annotation
    t_start = t_start * clip_duration
    
    video_duration = min(truncate_video_at, video_duration)
    t_end = min((t_end + 1) * clip_duration, video_duration)
    
    return t_start, t_end


class SyntheticDataset(Dataset):
    """Dataset definition for TEMPO."""
    def __init__(
        self,
        video_dir,
        split_path,
        feat_dir,
        use_time_bdry=False,
        print_example=False,
    ):
        super().__init__()
        self.video_dir = video_dir
        self.feat_dir = feat_dir
        self.print_example = print_example
        
        # load split
        data = pd.read_csv(split_path).to_dict(orient="records")
        
        # # make certain keys lowercase
        for i, d in enumerate(data):
            data[i]["sentence"] = d["sentence"].lower()
        
        # remove instances without npy features
        indices = []
        for i, d in enumerate(data):
            video_id = d["video_id"]
            if not os.path.exists(self.feature_path(video_id)):
                indices.append(i)

        # delete indices
        data = [d for i, d in enumerate(data) if i not in indices]
                
        # filter instances with temporal prepositions
        data = self.filter_data(data)

        # add start and end times
        self.use_time_bdry = use_time_bdry
        for i, d in enumerate(data):
            sx, ex = d["event_X"]["time_in_secs"]
            sy, ey = d["event_Y"]["time_in_secs"]
            s = min(sx, sy)
            e = max(ex, ey)
            data[i]["start_time"], data[i]["end_time"] = s, e

        self.data = data
        
        # load text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # aligner: converts video and text tokens into model-ingestible tensors
        self.aligner = DSAligner(max_video_len=32, max_len=96, bert_name="bert-base-uncased")
    
    def filter_data(self, data):
        """Filters data to only have instances with temporal prepositions."""
        print(">>> Filtering data based on temporal prepositions...")
        filtered_data = data
        
        # check if each instance has event X, event Y (description and times)
        final_data = []
        for i, d in enumerate(filtered_data):

            desc = d["sentence"]
            temporal_preposition = "before" if "before" in d["video_id"] else "after"
            
            instance = filtered_data[i]
            instance["temporal_preposition"] = temporal_preposition

            event_X_desc, event_Y_desc = desc.split(f" {temporal_preposition} ")
            if temporal_preposition == "before":
                event_X_time_in_secs = [0., d["end_time"] / 2.]
                event_Y_time_in_secs = [d["end_time"] / 2., d["end_time"]]
            else:
                event_Y_time_in_secs = [0., d["end_time"] / 2.]
                event_X_time_in_secs = [d["end_time"] / 2., d["end_time"]]
            
            instance["event_X"] = {
                "description": event_X_desc,
                "time_in_secs": event_X_time_in_secs,
            }
            instance["event_Y"] = {
                "description": event_Y_desc,
                "time_in_secs": event_Y_time_in_secs,
            }
            
            if instance["event_X"]["description"].strip() == "" or \
                instance["event_Y"]["description"].strip() == "":
                continue

            final_data.append(instance)

        print(">>> Number of filtered instances:", len(final_data), "\n")

        return final_data

    def __len__(self):
        return len(self.data)

    def video_path(self, video_id):
        candidate_path = os.path.join(self.video_dir, video_id + ".mp4")
        if os.path.exists(candidate_path):
            return candidate_path
        else:
            try:
                candidate_path = os.path.join(self.video_dir, video_id)
                assert exists(candidate_path), f"video {video_id} not found"
            except:
                raise ValueError(f"video {video_id} not found")
    
    def feature_path(self, video_id):
        return os.path.join(self.feat_dir, video_id + ".npy")
    
    def __getitem__(self, index):
        outputs = dict()
        instance = self.data[index]
        
        video_id = instance["video_id"]
        outputs["video_id"] = video_id
        
        outputs["temporal_preposition"] = instance["temporal_preposition"]

        # # process video: obtain video tokens (S3D features)
        # video_path = self.video_path(video_id)        
        # outputs["video_path"] = video_path

        feat_path = self.feature_path(video_id)
        video_tokens = np.load(feat_path)
        if self.use_time_bdry:
            start, end = instance["start_time"], instance["end_time"]
            start, end = int(start), int(end)
            video_tokens = video_tokens[start:end]
            outputs["start_time"] = start
            outputs["end_time"] = end
            
        midpoint = len(video_tokens) // 2
        video_tokens_swapped = np.concatenate(
            [video_tokens[midpoint:], video_tokens[:midpoint]], axis=0
        )
        
        # process text: obtain text tokens (BERT tokenizer)
        text = instance["sentence"]
        outputs["text"] = text
        text_tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]

        # align: convert video and text tokens into model-ingestible tensors
        inputs = self.aligner(video_id, video_tokens, text_tokens)
        outputs.update(inputs)
        
        text_swapped = swap_event_descriptions(
            text,
            instance["event_X"]["description"],
            instance["event_Y"]["description"],
            instance["temporal_preposition"],
        )
        outputs["text_swapped"] = text_swapped
        text_tokens = self.tokenizer(text_swapped, add_special_tokens=False)["input_ids"]
        inputs_swapped = self.aligner(video_id, video_tokens, text_tokens)
        outputs.update(
            dict(caps_swapped=inputs_swapped["caps"], cmasks_swapped=inputs_swapped["cmasks"])
        )

        # align: convert video and text tokens into model-ingestible tensors
        inputs = self.aligner(video_id, video_tokens_swapped, text_tokens)
        outputs.update(
            {
                "vfeats_swapped": inputs["vfeats"],
                "vmasks_swapped": inputs["vmasks"],
            }
        )

        if self.print_example and index == 0:
            print(":::::: Example instance ::::::")
            print("Text:", outputs["text"])
            print("Text swapped:", outputs["text_swapped"])

            print("Event X: ", self.data[index]["event_X"])
            print("Relation:", instance["temporal_preposition"])
            print("Event Y: ", self.data[index]["event_Y"])
            print("::::::::::::::::::::::::::::::")
            print()

        return outputs


def load_dataset(
        data_root,
        video_dir="ToT-syn-v2.0/videos",
        feat_dir="ToT-syn-v2.0/feat/feat_how2_s3d_fps3",
        split_dir="ToT-syn-v2.0/splits/",
        mode="val", subset="v2.0",
    ):
    assert mode in ["test"], f"mode {mode} not supported for SyntheticDataset"
    video_dir = os.path.join(data_root, video_dir)
    feat_dir = os.path.join(data_root, feat_dir)
    split_dir = os.path.join(data_root, split_dir)
    split_path = os.path.join(split_dir, f"{mode}_{subset}.csv")
    assert exists(split_path), f"split {split_path} not found"
    dataset = SyntheticDataset(
        video_dir=video_dir,
        feat_dir=feat_dir,
        split_path=split_path,
    )
    return dataset


if __name__ == "__main__":
    from tqdm import tqdm
    from package.utils.log import print_update

    print_update("> TESTING Synthetic DATASET", color="green")
    dataset = load_dataset(
        data_root="/ssd/pbagad/datasets/",
        mode="test",
        subset="v2.0",
        print_example=True,
    )

    instance = dataset[0]
    print(instance.keys())
    print(instance["vfeats"].shape)
    print_update("< FINISHED TESTING Synthetic DATASET", color="green")
