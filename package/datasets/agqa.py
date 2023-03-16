"""ACQA benchmark subset with before/after questions."""
from genericpath import exists
from os.path import join
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
from termcolor import colored
import pandas as pd
from collections import defaultdict

from package.utils.io import load_json
from external.utils_videoclip import DSAligner
# from package.datasets.tempo import swap_event_descriptions
# from midtraining.tempo import DSAligner, swap_event_descriptions

import warnings
warnings.filterwarnings("ignore")

# from posttraining.nextqa_mcq import VQAAligner


class VQAAligner(DSAligner):
    """VQA dataset.
    similar to sample in how2.
    we call __call__ multiple times.
    """

    def __call__(self, video_id, video_feature, text_feature, wps=0.7):
        caps = []
        cmasks = []
        answer = text_feature[0]
        for ans_idx, _text_feature in enumerate(text_feature[1]):
            output = super().__call__(
                video_id, video_feature, _text_feature, wps)
            caps.append(output["caps"])
            cmasks.append(output["cmasks"])
        output.update({
            "caps": torch.stack(caps),
            "cmasks": torch.stack(cmasks),
            "answers": torch.LongTensor([answer]),
        })
        return output


class AGQATemporal(Dataset):
    def __init__(
            self,
            data_root,
            split_file="AGQA/splits/test_unbalanced_subset-temporal-v1.0.csv",
            video_dir="Charades/Charades_v1_480/",
            vfeat_dir="Charades/feat/feat_how2_s3d",
        ):

        self.video_dir = join(data_root, video_dir)
        self.vfeat_dir = join(data_root, vfeat_dir)
        self.split_path = join(data_root, split_file)
        assert exists(self.split_path), f"Split file {self.split_path} does not exist."
        
        print()
        print(colored(f">>> Loading split from {os.path.basename(split_file)} ...", "yellow"))

        df = pd.read_csv(self.split_path)
        
        df["video_path"] = df["video_id"].apply(lambda x: join(self.video_dir, x + ".mp4"))
        # df = df[df["video_path"].apply(exists)]
        
        print(">>> Number of unique videos:", len(df["video_id"].unique()))
        
        df["vfeat_path"] = df["video_id"].apply(lambda x: join(self.vfeat_dir, x + ".npy"))
        df = df[df["vfeat_path"].apply(exists)]
        nv = len(df["video_id"].unique())
        print(">>> Number of unique videos with vfeat available:", nv)
        assert nv > 0, "No videos with vfeat available."

        df["answer_idx"] = df["answer"].apply(lambda x: int(x == "after"))
        df["category"] = "temporal"
        
        self.data = df

        print(colored(f">>> Number of videos: {len(self.data)}", "yellow"))
        print()
        
        # load text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # aligner: converts video and text tokens into model-ingestible tensors
        self.aligner = VQAAligner(max_video_len=32, max_len=96, bert_name="bert-base-uncased")

        # print and example
        print("::: Displaying an example :::\n")
        sample = self.__getitem__(1500)
        video_path = sample["video_path"]
        print("Video path:", video_path)
        print(colored(">>> Question: ", "blue"), sample["question"])
        print(colored(">>> Answer: ", "green"), sample["answer"])
        print(colored(">>> Statements: ", "magenta"))
        print("\n".join(sample["statement"]))
        print()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        outputs = dict()
        instance = dict(self.data.iloc[index])
        
        video_id = instance["video_id"]
        outputs["video_id"] = video_id
        
        video_path = instance["video_path"]
        outputs["video_path"] = video_path

        category = instance["category"]
        outputs["category"] = category
        
        # get video tokens
        vfeat_path = join(self.vfeat_dir, video_id + '.npy')
        video_tokens = np.load(vfeat_path)

        # get text tokens
        answer = instance["answer_idx"]
        outputs["answer"] = answer
        question = instance["question"]
        outputs["question"] = question
        outputs["statement"] = []
        ans_cands = ["before", "after"]
        for ans_idx, ans in enumerate(ans_cands):
            text = question.replace("before or after", ans)
            outputs["statement"].append(text)
            ans_cands[ans_idx] = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        text_feature = (answer, ans_cands)

        # gather them all
        inputs = self.aligner(video_id, video_tokens, text_feature)
        outputs.update(inputs)
        
        return outputs        


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # load splits
    data_root = "/ssd/pbagad/datasets/"    
    dataset = AGQATemporal(
        data_root=data_root,
        split_file="AGQA/splits/test_unbalanced_subset-temporal-v1.0.csv",
    )
    sample = dataset.__getitem__(0)
    print(sample.keys())
    print(sample["caps"].shape)
