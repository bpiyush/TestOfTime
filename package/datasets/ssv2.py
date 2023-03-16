"""Something-Something v2 dataset for action recognition."""
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

from package.utils.io import load_json, save_json
from external.utils_videoclip import DSAligner

import warnings
warnings.filterwarnings("ignore")


class SSv2(Dataset):
    def __init__(
            self,
            data_root,
            video_dir="SSv2/something-something-v2-videos_avi",
            vfeat_dir="SSv2/feat/feat_how2_s3d/",
            split_dir="SSv2/something-something-v2-annotations/",
            split_file="something-something-v2-validation-tmpl-ret-singularity.json",
        ) -> None:
        super().__init__()

        self.data_root = data_root
        self.video_dir = join(data_root, video_dir)
        self.vfeat_dir = join(data_root, vfeat_dir)
        self.split_path = join(data_root, split_dir, split_file)

        print()
        print(colored(f">>> Loading split from {os.path.basename(split_file)} ...", "yellow"))
        self.data = load_json(self.split_path)

        # append video path
        self.data = list(
            map(
                lambda x: {**x, 'video_path': os.path.join(self.video_dir, x['id'] + '.avi')},
                self.data
            )
        )
        
        # # filter data whose video does not exist
        # indices = [i for i, x in enumerate(self.data) if exists(x['video_path'])]
        # self.data = [self.data[i] for i in indices]
        
        # filter data whose vfeat does not exist
        indices = [i for i, x in enumerate(self.data) if exists(os.path.join(self.vfeat_dir, x['id'] + '.npy'))]
        self.data = [self.data[i] for i in indices]
        
        # class indices file
        split_file_name = os.path.basename(self.split_path)
        df = pd.read_csv(self.split_path.replace(split_file_name, "fine_grained_classes.csv"))
        self.template_to_idx = dict(zip(df.class_name,df.class_index))
        
        print(colored(f">>> Number of videos: {len(self.data)}", "yellow"))
        print()
        
        # create a dict: text2videos: {text: [video_id, ...]}
        self.text2videos = defaultdict(list)
        for i, d in enumerate(self.data):
            self.text2videos[self.template_to_idx[d['template']]].append(d['id'])
        num_videos_per_class = np.array(list(map(len, self.text2videos.values())))
        class_priors = num_videos_per_class / num_videos_per_class.sum()
        class_random_ranks = num_videos_per_class / len(self.data)
        random_metric = 100. * (class_random_ranks * class_priors).sum()
        print("Random retrieval metric: {:.4f}".format(random_metric))
        
        # load text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # aligner: converts video and text tokens into model-ingestible tensors
        self.aligner = DSAligner(max_video_len=32, max_len=96, bert_name="bert-base-uncased")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        outputs = dict()
        instance = self.data[index]
        
        video_id = instance["id"]
        # outputs["video_id"] = video_id
        
        video_path = instance["video_path"]
        # outputs["video_path"] = video_path
        
        # TODO: Prompt engineering
        text = instance["template"]
        text += "A video that shows " + text
        # outputs["text"] = text
        outputs["raw_label_idx"] = torch.tensor(self.template_to_idx[instance["template"]]).long().unsqueeze(0)
        outputs["template"] = instance["template"]
        
        # get video tokens
        vfeat_path = os.path.join(self.vfeat_dir, video_id + '.npy')
        video_tokens = np.load(vfeat_path)

        # import torchvision as tv
        # x, a, fps = tv.io.read_video(video_path)
        # num_secs = x.shape[0] / fps["video_fps"]
        # # assert int(num_secs) == video_tokens.shape[0]
        # print(video_path, video_tokens.shape, num_secs)

        # get text tokens
        text_tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        
        # gather them all
        inputs = self.aligner(video_id, video_tokens, text_tokens)
        outputs.update(inputs)
        
        return outputs


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    data_root = "/ssd/pbagad/datasets"
    dataset = SSv2(
        data_root=data_root,
        split_file="something-something-v2-validation-tmpl-ret-singularity.json",
    )
    for i in range(min(len(dataset), 10)):
        instance = dataset[i]
    
    template = "Spinning [something] that quickly stops spinning"
    indices = [i for i, x in enumerate(dataset.data) if x["template"] == template]
    idx = indices[4]
    instance = dataset[idx]
    print(instance["template"])