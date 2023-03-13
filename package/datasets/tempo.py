"""Creates a temporal version of the TEMPO Dataset."""
import os
import re
import editdistance

import pandas as pd
import numpy as np
from termcolor import colored
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from package.utils.io import load_json
from external.utils_videoclip import DSAligner


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


class TEMPOTL(Dataset):
    """Dataset definition for TEMPO."""
    def __init__(
            self,
            video_dir,
            split_path,
            feat_dir,
            subset_path=None,
            use_time_bdry=True,
            add_metadata=True,
            print_example=False,
        ):
        super().__init__()
        self.video_dir = video_dir
        self.feat_dir = feat_dir
        self.add_metadata = add_metadata
        self.subset_path = subset_path
        self.print_example = print_example
        
        print()
        print(colored(">>> Loading data from {}".format(split_path), "yellow"))
        
        # load split
        data = load_json(split_path)
        
        if subset_path is not None:
            print(colored(">>> Loading subset from {}".format(subset_path), "blue"))
            subset = pd.read_csv(subset_path)
            video_ids = subset["video"].tolist()
            annot_ids = subset["annotation_id"].tolist()
            data = [
                d for d in data if d["video"] in video_ids and d["annotation_id"] in annot_ids
            ]
        
        # remove non temporal instances
        data = [
            d for d in data if "before" in d["annotation_id"] or "after" in d["annotation_id"]
        ]
        for i in range(len(data)):
            data[i]["temporal_preposition"] = data[i]["annotation_id"].split("_")[0]
        
        # remove instances with multiple temporal relations
        data = [d for d in data if d["description"].count(d["temporal_preposition"]) == 1]
        
        # make certain keys lowercase
        for i, d in enumerate(data):
            data[i]["annotation_id"] = data[i]["annotation_id"].lower()
            data[i]["description"] = data[i]["description"].lower()
            data[i]["reference_description"] = data[i]["reference_description"].lower()
            
            # remove newline
            data[i]["description"] = data[i]["description"].strip("\n")
            data[i]["reference_description"] = data[i]["reference_description"].strip("\n")
        
            # remove commas
            data[i]["description"] = remove_multiple_spaces(
                data[i]["description"].replace(",", "")
            ).strip()
            data[i]["reference_description"] = remove_multiple_spaces(
                data[i]["reference_description"].replace(",", "")
            ).strip()
        
        # remove instances without npy features
        indices = []
        for i, d in enumerate(data):
            video_id = d["video"]
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
            
            # interval = d["train_times"][0]
            # interval = process_time_annotation(interval, video_duration=30.)
            # data[i]["start_time"], data[i]["end_time"] = interval

        # add metadata
        if self.add_metadata:
            for i, d in enumerate(data):
                data[i]["metadata"] = {
                    "video_id": d["video"],
                    "temporal_relation": d["temporal_preposition"],
                    "text_x": d["event_X"]["description"],
                    "start_x": d["event_X"]["time_in_secs"][0],
                    "end_x": d["event_X"]["time_in_secs"][1],
                    "text_y": d["event_Y"]["description"],
                    "start_y": d["event_Y"]["time_in_secs"][0],
                    "end_y": d["event_Y"]["time_in_secs"][1],
                }
        
        use_balanced=False
        if use_balanced:
            df = pd.DataFrame(data)
            
            t_to_count = df["temporal_preposition"].value_counts()
            min_count = t_to_count.min()
            # sample min_count from each temporal preposition
            indices = []
            for t in t_to_count.index:
                indices.extend(
                    df[df.temporal_preposition == t].sample(n=min_count).index.tolist()
                )
            
            data = [d for i, d in enumerate(data) if i in indices]
            print(">>> Using balanced set with {} instances.".format(len(data)))
        
        self.data = data
        video_ids = np.unique([d["video"] for d in data])

        print(
            colored(
                ">>> Loaded {} instances from {} unique videos.".format(
                    len(data), len(video_ids)
                ),
                "yellow"
            )
        )
        
        # load text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # aligner: converts video and text tokens into model-ingestible tensors
        self.aligner = DSAligner(max_video_len=32, max_len=96, bert_name="bert-base-uncased")
    
    def filter_data(self, data):
        """Filters data to only have instances with temporal prepositions."""
        print(">>> Filtering data based on temporal prepositions...")
        filtered_data = []
        for instance in data:
            if "before" in instance["annotation_id"] or "after" in instance["annotation_id"]:
                filtered_data.append(instance)
        
        # check if each instance has event X, event Y (description and times)
        final_data = []
        num_exceptions = 0
        for i, d in enumerate(filtered_data):
            assert len(d["context"]) == 2 and len(d["train_times"]) == 1
            
            temporal_preposition = d["annotation_id"].split("_")[0]
            
            instance = filtered_data[i]
            instance["temporal_preposition"] = temporal_preposition
            # filtered_data[i]["temporal_preposition"] = temporal_preposition
            
            desc = d["description"]
            instance["description"] = desc
            
            # if the sentence starts with the temporal_preposition
            # then need to get the event X and Y carefully
            if desc.startswith(temporal_preposition):
                desc = desc.replace(temporal_preposition, "", 1).strip()

                if temporal_preposition == "before":
                    num_exceptions += 1
                    event_X_desc = d["reference_description"]
                    event_Y_desc = desc.split(event_X_desc)[1].replace(",", "").strip()
                    # BUGFIX: swap event X and Y
                    event_X_desc, event_Y_desc = event_Y_desc, event_X_desc
                else:
                    event_X_desc = d["reference_description"]
                    event_Y_desc = desc.split(event_X_desc)[1].replace(",", "").strip()

                event_Y_desc = event_Y_desc.strip()
                event_X_desc = event_X_desc.strip()
                # IMPORTANT
                instance["description"] = f"{event_X_desc} {temporal_preposition} {event_Y_desc}"

            else:
                if temporal_preposition == "before":
                    event_Y_desc = d["reference_description"]
                    event_X_desc = desc.split(" " + temporal_preposition + " " + event_Y_desc)[0]
                else:
                    event_Y_desc = d["reference_description"]
                    event_X_desc = desc.split(" " + temporal_preposition + " " + event_Y_desc)[0]
            
            event_X_times = d["train_times"][0]
            event_Y_times = d["context"]

            # if temporal_preposition == "before":
            #     event_X_times = d["train_times"][0]
            #     event_Y_times = d["context"]
            # else:
            #     event_Y_times = d["train_times"][0]
            #     event_X_times = d["context"]

            event_X_time_in_secs = process_time_annotation(
                event_X_times, video_duration=30.,
            )
            event_Y_time_in_secs = process_time_annotation(
                event_Y_times, video_duration=30.,
            )
            
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

        print(">>> {} exceptions".format(num_exceptions))
        print(colored(">>> Number of filtered instances: " + str(len(final_data)) + "\n", "yellow"))

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
                assert os.path.exists(candidate_path), f"video {video_id} not found"
            except:
                raise ValueError(f"video {video_id} not found")
    
    def feature_path(self, video_id):
        return os.path.join(self.feat_dir, video_id + ".npy")
    
    def __getitem__(self, index):
        outputs = dict()
        instance = self.data[index]
        
        video_id = instance["video"]
        outputs["video_id"] = video_id
        
        outputs["temporal_relation"] = instance["temporal_preposition"]

        # # process video: obtain video tokens (S3D features)
        # video_path = self.video_path(video_id)
        # outputs["video_path"] = video_path
        
        if self.add_metadata:
            outputs["metadata"] = instance["metadata"]

        feat_path = self.feature_path(video_id)
        video_tokens = np.load(feat_path)
        if self.use_time_bdry:
            # start, end = instance["start_time"], instance["end_time"]
            # start, end = int(start), int(end)
            # video_tokens = video_tokens[start:end]
            # outputs["start_time"] = start
            # outputs["end_time"] = end
            
            start_X, end_X = instance["event_X"]["time_in_secs"]
            video_tokens_X = video_tokens[int(start_X):int(end_X)]
            
            start_Y, end_Y = instance["event_Y"]["time_in_secs"]
            video_tokens_Y = video_tokens[int(start_Y):int(end_Y)]
            
            if instance["temporal_preposition"] == "before":
                video_tokens = np.concatenate([video_tokens_X, video_tokens_Y])
                video_tokens_swapped = np.concatenate([video_tokens_Y, video_tokens_X])
            else:
                video_tokens = np.concatenate([video_tokens_Y, video_tokens_X])
                video_tokens_swapped = np.concatenate([video_tokens_X, video_tokens_Y])
        
        # process text: obtain text tokens (BERT tokenizer)

        text_event_x = instance["event_X"]["description"]
        text_event_y = instance["event_Y"]["description"]
        temporal_rel = instance["temporal_preposition"]
        
        use_first_then = False
        if use_first_then:
            if temporal_rel == "before":
                text = "First, " + text_event_x + ". Then, " + text_event_y + "."
            else:
                text = "First, " + text_event_y + ". Then, " + text_event_x + "."
        else:
            text = instance["description"]

        outputs["text"] = text
        text_tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]

        # add forward video tokens
        # align: convert video and text tokens into model-ingestible tensors
        inputs = self.aligner(video_id, video_tokens, text_tokens)
        outputs.update(inputs)
        
        # add reverse video tokens
        inputs = self.aligner(video_id, video_tokens_swapped, text_tokens)
        outputs.update({"vfeats_swapped": inputs["vfeats"], "vmasks_swapped": inputs["vmasks"]})
        
        if use_first_then:
            if temporal_rel == "before":
                text_swapped = "First, " + text_event_y + ". Then, " + text_event_x + "."
            else:
                text_swapped = "First, " + text_event_x + ". Then, " + text_event_y + "."
        else:
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

        if self.print_example and index == 701:
            print(":::::: Example instance ::::::")
            print("Text:", outputs["text"])
            print("Text swapped:", outputs["text_swapped"])

            print("Event X: ", self.data[index]["event_X"])
            print("Relation:", outputs["temporal_relation"])
            print("Event Y: ", self.data[index]["event_Y"])
            print("::::::::::::::::::::::::::::::")
            print()

        return outputs


def load_dataset(
        data_root,
        video_dir="DiDeMo/videos/",
        feat_dir="DiDeMo/feat/feat_how2_s3d/",
        split_dir="TEMPO/initial_release_data/",
        mode="val", subset="temporal_1k",
    ):
    assert mode in ["train", "val", "test"]
    video_dir = os.path.join(
        data_root, video_dir,
    )
    feat_dir = os.path.join(
        data_root, feat_dir,
    )

    split_dir = os.path.join(
        data_root, split_dir,
    )
    split_path = os.path.join(
        split_dir, f"tempoTL+didemo_{mode}.json",
    )
    if mode == "train":
        # assert subset is None, \
        #     "subset should be None for training"
        subset_path = None
    else:
        assert subset is not None, \
            "subset should not be None for validation/test"
        subset_path = os.path.join(
            split_dir, f"tempoTL+didemo_{mode}_{subset}.csv",
        )
    dataset = TEMPOTL(
        video_dir=None,
        split_path=split_path,
        feat_dir=feat_dir,
        subset_path=subset_path,
        use_time_bdry=True,
    )
    return dataset


if __name__ == "__main__":
    from tqdm import tqdm
    from package.utils.log import print_update

    print_update("> TESTING TEMPO DATASET", color="green")

    data_root = "/ssd/pbagad/datasets/"
    dataset = load_dataset(data_root=data_root, mode="train", subset=None, print_example=True)
    # dataset = load_dataset(data_root=data_root, mode="val", subset="temporal_1k")
    # dataset = load_dataset(data_root=data_root, mode="test", subset="temporal_1k")

    i = 701
    instance = dataset[i]
    assert instance["vfeats"].shape == instance["vfeats_swapped"].shape
    assert instance["vmasks"].shape == instance["vmasks_swapped"].shape
    assert instance["caps"].shape == instance["caps_swapped"].shape
    assert instance["cmasks"].shape == instance["cmasks_swapped"].shape
    
    print(">>> Running a sanity check...")
    data = dataset.data
    for i in tqdm(range(len(data)), desc="Sanity check"):
        d = data[i]

        desc = d["description"]
        X_desc = d["event_X"]["description"]
        X_time = d["event_X"]["time_in_secs"]
        Y_desc = d["event_Y"]["description"]
        Y_time = d["event_Y"]["time_in_secs"]
        tau = d["temporal_preposition"]
        # print(tau, X_time, Y_time)

        if tau == "before":
            assert X_time[1] <= Y_time[0]
        else:
            assert X_time[0] >= Y_time[1]
        
        # check feature existance
        feat_path = dataset.feature_path(d["video"])
        assert os.path.exists(feat_path), f"feature not found at {feat_path}"
        
        start_X, end_X = d["event_X"]["time_in_secs"]
        start_Y, end_Y = d["event_Y"]["time_in_secs"]
        
        if tau == "before":
            assert end_X <= start_Y, f"error in time boundaries at index {i}"
        else:
            assert end_Y <= start_X, f"error in time boundaries at index {i}"

    print_update("< FINISHED TESTING TEMPO DATASET", color="green")
