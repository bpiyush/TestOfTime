"""Zero-shot evaluation on a downstream task for a TACT adapted model."""
import os
from os.path import exists, join, basename, dirname
import pprint
import time

from termcolor import colored
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

from package.utils.log import repo_path
from package.utils.io import load_json, save_json
from package.utils.misc import ignore_warnings
ignore_warnings()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        "Zero-shot evaluation of a given model on a given dataset (task).",
    )
    # model args
    parser.add_argument(
        "--config", type=str,
        default="external/fairseq/examples/MMPT/"\
            "projects/retri/videoclip/test_vtt_zs.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--task", type=str, default="videoqa",
        help="Task to be evaluated upon name.",
        choices=["videoqa", "action_retrieval"],
    )
    parser.add_argument(
        "-c", "--ckpt_path", type=str,
        default=None, help="Ckpt path. This should be a lightning checkpoint.",
    )
    # dataset args
    parser.add_argument(
        "--data_root", required=True, type=str,
        help="Data root directory."
    )
    parser.add_argument(
        "--dataset", type=str, default="agqa",
        choices=["agqa", "ssv2"], help="Dataset name.",
    )
    parser.add_argument(
        "--split", type=str, default=None, help="split",
    )
    # misc args
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--save_dir", type=str, default=join(repo_path, "results"),
        help="Save directory",
    )
    parser.add_argument(
        "--no_save", action="store_true",
        help="Do not save results",
    )
    args = parser.parse_args()
    
    start = time.time()
    
    # pretty print args
    print(">>> Running zero-shot evaluation with the following args:")
    pp = pprint.PrettyPrinter(width=41, compact=True)
    pp.pprint(args)
    
    # load dataset & create dataloader
    sep = ":" * 40
    print(f"\n {sep} >>> Loading dataset <<< {sep}\n")
    additional_args = dict()


    data_root = args.data_root
    if args.dataset == "agqa":
        from package.datasets.agqa import AGQATemporal
        if args.split is None:
            args.split = "test_unbalanced_subset-temporal-v1.0"
        split_file = f"AGQA/splits/{args.split}.csv"
        dataset = AGQATemporal(data_root=data_root, split_file=split_file)
        additional_args.update(dict(num_answer_candidates=2, log_csv=False))
    elif args.dataset == "ssv2":
        from package.datasets.ssv2 import SSv2
        assert args.split in ["validation", "validation_2k", "validation-tmpl-ret-singularity"]
        split_file = f"something-something-v2-{args.split}.json"
        dataset = SSv2(data_root=data_root, split_file=split_file)
    else:
        raise ValueError("Invalid dataset")
    
    if args.debug:
        print(">>> Debug mode ON: using only 200 samples")
        dataset =  Subset(dataset, range(200))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


    # load videoclip model & evaluator
    print(f"\n {sep} >>> Loading model <<< {sep}\n")
    from external.utils_videoclip import load_videoclip_model
    config, model = load_videoclip_model(
        cfg_path=args.config,
    )
    model = model.to('cpu')
    model.eval()
    config.ckpt_path = args.ckpt_path
    config.no_save = args.no_save
    config.dataset = (args.dataset).lower()
    config.batch_size = args.batch_size
    
    # initialize a PL module (evaluator)
    print(f"\n {sep} >>> Loading task evaluator <<< {sep}\n")
    if args.task == "videoqa":
        assert args.dataset in ["nextqa", "agqa"], \
            "You can only use videoqa on NextQA or AGQA."\
                "But you are using it on {}".format(args.dataset)
        from package.evaluators.videoclip_videoqa_mcq import VideoQAMCQ
        evaluator = VideoQAMCQ(config, model, **additional_args)
    
    elif args.task == "action_retrieval":
        assert args.dataset in ["ssv2", "temporal"], \
            "You can only use action retrieval on SSv2 or Temporal."\
                "But you are using it on {}".format(args.dataset)
        from package.evaluators.videoclip_action_retrieval import VideoActionRetrieval
        evaluator = VideoActionRetrieval(config, model, **additional_args)

    else:
        raise ValueError("Invalid task")
    
    # initialize with a checkpoint
    if args.ckpt_path is not None:
        print(colored(f">>> Evaluating checkpoint: {args.ckpt_path}", "magenta"))
        state_dict = torch.load(args.ckpt_path, map_location='cpu')['state_dict']
        evaluator.load_state_dict(state_dict)
        

    print(f"\n {sep} >>> Evaluating <<< {sep}\n")
    # always evaluating with a single GPU for reproducibility
    trainer = pl.Trainer(
        gpus=[0],
    )
    metrics = trainer.validate(evaluator, dataloader)
    
    # save results
    results = {
        "dataset": args.dataset,
        "task": args.task,
        "model": "VideoCLIP",
        "metrics": metrics,
        "script": (basename(__file__)).split(".py")[0],
        "checkpoint": args.ckpt_path,
    }
    ckpt_id = "none"
    if args.ckpt_path is not None:
        ckpt_id = args.ckpt_path.split("test-of-time/")[1].split("/")[0]
    
    if not args.no_save:
        split = "" if args.split is None else "-" + basename(args.split)
        filename = f"{args.dataset}{split}-{args.task}-{ckpt_id}.json"
        save_path = join(
            args.save_dir,
            results['script'],
            filename,
        )
        os.makedirs(dirname(save_path), exist_ok=True)
        save_json(results, save_path)
        print(colored(f">>> Results saved to {save_path}", "green"))
    
    end = time.time()
    print(">>> Total time taken: {:.2f} mins".format((end-start)/60))