"""PyTorch Lightning training script for the midtraining task with weighted contrastive loss."""

import os

import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from termcolor import colored

from external.utils_videoclip import load_videoclip_model
import package.datasets as datasets
import package.models as models
from package.utils.log import print_update

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def check_args(args):
    """Checks validity of arguments."""
    
    if args.only_eval:
        args.only_train = False
    if args.only_train:
        args.only_eval = False
    
    if not args.only_eval:
        assert args.split is None, \
            "No need to specify --split for training. Only useful for evaluation."
        args.split = "train"
    
    if args.dataset == "synthetic":
        assert args.split == "test", \
            f"Not a valid split(={args.split}) for synthetic dataset."\
    
    # if args.split == "train":
    #     assert args.subset is None, \
    #         "--subset should be None for training."
    
    if args.dataset == "synthetic" and args.split == "test":
        assert args.subset in ["v2.0"], \
            f"Not a valid subset(={args.subset}) for synthetic dataset."\
                "Only --subset v2.0 is supported."
    
    if args.dataset == "tempo" and args.split in ["val", "test"]:
        assert args.subset in ["temporal_1k"], \
            f"Not a valid subset(={args.subset}) for tempo dataset."\
                "Only --subset temporal_1k is supported."
    
    if args.gpus is None:
        if torch.cuda.is_available():
            args.gpus = torch.cuda.device_count()
        else:
            args.gpus = None
    
    return args
    

def update_config(config, args):
    config.lr = args.lr
    # config.contrastive_lambda = args.w_contrastive
    config.contrastive_lambda = 1.0
    # config.temporal_lambda = args.w_temporal
    config.batch_size = args.batch_size
    config.epoch = args.epochs
    config.freeze_layers = args.freeze_layers
    config.video_freeze_layers = args.video_freeze_layers
    config.text_freeze_layers = args.text_freeze_layers
    # config.no_reverse = args.no_reverse
    config.alpha_same = args.alpha_same
    config.alpha_cross = args.alpha_cross
    config.beta = args.beta
    return config


def freeze_required_layers(model, args):

    # freeze layers (these layers are frozen by default)
    modules_to_freeze = [
        model.video_encoder.bert.encoder.layer[:args.video_freeze_layers],
        model.text_encoder.encoder.layer[:args.text_freeze_layers],
        model.video_encoder.bert.embeddings.word_embeddings,
        model.text_encoder.embeddings.word_embeddings,
    ]

    if args.freeze_videomlp:
        print(">>> Freezing video MLP")
        modules_to_freeze += [model.video_encoder.videomlp]

    if args.freeze_pooler:
        print(">>> Freezing pooler for video/text")
        modules_to_freeze += [
            model.video_encoder.bert.pooler,
            model.text_encoder.pooler,
        ]

    if args.freeze_pos_emb:
        print("\n>>> Freezing positional embeddings")
        modules_to_freeze.extend([
            model.video_encoder.bert.embeddings.position_embeddings,
            model.text_encoder.embeddings.position_embeddings,
        ])

        if args.remove_pos_emb:
            print(">>> Removing positional embeddings")
            # also set them to 0
            model.video_encoder.bert.embeddings.position_embeddings.weight.data.fill_(0)
            model.text_encoder.embeddings.position_embeddings.weight.data.fill_(0)
            print(">>> Positional embeddings set to 0")
            print(model.video_encoder.bert.embeddings.position_embeddings.weight.data)

    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False
    
    # sanity check
    print(">>> Parameters to train:")
    for name, params in model.named_parameters():
        if params.requires_grad:
            print(name, params.shape)
    
    return model


if __name__ == "__main__":
    
    # read arguments
    import argparse
    parser = argparse.ArgumentParser("Train a model")
    
    # Model args
    parser.add_argument(
        "--model", type=str, default="videoclip",
        choices=["videoclip"],
    )
    parser.add_argument(
        "--config", type=str,
        default="external/fairseq/examples/MMPT/"\
            "projects/retri/videoclip/test_vtt_zs.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--freeze_videomlp", action="store_true",
        help="Freeze video MLP",
    )
    parser.add_argument(
        "--freeze_pooler", action="store_true",
        help="Freeze pooler for both video/text",
    )
    parser.add_argument(
        "--freeze_pos_emb", action="store_true",
        help="Freeze positional embeddings or not",
    )
    parser.add_argument(
        "--freeze_layers", type=int, default=5,
        help="Number of layers to freeze",
    )
    parser.add_argument(
        "--video_freeze_layers", type=int, default=5,
        help="Number of layers to freeze in video transformer",
    )
    parser.add_argument(
        "--text_freeze_layers", type=int, default=5,
        help="Number of layers to freeze in text transformer",
    )
    parser.add_argument(
        "--alpha_same", type=float, default=1.0,
        help="Alpha for same-sample time-reversal",
    )
    parser.add_argument(
        "--alpha_cross", type=float, default=1.0,
        help="Alpha for cross-sample time-reversal",
    )
    parser.add_argument(
        "--beta", type=float, default=1.0,
        help="Beta for the contrastive loss",
    )
    parser.add_argument(
        "--remove_pos_emb", action="store_true",
        help="Remove positional embeddings at all",
    )
    parser.add_argument(
        "-c", "--ckpt_path", type=str,
        default=None,
        help="Path to checkpoint (only used for evaluation)",
    )

    # Dataset args
    parser.add_argument(
        "--dataset", type=str, default="tempo",
        help="Dataset name", choices=["tempo", "synthetic"],
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
    )
    parser.add_argument(
        "--split", type=str, default=None,
        help="Split name", choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--subset", type=str, default=None,
    )

    # Optimization and other args
    parser.add_argument(
        "--lr", type=float, default=5.0e-06,
        help="Learning rate",
    )
    parser.add_argument(
        "--gpus", type=int, default=None, nargs="+",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs", type=int, default=40,
        help="Number of epochs",
    )
    parser.add_argument(
        "--no_wandb", action="store_true",
        help="Force not to use wandb",
    )
    parser.add_argument(
        "--overfit_batches", type=float, default=0.0,
        help="Overfit batches",
    )
    parser.add_argument(
        "--suffix", type=str, default="",
        help="Suffix to add to the name of the run",
    )
    parser.add_argument(
        "--save_every", type=int, default=10,
        help="Save every n epochs",
    )
    parser.add_argument(
        "--only_eval", action="store_true",
        help="Only evaluate the model",
    )
    parser.add_argument(
        "--only_train", action="store_true",
        help="Only train the model",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode",
    )
    args = parser.parse_args()
    
    args = check_args(args)


    # 1. Load the datasets
    dataset_load_function = getattr(datasets, f"load_{args.dataset}_dataset")
    dataset_load_args = dict(
        data_root=args.data_root,
    )
    if not args.only_eval:
        print_update(">>> Loading train set")
        # 1.A. Load train set (only if not only_eval)
        dataset_load_args.update(dict(mode="train"))
        train_dataset = dataset_load_function(**dataset_load_args)
    # 1.B. Load val set
    dataset_load_args.update(dict(mode=args.split, subset=args.subset))
    valid_dataset = dataset_load_function(**dataset_load_args)
    
    if args.debug:
        train_dataset = Subset(train_dataset, range(1000))
        val_dataset = Subset(valid_dataset, range(500))

    # 1.C. Load the dataloaders
    if not args.only_eval:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )


    # 2. Load the model
    # 2.A. Load the base VideoCLIP model
    config, model = load_videoclip_model(
        cfg_path=args.config,
    )
    config = update_config(config, args)
    model = freeze_required_layers(model, args)
    
    # 2.B. Load PL module
    model_loading_function = getattr(models, "VideoCLIP")
    pl_module = model_loading_function(config, model)


    # 3. Run the experiment (train/eval)
    # 3.A. Setup logging/other cosmetics
    log = not args.no_wandb
    logger = None
    if log:
        run_name = f"ppt-{args.model}-{args.dataset}-bs_{args.batch_size}"\
            f"-frozen-lr_{args.lr}-ep{args.epochs}"
        run_name += "-overfit" if args.overfit_batches > 0 else ""
        run_name += "-" + args.suffix
        run_name += "-alpha_same_" + str(args.alpha_same) \
            + "-alpha_cross_" + str(args.alpha_cross) \
                + "-beta_" + str(args.beta)
        print("WARNING: If you need to log to W&B, "\
            "you need to change entity & project.")
        logger = pl_loggers.WandbLogger(
            project="test-of-time",
            entity="bpiyush",
            name=run_name,
        )
    callbacks = []
    if not args.only_eval:
        # 3.B. Save the model every 5 epochs
        save_every_k_epochs = ModelCheckpoint(
            every_n_epochs=args.save_every,
            save_top_k=-1,
            save_last=True,
        )
        callbacks.append(save_every_k_epochs)

    # 3.C. Define the trainer
    trainer = pl.Trainer(
        logger=logger,
        gpus=args.gpus,
        max_epochs=args.epochs,
        log_every_n_steps=2,
        callbacks=callbacks,
        overfit_batches=args.overfit_batches,
    )
    
    # 3.D. Load the checkpoint if required
    if args.ckpt_path is not None:
        print(
            colored(
                f">>> Initializing with checkpoint: {args.ckpt_path}",
                "magenta"
            )
        )
        state_dict = torch.load(args.ckpt_path, map_location='cpu')['state_dict']
        pl_module.load_state_dict(state_dict)
    
    # 3.E. Run the evaluation (before training)
    if not args.only_train:
        trainer.validate(pl_module, dataloaders=valid_dataloader)
    
    # 3.F. Run the training
    if not args.only_eval:
         trainer.fit(
            model=pl_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )
