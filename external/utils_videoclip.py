"""Helper functions to access VideoCLIP model and its data processors."""

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


if __name__ == "__main__":
    from package.utils.log import print_update

    print_update("TEST 1: Loading VideoCLIP model")
    cfg_path = "external/fairseq/examples/MMPT/projects/retri/videoclip/test_vtt_zs.yaml"
    checkpoint_path="external/requirements/fairseq/runs/retri/videoclip/checkpoint_best.pt"
    config, model = load_videoclip_model(
        cfg_path=cfg_path,
        checkpoint_path=checkpoint_path,
    )
    print_update("TEST 1 FINISHED: Successful!")
