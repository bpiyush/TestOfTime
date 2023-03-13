"""Loggers."""
import os
from os.path import dirname, realpath
from tqdm import tqdm
import numpy as np
from termcolor import colored


repo_path = dirname(dirname(dirname(realpath(__file__))))

def tqdm_iterator(items, desc=None, bar_format=None, **kwargs):
    tqdm._instances.clear()
    iterator = tqdm(
        items,
        desc=desc,
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
        **kwargs,
    )
    
    return iterator


def print_retrieval_metrics_for_csv(metrics, scale=100):
    print_string = [
        np.round(scale * metrics["R1"], 3),
        np.round(scale * metrics["R5"], 3),
        np.round(scale * metrics["R10"], 3),
    ]
    if "MR" in metrics:
        print_string += [metrics["MR"]]
    print()
    print("Final metrics: ", ",".join([str(x) for x in print_string]))
    print()


def color(string: str, color_name: str = 'yellow') -> str:
    """Returns colored string for output to terminal"""
    return colored(string, color_name)


def print_update(message: str, width: int = 140, fillchar: str = ":", color="yellow") -> str:
    """Prints an update message
    Args:
        message (str): message
        width (int): width of new update message
        fillchar (str): character to be filled to L and R of message
    Returns:
        str: print-ready update message
    """
    terminal_width = os.get_terminal_size().columns
    width = min(terminal_width, width)
    
    message = message.center(len(message) + 2, " ")
    print(colored(message.center(width, fillchar), color))


if __name__ == "__main__":
    print("Repo path:", repo_path)