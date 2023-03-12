"""Misc utils."""
import os


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        

def ignore_warnings(type="ignore"):
    import warnings
    warnings.filterwarnings(type)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
