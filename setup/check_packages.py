"""Checks installed packages."""

packages = [
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "pytorch_lightning",
    "wandb"
]

print("-" * 80)
for package in packages:
    try:
        __import__(package)
        print("Package {} is installed with {}.".format(package, __import__(package).__version__))
        print("-" * 80)
    except ImportError:
        raise ImportError(
            f"Please install {package} to run this example."
        )