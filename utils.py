import torch
import os
import logging

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def safe_exp(x, max_val=10.0):
    return torch.exp(torch.clamp(x, max=max_val))


def safe_divide(numerator, denominator):
    return (
        numerator
        / (torch.abs(denominator) + HyperParameters.EPSILON)
        * torch.sign(denominator)
    )


def get_ending_index(sketch):
    for i, stroke in enumerate(sketch):
        if stroke[-1] == 1:
            return i
    return -1


def get_available_folder_name(base_name, directory="."):
    # Start checking from the original folder name
    available_name = base_name
    counter = 1

    # Keep incrementing the counter and checking for folder existence
    while os.path.exists(os.path.join(directory, available_name)):
        available_name = f"{base_name}_{counter}"
        counter += 1

    return available_name


def get_max_strokes(data):
    return max([len(sketch) for sketch in data]) + 1


def replace_last(s, old, new):
    parts = s.rsplit(old, 1)  # Split from the right, at most once
    return new.join(parts)


class HyperParameters:
    DATA_CATEGORY = "octopus"
    DEVICE = device

    # MODEL ARCHITECTURE (cannot change while training the same model)
    INPUT_SIZE = 5
    # ∆x,
    # ∆y,
    # pen down (pen is currently down),
    # pen up (after this stroke, pen goes up),
    # end (current point and subsequent points are voided)
    HIDDEN_SIZE = 512
    BIAS = True
    MAX_STROKES = 132
    NUM_MIXTURES = 20
    LATENT_VECTOR_SIZE = 128  # not used

    # HYPER PARAMETERS (can change while training the same model)
    LEARNING_RATE = 1e-3
    LEARNING_RATE_DECAY = 0.996
    MIN_LEARNING_RATE = 1e-5
    BATCH_SIZE = 100
    DROPOUT = 0.0
    GRAD_CLIP = 1.0
    EPSILON = 1e-5

    def state_dict():
        return {
            key.lower(): value
            for key, value in vars(HyperParameters).items()
            if not key.startswith("__") or key == "state_dict"
        }

    def input_state(self, state):
        for key, value in state.items():
            # Update the instance attribute if it exists
            if hasattr(self, key):
                setattr(self, key, value)


LOG_FORMAT = "%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d): %(message)s"
formatter = logging.Formatter(LOG_FORMAT)


def get_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.FileHandler(filename=log_file)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def log_tensor_detailed_stats(logger, tensor, tensor_name="", level=logging.INFO):
    """Provide comprehensive tensor statistics."""
    if logger:
        if tensor is None:
            logger.warning(f"{tensor_name}: Tensor is None!")
            return

        # tensor stats
        stats = {
            "shape": tensor.shape,
            "dtype": tensor.dtype,
            "min": tensor.min().item() if tensor.numel() > 0 else "N/A",
            "max": tensor.max().item() if tensor.numel() > 0 else "N/A",
            "mean": tensor.mean().item() if tensor.numel() > 0 else "N/A",
            "std": tensor.std().item() if tensor.numel() > 0 else "N/A",
            "nan_count": torch.isnan(tensor).sum().item(),
            "inf_count": torch.isinf(tensor).sum().item(),
        }

        log_message = f"{tensor_name} Tensor Statistics:\n" + "\n".join(
            f"  {k}: {v}" for k, v in stats.items()
        )

        logger.log(level, log_message)

        # error checking
        if stats["nan_count"] > 0:
            logger.critical(f"NaN DETECTED in {tensor_name}!")

        if stats["inf_count"] > 0:
            logger.critical(f"Inf DETECTED in {tensor_name}!")


if __name__ == "__main__":
    import numpy as np

    load_path = f"data/{HyperParameters.DATA_CATEGORY}.npz"
    drawings = np.load(load_path, allow_pickle=True, encoding="latin1")

    print(get_max_strokes(drawings["train"]))
