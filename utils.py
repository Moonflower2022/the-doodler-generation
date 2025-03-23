import torch
import os
import logging

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
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


class HyperParameters:
    DATA_CATEGORY = "apple"
    DEVICE = device
    INPUT_SIZE = 5
    # ∆x,
    # ∆y,
    # pen down (pen is currently down),
    # pen up (after this stroke, pen goes up),
    # end (current point and subsequent points are voided)
    HIDDEN_SIZE = 512
    BIAS = True
    MAX_STROKES = 63
    LEARNING_RATE = 1e-3
    LEARNING_RATE_DECAY = 1 # 0.9999
    MIN_LEARNING_RATE = 0.00001
    BATCH_SIZE = 200
    DROPOUT = 0.9
    LATENT_VECTOR_SIZE = 128

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

if __name__ == "__main__":
    import numpy as np

    load_path = f"data/{HyperParameters.DATA_CATEGORY}.npz"
    drawings = np.load(load_path, allow_pickle=True, encoding="latin1")

    print(get_max_strokes(drawings["train"]))
