import torch
import os

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
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

class HyperParameters:
    DEVICE = device
    INPUT_SIZE = 5
    # ∆x,
    # ∆y,
    # pen down (pen is currently down),
    # pen up (after this stroke, pen goes up),
    # end (current point and subsequent points are voided)

    HIDDEN_SIZE = 256
    BIAS = True
    MAX_STROKES = 2016

    def state_dict():
        return {key.lower(): value for key, value in vars(HyperParameters).items() if not key.startswith('__') or key == 'state_dict'}