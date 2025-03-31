from model import SketchDecoder as ModelClass
from utils import HyperParameters
from clean_data import draw_sketch

import torch
import sys
import os
import argparse

def load_model(model_path=None):
    if model_path is None:
        with open("models/latest_experiment.txt", "r") as file:
            model_path = file.read()

    info = torch.load(model_path, weights_only=False)

    hyper_parameters = HyperParameters()
    hyper_parameters.input_state(info["hyper_parameters"])
    hyper_parameters.DEVICE = "cpu"

    model = ModelClass(hyper_parameters)
    model.load_state_dict(info["state_dict"])

    outputs = []

    for i in range(hyper_parameters.MAX_STROKES):
        output, hidden_cell = model.generate_stroke(
            last_stroke=None if i == 0 else outputs[-1],
            hidden_cell=None if i == 0 else hidden_cell,
        )
        outputs.append(output)
        if torch.argmax(output[2:]) == 2:
            break

    torchized_outputs = torch.stack(outputs, dim=1).T
    print(torchized_outputs.size())
    print(torchized_outputs)
    draw_sketch(torchized_outputs.detach().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="load a decoder"
    )
    parser.add_argument(
        "-l",
        "--load",
        type=str,
        metavar="MODEL_PATH",
        help="Specify the model path to load.",
    )
    args = parser.parse_args()

    model_path = args.load
    load_model(model_path=model_path)
    