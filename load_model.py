from model import SketchDecoder as ModelClass
from utils import HyperParameters
from clean_data import draw_sketch
import torch

if __name__ == "__main__":
    with open("models/latest_experiment.txt", "r") as file:
        model_path = file.read()

    info = torch.load(model_path)

    hyper_parameters = HyperParameters()
    hyper_parameters.input_state(info["hyper_parameters"])
    hyper_parameters.DEVICE = "cpu"

    model = ModelClass(hyper_parameters)
    model.load_state_dict(info["state_dict"])

    output = model()
    print(output)
    draw_sketch(output.detach().numpy())
