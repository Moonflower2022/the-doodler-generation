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

    outputs = []

    for i in range(hyper_parameters.MAX_STROKES):
        output, hidden_cell = model.generate_stroke(
            last_stroke=None if i == 0 else outputs[-1],
            hidden_cell=None if i == 0 else hidden_cell,
        )
        if torch.argmax(output[2:]) == 2:
            break
        outputs.append(output)

    torchized_outputs = torch.stack(outputs, dim=1).T
    print(torchized_outputs.size())
    print(torchized_outputs)
    draw_sketch(torchized_outputs.detach().numpy())
