from utils import get_ending_index, get_available_folder_name, HyperParameters
from clean_data import SketchesDataset
from model import SketchDecoder

from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch
import numpy as np
import time
import os


def reconstruction_loss(predictions, labels):
    predicted_pen_states = predictions[:, 2:]
    label_pen_states = labels[:, 2:]

    predicted_offsets = predictions[:, :2]
    label_offsets = labels[:, :2]

    pen_state_loss = torch.nn.functional.binary_cross_entropy(
        predicted_pen_states, label_pen_states
    )
    offset_loss = torch.nn.functional.binary_cross_entropy(
        predicted_offsets, label_offsets
    )

    # average loss over MAX_STROKES
    total_loss = -(pen_state_loss + offset_loss) / HyperParameters.MAX_STROKES

    return total_loss


def train_model():
    print("loading data...", end="\r")

    load_path = f"data/processed_{HyperParameters.DATA_CATEGORY}.npz"

    sketches = np.load(load_path, allow_pickle=True, encoding="latin1")["train"]

    sketches_dataset = SketchesDataset(sketches)
    sketches_data_loader = DataLoader(
        sketches_dataset, batch_size=HyperParameters.BATCH_SIZE
    )

    print("loaded data.   ")

    for batch in sketches_data_loader:
        print("input batch size:", batch.size())
        break

    model = SketchDecoder(HyperParameters()).to(HyperParameters.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=HyperParameters.LEARNING_RATE)

    base_folder_name = f"models/decoder_{HyperParameters.DATA_CATEGORY}"
    folder_name = get_available_folder_name(base_folder_name)
    os.makedirs(folder_name, exist_ok=True)

    print_frequency = 1
    save_frequency = 100

    start_time = time.time()

    num_epochs = 5000

    for epoch in range(num_epochs):
        for i, batch in enumerate(sketches_data_loader):
            label = batch.to(HyperParameters.DEVICE)

            optimizer.zero_grad()

            output, _ = model()
            loss = reconstruction_loss(output, label[:, 0, :])
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % print_frequency == 0:
            print(f"[Epoch {epoch + 1}] loss: {loss.item():.2f}")

        if (epoch + 1) % save_frequency == 0:
            file_name = f"epoch_{epoch+1}_loss_{loss.item():.2f}"

            torch.save(
                {
                    "model_class_name": model.__class__.__name__,
                    "hyper_parameters": HyperParameters.state_dict(),
                    "state_dict": model.state_dict(),
                },
                f"{folder_name}/{file_name}.pth",
            )
            with open("models/latest_experiment.txt", "w") as file:
                file.write(f"{folder_name}/{file_name}.pth")

    print(f"Finished Training, spent {time.time() - start_time}s running {num_epochs} epochs")


if __name__ == "__main__":
    train_model()
