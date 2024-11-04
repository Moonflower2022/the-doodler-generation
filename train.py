from utils import get_ending_index, get_available_folder_name, HyperParameters
from clean_data import SketchesDataset
from model import SketchDecoder
from torch import nn
from torch import optim
import torch
import numpy as np
import os

def reconstruction_loss(prediction, label):
    
    # ending_index is the last index up to which offset_loss will be applied.
    ending_index = get_ending_index(label)
    
    # Separate pen states and offsets for predictions and labels
    pred_pen_states = prediction[:, 2:]
    label_pen_states = label[:, 2:]
    
    pred_offsets = prediction[:, :2]
    label_offsets = label[:, :2]
    
    # Calculate pen state loss
    pen_state_loss = torch.nn.functional.binary_cross_entropy(pred_pen_states, label_pen_states, reduction='sum')
    
    # Apply mask for offset loss
    mask = torch.arange(label.size(0), device=label.device) <= ending_index
    offset_loss = torch.nn.functional.binary_cross_entropy(pred_offsets[mask], label_offsets[mask], reduction='sum')
    
    # Average loss over MAX_STROKES for normalization
    total_loss = -(pen_state_loss + offset_loss) / HyperParameters.MAX_STROKES
    
    return total_loss



def train_model():
    print("loading data...", end="\r")
    category = "airplane"

    load_path = f"data/processed_{category}.npz"

    sketches = np.load(load_path, allow_pickle=True, encoding='latin1')["train"]


    sketches_dataset = SketchesDataset(sketches)
    print("loaded data.   ")

    for sketch in sketches_dataset:
        print(sketch.size())
        break

    model = SketchDecoder(HyperParameters()).to(HyperParameters.DEVICE)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=HyperParameters.LEARNING_RATE)

    with open("models/latest_experiment.txt", "w") as file:
        file.write("")

    base_folder_name = f'models/decoder'
    folder_name = get_available_folder_name(base_folder_name)
    os.makedirs(folder_name, exist_ok=True)

    print_frequency = 100

    for epoch in range(60):
        for i, data in enumerate(sketches_dataset):
            print(epoch, i)
            label = data.to(HyperParameters.DEVICE)

            optimizer.zero_grad()

            output = model()
            loss = reconstruction_loss(output, label)
            loss.backward()
            optimizer.step()

            if (i + 1) % print_frequency == 0:    # print every [print_frequency] mini-batches
                print(f'[{epoch + 1}, {i + 1}] loss: {loss.item():.3f}')
            
        file_name = f"epoch_{epoch+1}_loss_{loss.item():.4f}"

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
            
    print('Finished Training')

if __name__ == '__main__':
    train_model()