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


def log_normal_pdf(mu, sigma, x):
    return -0.5 * (
        torch.log(torch.tensor(2 * torch.pi))
        + 2 * torch.log(sigma)
        + torch.pow(x - mu, 2) / torch.pow(sigma, 2)
    )


def reconstruction_loss(predictions, labels):
    predicted_offset_distributions = predictions[:, :4]
    label_offsets = labels[:, :2]

    predicted_pen_states = predictions[:, 4:]
    label_pen_states = labels[:, 2:]

    # Find where [0,0,0,0,1] occurs - looking for the first occurrence of pen_state[4] == 1
    end_mask = label_pen_states[:, 2] == 1  # Find where fifth state is 1
    end_indices = torch.where(end_mask)[0]  # Get indices where True

    if len(end_indices) > 0:
        N_s = end_indices[0]  # Take first occurrence
    else:
        N_s = len(labels)  # If no end token, use full sequence

    offset_loss = -torch.sum(
        log_normal_pdf(
            predicted_offset_distributions[:N_s, 0],
            predicted_offset_distributions[:N_s, 1],
            label_offsets[:N_s, 0],
        )
        + log_normal_pdf(
            predicted_offset_distributions[:N_s, 2],
            predicted_offset_distributions[:N_s, 3],
            label_offsets[:N_s, 1],
        )
    )

    pen_state_loss = -torch.sum(label_pen_states * torch.log(predicted_pen_states))

    # average loss over MAX_STROKES
    total_loss = (pen_state_loss + offset_loss) / HyperParameters.MAX_STROKES

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
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=HyperParameters.LEARNING_RATE_DECAY)

    base_folder_name = f"models/decoder_{HyperParameters.DATA_CATEGORY}"
    folder_name = get_available_folder_name(base_folder_name)
    os.makedirs(folder_name, exist_ok=True)

    print_frequency = 1
    save_frequency = 100

    start_time = time.time()

    num_epochs = 5000

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in sketches_data_loader:
            batch = batch.to(HyperParameters.DEVICE)
            
            optimizer.zero_grad()
            
            # Get model predictions (PDF parameters and pen states)
            predictions, hidden_cell = model()
            
            # Get target
            target = batch[:, 0, :]  # First timestep as target
            
            # Compute loss using the PDF parameters
            loss = reconstruction_loss(predictions, target)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            # scheduler.step()

            # # Clamp learning rate if it goes below min_lr
            # for param_group in optimizer.param_groups:
            #     if param_group['lr'] < HyperParameters.MIN_LEARNING_RATE:
            #         param_group['lr'] = HyperParameters.MIN_LEARNING_RATE
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        
        if (epoch + 1) % print_frequency == 0:
            print(f"[Epoch {epoch + 1}] loss: {avg_epoch_loss:.2f}")
            
        if (epoch + 1) % save_frequency == 0:
            torch.save(
                {
                    "model_class_name": model.__class__.__name__,
                    "hyper_parameters": HyperParameters.state_dict(),
                    "state_dict": model.state_dict(),
                },
                f"models/decoder_{HyperParameters.DATA_CATEGORY}/epoch_{epoch+1}_loss_{avg_epoch_loss:.2f}.pth",
            )
    print(f"finished training in {start_time - time.time()}s")

if __name__ == "__main__":
    train_model()