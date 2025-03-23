from utils import get_available_folder_name, get_logger, HyperParameters
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
    predicted_offset_distributions = predictions[:, :, :4]
    label_offsets = labels[:, :, :2]

    predicted_pen_states = predictions[:, :, 4:]
    label_pen_states = labels[:, :, 2:]

    end_mask = label_pen_states[:, :, 2] == 1

    batch_size = labels.size(0)
    N_s = torch.zeros(batch_size, dtype=torch.long, device=labels.device)

    # find where the sketches stop
    for b in range(batch_size):
        end_indices = torch.where(end_mask[b])[0]
        if len(end_indices) > 0:
            N_s[b] = end_indices[0]
        else:
            N_s[b] = len(labels[b])

    offset_loss = torch.zeros(1, device=labels.device)
    pen_state_loss = torch.zeros(1, device=labels.device)

    for b in range(batch_size):
        offset_loss += -torch.sum(
            log_normal_pdf(
                predicted_offset_distributions[b, : N_s[b], 0],
                predicted_offset_distributions[b, : N_s[b], 1],
                label_offsets[b, : N_s[b], 0],
            )
            + log_normal_pdf(
                predicted_offset_distributions[b, : N_s[b], 2],
                predicted_offset_distributions[b, : N_s[b], 3],
                label_offsets[b, : N_s[b], 1],
            )
        )

        logits = nn.functional.softmax(predicted_pen_states[b], dim=1)
        pen_state_loss += -torch.sum(label_pen_states[b] * torch.log(logits))

    total_loss = (pen_state_loss + offset_loss) / (
        batch_size * HyperParameters.MAX_STROKES
    )
    return total_loss


def train_model():
    print("loading data:", end="\r")

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
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=HyperParameters.LEARNING_RATE_DECAY
    )

    base_folder_name = f"models/decoder_{HyperParameters.DATA_CATEGORY}"
    folder_name = get_available_folder_name(base_folder_name)
    os.makedirs(folder_name, exist_ok=True)

    logger = get_logger(f"{__name__}.info", f"{folder_name}/train.log")
    io_logger = get_logger(f"{__name__}.io", f"{folder_name}/input_output.log")

    logger.info("Training started")
    io_logger.info("IO logging initialized")

    print_frequency = 1
    save_frequency = 1
    log_frequency = 1

    start_time = time.time()

    num_epochs = 5000

    for epoch in range(num_epochs):
        for i, sketch_batch in enumerate(sketches_data_loader):
            sketch_batch = sketch_batch.to(HyperParameters.DEVICE)

            optimizer.zero_grad()

            outputs, _ = model(sketch_batch)
            loss = reconstruction_loss(outputs, sketch_batch)

            loss.backward()
            optimizer.step()
            
            for param_group in optimizer.param_groups:
                if param_group["lr"] < HyperParameters.MIN_LEARNING_RATE:
                    param_group["lr"] = HyperParameters.MIN_LEARNING_RATE

        if (epoch + 1) % print_frequency == 0:
            print(f"[Epoch {epoch + 1}] loss: {loss.item():.2f}")

        if (epoch + 1) % log_frequency == 0:
            io_logger.info(f"inputs[0]: {sketch_batch[0]}")
            io_logger.info(f"outputs[0]: {outputs[0]}")
            logger.info(f"EPOCH {epoch}")
            logger.info(f"loss: {loss.item():.2f}")
            logger.info(f"learning rate: {optimizer.param_groups[0]["lr"]}")
            logger.info(f"training time so far: {time.time() - start_time}s")

        if (epoch + 1) % save_frequency == 0:
            save_filename = f"{folder_name}/epoch_{epoch+1}_loss_{loss.item():.2f}.pth"
            torch.save(
                {
                    "model_class_name": model.__class__.__name__,
                    "hyper_parameters": HyperParameters.state_dict(),
                    "state_dict": model.state_dict(),
                },
                save_filename
            )
            with open("models/latest_experiment.txt", "w") as output_file:
                output_file.write(save_filename)

        scheduler.step()


if __name__ == "__main__":
    train_model()
