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

torch.backends.cudnn.benchmark = True

log_2pi = torch.log(torch.tensor(2 * torch.pi))


def bivariate_normal_pdf(dx, dy, sigma_x, sigma_y, mu_x, mu_y, rho_xy):
    z_x = ((dx-mu_x)/sigma_x)**2
    z_y = ((dy-mu_y)/sigma_y)**2
    z_xy = (dx-mu_x)*(dy-mu_y)/(sigma_x*sigma_y)
    z = z_x + z_y - 2*rho_xy*z_xy
    exp = torch.exp(-z/(2*(1-rho_xy**2)))
    norm = 2*np.pi*sigma_x*sigma_y*torch.sqrt(1-rho_xy**2)
    return exp/norm


def vectorized_reconstruction_loss(predictions, labels):
    predicted_offset_parameters = predictions[:, :, :-3]
    label_offsets = labels[:, :, :2]
    predicted_pen_states = predictions[:, :, -3:]
    label_pen_states = labels[:, :, 2:]

    batch_size = labels.size(0)
    seq_length = labels.size(1)
    device = labels.device

    # Vectorized mask creation for valid points
    end_mask = label_pen_states[:, :, 2] == 1

    # Create indices tensor for each sequence position
    position_indices = torch.arange(
        seq_length, device=device).unsqueeze(0).expand(batch_size, -1)

    # Find the first occurrence of end_mask for each batch
    # If no end marker, use the max sequence length
    has_end = torch.any(end_mask, dim=1)
    first_end_indices = torch.argmax(end_mask.long(), dim=1)
    # For sequences without end marker, set to seq_length
    first_end_indices = torch.where(
        has_end,
        first_end_indices,
        torch.tensor(seq_length, device=device).expand(batch_size)
    )

    # Create valid mask: positions less than the first end index
    valid_mask = position_indices < first_end_indices.unsqueeze(1)

    m = HyperParameters.NUM_MIXTURES

    gaussian_mixture_probabilities = bivariate_normal_pdf(
        label_offsets[:, :, 0].unsqueeze(-1).repeat(1, 1, m), 
        label_offsets[:, :, 1].unsqueeze(-1).repeat(1, 1, m), 
        predicted_offset_parameters[:, :, m:2 * m],
        predicted_offset_parameters[:, :, 2 * m:3 * m],
        predicted_offset_parameters[:, :, 4 * m:5 * m],
        predicted_offset_parameters[:, :, 5 * m:6 * m],
        predicted_offset_parameters[:, :, 3 * m:4 * m],
    )
    weighted_gaussian_mixture_probabilities = predicted_offset_parameters[:, :, :m] * gaussian_mixture_probabilities

    epsilon = 1e-5

    offset_loss = -torch.sum(torch.log(torch.sum(valid_mask.unsqueeze(2) * weighted_gaussian_mixture_probabilities, 2) + epsilon))


    logits = nn.functional.softmax(predicted_pen_states, dim=-1)

    pen_state_loss = -torch.sum(label_pen_states * torch.log(logits + epsilon))

    total_loss = (pen_state_loss + offset_loss) / (batch_size * HyperParameters.MAX_STROKES)
    return total_loss


def train_model():
    print("loading data:", end="\r")

    load_path = f"data/processed_{HyperParameters.DATA_CATEGORY}.npz"

    sketches = np.load(load_path, allow_pickle=True,
                       encoding="latin1")["train"]

    sketches_dataset = SketchesDataset(sketches)
    sketches_data_loader = DataLoader(
        sketches_dataset, batch_size=HyperParameters.BATCH_SIZE)

    print("loaded data.   ")

    for batch in sketches_data_loader:
        print("input batch size:", batch.size())
        break


    base_folder_name = f"models/decoder_{HyperParameters.DATA_CATEGORY}"
    folder_name = get_available_folder_name(base_folder_name)
    os.makedirs(folder_name, exist_ok=True)

    logger = get_logger(f"{__name__}.info", f"{folder_name}/train.log")
    model_logger = get_logger(f"{__name__}.model", f"{folder_name}/model.log")

    logger.info("Training started")
    model_logger.info("Model logging initialized")

    model = SketchDecoder(HyperParameters(), model_logger, debug=False).to(HyperParameters.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=HyperParameters.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=HyperParameters.LEARNING_RATE_DECAY
    )

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
            loss = vectorized_reconstruction_loss(outputs, sketch_batch)

            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), HyperParameters.GRAD_CLIP)

            for param_group in optimizer.param_groups:
                if param_group["lr"] < HyperParameters.MIN_LEARNING_RATE:
                    param_group["lr"] = HyperParameters.MIN_LEARNING_RATE

        if (epoch + 1) % print_frequency == 0:
            print(f"[Epoch {epoch + 1}] loss: {loss.item():.2f}")

        if (epoch + 1) % log_frequency == 0:
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
