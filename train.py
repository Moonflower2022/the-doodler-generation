from utils import (
    get_available_folder_name,
    get_logger,
    HyperParameters,
    log_tensor_detailed_stats,
    safe_divide,
    replace_last,
)
from clean_data import SketchesDataset
from model import SketchDecoder

from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch
import numpy as np
import time
import os
import logging
import argparse


def bivariate_normal_pdf(dx, dy, sigma_x, sigma_y, mu_x, mu_y, rho_xy, logger=None):
    z_x_top = dx - mu_x
    log_tensor_detailed_stats(logger, z_x_top, "z_x_top")
    z_x_bottom = sigma_x
    log_tensor_detailed_stats(logger, z_x_bottom, "z_x_bottom")
    z_x = (safe_divide(z_x_top, z_x_bottom)) ** 2
    log_tensor_detailed_stats(logger, z_x, "z_x")

    z_y_top = dy - mu_y
    log_tensor_detailed_stats(logger, z_y_top, "z_y_top")
    z_y_bottom = sigma_y
    log_tensor_detailed_stats(logger, z_y_bottom, "z_y_bottom")
    z_y = (safe_divide(z_y_top, z_y_bottom)) ** 2
    log_tensor_detailed_stats(logger, z_y, "z_y")

    z_xy_top = (dx - mu_x) * (dy - mu_y)
    log_tensor_detailed_stats(logger, z_xy_top, "z_xy_top")
    z_xy_bottom = sigma_x * sigma_y
    log_tensor_detailed_stats(logger, z_xy_bottom, "z_xy_bottom")
    z_xy_factor = 2 * rho_xy
    log_tensor_detailed_stats(logger, z_xy_factor, "z_xy_factor")
    z_xy = safe_divide(z_xy_top, z_xy_bottom) * z_xy_factor
    log_tensor_detailed_stats(logger, z_xy, "z_xy")

    z = z_x + z_y - z_xy
    log_tensor_detailed_stats(logger, z, "z")

    clipped_z = torch.clamp(z, 1e-5, 1e7)
    log_tensor_detailed_stats(logger, clipped_z, "clipped_z")

    top = torch.exp(safe_divide(-clipped_z, 2 * (1 - rho_xy**2)))
    log_tensor_detailed_stats(logger, top, "top")

    norm = 2 * np.pi * sigma_x * sigma_y * torch.sqrt(1 - rho_xy**2)
    log_tensor_detailed_stats(logger, norm, "bottom")

    return safe_divide(top, norm)


def vectorized_reconstruction_loss(predictions, labels, logger=None):
    predicted_offset_parameters = predictions[:, :, :-3]
    label_offsets = labels[:, :, :2]
    predicted_pen_states = predictions[:, :, -3:]
    label_pen_states = labels[:, :, 2:]

    batch_size = labels.size(0)

    # Vectorized mask creation for valid points
    end_mask = label_pen_states[:, :, 2] == 1
    valid_mask = end_mask == False

    m = HyperParameters.NUM_MIXTURES

    gaussian_mixture_probabilities = bivariate_normal_pdf(
        label_offsets[:, :, 0].unsqueeze(-1).repeat(1, 1, m),
        label_offsets[:, :, 1].unsqueeze(-1).repeat(1, 1, m),
        predicted_offset_parameters[:, :, m : 2 * m],
        predicted_offset_parameters[:, :, 2 * m : 3 * m],
        predicted_offset_parameters[:, :, 4 * m : 5 * m],
        predicted_offset_parameters[:, :, 5 * m : 6 * m],
        predicted_offset_parameters[:, :, 3 * m : 4 * m],
        logger=logger,
    )

    log_tensor_detailed_stats(
        logger, gaussian_mixture_probabilities, "gaussian_mixture_probabilities"
    )

    weighted_gaussian_mixture_probabilities = (
        predicted_offset_parameters[:, :, :m] * gaussian_mixture_probabilities
    )

    log_tensor_detailed_stats(
        logger,
        weighted_gaussian_mixture_probabilities,
        "weighted_gaussian_mixture_probabilities",
    )

    log_weighted_gaussian_mixture_probabilities = torch.log(
        HyperParameters.EPSILON + torch.sum(weighted_gaussian_mixture_probabilities, -1)
    )

    log_tensor_detailed_stats(
        logger,
        log_weighted_gaussian_mixture_probabilities,
        "log_weighted_gaussian_mixture_probabilities",
    )

    offset_loss = -torch.sum(valid_mask * log_weighted_gaussian_mixture_probabilities)

    logits = nn.functional.softmax(predicted_pen_states, dim=-1)

    pen_state_loss = -torch.sum(label_pen_states * torch.log(logits))

    total_loss = (pen_state_loss + offset_loss) / (
        batch_size * HyperParameters.MAX_STROKES
    )

    # Optional logging of loss components
    if logger:
        logger.info(f"Offset Loss: {offset_loss.item():.4f}")
        logger.info(f"Pen State Loss: {pen_state_loss.item():.4f}")
        logger.info(f"Total Loss: {total_loss.item():.4f}")

    return total_loss


def log_gradient_statistics(model, logger):
    """
    log gradient statistics for model parameters
    """
    total_norm = 0
    parameter_norms = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            # Gradient norm
            param_norm = param.grad.detach().norm(2).item()
            total_norm += param_norm**2
            parameter_norms[name] = param_norm

    total_norm = total_norm**0.5

    # Log total gradient norm
    logger.info(f"Total Gradient Norm: {total_norm:.4f}")

    # Log top 5 parameter gradients by norm
    sorted_grads = sorted(parameter_norms.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, norm in sorted_grads:
        logger.info(f"Gradient Norm - {name}: {norm:.4f}")


def train_model(debug, model_path=None):
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

    base_folder_name = (
        f"models/decoder_{HyperParameters.DATA_CATEGORY}"
        if not model_path
        else replace_last(f"{model_path}+", "/", "_")
    )
    folder_name = get_available_folder_name(base_folder_name)
    os.makedirs(folder_name, exist_ok=True)

    logger = get_logger(
        f"{__name__}.info",
        f"{folder_name}/train.log",
        level=logging.INFO if debug else logging.CRITICAL,
    )
    model_logger = get_logger(
        f"{__name__}.model",
        f"{folder_name}/model.log",
        level=logging.INFO if debug else logging.CRITICAL,
    )
    gradient_logger = get_logger(
        f"{__name__}.gradient",
        f"{folder_name}/gradient.log",
        level=logging.INFO if debug else logging.CRITICAL,
    )

    logger.info("Training started")
    model_logger.info("Model logging initialized")
    gradient_logger.info("Gradient logging initialized")

    if model_path:
        info = torch.load(model_path, weights_only=False)

        hyper_parameters = HyperParameters()
        hyper_parameters.input_state(info["hyper_parameters"])

        model = SketchDecoder(hyper_parameters, model_logger, debug=debug).to(
            HyperParameters.DEVICE
        )
        model.load_state_dict(info["state_dict"])
    else:
        model = SketchDecoder(HyperParameters(), model_logger, debug=debug).to(
            HyperParameters.DEVICE
        )
    optimizer = optim.Adam(model.parameters(), lr=HyperParameters.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=HyperParameters.LEARNING_RATE_DECAY
    )

    print_frequency = 1
    save_frequency = 1
    log_frequency = 1
    gradient_log_frequency = 10

    start_time = time.time()

    num_epochs = 5000

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        for i, sketch_batch in enumerate(sketches_data_loader):
            sketch_batch = sketch_batch.to(HyperParameters.DEVICE)

            optimizer.zero_grad()

            outputs, _ = model(sketch_batch)
            loss = vectorized_reconstruction_loss(
                outputs, sketch_batch, logger=model_logger if model.debug else None
            )

            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), HyperParameters.GRAD_CLIP
            )

            optimizer.step()

            for param_group in optimizer.param_groups:
                if param_group["lr"] < HyperParameters.MIN_LEARNING_RATE:
                    param_group["lr"] = HyperParameters.MIN_LEARNING_RATE

        # Gradient logging
        if (epoch + 1) % gradient_log_frequency == 0:
            log_gradient_statistics(model, gradient_logger)

        if (epoch + 1) % print_frequency == 0:
            print(f"[Epoch {epoch + 1}] loss: {loss.item():.2f}")

        if (epoch + 1) % log_frequency == 0:
            logger.info(f"EPOCH {epoch}")
            logger.info(f"Loss: {loss.item():.2f}")
            logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
            logger.info(f"Epoch Training Time: {time.time() - epoch_start_time:.2f}s")
            logger.info(f"Total Training Time: {time.time() - start_time:.2f}s")

        if (epoch + 1) % save_frequency == 0:
            save_filename = f"{folder_name}/epoch_{epoch}_loss_{loss.item():.2f}.pth"
            torch.save(
                {
                    "model_class_name": model.__class__.__name__,
                    "hyper_parameters": HyperParameters.state_dict(),
                    "state_dict": model.state_dict(),
                },
                save_filename,
            )
            with open("models/latest_experiment.txt", "w") as output_file:
                output_file.write(save_filename)

        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a decoder")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "-l",
        "--load",
        type=str,
        metavar="MODEL_PATH",
        help="Specify the model path to load and train.",
    )

    args = parser.parse_args()

    if args.debug:
        print("Debug mode enabled.")

    model_path = args.load

    train_model(args.debug, model_path=model_path)
