import torch
from utils import HyperParameters, get_logger
from torch import nn
import logging

torch.autograd.set_detect_anomaly(True)

def safe_exp(x, max_val=88.0):
    return torch.exp(torch.clamp(x, max=max_val))

def sample_bivariate_normal(sigma_x, sigma_y, rho_xy, mu_x, mu_y):
    mean = torch.tensor([mu_x, mu_y])
    covariance_matrix = torch.tensor([
        [sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],
        [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]
    ])

    mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix)
    
    samples = mvn.sample((1,))
    
    return samples[0]


class SketchDecoder(nn.Module):
    def __init__(self, hyper_parameters, logger=None, debug=False):
        super(SketchDecoder, self).__init__()
        self.hyper_parameters = hyper_parameters
        self.debug = debug
        
        # set up logging
        self.logger = logger or get_logger('SketchDecoder')
        
        self.log_model_configuration()

        # neural network layers
        self.lstm = nn.LSTM(5, hidden_size=hyper_parameters.HIDDEN_SIZE, batch_first=True)

        self.dropout = nn.Dropout(hyper_parameters.DROPOUT)
        self.linear = nn.Linear(hyper_parameters.HIDDEN_SIZE, 3 + 6 * hyper_parameters.NUM_MIXTURES)

    def log_model_configuration(self):
        """Log detailed model configuration for debugging."""
        config_info = f"""
        SketchDecoder Configuration:
        -------------------------
        HIDDEN_SIZE: {self.hyper_parameters.HIDDEN_SIZE}
        NUM_MIXTURES: {self.hyper_parameters.NUM_MIXTURES}
        DROPOUT: {self.hyper_parameters.DROPOUT}
        DEVICE: {self.hyper_parameters.DEVICE}
        DEBUG MODE: {self.debug}
        """
        self.logger.info(config_info)

    def debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def log_tensor_detailed_stats(self, tensor, message="", level=logging.INFO):
        """Provide comprehensive tensor statistics."""
        if tensor is None:
            self.logger.warning(f"{message}: Tensor is None!")
            return

        # tensor stats
        stats = {
            'shape': tensor.shape,
            'dtype': tensor.dtype,
            'min': tensor.min().item() if tensor.numel() > 0 else 'N/A',
            'max': tensor.max().item() if tensor.numel() > 0 else 'N/A', 
            'mean': tensor.mean().item() if tensor.numel() > 0 else 'N/A',
            'std': tensor.std().item() if tensor.numel() > 0 else 'N/A',
            'nan_count': torch.isnan(tensor).sum().item(),
            'inf_count': torch.isinf(tensor).sum().item()
        }

        log_message = f"{message} Tensor Statistics:\n" + \
                      "\n".join(f"  {k}: {v}" for k, v in stats.items())
        
        self.logger.log(level, log_message)

        # error checking
        if stats['nan_count'] > 0:
            self.logger.critical(f"NaN DETECTED in {message}!")
            self.debug_print(f"NaN Tensor: {tensor}")
        
        if stats['inf_count'] > 0:
            self.logger.critical(f"Inf DETECTED in {message}!")
            self.debug_print(f"Inf Tensor: {tensor}")

    def forward(self, x, hidden_cell=None):
        self.log_tensor_detailed_stats(x, "Input Tensor")

        start_of_sequence = torch.zeros(x.size(0), 1, 5).to(
            self.hyper_parameters.DEVICE
        )
        start_of_sequence[:, :, 2] = 1

        x_shifted = x[:, :-1, :]

        inputs = torch.cat([start_of_sequence, x_shifted], dim=1)
        
        self.log_tensor_detailed_stats(inputs, "Processed Input Sequence")

        if hidden_cell is None:
            hidden = torch.zeros(
                (1, self.hyper_parameters.BATCH_SIZE, self.hyper_parameters.HIDDEN_SIZE),
                dtype=torch.float32
            ).to(device=self.hyper_parameters.DEVICE)
            cell = torch.zeros_like(hidden)
            hidden_cell = (hidden, cell)

        lstm_outputs, hidden_cell = self.lstm(inputs, hidden_cell)
        lstm_outputs = self.dropout(lstm_outputs)

        self.log_tensor_detailed_stats(lstm_outputs, "LSTM Outputs")

        stroke_parameters = self.linear(lstm_outputs)
        
        self.log_tensor_detailed_stats(stroke_parameters, "Raw Stroke Parameters")

        m = self.hyper_parameters.NUM_MIXTURES


        mixture_weights = torch.softmax(stroke_parameters[:, :, :m], dim=-1)
        self.log_tensor_detailed_stats(stroke_parameters[:, :, :m], "Transformed Mixture Weights (pi)")

        sigmas = safe_exp(stroke_parameters[:, :, m:3 * m])
        self.log_tensor_detailed_stats(stroke_parameters[:, :, m:3 * m], "Transformed Sigmas")

        rhos = torch.tanh(stroke_parameters[:, :, 3 * m:4 * m])
        self.log_tensor_detailed_stats(stroke_parameters[:, :, 3 * m:4 * m], "Transformed Rho's")

        mus = stroke_parameters[:, :, 4 * m:6 * m]
        self.log_tensor_detailed_stats(stroke_parameters[:, :, 3 * m:4 * m], "Transformed Mu's")
                         
        pen_states = torch.softmax(stroke_parameters[:, :, 6 * m:], dim=-1)
        self.log_tensor_detailed_stats(stroke_parameters[:, :, 6 * m:], "Transformed Pen States")

        processed_stroke_parameters = torch.cat([mixture_weights, sigmas, rhos, mus, pen_states], dim=2)

        self.log_tensor_detailed_stats(processed_stroke_parameters, "Final Processed Stroke Parameters")

        return (processed_stroke_parameters, hidden_cell)

    def generate_stroke(self, last_stroke=None, hidden_cell=None):
        if last_stroke is None:
            last_stroke = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32).to(
                self.hyper_parameters.DEVICE
            )

        if hidden_cell is None:
            hidden = torch.zeros(1, self.hyper_parameters.BATCH_SIZE, dtype=torch.float32) \
                .to(device=self.hyper_parameters.DEVICE)
            cell = torch.zeros_like(hidden)
            hidden_cell = (hidden, cell)

        self.log_tensor_detailed_stats(last_stroke, "Last Stroke Input")
        
        lstm_outputs, hidden_cell = self.lstm(last_stroke.unsqueeze(0), hidden_cell)
        lstm_outputs = self.dropout(lstm_outputs)

        self.log_tensor_detailed_stats(lstm_outputs, "Generate Stroke LSTM Outputs")

        stroke_parameters = self.linear(lstm_outputs[-1])
        
        self.log_tensor_detailed_stats(stroke_parameters, "Generate Stroke Parameters")

        gaussian_parameters, pen_state_logits = (
            stroke_parameters[:-3],
            stroke_parameters[-3:]
        )

        m = self.hyper_parameters.NUM_MIXTURES

        gaussian_parameters[:m] = torch.softmax(gaussian_parameters[:m], dim=-1)
        gaussian_parameters[m:3 * m] = torch.exp(gaussian_parameters[m:3 * m])
        gaussian_parameters[3 * m:4 * m] = torch.tanh(gaussian_parameters[3 * m:4 * m])
        
        pen_state_probabilities = torch.softmax(pen_state_logits, dim=-1)

        self.log_tensor_detailed_stats(gaussian_parameters, "Processed Gaussian Parameters")
        self.log_tensor_detailed_stats(pen_state_probabilities, "Pen State Probabilities")

        index = torch.multinomial(gaussian_parameters[:m], num_samples=1)[0]

        sample_parameters = [
            gaussian_parameters[m * (i + 1) + index] for i in range(5)
        ]

        x, y = sample_bivariate_normal(*sample_parameters)

        return (
            torch.cat([x.view(-1), y.view(-1), pen_state_probabilities])
            .to(self.hyper_parameters.DEVICE),
            hidden_cell,
        )


if __name__ == "__main__":
    hyper_parameters = HyperParameters()
    model = SketchDecoder(hyper_parameters, debug=True).to(hyper_parameters.DEVICE)
    print("Model Initialized Successfully")
    print(model)