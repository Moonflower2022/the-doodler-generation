import torch
from utils import HyperParameters
from torch import nn
import numpy as np

def sample_bivariate_normal(sigma_x,sigma_y,rho_xy,mu_x,mu_y):
    mean = torch.tensor([mu_x, mu_y])
    covariance_matrix = torch.tensor([
        [sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],
        [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]
    ])

    mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix)
    
    # Sample from the distribution
    samples = mvn.sample((1,))
    
    return samples[0]

class SketchDecoder(nn.Module):
    def __init__(self, hyper_parameters):
        super(SketchDecoder, self).__init__()
        self.hyper_parameters = hyper_parameters

        self.lstm = nn.LSTM(5, hidden_size=hyper_parameters.HIDDEN_SIZE)
        self.dropout = nn.Dropout(hyper_parameters.DROPOUT)
        # input of lstm should be hyper_parameters.LATENT_VECTOR_SIZE + 5 if using encoder
        self.linear = nn.Linear(hyper_parameters.HIDDEN_SIZE, 3 + 6 * hyper_parameters.NUM_MIXTURES)
        # current output: first 4 are normal distribution parameters for ∆x and ∆y
        # last 3 are softmax logits for the three pen states
        # output size should be different if using Gausian Mixture Model

    def forward(self, x, hidden_cell=None):
        start_of_sequence = torch.zeros(x.size(0), 1, 5).to(
            self.hyper_parameters.DEVICE
        )
        start_of_sequence[:, :, 2] = 1  # Set middle feature to 1

        x_shifted = x[:, :-1, :]

        inputs = torch.cat([start_of_sequence, x_shifted], dim=1)
        # [HyperParameters.BATCH_SIZE, HyperParameters.MAX_STROKES, 5]

        if not hidden_cell:
            hidden = (
                torch.zeros(
                    (
                        self.hyper_parameters.MAX_STROKES,
                        self.hyper_parameters.HIDDEN_SIZE,
                    ),
                    dtype=torch.float32,
                )
                .to(device=self.hyper_parameters.DEVICE)
                .unsqueeze(0)
            )
            cell = (
                torch.zeros(
                    (
                        self.hyper_parameters.MAX_STROKES,
                        self.hyper_parameters.HIDDEN_SIZE,
                    ),
                    dtype=torch.float32,
                )
                .to(device=self.hyper_parameters.DEVICE)
                .unsqueeze(0)
            )

            hidden_cell = (hidden, cell)

        lstm_outputs, hidden_cell = self.lstm(inputs, hidden_cell)
        lstm_outputs = self.dropout(lstm_outputs)

        stroke_parameters = self.linear(lstm_outputs)

        m = self.hyper_parameters.NUM_MIXTURES

        stroke_parameters[:, :, :m] = torch.softmax(stroke_parameters[:, :, :m], dim=-1)        # pi_m
        stroke_parameters[:, :, m:3 * m] = torch.exp(stroke_parameters[:, :, m:3 * m])          # sigma_x, sigma_y
        stroke_parameters[:, :, 3 * m:4 * m] = torch.tanh(stroke_parameters[:, :, 3 * m:4 * m]) # rho_xy
                                                                                                # mu_x, mu_y
        stroke_parameters[:, :, 6 * m:] = torch.softmax(stroke_parameters[:, :, 6 * m:], dim=-1)# p1, p2, p3

        return (stroke_parameters, hidden_cell)

    def generate_stroke(self, last_stroke=None, hidden_cell=None):
        if last_stroke == None:
            last_stroke = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32).to(
                self.hyper_parameters.DEVICE
            )

        if hidden_cell == None:
            hidden = (
                torch.zeros(self.hyper_parameters.HIDDEN_SIZE, dtype=torch.float32)
                .to(device=self.hyper_parameters.DEVICE)
                .unsqueeze(0)
            )
            cell = (
                torch.zeros(self.hyper_parameters.HIDDEN_SIZE, dtype=torch.float32)
                .to(device=self.hyper_parameters.DEVICE)
                .unsqueeze(0)
            )

            hidden_cell = (hidden, cell)

        lstm_outputs, hidden_cell = self.lstm(last_stroke.unsqueeze(0), hidden_cell)
        lstm_outputs = self.dropout(lstm_outputs)

        stroke_parameters = self.linear(lstm_outputs[-1])
        # size (7)

        gaussian_parameters, pen_state_logits = (
            stroke_parameters[:-3],
            stroke_parameters[-3:],
        )

        m = self.hyper_parameters.NUM_MIXTURES

        gaussian_parameters[:m] = torch.softmax(gaussian_parameters[:m], dim=-1)        # pi_m
        gaussian_parameters[m:3 * m] = torch.exp(gaussian_parameters[m:3 * m])          # sigma_x, sigma_y
        gaussian_parameters[3 * m:4 * m] = torch.tanh(gaussian_parameters[3 * m:4 * m]) # rho_xy
                                                                                        # mu_x, mu_y
        pen_state_probabilities = torch.softmax(pen_state_logits, dim=-1)

        index = torch.multinomial(gaussian_parameters[:m], num_samples=1)[0]

        sample_parameters = []

        for i in range(5):
            sample_parameters.append(gaussian_parameters[m * (i + 1) + index])

        x, y = sample_bivariate_normal(*sample_parameters)

        return (
            torch.cat(
                [x.view(-1), y.view(-1), pen_state_probabilities]
            ).to(self.hyper_parameters.DEVICE),
            hidden_cell,
        )


if __name__ == "__main__":
    model = SketchDecoder(HyperParameters()).to(HyperParameters.DEVICE)

    print(model)

    print(model()[0].size())
