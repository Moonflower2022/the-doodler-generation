import torch
from utils import HyperParameters
from torch import nn

class SketchDecoder(nn.Module):
    def __init__(self, hyper_parameters):
        super(SketchDecoder, self).__init__()
        self.hyper_parameters = hyper_parameters

        self.lstm = nn.LSTM(5, hidden_size=hyper_parameters.HIDDEN_SIZE)
        # self.dropout = nn.Dropout(hyper_parameters.DROPOUT)
        # input of lstm should be hyper_parameters.LATENT_VECTOR_SIZE + 5 if using encoder
        self.linear = nn.Linear(hyper_parameters.HIDDEN_SIZE, 7)
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
        # lstm_outputs = self.dropout(lstm_outputs)

        stroke_parameters = self.linear(lstm_outputs)

        gaussian_parameters, pen_state_logits = (
            stroke_parameters[:, :, :4],
            stroke_parameters[:, :, 4:],
        )

        pen_state_probabilities = torch.softmax(pen_state_logits, dim=-1)

        mu_x, sigma_x = gaussian_parameters[:, :, 0], torch.exp(
            gaussian_parameters[:, :, 1] / 2
        )
        mu_y, sigma_y = gaussian_parameters[:, :, 2], torch.exp(
            gaussian_parameters[:, :, 3] / 2
        )

        output = torch.stack(
            [mu_x, sigma_x, mu_y, sigma_y],
            dim=-1,  # [HyperParameters.BATCH_SIZE, HyperParameters.MAX_STROKES, 4]
        )
        output = torch.cat([output, pen_state_probabilities], dim=-1)

        return (output, hidden_cell)

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
        # lstm_outputs = self.dropout(lstm_outputs)

        stroke_parameters = self.linear(lstm_outputs[-1])
        # size (7)

        gaussian_parameters, pen_state_logits = (
            stroke_parameters[:4],
            stroke_parameters[4:],
        )
        pen_state_probabilities = torch.softmax(pen_state_logits, dim=-1)

        # gaussian sampling for ∆x and ∆y
        mu_x, sigma_x = gaussian_parameters[0], torch.exp(gaussian_parameters[1] / 2)
        mu_y, sigma_y = gaussian_parameters[2], torch.exp(gaussian_parameters[3] / 2)
        gaussian_sample_x = torch.normal(mu_x, sigma_x).unsqueeze(0)
        gaussian_sample_y = torch.normal(mu_y, sigma_y).unsqueeze(0)

        return (
            torch.cat(
                [gaussian_sample_x, gaussian_sample_y, pen_state_probabilities]
            ).to(self.hyper_parameters.DEVICE),
            hidden_cell,
        )


if __name__ == "__main__":
    model = SketchDecoder(HyperParameters()).to(HyperParameters.DEVICE)

    print(model)

    print(model()[0].size())
