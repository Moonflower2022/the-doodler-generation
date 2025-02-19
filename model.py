import torch
from utils import HyperParameters
from torch import nn

# def lstm_orthogonal_init(lstm_layer, gain=1.0):
#     """
#     Initializes LSTM layer weights using orthogonal initialization.
    
#     Args:
#         lstm_layer: torch.nn.LSTM layer
#         gain: scaling factor for the weights
#     """
#     with torch.no_grad():
#         for name, param in lstm_layer.named_parameters():
#             if 'weight_ih' in name:  # Input-to-hidden weights
#                 for idx in range(4):
#                     # Initialize each gate's weights separately
#                     nn.init.orthogonal_(
#                         param.data[idx*lstm_layer.hidden_size:(idx+1)*lstm_layer.hidden_size],
#                         gain=gain
#                     )
                    
#             elif 'weight_hh' in name:  # Hidden-to-hidden weights
#                 for idx in range(4):
#                     # Initialize each gate's weights separately
#                     nn.init.orthogonal_(
#                         param.data[idx*lstm_layer.hidden_size:(idx+1)*lstm_layer.hidden_size],
#                         gain=gain
#                     )
                    
#             elif 'bias' in name:
#                 param.data.fill_(0)
#                 # Set forget gate bias to 1
#                 param.data[lstm_layer.hidden_size:2*lstm_layer.hidden_size].fill_(1)

class SketchDecoder(nn.Module):
    def __init__(self, hyper_parameters):
        super(SketchDecoder, self).__init__()
        self.hyper_parameters = hyper_parameters

        self.hidden_initiation_linear = nn.Linear(
            hyper_parameters.LATENT_VECTOR_SIZE, 2 * hyper_parameters.HIDDEN_SIZE
        )

        self.lstm = nn.LSTM(
            5,
            hidden_size=hyper_parameters.HIDDEN_SIZE
        )
        # lstm_orthogonal_init(self.lstm)
        self.dropout = nn.Dropout(hyper_parameters.DROPOUT)
        # input of lstm should be hyper_parameters.LATENT_VECTOR_SIZE + 5 if using encoder
        self.linear = nn.Linear(hyper_parameters.HIDDEN_SIZE, 7)
        # current output: first 4 are normal distribution parameters for ∆x and ∆y
        # last 3 are softmax logits for the three pen states
        # output size should be different if using Gausian Mixture Model

    def forward(self, hidden_cell=None):
        start_of_sequence = torch.stack(
            [torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32)]
            * self.hyper_parameters.BATCH_SIZE
        ).to(self.hyper_parameters.DEVICE)
        # size (100, 5)

        if not hidden_cell:
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

        lstm_outputs, hidden_cell = self.lstm(start_of_sequence, hidden_cell)
        lstm_outputs = self.dropout(lstm_outputs) 

        stroke_parameters = self.linear(lstm_outputs)
        # size (100, 7)

        gaussian_parameters, pen_state_logits = (
            stroke_parameters[:, :4],
            stroke_parameters[:, 4:],
        )
        pen_state_probabilities = torch.softmax(pen_state_logits, dim=0)

        mu_x, sigma_x = gaussian_parameters[:, 0], torch.exp(
            gaussian_parameters[:, 1] / 2
        )
        mu_y, sigma_y = gaussian_parameters[:, 2], torch.exp(
            gaussian_parameters[:, 3] / 2
        )
        # # gaussian sampling for ∆x and ∆y
        # gaussian_sample_x = torch.normal(mu_x, sigma_x).unsqueeze(1)
        # gaussian_sample_y = torch.normal(mu_y, sigma_y).unsqueeze(1)

        return (
            torch.cat(
                [mu_x.unsqueeze(1), sigma_x.unsqueeze(1), mu_y.unsqueeze(1), sigma_y.unsqueeze(1), pen_state_probabilities], dim=1
            ).to(self.hyper_parameters.DEVICE),
            hidden_cell,
        )

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
            stroke_parameters[:4],
            stroke_parameters[4:],
        )
        pen_state_probabilities = torch.softmax(pen_state_logits, dim=0)

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
