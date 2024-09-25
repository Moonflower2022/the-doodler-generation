import torch
from train import HyperParameters
from torch import nn

class SketchCell(nn.Module):
    def __init__(self, hyper_parameters):
        super(SketchCell, self).__init__()
        self.hyper_parameters = hyper_parameters
        self.lstm_cell = nn.LSTMCell(input_size=hyper_parameters.INPUT_SIZE, hidden_size=hyper_parameters.HIDDEN_SIZE, bias=hyper_parameters.BIAS)
        self.linear = nn.Linear(hyper_parameters.HIDDEN_SIZE, 7)
        self.tanh = nn.Tanh()

    def forward(self, x):
        lstm_output, _ = self.lstm_cell(x)
        linear_output = self.linear(lstm_output)
        logits = self.tanh(linear_output)
        return logits

class SketchDecoder(nn.Module):
    def __init__(self, hyper_parameters):
        super(SketchDecoder, self).__init__()
        self.hyper_parameters = hyper_parameters
        self.lstm_cell = SketchCell(hyper_parameters)

    def forward(self):
        output = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32).unsqueeze(0)

        i = 0
        while torch.argmax(output[-1][2:]) != torch.tensor(2) and i < hyper_parameters.MAX_STROKES:
            cell_output = self.lstm_cell(output[-1])

            gaussian_params = cell_output[:4]
            softmax_values = cell_output[4:]

            softmax = torch.softmax(softmax_values, dim=0)

            mu_x, sigma_x = gaussian_params[0], torch.exp(gaussian_params[1] / 2)
            mu_y, sigma_y = gaussian_params[2], torch.exp(gaussian_params[3] / 2)

            # Sample from Gaussian distributions
            gaussian_sample_x = torch.normal(mu_x, sigma_x)
            gaussian_sample_y = torch.normal(mu_y, sigma_y)

            # Concatenate the two Gaussian samples with the max value
            output = torch.cat((output, torch.cat([torch.tensor([gaussian_sample_x, gaussian_sample_y]), softmax]).unsqueeze(0)), axis=0)
            i += 1

        return output

# Reconstruction loss: You calculate the likelihood of the true ∆x and ∆y based on the predicted Gaussian mixture model.
# Pen state loss: You use a categorical cross-entropy loss to predict the correct pen state.

if __name__ == '__main__':
    hyper_parameters = HyperParameters()

    model = SketchDecoder(hyper_parameters)

    print(model)

    print(model().size())