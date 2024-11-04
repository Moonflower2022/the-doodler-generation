import torch
from utils import HyperParameters
from torch import nn

class SketchCell(nn.Module):
    def __init__(self, hyper_parameters):
        super(SketchCell, self).__init__()
        self.hyper_parameters = hyper_parameters
        self.lstm_cell = nn.LSTMCell(input_size=hyper_parameters.INPUT_SIZE, hidden_size=hyper_parameters.HIDDEN_SIZE, bias=hyper_parameters.BIAS)
        self.linear = nn.Linear(hyper_parameters.HIDDEN_SIZE, 7)
        self.tanh = nn.Tanh()

    def forward(self, x, hidden_state=None):
        lstm_output, _ = self.lstm_cell(x, hidden_state)
        linear_output = self.linear(lstm_output)
        logits = self.tanh(linear_output)
        return logits, hidden_state

class SketchDecoder(nn.Module):
    def __init__(self, hyper_parameters):
        super(SketchDecoder, self).__init__()
        self.hyper_parameters = hyper_parameters
        self.lstm_cell = SketchCell(hyper_parameters)

    def forward(self):
        # Pre-allocate the output tensor
        output = torch.zeros((self.hyper_parameters.MAX_STROKES + 1, 5), dtype=torch.float32, device=self.hyper_parameters.DEVICE)
        output[0] = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32, device=self.hyper_parameters.DEVICE)

        hidden_state = None

        for i in range(1, self.hyper_parameters.MAX_STROKES + 1):
            # Stop condition based on the previous row
            if torch.argmax(output[i - 1][2:]) == 2:
                break

            # LSTM cell and softmax calculation
            cell_output, hidden_state = self.lstm_cell(output[i - 1], hidden_state)
            gaussian_params, softmax_values = cell_output[:4], cell_output[4:]
            softmax = torch.softmax(softmax_values, dim=0)

            # Gaussian sampling for x and y
            mu_x, sigma_x = gaussian_params[0], torch.exp(gaussian_params[1] / 2)
            mu_y, sigma_y = gaussian_params[2], torch.exp(gaussian_params[3] / 2)
            gaussian_sample_x = torch.normal(mu_x, sigma_x)
            gaussian_sample_y = torch.normal(mu_y, sigma_y)

            # Populate row `i` of the output directly
            output[i] = torch.tensor([gaussian_sample_x, gaussian_sample_y, *softmax.tolist()], device=self.hyper_parameters.DEVICE, requires_grad=True)

        return output[1:]

if __name__ == '__main__':
    model = SketchDecoder(HyperParameters()).to(HyperParameters.DEVICE)

    print(model)

    print(model().size())