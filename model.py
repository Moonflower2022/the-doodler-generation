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
        output = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32).to(self.hyper_parameters.DEVICE).unsqueeze(0)

        hidden_state = None

        i = 0
        while torch.argmax(output[-1][2:]) != torch.tensor(2, device=self.hyper_parameters.DEVICE) and i < self.hyper_parameters.MAX_STROKES:
            cell_output, hidden_state = self.lstm_cell(output[-1], hidden_state)

            gaussian_params = cell_output[:4]
            softmax_values = cell_output[4:]

            softmax = torch.softmax(softmax_values, dim=0)

            mu_x, sigma_x = gaussian_params[0], torch.exp(gaussian_params[1] / 2)
            mu_y, sigma_y = gaussian_params[2], torch.exp(gaussian_params[3] / 2)

            gaussian_sample_x = torch.normal(mu_x, sigma_x).to(self.hyper_parameters.DEVICE)
            gaussian_sample_y = torch.normal(mu_y, sigma_y).to(self.hyper_parameters.DEVICE)

            new_output = torch.cat([torch.tensor([gaussian_sample_x, gaussian_sample_y], device=self.hyper_parameters.DEVICE), softmax])
            if i > 0:
                output = torch.cat((output, new_output.unsqueeze(0)), axis=0)
            i += 1
        
        pad_tensor = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32, device=self.hyper_parameters.DEVICE).unsqueeze(0)
        padding_needed = self.hyper_parameters.MAX_STROKES - output.size(0)
        if padding_needed > 0:
            padding = pad_tensor.repeat(padding_needed, 1)
            output = torch.cat((output, padding), axis=0)

        return output

if __name__ == '__main__':
    model = SketchDecoder(HyperParameters()).to(HyperParameters.DEVICE)

    print(model)

    print(model().size())