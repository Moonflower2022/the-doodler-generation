from utils import device
from torch import nn
import ndjson


class HyperParameters:
    INPUT_SIZE = 5
    # ∆x,
    # ∆y,
    # pen down (pen is currently down),
    # pen up (after this stroke, pen goes up),
    # end (current point and subsequent points are voided)

    HIDDEN_SIZE = 256
    BIAS = True
    MAX_STROKES = 2016


log_loss = nn.CrossEntropyLoss()


def get_ending_index(sketch):
    for i, stroke in enumerate(sketch):
        if stroke[-1] == 1:
            return i
    return -1


def reconstruction_loss(predictions, labels):
    # sum of log loss of ∆x, ∆y, and pen states p_1, p_2, and p_3.
    pen_state_loss = 0

    offset_loss = 0

    ending_index = get_ending_index(labels)

    for i, (predicted_stroke, label_stroke) in enumerate(zip(predictions, labels)):
        pen_state_loss += log_loss(predicted_stroke[2:], label_stroke[2:])

        if i <= ending_index:
            offset_loss += log_loss(predicted_stroke[:2], label_stroke[:2])

    return pen_state_loss + offset_loss


# training: if len(S) < MAX_STROKES, pad it with (0, 0, 0, 0, 1) until the length equals MAX_SROKES

hyper_parameters = HyperParameters()

category = "airplane"

with open("data/processed_airplane.ndjson") as f:
    drawings = ndjson.load(f)