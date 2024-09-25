from utils import device
from torch import nn

class HyperParameters():
    INPUT_SIZE = 5 
    # ∆x, 
    # ∆y, 
    # pen down (pen is currently down), 
    # pen up (after this stroke, pen goes up), 
    # end (current point and subsequent points are voided)

    HIDDEN_SIZE = 256
    BIAS = True
    MAX_STROKES = 1000 # TODO: change this to the actual max strokes of a sketch after getting and cleaning dataset

log_loss = nn.CrossEntropyLoss()

def reconstruction_loss(predictions, labels):
    # sum of log loss of ∆x, ∆y, and pen states p_1, p_2, and p_3.
    pass
    

# training: if len(S) < MAX_STROKES, pad it with (0, 0, 0, 0, 1) until the length equals MAX_SROKES