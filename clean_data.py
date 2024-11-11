import numpy as np
from utils import HyperParameters, get_max_strokes
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch


class SketchesDataset(Dataset):
    def __init__(self, sketches):
        self.sketches = sketches

    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, idx):
        return torch.tensor(self.sketches[idx], dtype=torch.float32)


def draw_sketch(stroke_data):
    """
    Draws the sketch from stroke data.

    stroke_data: List of strokes where each stroke is of the form [∆x, ∆y, p1, p2, p3].
    ∆x, ∆y are the pen movements, and p1, p2, p3 indicate pen states:
        p1 = 1: pen down, draw a line.
        p2 = 1: pen up, move without drawing.
        p3 = 1: end of drawing.
    """
    # Starting point
    x, y = 0, 0
    lines = []  # Store all line segments as [x_start, x_end, y_start, y_end]

    fig, ax = plt.subplots()

    for stroke in stroke_data:
        dx, dy, p1, p2, p3 = stroke

        # Compute the new point
        new_x = x + dx
        new_y = y + dy

        if p1 == 1:
            # Pen down: draw a line
            lines.append([x, new_x, y, new_y])
        elif p3 == 1:
            # End of drawing
            break

        # Update the current position
        x, y = new_x, new_y

    # Draw all line segments
    for line in lines:
        ax.plot([line[0], line[1]], [line[2], line[3]], "k-")

    # Adjust the aspect ratio
    ax.set_aspect("equal")

    # Turn off the axes
    plt.axis("off")

    # Show the drawing
    plt.show()


def to_big_strokes(sketch, max_len=HyperParameters.MAX_STROKES):
    """Converts from stroke-3 format sketch to stroke-5 format sketch and pads to given length."""
    # (But does not insert special start token).
    result = np.zeros((max_len if max_len else len(sketch), 5), dtype=float)
    l = len(sketch)
    assert l <= max_len
    result[0:l, 0:2] = sketch[:, 0:2]
    result[0:l, 3] = sketch[:, 2]
    result[0:l, 2] = 1 - result[0:l, 3]
    result[l:, 4] = 1
    return result


def process_drawings(drawings):
    for drawing in drawings:
        yield to_big_strokes(drawing)


def clean_and_output(category):
    load_path = f"data/{category}.npz"

    print(f"loading data from {load_path}...")
    drawings = np.load(load_path, allow_pickle=True, encoding="latin1")
    print("loaded object:", drawings.items)

    for sketch in drawings['train']:
        has_stop = False
        for stroke in to_big_strokes(sketch):
            if stroke[4] == 1:
                has_stop = True
        if not has_stop:
            print("no stop marker!")

    print("converting 3-stroke data to 5-stroke data...")

    output = {
        data_section: np.asarray(
            list(process_drawings(drawings[data_section])), dtype=np.int16
        )
        for data_section in ["train", "test", "valid"]
    }

    draw_sketch(output["train"][100])

    file_path = f"data/processed_{category}.npz"

    print(f"writing output to {file_path}...")

    with open(file_path, "wb") as output_file:
        np.savez(output_file, **output)


if __name__ == "__main__":
    clean_and_output(HyperParameters.DATA_CATEGORY)
