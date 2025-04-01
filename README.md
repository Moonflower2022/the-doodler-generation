# the-doodler-generation

Generation models for the quick, draw! dataset by google that are based on the SketchRNN model.

## usage

1. go to `utils.py` and change `HyperParameters.DATA_CATEGORY` to be the string matching the type of images you want to download.
2. run `get_data.py`
3. run `utils.py`, and there should be a number that is outputted
4. go to `utils.py`, and change `HyperParameters.MAX_STROKES` to be that number
5. run `clean_data.py`
6. run `train.py`
7. run `load_model.py`
8. success

## todo

-   data
    -   find only the data that google's model was able to classify
-   generate lots of images
-   make game

## references

-   <https://nn.labml.ai/sketch_rnn/index.html>
    -   annotated pytorch implementation of sketchrnn
-   <https://github.com/alexis-jacq/Pytorch-Sketch-RNN>
    -   another pytorch implementation of sketchrnn
-   <https://github.com/rfeinman/Sketch-RNN>
    -   another pytorch implementation of sketchrnn
-   <https://arxiv.org/pdf/1308.0850>
    -   generalized strokes training
-   <https://arxiv.org/pdf/1704.03477>
    -   sketch rnn paper
-   <https://github.com/magenta/magenta/tree/main/magenta/models/sketch_rnn>
    -   TF implementation of sketch rnn
