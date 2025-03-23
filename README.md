# the-doodler-generation

Generation models for the quick, draw! dataset by google that are based on the SketchRNN model.

## usage

1. go to `utils.py` and change `HyperParameters.DATA_CATEGORY` to be the string matching the type of images you want to download.
2. run `get_data.py`
3. run `clean_data.py` <- not sure if this is needed
4. run `utils.py`, and there should be a number that is outputted
5. go to `utils.py`, and change `HyperParameters.MAX_STROKES` to be that number
6. run `clean_data.py`
7. run `train.py`
8. run `load_model.py`
9. success

## todo

* fix model
  * currently really slow
  * is able to generate a variety of random shapes at ~180 epochs
  * 
* data
  * find only the data that google's model was able to classify
* generate lots of images
* make game