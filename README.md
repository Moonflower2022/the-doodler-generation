# the-doodler-generation

Generation models for the quick, draw! dataset by google that are based on the SketchRNN model.

## todo

* fix model
  * model does not understand the ordered nature of the strokes
  * currently terminates very early because disregarding position, most of the strokes are the termination stroke ([0, 0, 0, 0, 1])
  * something to do with how the model is made
* generate lots of images
* make game