# dataset.py
## class Dataset
This class is used for generating a training dataset, which cosists of images of 2x2 pixels.
Any one of four pixels has a value of 1, and others have 0s.

# sampler.py
## class Sampler
This class is used for sampling values from an input dataset and a gaussian distribution.

# net.py
## class Encoder
Encoder takes 'x' and 'eps' as inputs.
After 'eps' is transformed by the linear transformation, it is added to x.
An altenative method to merge 'x' and 'eps' is to concatenate them.


