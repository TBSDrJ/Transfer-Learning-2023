import numpy as np
import tensorflow.keras.applications.resnet as resnet
import tensorflow.keras.applications.resnet50 as resnet50
import tensorflow.keras.preprocessing.image as preprocimage
import tensorflow.image as image
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.utils as utils
from tensorflow.data import Dataset

train = utils.image_dataset_from_directory(
    'people',
    labels = 'inferred',
    label_mode = 'categorical',
    class_names = None,
    color_mode = 'rgb',
    batch_size = 32,
    image_size = (250, 250),
    shuffle = True,
    seed = 8008,
    validation_split = 0.3,
    subset = 'training',
)

test = utils.image_dataset_from_directory(
    'people',
    labels = 'inferred',
    label_mode = 'categorical',
    class_names = None,
    color_mode = 'rgb',
    batch_size = 32,
    image_size = (250, 250),
    shuffle = True,
    seed = 8008,
    validation_split = 0.3,
    subset = 'validation',
)
