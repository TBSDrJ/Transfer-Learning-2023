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
    image_size = (224, 224),
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
    image_size = (224, 224),
    shuffle = True,
    seed = 8008,
    validation_split = 0.3,
    subset = 'validation',
)

# Prepare train and test sets for input into ResNet
train = train.map(lambda x, y: (resnet.preprocess_input(x), y))
test = test.map(lambda x, y: (resnet.preprocess_input(x), y))

# Load the ResNet architecture
res = resnet50.ResNet50(
    # This skips the classification layer
    include_top = False,
    # Load trained weights, not just the architecture
    weights = 'imagenet',
    # Gotta use this size if include_top = True
    input_shape = (224, 224, 3),
    )

# Freeze the ResNet weights so we only train the transfer layer
res.trainable = False
