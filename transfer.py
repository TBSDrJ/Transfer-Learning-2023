# Requires Tensorflow 2.10 or later
import tensorflow as tf 
import tensorflow.keras.applications.resnet50 as resnet50
import tensorflow.keras.preprocessing.image as preprocimage
import tensorflow.image as image
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.utils as utils
import tensorflow.data as data
from tensorflow.data import Dataset

BATCH_SIZE = 32

train, validation = utils.image_dataset_from_directory(
    'defungi',
    labels = 'inferred',
    label_mode = 'categorical',
    class_names = None,
    color_mode = 'rgb',
    batch_size = BATCH_SIZE,
    image_size = (224, 224),
    shuffle = True,
    seed = 8008,
    validation_split = 0.2,
    subset = 'both',
)

# Prepare train and test sets for input into ResNet
train = train.map(lambda x, y: (resnet50.preprocess_input(x), y))
validation = validation.map(lambda x, y: (resnet50.preprocess_input(x), y))
train = train.cache().prefetch(buffer_size = data.AUTOTUNE)
validation = validation.cache().prefetch(buffer_size = data.AUTOTUNE)

# Load the ResNet architecture
res = resnet50.ResNet50(
        # Remove the final fully-connected/linear layer by setting include_top
        #   to False
        include_top = False,
        classifier_activation = None,
        # Load trained weights, not just the architecture
        weights = 'imagenet',
        # This is the expected input size for Resnet
        input_shape = (224, 224, 3),
    )

# Freeze the ResNet weights so we only train the transfer layer
res.trainable = False

# Now build the model, first set up inputs:
inputs = tf.keras.Input(shape = (224, 224, 3))

# The next part sequentially builds the output.  We can't use
#   The Sequential model because resnet is not a layer in that.

# First, send image through ResNet:
outputs = res(inputs)
# Flatten so we can send through the Dense layer
outputs = layers.Flatten()(outputs)
# Then send it to a classification layer
outputs = layers.Dense(5, activation = 'softmax')(outputs)

# Set optimizer and loss
optimizer = optimizers.legacy.Adam(learning_rate = .0001)
loss = losses.CategoricalCrossentropy()

# Now, define the model, using above inputs and outputs
model = tf.keras.Model(inputs, outputs)
# Compile with optimizer, loss and metrics using above variables
model.compile(
    optimizer = optimizer,
    loss = loss,
    metrics = ['accuracy'],
)

model.fit(
    train,
    batch_size = BATCH_SIZE,
    epochs = 3,
    verbose = 1,
    validation_data = validation,
)