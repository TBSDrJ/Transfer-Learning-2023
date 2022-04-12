import numpy as np
import tensorflow.keras.applications.resnet as resnet
import tensorflow.keras.applications.resnet50 as resnet50
import tensorflow.keras.preprocessing.image as preprocimage
import tensorflow.image as image

# Load image as Tensor
img = preprocimage.load_img('can_opener.jpg')
# Convert to numpy array so we can mess with it a bit
img = preprocimage.img_to_array(img)
# Resize using padding so we don't mess with aspect ratio
img = image.resize_with_pad(img, 224, 224)
# Add a batch dimension to make the next function happy
img = np.array([img])
# ResNet comes with a built-in function to preprocess
# images so that they are properlay set up for its architecture
img = resnet.preprocess_input(img)

# Load the ResNet architecture
res = resnet50.ResNet50(
    # This puts on the last dense classification layer
    include_top = True,
    # Load trained weights, not just the architecture
    weights = 'imagenet',
    # Gotta use this size if include_top = True
    input_shape = (224, 224, 3),
    classifier_activation = 'softmax',
    )

# Take this apart a bit:
#   res(img) will give a 1-dim Tensor of length 1000 with
#       probabilities for each of the 1000 object classes
#   But decode_predictions() wants a numpy array, not a Tensor
#       so convert to a numpy array
#   decode_predictions() finds the top 5 probabilities, and
#       matches those with the class names for us.
print(resnet50.decode_predictions(res(img).numpy()))
