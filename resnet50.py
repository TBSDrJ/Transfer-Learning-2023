import numpy as np
import tensorflow.keras.applications.resnet as resnet
import tensorflow.keras.applications.resnet50 as resnet50
import tensorflow.keras.preprocessing.image as preprocimage
import tensorflow.image as image

img = preprocimage.load_img('can_opener.jpg')
img = preprocimage.img_to_array(img)
img = image.resize_with_pad(img, 224, 224)
img = np.array([img])
img = resnet.preprocess_input(img)

res = resnet50.ResNet50(
    include_top = True,
    weights = 'imagenet',
    input_shape = (224, 224, 3),
    classifier_activation = 'softmax',
    )

print(resnet50.decode_predictions(res(img).numpy()))
