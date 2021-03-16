import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "3";
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
from tensorflow.keras import optimizers
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
print("keras      {}".format(keras.__version__))
print("tensorflow {}".format(tf.__version__))





from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
model = VGG16(weights='imagenet')
model.summary()
for ilayer, layer in enumerate(model.layers):
    print("{:3.0f} {:10}".format(ilayer, layer.name))
    
    
from tensorflow.vis.utils import utils
# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'predictions')
# Swap softmax with linear
model.layers[layer_idx].activation = tf.keras.activations.linear
model = utils.apply_modifications(model)

from vis.visualization import visualize_cam

from keras.vis.utils import utils
from vis.losses import ActivationMaximization

import tensorflow.keras.utils as utils



import vis 

import vis.utils as utils

layer_idx = utils.find_layer_idx(model, 'predictions')
# Swap softmax with linear
model.layers[layer_idx].activation = keras.activations.linear
model = utils.apply_modifications(model)

model.layers[10].activation = keras.activations.linear
model = utils.apply_modifications(model)



from vis.visualization import visualize_cam

from vis.visualization import visualize_saliency






import cv2
import numpy as np
import tensorflow as tf

IMAGE_PATH = '/home/user01/data_ssd/Abbas/dog3/train/1/file_4.jpg'
LAYER_NAME = 'block5_conv3'
CAT_CLASS_INDEX = 0

img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(400, 400))
img = tf.keras.preprocessing.image.img_to_array(img)

#model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)

from Models import Model_P
model= Model_P()  
model.summary()
model.load_weights("MODEL1.h5")


for i in range(len(model.layers)):
	layer = model.layers[i]
	# check for convolutional layer
	# summarize output shape
	print(i, layer.name, layer.output.shape)
    
grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])

print(grad_model)
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(np.array([img]))
    predictions=predictions[1]
    print(predictions)
    loss = predictions[:, 0]

output = conv_outputs[0]
grads = tape.gradient(loss, conv_outputs)[0]

gate_f = tf.cast(output > 0, 'float32')
gate_r = tf.cast(grads > 0, 'float32')
guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

weights = tf.reduce_mean(guided_grads, axis=(0, 1))

cam = np.ones(output.shape[0: 2], dtype = np.float32)

for i, w in enumerate(weights):
    cam += w * output[:, :, i]

cam = cv2.resize(cam.numpy(), (400, 400))
cam = np.maximum(cam, 0)
heatmap = (cam - cam.min()) / (cam.max() - cam.min())

cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)

cv2.imwrite('cam.png', output_image)


plt.figure()
plt.imshow(cam)


import cv2
import numpy as np
import tensorflow as tf

IMAGE_PATH = '/home/user01/data_ssd/Abbas/dog3/train/10/file_3.jpg'

LAYER_NAME = 'block5_conv3'
CAT_CLASS_INDEX = 281

img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
img = tf.keras.preprocessing.image.img_to_array(img)

model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)

grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(np.array([img]))
    print(conv_outputs)
    print(predictions)
    loss = predictions[:, 10]

output = conv_outputs[0]
grads = tape.gradient(loss, conv_outputs)[0]

gate_f = tf.cast(output > 0, 'float32')
gate_r = tf.cast(grads > 0, 'float32')
guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

weights = tf.reduce_mean(guided_grads, axis=(0, 1))

cam = np.ones(output.shape[0: 2], dtype = np.float32)

for i, w in enumerate(weights):
    cam += w * output[:, :, i]

cam = cv2.resize(cam.numpy(), (224, 224))
cam = np.maximum(cam, 0)
heatmap = (cam - cam.min()) / (cam.max() - cam.min())

cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)

cv2.imwrite('cam1.png', output_image)