import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "2";
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


from Models import Model
model= Model()  
model.summary()

from Models import Model_P
model= Model_P()  
model.summary()


from read_data1 import data_readV

img_,cor_,clas_=data_readV()





def Loss_clas(y_true, y_pred):

    alpha_factor = K.ones_like(y_true) * 0.25
    alpha_factor = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    focal_weight = tf.where(K.equal(y_true, 1), 1 - y_pred, y_pred)
    focal_weight = alpha_factor * focal_weight ** 2
    cls_loss = focal_weight * K.binary_crossentropy(y_true, y_pred)
    normalizer = K.cast(K.shape(y_pred)[1], K.floatx())
    normalizer = K.maximum(K.cast_to_floatx(1.0), normalizer)

    class_=K.sum(cls_loss) / normalizer

    return class_

def calculate_iou(target_boxes, pred_boxes):
			xA = K.maximum(target_boxes[..., 0], pred_boxes[..., 0])
			yA = K.maximum(target_boxes[..., 1], pred_boxes[..., 1])
			xB = K.minimum(target_boxes[..., 2], pred_boxes[..., 2])
			yB = K.minimum(target_boxes[..., 3], pred_boxes[..., 3])
			interArea = K.maximum(0.0, xB - xA) * K.maximum(0.0, yB - yA)
			boxAArea = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
			boxBArea = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
			iou = interArea / (boxAArea + boxBArea - interArea)
			return iou

def custom_loss(y_true, y_pred):
    mse = tf.losses.mean_squared_error(y_true, y_pred)
    iou = calculate_iou(y_true, y_pred)
    return mse + (1 - iou)
        
        
epochs=50
Adam = optimizers.Adam(lr=0.0001,  beta_1=0.9, beta_2=0.99)
model.compile(optimizer=Adam, loss=[custom_loss,Loss_clas],loss_weights = [1,1])
model.fit(img_,[cor_,clas_],batch_size=2, 
                    epochs=epochs)
model.save_weights("MODEL3.h5")



model.load_weights("MODEL2.h5")

## Testing ###
img_p,_,_=data_readV()

result = model.predict(img_p)
cor_p=result[0]
clas_p=result[1]

import numpy as np
import cv2
import os
save_path='/home/user01/data_ssd/Abbas/Dog1/Evaluation/pred/' 
path_img='/home/user01/data_ssd/Abbas/Dog1/Evaluation/pred_imgs/'

fontScale = 1   
color = (0, 0, 1)   
thickness = 2  
font = cv2.FONT_HERSHEY_SIMPLEX   

for i in range(35,37):
    name=str(i)
    file = os.path.join(save_path, name +".txt") 
    file = open(file, "w")
    class_name=np.argmax(clas_p[i])
    class_prob=np.max(clas_p[i])
    xmin=cor_p[i][0]*400
    ymin=cor_p[i][1]*400
    xmax=cor_p[i][2]*400
    ymax=cor_p[i][3]*400
    file.write(str(class_name) + ' ')
    file.write(str(class_prob) + ' ')
    file.write(str(xmin) + ' ') 
    file.write(str(ymin ) + ' ') 
    file.write(str(xmax ) + ' ') 
    file.write(str(ymax))
    file.close() 
    org = (int(xmin), int(ymin))   
    image=img_p[i,:,:,:]
    cv2.rectangle(image, (int(xmin), int(ymin)),
                                 (int(xmax), int(ymax)),
                                 (1, 0, 0),thickness = 2)
    image = cv2.putText(image, str(class_name), org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
    org1 = (int(xmin+20), int(ymin-10)) 
    image = cv2.putText(image, str(class_prob), org1, font,  
                   0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(path_img , str(i)+".png"),image*255)
    
    
    
### write Grond Truths ##
import numpy as np
import cv2
import os
save_path='/home/user01/data_ssd/Abbas/Dog1/Evaluation/ground/' 
path_img='/home/user01/data_ssd/Abbas/Dog1/Evaluation/ground_imgs/'
fontScale = 1   
color = (0, 0, 1)   
thickness = 2  
font = cv2.FONT_HERSHEY_SIMPLEX   

for i in range(50,55):
    name=str(i)
    file = os.path.join(save_path, name +".txt") 
    file = open(file, "w")
    class_name=np.argmax(clas_[i])
    class_prob=np.max(clas_[i])
    xmin=cor_[i][0]*400
    ymin=cor_[i][1]*400
    xmax=cor_[i][2]*400
    ymax=cor_[i][3]*400
    file.write(str(class_name) + ' ')
    file.write(str(xmin) + ' ') 
    file.write(str(ymin ) + ' ') 
    file.write(str(xmax ) + ' ') 
    file.write(str(ymax))
    file.close() 
    org = (int(xmin), int(ymin))   
    image=img_[i,:,:,:]
    cv2.rectangle(image, (int(xmin), int(ymin)),
                                 (int(xmax), int(ymax)),
                                 (1, 0, 0),thickness = 2)
    
    image = cv2.putText(image, str(class_name), org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
    cv2.imwrite(os.path.join(path_img , str(i)+".png"),image*255)
    
    
    
    

