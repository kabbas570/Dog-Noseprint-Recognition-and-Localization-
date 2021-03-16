import numpy as np
import cv2
import matplotlib.pyplot as plt


num_calss=5
        
def data_readV():   
    image_w=400
    image_h=400
    img_ = []
    cor_=[]
    clas_=[]    
    with open('data_train.txt') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
    for i in range(len(content)):
        coor = np.zeros([4])  # 7 x 7 x num_classes + 5
        clas_t=np.zeros([num_calss])
        img_path=content[i].partition(" ")[0]
        img=cv2.imread(img_path)
        #img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize (img, (image_w,image_h), interpolation = cv2.INTER_AREA) 
        img=img/255
        img_.append(img)
        q=0
        for j in content[i]:
            if j==" ":
                q=q+1
        for c in range(q):
            label=content[i].split(" ")[c+1]      
            for  l  in  label :
                l  = label. split ( ',' )
                l = np.array(l, dtype=np.int)
                xmin = l[0]*(image_w/3888)
                ymin  = l[1]*(image_h/2592)
                xmax = l[2]*(image_h/3888)
                ymax = l[3]*(image_w/2592)
                clas  = l[4]
                coor[0]=xmin/400
                coor[1]=ymin/400
                coor[2]=xmax/400
                coor[3]=ymax/400
                clas_t[clas]=1
        cor_.append(coor)
        clas_.append(clas_t)
    img_ = np.array(img_)
    cor_ = np.array(cor_)
    clas_ = np.array(clas_)
    return img_,cor_,clas_


# =============================================================================
# index=106
# image=img_[index,:,:,:]
# xmin=cor_[index][0]
# ymin=cor_[index][1]
# xmax=cor_[index][2]
# ymax=cor_[index][3]
# class_name=np.argmax(clas_[index])
# cv2.rectangle(image, (int(xmin), int(ymin)),
#                                  (int(xmax), int(ymax)),
#                                  (255, 0, 0),thickness = 2)
# 
# font = cv2.FONT_HERSHEY_SIMPLEX   
# org = (int(xmin), int(ymin))   
# fontScale = 1   
# color = (0, 0, 255)   
# thickness = 2  
# image = cv2.putText(image, str(class_name), org, font,  
#                    fontScale, color, thickness, cv2.LINE_AA)
# 
# 
# plt.figure()
# plt.imshow(image)
# =============================================================================




def data_readVt():   
    image_w=400
    image_h=400
    img_ = []
    cor_=[]
    clas_=[]    
    with open('data_train.txt') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
    for i in range(len(content)):
        coor = np.zeros([4])  # 7 x 7 x num_classes + 5
        clas_t=np.zeros([num_calss])
        img_path=content[i].partition(" ")[0]
        img=cv2.imread(img_path)
        img = cv2.resize (img, (image_w,image_h), interpolation = cv2.INTER_AREA) 
        img=img/255
        img_.append(img)
        q=0
        for j in content[i]:
            if j==" ":
                q=q+1
        for c in range(q):
            label=content[i].split(" ")[c+1]      
            for  l  in  label :
                l  = label. split ( ',' )
                l = np.array(l, dtype=np.int)
                xmin = l[0]*(image_w/3888)
                ymin  = l[1]*(image_h/2592)
                xmax = l[2]*(image_h/3888)
                ymax = l[3]*(image_w/2592)
                clas  = l[4]
                coor[0]=xmin
                coor[1]=ymin
                coor[2]=xmax
                coor[3]=ymax
                clas_t[clas]=1
        cor_.append(coor)
        clas_.append(clas_t)
    img_ = np.array(img_)
    cor_ = np.array(cor_)
    clas_ = np.array(clas_)
    return img_,cor_,clas_





