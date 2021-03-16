import tensorflow.keras as keras

classes=5
def Model(input_size = (400,400,3)):
    inputs=keras.layers.Input(input_size)
     # Block 1
    x = keras.layers.Conv2D(16, (3, 3),activation='relu',padding='same',name='block1_conv1')(inputs)
    x = keras.layers.Conv2D(16, (3, 3),activation='relu',padding='same',name='block1_conv2')(x)
    x=keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # Block 2
    x = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same',name='block2_conv1')(x)
    x = keras.layers.Conv2D(64, (3, 3),activation='relu',padding='same',name='block2_conv2')(x)
    x=keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
     # Block 3
    x = keras.layers.Conv2D(128, (3, 3),activation='relu',padding='same',name='block3_conv1')(x)
    x = keras.layers.Conv2D(128, (3, 3),activation='relu',padding='same',name='block3_conv2')(x)
    x=keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # Block 4
    x = keras.layers.Conv2D(256, (3, 3),activation='relu',padding='same',name='block4_conv2')(x)
    x = keras.layers.Conv2D(256, (3, 3),activation='relu',padding='same',name='block4_conv3')(x)
    x=keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    
    # Block 5
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv2')(x)
    x=keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    #x = keras.layers.Dropout(0.5)(x)
    #fully connected
    x1 = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(4096, activation='relu', name='fc1')(x1)
    x = keras.layers.Dense(1000, activation='relu', name='fc2')(x)
    x = keras.layers.Dense(100, activation='relu', name='fc3')(x)
    x = keras.layers.Dense(4, activation='linear', name='coordinates')(x)
    
    
    
    y= keras.layers.Dense(4096, activation='relu', name='fc4')(x1)
    y=keras.layers.BatchNormalization()(y)
    y = keras.layers.Dense(1000, activation='relu', name='fc5')(y)
    y=keras.layers.BatchNormalization()(y)
    y = keras.layers.Dense(100, activation='relu', name='fc6')(y)
    y=keras.layers.BatchNormalization()(y)
    y = keras.layers.Dense(classes, activation='softmax', name='class')(y)
    model = keras.models.Model(inputs, [x,y])
    return model
classes=5
def Model_j(input_size = (400,400,3)):
    inputs=keras.layers.Input(input_size)
     # Block 1
    x = keras.layers.Conv2D(16, (3, 3),activation='relu',padding='same',name='block1_conv1')(inputs)
    x = keras.layers.Conv2D(16, (3, 3),activation='relu',padding='same',name='block1_conv2')(x)
    x=keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # Block 2
    x = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same',name='block2_conv1')(x)
    x = keras.layers.Conv2D(64, (3, 3),activation='relu',padding='same',name='block2_conv2')(x)
    x=keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
     # Block 3
    x = keras.layers.Conv2D(128, (3, 3),activation='relu',padding='same',name='block3_conv1')(x)
    x = keras.layers.Conv2D(128, (3, 3),activation='relu',padding='same',name='block3_conv2')(x)
    x=keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # Block 4
    x = keras.layers.Conv2D(256, (3, 3),activation='relu',padding='same',name='block4_conv2')(x)
    x = keras.layers.Conv2D(256, (3, 3),activation='relu',padding='same',name='block4_conv3')(x)
    x=keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    
    # Block 5
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv2')(x)
    x=keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = keras.layers.Dropout(0.5)(x)
    #fully connected
    x1 = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(4096, activation='relu', name='fc1')(x1)
    x=keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(1000, activation='relu', name='fc2')(x)
    x = keras.layers.Dense(100, activation='relu', name='fc3')(x)
    x = keras.layers.Dense(25, activation='relu', name='coordinates')(x)
    model = keras.models.Model(inputs,x)
    return model

def Model_P(input_size = (400,400,3)):
    inputs=keras.layers.Input(input_size)
     # Block 1
    x = keras.layers.Conv2D(16, (3, 3),activation='relu',padding='same',name='block1_conv1')(inputs)
    x = keras.layers.Conv2D(16, (3, 3),activation='relu',padding='same',name='block1_conv2')(x)
    x = keras.layers.Conv2D(16, (3, 3),activation='relu',strides=(2, 2),padding='same')(x)
    x=keras.layers.BatchNormalization()(x)
    
    # Block 2
    x = keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same',name='block2_conv1')(x)
    x = keras.layers.Conv2D(64, (3, 3),activation='relu',padding='same',name='block2_conv2')(x)
    x = keras.layers.Conv2D(64, (3, 3),activation='relu',strides=(2, 2),padding='same')(x)
    x=keras.layers.BatchNormalization()(x)
     # Block 3
    x = keras.layers.Conv2D(128, (3, 3),activation='relu',padding='same',name='block3_conv1')(x)
    x = keras.layers.Conv2D(128, (3, 3),activation='relu',padding='same',name='block3_conv2')(x)
    x = keras.layers.Conv2D(128, (3, 3),activation='relu',strides=(2, 2),padding='same')(x)
    x=keras.layers.BatchNormalization()(x)
    # Block 4
    x = keras.layers.Conv2D(256, (3, 3),activation='relu',padding='same',name='block4_conv2')(x)
    x = keras.layers.Conv2D(256, (3, 3),activation='relu',padding='same',name='block4_conv3')(x)
    x = keras.layers.Conv2D(256, (3, 3),activation='relu',strides=(2, 2),padding='same')(x)
    x=keras.layers.BatchNormalization()(x)
    
    # Block 5
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv2')(x)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',strides=(2, 2),padding='same')(x)
    x=keras.layers.BatchNormalization()(x)
    #fully connected
    x1 = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(4096, activation='relu', name='fc1')(x1)
    x = keras.layers.Dense(1000, activation='relu', name='fc2')(x)
    x = keras.layers.Dense(100, activation='relu', name='fc3')(x)
    x = keras.layers.Dense(4, activation='linear', name='coordinates')(x)
    

    y= keras.layers.Dense(4096, activation='relu', name='fc4')(x1)
    y=keras.layers.BatchNormalization()(y)
    y = keras.layers.Dense(1000, activation='relu', name='fc5')(y)
    y=keras.layers.BatchNormalization()(y)
    y = keras.layers.Dense(100, activation='relu', name='fc6')(y)
    y=keras.layers.BatchNormalization()(y)
    y = keras.layers.Dense(classes, activation='softmax', name='class')(y)
    model = keras.models.Model(inputs, [x,y])
    return model
