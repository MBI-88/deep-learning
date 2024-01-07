#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np 
import math,os
import matplotlib.pyplot as plt


# In[2]:


# Modelo
class ResNet_V1_V2():
    def __init__(self,version='v1',inputs=None,output=None,depth=0,lr=1e-3):
        self.output_dim=output
        self.lr=lr
        self.version=version
        self.input_dim=inputs
        self.n_blocks=0
        self.filepath=os.getcwd()
        self.lr_scheduler=tf.keras.callbacks.LearningRateScheduler(self.lr_schedule)
        self.lr_reducer=tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=5,min_lr=0.5e-6)
        self.checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath=self.filepath,monitor='val_acc',verbose=1,save_best_only=True)
        self.callback=[self.checkpoint,self.lr_scheduler,self.lr_reducer]
        if self.version == 'v1':
            if (depth-2)%6 != 0:
                raise  ValueError('depth should be 6n+2 (eg 20, 32, in [a])')
            self.n_blocks = int((depth-2)/6)
            self.modelo_resnet=self.resnet_v1()
    
        elif self.version == 'v2':
            if (depth-2)%9 != 0:
                raise  ValueError('depth should be 9n+2 (eg 110 in [b])')
            self.n_blocks = int((depth-2)/9)
            self.modelo_resnet=self.resnet_v2()

        else:
            raise ValueError('Vertion no valid')
        self.modelo_resnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr_schedule(0)),loss='categorical_crossentropy',metrics=['acc'])
        self.modelo_resnet.summary()
    
    def resnet_layers(self,inputs,n_filters=16,kernel=3,stride=1,activation='relu',batch_nor=True,conv_firt=True):
        conv=tf.keras.layers.Conv2D(n_filters,kernel_size=kernel,strides=stride,padding='same',kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        x=inputs
        if conv_firt:
            x=conv(x)
            if batch_nor:
                x=tf.keras.layers.BatchNormalization()(x)
            if activation is not None:
                x=tf.keras.layers.Activation(activation)(x)
        else:
            if batch_nor:
                x=tf.keras.layers.BatchNormalization()(x)
            if activation is not None:
                x=tf.keras.layers.Activation(activation)(x)
            x=conv(x)
        return x
    
    def resnet_v1(self):
        n_filters=16
        inputs=tf.keras.Input(shape=self.input_dim)
        x=self.resnet_layers(inputs=inputs)
        for stack in range(3):
            for nblocks in range(self.n_blocks):
                stride=1
                if stack > 0 and nblocks == 0 :
                    stride=2
                y=self.resnet_layers(inputs=x,n_filters=n_filters,stride=stride)
                y=self.resnet_layers(inputs=y,n_filters=n_filters,activation=None)
                if stack > 0 and nblocks == 0:
                    x=self.resnet_layers(inputs=x,n_filters=n_filters,kernel=1,stride=stride,activation=None,batch_nor=False)
                x=tf.keras.layers.Add()([x,y])
                x=tf.keras.layers.Activation('relu')(x)

            n_filters *= 2
        
        x=tf.keras.layers.AveragePooling2D(pool_size=8)(x)
        y=tf.keras.layers.Flatten()(x)
        outputs=tf.keras.layers.Dense(units=self.output_dim,activation='softmax',kernel_initializer='he_normal')(y)

        model=tf.keras.Model(inputs,outputs)
        return model
    
    def resnet_v2(self):
        n_filters=16
        inputs=tf.keras.Input(shape=self.input_dim)
        x=self.resnet_layers(inputs=inputs)
        
        for stack in range(3):
            for nblock in range(self.n_blocks):
                activation='relu'
                batch_nor=True
                stride=1
                if stack == 0:
                    n_filters_out = n_filters * 4
                    if  nblock == 0:
                        activation = None
                        batch_nor= None
                else:
                    n_filters_out = n_filters * 2
                    if nblock == 0:
                        stride=2
                
                y=self.resnet_layers(x,n_filters=n_filters,kernel=1,stride=stride,activation=activation,batch_nor=batch_nor,conv_firt=False)
                y=self.resnet_layers(y,n_filters=n_filters,conv_firt=False)
                y=self.resnet_layers(y,n_filters=n_filters_out,conv_firt=False,kernel=1)

                if nblock == 0:
                    x=self.resnet_layers(x,n_filters=n_filters_out,kernel=1,activation=None,batch_nor=False,stride=stride)

                x=tf.keras.layers.Add()([x,y])
            
            n_filters = n_filters_out
        
        x=tf.keras.layers.AveragePooling2D(pool_size=8)(x)
        y=tf.keras.layers.Flatten()(x)
        outputs=tf.keras.layers.Dense(units=self.output_dim,activation='softmax',kernel_initializer='he_normal')(y)
        model=tf.keras.Model(inputs,outputs)
        return model    
    
    def lr_schedule(self,epoch):
        if epoch > 180:
            self.lr *= 0.5e-3
        elif  epoch > 160:
            self.lr *= 1e-3
        elif epoch > 120:
            self.lr *= 1e-2
        elif epoch > 80:
            self.lr *= 1e-1
        print('Learning_rate ',self.lr)
        return self.lr
    
    def plot_resnet(self):
        tf.keras.utils.plot_model(self.modelo_resnet,to_file='ResNet'+self.version+'.png',dpi=100,show_shapes=True)
    
    def train_resnet(self,x_train,y_train,x_val,y_val,x_test,y_test,argument=True,batch=32,epo=10):
        if argument:
            gent=tf.keras.preprocessing.image.ImageDataGenerator( featurewise_center=False,
                                                                    # set each sample mean to 0
                                                                    samplewise_center=False,
                                                                    # divide inputs by std of dataset
                                                                    featurewise_std_normalization=False,
                                                                    # divide each input by its std
                                                                    samplewise_std_normalization=False,
                                                                    # apply ZCA whitening
                                                                    zca_whitening=False,
                                                                    # randomly rotate images in the range (deg 0 to 180)
                                                                    rotation_range=0,
                                                                    # randomly shift images horizontally
                                                                    width_shift_range=0.1,
                                                                    # randomly shift images vertically
                                                                    height_shift_range=0.1,
                                                                    # randomly flip images
                                                                    horizontal_flip=True,
                                                                    # randomly flip images
                                                                    vertical_flip=False)

            gent.fit(x_train)
            step_per_epo=math.ceil(len(x_train)/batch)
            self.modelo_resnet.fit(x=gent.flow(x_train,y_train,batch_size=batch),validation_data=(x_val,y_val),verbose=1,callbacks=self.callback,epochs=epo,steps_per_epoch=step_per_epo)

        else:
            self.model_resnet.fit(x=x_train,y=y_train,validation_data=(x_val,y_val),epochs=epo,shuffle=True,callbacks=self.callback)
        

        score=self.modelo_resnet.evaluate(x=x_test,y=y_test,batch_size=batch,verbose=0)
        print('Test Accuracy ',score[1])
        print('Test Loss ',score[0])


# In[3]:


# Procesado de datos 
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()


# In[4]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[7]:


# Mostrar imagenes
plt.figure(figsize=(8,8))
num_image=10
for i in range(num_image):
    ax=plt.subplot(5,5,i + 1)
    image=x_train[i,:,:,:]
    image=np.reshape(image,[32,32,3])
    plt.imshow(image)
    plt.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 

plt.show()



# In[5]:


X_train=x_train[:30000,:]
X_val=x_train[30000:,:]
Y_train=y_train[:30000]
Y_val=y_train[30000:]
X_train.shape,Y_train.shape,X_val.shape,Y_val.shape


# In[6]:


num_clase=10

X_train=X_train.astype('float32')/255
X_val=X_val.astype('float32')/255
x_test=x_test.astype('float32')/255

Y_train=tf.keras.utils.to_categorical(Y_train,num_classes=num_clase)
Y_val=tf.keras.utils.to_categorical(Y_val,num_classes=num_clase)
y_test=tf.keras.utils.to_categorical(y_test,num_classes=num_clase)

subtract_pixel_mean=True
if subtract_pixel_mean:
    X_train_mean=np.mean(X_train,dtype=np.float32)
    X_val_mean=np.mean(X_val,dtype=np.float32)
    x_test_mean=np.mean(x_test,dtype=np.float32)

    X_train -= X_train_mean
    X_val -= X_val_mean
    x_test -= x_test_mean

epochs=100
n=3
batch=32

inputs=X_train.shape[1:]


# In[7]:


resnet=ResNet_V1_V2(inputs=inputs,output=num_clase,depth=20)


# In[9]:


resnet.train_resnet(X_train,Y_train,X_val,Y_val,x_test,y_test,batch=batch,epo=epochs)


# In[8]:


resnet.plot_resnet()

