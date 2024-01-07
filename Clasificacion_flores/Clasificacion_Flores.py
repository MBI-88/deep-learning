#!/usr/bin/env python
# coding: utf-8

# In[1]:


import PIL.Image as imagen 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import pathlib
from sklearn.metrics import roc_curve
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(42)
np.random.seed(42)


# In[2]:


directorio='C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/flower_photos'
data_dir=pathlib.Path(directorio)


# In[3]:


imagen_count=len(list(data_dir.glob('*/*.jpg')))
print(imagen_count)


# In[4]:


rosas=list(data_dir.glob('roses/*'))
imagen.open(str(rosas[0]))


# In[5]:


flores={'roses':0,'daisy':1,'dandelion':2,'sunflowers':3,'tulips':4} # Nota las clases tienen que ser etiquetadas empezando por 0
Code={0:'roses',1:'daisy',2:'dandelion',3:'sunflowers',4:'tulips'}


# In[6]:


lista_flores=['roses','daisy','dandelion','sunflowers','tulips']
path_lits=[]
labels=[]
for f in lista_flores:
    path_dir=os.path.join(data_dir,f)
    for i in os.listdir(path_dir):
        path_lits.append(path_dir+'/'+i)
        labels.append(flores[f])
    


# In[7]:


path_lits[1000:1001],labels[1000:1001]


# In[8]:


len(path_lits),len(labels)


# In[43]:


ds_file=tf.data.Dataset.from_tensor_slices((path_lits,labels))
ds_file=ds_file.shuffle(buffer_size=3670,reshuffle_each_iteration=True)
ds_train=ds_file.take(1970)
#ds_test=ds_file.skip(270)
ds_val=ds_file.skip(1970)


# In[44]:


def load_prepro(path,labels,size=(64,64),mode='train'):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image=tf.image.resize(image,size=size)
    image /=255.0
    image=image*2-1.0
    
    if mode == 'train':
        image_crop=tf.image.random_crop(value=image,size=(59,59,3))
        image_rizased=tf.image.resize(image_crop,size=size)
        image_b=tf.image.adjust_brightness(image_rizased,delta=0.45)
        image_c=tf.image.adjust_contrast(image_b,contrast_factor=0.50)
        image_s=tf.image.adjust_saturation(image_c,saturation_factor=0.36)
        image_flip=tf.image.flip_left_right(image_s)
        return image_flip,labels
    else:
        image_crop=tf.image.crop_to_bounding_box(image,offset_height=4,offset_width=4,target_height=60,target_width=60)
        image_rizased=tf.image.resize(image_crop,size=size)
        return image_rizased,labels
  


# In[45]:


ds_train=ds_train.map(lambda path,label:load_prepro(path,label))
ds_val=ds_val.map(lambda path,label:load_prepro(path,label,mode='val'))
#ds_test=ds_test.map(lambda path,label:load_prepro(path,label,mode='test'))


# In[46]:


iteracion,lab=next(iter(ds_val))
imm=iteracion[0]
print(np.min(imm),' ',np.max(imm))


# In[47]:


fig=plt.figure(figsize=(10,5))
for i , exa in enumerate(ds_train.take(6)):
    print(exa[0].shape, exa[1].numpy())
    ax=fig.add_subplot(2,3,i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(exa[0])
    ax.set_title('{}'.format(Code[exa[1].numpy()]),size=10)
plt.tight_layout()
plt.show()


# In[48]:


# Modelo
modelo_clf=tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32,kernel_size=2,padding='same',activation='relu',data_format='channels_last'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(rate=0.40),
    
   
    tf.keras.layers.Conv2D(filters=32,kernel_size=2,padding='same',activation='relu',data_format='channels_last'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(rate=0.45),
    
   
    tf.keras.layers.Conv2D(filters=128,kernel_size=2,padding='same',activation='relu',data_format='channels_last'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(rate=0.68),
    
  ])


# In[49]:


modelo_clf.compute_output_shape(input_shape=(None,64,64,3))


# In[50]:


modelo_clf.add(tf.keras.layers.Flatten())
modelo_clf.compute_output_shape(input_shape=(None,64,64,3))


# In[17]:


modelo_clf.add(tf.keras.layers.Dense(units=8192,activation='relu'))
modelo_clf.add(tf.keras.layers.Dropout(rate=0.5))
modelo_clf.add(tf.keras.layers.Dense(units=100,activation='relu'))
modelo_clf.add(tf.keras.layers.Dense(units=5,activation='softmax'))
modelo_clf.build(input_shape=(None,64,64,3))

modelo_clf.summary()


# In[18]:


modelo_clf.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
Epocas=25
batch_size=32


# In[51]:


Autonomo=tf.data.experimental.AUTOTUNE

ds_train=ds_train.cache().prefetch(buffer_size=Autonomo).shuffle(buffer_size=2000,reshuffle_each_iteration=True).batch(batch_size=batch_size)
ds_val=ds_val.cache().prefetch(buffer_size=Autonomo).batch(batch_size=batch_size)
#ds_test=ds_test.cache().prefetch(buffer_size=Autonomo).batch(batch_size=batch_size)


# In[20]:


hist=modelo_clf.fit(ds_train,epochs=Epocas,validation_data=ds_val)


# In[23]:


historia=hist.history
x_arr=np.arange(len(historia['loss']))+1

fig=plt.figure(figsize=(10,4))
ax=fig.add_subplot(1,2,1)
ax.plot(x_arr,historia['loss'],'-o',label='Train_loss')
ax.plot(x_arr,historia['val_loss'],'--<',label='Validation_loss')
ax.legend(fontsize=15)
ax=fig.add_subplot(1,2,2)
ax.plot(x_arr,historia['accuracy'],'-o',label='Train_acc')
ax.plot(x_arr,historia['val_accuracy'],'--<',label='Validation_acc')
ax.legend(fontsize=15)
plt.show()


# In[24]:


result=modelo_clf.evaluate(ds_val.take(1))
print(result)


# In[25]:


modelo_clf.save('C:/Users/MBI/Documents/Python_Scripts/Deep_Learning_Ejercicios/Clasificacion_flores.h5')


# In[52]:


pp=tf.keras.models.load_model('C:/Users/MBI/Documents/Python_Scripts/Deep_Learning_Ejercicios/Clasificacion_flores.h5')


# In[58]:


pp.predict_classes(ds_val.take(1),batch_size=None)


# In[ ]:




