#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time as tm 
import tensorflow as tf  
import numpy as np 
import matplotlib.pyplot as plt 
from IPython import display
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(42)
np.random.seed(42)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


(train_image,test_image),(_,_)=tf.keras.datasets.fashion_mnist.load_data()


# In[3]:


train_image.shape


# In[4]:


plt.imshow(train_image[5])
plt.show()


# In[5]:


def procesado(ex,mode='uniforme'):
    image=ex
    image=tf.image.convert_image_dtype(image,tf.float32)
    image = image*2.-1.0
    if mode=='uniforme':
        vector_latente=tf.random.uniform(shape=(100,),minval=-1.0,maxval=1.0)
    else:
        vector_latente=tf.random.normal(shape=(100,),mean=0.0,stddev=1)
    
    return vector_latente,image

train_image=train_image.reshape((train_image.shape[0],28,28,1))
set_train=tf.data.Dataset.from_tensor_slices(train_image)
set_train=set_train.map(procesado)


# In[6]:


# Modelo DGAN
class DGAN(tf.keras.Model):
    def __init__(self,ouput=(28,28,1),n_filters=128,n_blocks=2,vector_size=100):
       super(DGAN,self).__init__()
       self.ouput=ouput
       self.n_blocks=n_blocks
       self.n_filters=n_filters
       self.vector_size=vector_size
       self.factor=2**self.n_blocks
       self.hidden_size=(self.ouput[0]//self.factor,self.ouput[1]//self.factor)

    def generar(self):
       self.generador=tf.keras.Sequential([
           tf.keras.layers.Input(shape=(self.vector_size,)),
           tf.keras.layers.Dense(units=self.n_filters*np.prod(self.hidden_size),use_bias=False),
           tf.keras.layers.BatchNormalization(),
           tf.keras.layers.LeakyReLU(),
           tf.keras.layers.Reshape((self.hidden_size[0],self.hidden_size[1],self.n_filters)),
           tf.keras.layers.Conv2DTranspose(filters=self.n_filters,kernel_size=5,padding='same',use_bias=False),
           tf.keras.layers.BatchNormalization(),
           tf.keras.layers.LeakyReLU()
       ])

       nf=self.n_filters
       for i in range(self.n_blocks):
           nf=nf//2
           self.generador.add(tf.keras.layers.Conv2DTranspose(filters=nf,kernel_size=3,strides=2,padding='same',use_bias=False))
           self.generador.add(tf.keras.layers.BatchNormalization())
           self.generador.add(tf.keras.layers.LeakyReLU())
        
       self.generador.add(tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=3,padding='same',use_bias=False,activation='tanh'))

       return self.generador
    
    def discriminar(self):
        self.discriminador=tf.keras.Sequential([
            tf.keras.Input(shape=self.ouput),
            tf.keras.layers.Conv2D(filters=self.n_filters,kernel_size=5,padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU()
        ])

        nf=self.n_filters
        for i in range(self.n_blocks):
            nf=nf*2
            self.discriminador.add(tf.keras.layers.Conv2D(filters=nf,kernel_size=5,padding='same',strides=2))
            self.discriminador.add(tf.keras.layers.BatchNormalization())
            self.discriminador.add(tf.keras.layers.LeakyReLU())
            self.discriminador.add(tf.keras.layers.Dropout(rate=0.3))
        
        self.discriminador.add(tf.keras.layers.Conv2D(filters=1,kernel_size=7,padding='valid'))
        return self.discriminador


# In[7]:


generar=DGAN()
generador=generar.generar()
generador.summary()


# In[8]:


discriminador=generar.discriminar()
discriminador.summary()


# In[9]:


num_example=16
image_size=(28,28)
num_dim=100
nois=tf.random.uniform(shape=(num_example,num_dim))


# In[10]:


# Funcion para alimentar el modelo, funciones de perdida
crossentropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generador_losses(fake_image):
    return crossentropy(tf.ones_like(fake_image),fake_image)

def discriminador_losses(real_output,fake_image):
    real_losses=crossentropy(tf.ones_like(real_output),real_output)
    fake_losses=crossentropy(tf.zeros_like(fake_image),fake_image)
    total_losses=real_losses+fake_losses
    return total_losses

generador_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
discriminador_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)


all_gen_losses,all_disc_losses=[],[]

@tf.function
def Gradiente(vector,image):
    with tf.GradientTape() as g_tape , tf.GradientTape() as d_tape:
        imagen_generada=generador(vector,training=True)

        real_output=discriminador(image,training=True)
        fake_image=discriminador(imagen_generada,training=True)

        gen_losses=generador_losses(fake_image)
        disc_losses=discriminador_losses(real_output,fake_image)

    generador_gradient=g_tape.gradient(gen_losses,generador.trainable_variables)
    discriminador_gradient=d_tape.gradient(disc_losses,discriminador.trainable_variables)

    generador_optimizer.apply_gradients(zip(generador_gradient,generador.trainable_variables))
    discriminador_optimizer.apply_gradients(zip(discriminador_gradient,discriminador.trainable_variables))
    all_disc_losses.append(disc_losses)
    all_gen_losses.append(gen_losses)


# In[11]:


# Entrenando al modelo

def generador_image(modelo,epoch,test_input):
    prediccion=modelo(test_input,training=False)

    figura=plt.figure(figsize=(8,8))
    for i in range(prediccion.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(prediccion[i, :, :, 0] * 127.5 + 127.5 , cmap='gray')
        plt.axis('off')
    plt.show()
  


def train(dataset,epoch,batch_size=32):
    dataset=dataset.shuffle(buffer_size=60000).batch(batch_size=batch_size,drop_remainder=True)
    for i in range(epoch):
        start=tm.time()
        for vector,image in dataset:
            Gradiente(vector,image)

        display.clear_output(wait=True)
        generador_image(generador,i+1,nois)

        display.clear_output(wait=True)
        generador_image(generador,i+1,nois)
        print('Epoca -> {} tiempo -> {:.2f} min '.format(i+1,(tm.time()-start)/60))


# In[12]:


Epoch=10
batch_size=50
train(set_train,Epoch,batch_size=batch_size)


# In[ ]:




