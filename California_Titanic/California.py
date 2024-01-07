#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(1)
np.random.seed(1)


# In[2]:


datos_train=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/california_housing_train.csv',index_col=False,sep=',')
datos_test=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/california_housing_test.csv',index_col=False,sep=',')


# In[3]:


datos_train.head()


# In[4]:


Metricas=datos_train.describe().T.copy()
Metricas


# In[5]:


datos_train['median_house_value']=datos_train['median_house_value']/1000.0
datos_train=datos_train.reindex(np.random.permutation(datos_train.index))
datos_train.head()


# In[6]:


datos_train.corr()


# In[7]:


datos_train.plot('longitude','latitude');


# In[8]:


datos_train.hist(['median_house_value','housing_median_age']);


# In[9]:


datos_train['median_house_value'].plot.kde();


# In[10]:


datos_train['housing_median_age'].plot.kde();


# In[11]:


datos_train.columns


# In[12]:


label='median_house_value'
bukatizar='housing_median_age'
Columns_extandar=datos_train.drop([label,bukatizar],axis=1).columns
Columns_extandar


# In[13]:


datos_train.isna().sum(),datos_train.duplicated().sum()


# In[14]:


housing_median_age=tf.feature_column.numeric_column(key='housing_median_age')
feature_bukatize=[]
feature_bukatize.append(tf.feature_column.bucketized_column(source_column=housing_median_age,boundaries=[20,30,40]))


# In[15]:


datos_train_norm,datos_test_norm=datos_train.copy(),datos_test.copy()

for col in Columns_extandar:
    mean=Metricas.loc[col,'mean']
    std=Metricas.loc[col,'std']
    datos_train_norm.loc[:,col]=(datos_train.loc[:,col]-mean)/std
    datos_test_norm.loc[:,col]=(datos_test.loc[:,col]-mean)/std
datos_train_norm.head()


# In[16]:


X_train,X_val,y_train,y_val=train_test_split(datos_train_norm.drop(label,axis=1),datos_train_norm[label],train_size=0.80,random_state=42,shuffle=True)
X_train.shape,y_train.shape,X_val.shape,y_val.shape


# In[17]:


feature_columns=[]
for col in Columns_extandar:
  feature_columns.append(tf.feature_column.numeric_column(key=col))

all_columns=(feature_columns+feature_bukatize)


# In[18]:


def my_input_fn(X,y,batch_size=32,shuffle=True,num_epochs=None):
    
    if y is  None:
        ds=tf.data.Dataset.from_tensor_slices((dict(X))).batch(batch_size).repeat(count=1)
        return ds
    else:
        ds=tf.data.Dataset.from_tensor_slices((dict(X),y)).batch(batch_size).repeat(count=1)
    if shuffle:
        ds=ds.shuffle(len(y))
    ds=ds.batch(batch_size=batch_size).repeat(count=num_epochs)
    return  ds

def metrica(y,pred):
    train_metric=np.sqrt(mean_squared_error(y,pred))
    return train_metric


# In[19]:


layers=[10,10]
epochs=20
batch_size=50
step_per_epochs=int(np.ceil(len(y_train)/batch_size))*epochs
print('Pasos en cada epoca ',step_per_epochs)
stimator=tf.estimator.DNNRegressor(hidden_units=layers,feature_columns=all_columns,optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.01),dropout=0.08)


# In[25]:


stimator.train(
    input_fn=lambda:my_input_fn(X_train,y_train,batch_size=batch_size),
    max_steps=step_per_epochs)

resultado=stimator.evaluate(input_fn=lambda:my_input_fn(X_val,y_val,batch_size=batch_size,shuffle=False),steps=step_per_epochs)
print(resultado)


# In[26]:


predict=stimator.predict(input_fn=lambda:my_input_fn(X_val,y_val,batch_size=batch_size,shuffle=False,num_epochs=1))
predict=np.array([item['predictions'][0] for item in predict])
metric=metrica(y_val,predict)
print('Raiz del errror cuadratico medio :',metric)


# In[28]:


predict_test=stimator.predict(input_fn=lambda:my_input_fn(datos_test_norm.drop(label,axis=1),datos_test_norm[label],shuffle=False,num_epochs=1))
predict_test=np.array([item['predictions'][0] for item in predict_test])
datos_test_norm[label]=datos_test_norm[label]/1000.0
metric=metrica(datos_test_norm[label],predict_test)
print('Raiz del errror cuadratico medio :',metric)


# In[ ]:




