#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.compose import make_column_selector,make_column_transformer
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from category_encoders import OrdinalEncoder
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(1)
np.random.seed(1)


# In[19]:


dato_train=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/titanic_train.csv',index_col=False,sep=',')
dato_test=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/titanic_eval.csv',index_col=False,sep=',')


# In[20]:


dato_test.pop('jj'),dato_train.pop('ii')


# In[21]:


dato_train.isna().sum(),dato_test.isna().sum()


# In[32]:


dato_test_copy=dato_test.drop_duplicates().copy()
dato_train_copy=dato_train.drop_duplicates().copy()
dato_train_copy=dato_train_copy.reindex(np.random.permutation(dato_train_copy.index))


# In[33]:


label='survived'
X_train,X_val,y_train,y_val=train_test_split(dato_train_copy.drop(label,axis=1),dato_train_copy[label],train_size=0.70,random_state=42,shuffle=True)
X_train.shape,y_train.shape,X_val.shape,y_val.shape


# In[162]:



transformer=make_column_transformer(
    (StandardScaler(),make_column_selector(dtype_include=np.number)),
    (OrdinalEncoder(),make_column_selector(dtype_include=np.object)),
)


# In[56]:


def my_input_fn(X,y,batch_size=20,shuffle=True,num_epochs=None):
    X_trans=transformer.fit_transform(X)
    if y is None:
        ds=tf.data.Dataset.from_tensor_slices((X_trans)).batch(batch_size)
        return ds
    else:
        etiqueta=np.array(y)
        ds=tf.data.Dataset.from_tensor_slices((X_trans,etiqueta))
    if shuffle :
        ds=ds.shuffle(len(y))
    ds=ds.batch(batch_size).repeat(num_epochs)
    return ds 


# In[128]:


# Modelo 
model=tf.keras.Sequential([
    tf.keras.layers.Dense(units=2,activation='relu'),
    # Output layer
    tf.keras.layers.Dense(units=1,activation='sigmoid')
])


# In[129]:


model.build(input_shape=(None,X_train.shape[1]))
model.summary()


# In[146]:


model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01,rho=0.9),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy','mse']
)


# In[147]:


epocas=50
batch_size=32
step_x_epocas=int(np.ceil(len(y_train)/batch_size))*epocas

trainig=my_input_fn(X_train,y_train,batch_size=batch_size,num_epochs=None)
evaluate=my_input_fn(X_val,y_val,shuffle=False,num_epochs=None,batch_size=batch_size)

hist=model.fit(trainig,epochs=epocas,steps_per_epoch=step_x_epocas,validation_data=evaluate,validation_steps=step_x_epocas)


# In[148]:


X_test=dato_test_copy.drop(label,axis=1)
test=my_input_fn(X_test,y=None)


# In[149]:


predict=list(model.predict(test))
data_short=pd.Series([item for item in predict ])
fps,tpr,_=roc_curve(dato_test_copy[label],data_short)
plt.plot(fps,tpr)
plt.title('ROC curve')
plt.xlabel('falsos positivos')
plt.ylabel('verdaderos positivos')
plt.xlim(0,)
plt.ylim(0,)
plt.show();


# In[153]:


# Probando con modelo  lineal

Columns_cat=X_train.select_dtypes([np.object]).columns
Columns_num=X_train.drop('age',axis=1).select_dtypes([np.number]).columns

feature_num,feature_cat,feature_bukat=[],[],[]

age_bukat=tf.feature_column.numeric_column(key='age')
feature_bukat.append(tf.feature_column.bucketized_column(source_column=age_bukat,boundaries=[20,30,60]))

for col in Columns_cat:
    dtype=X_train[col].unique()
    feature_cat.append(tf.feature_column.categorical_column_with_vocabulary_list(key=col,vocabulary_list=dtype))

for col in Columns_num:
    feature_num.append(tf.feature_column.numeric_column(key=col))

