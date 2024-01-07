"""
@autor: MBI
Description: Script to develo SSD model
Tips (SSD):

* El eje de coordenadas comineza upper left y termina botton right

* Si existe mulitples objetos en la escena el detector podra identifiar los que halla entrenado
todo los demas se clasifica como background y no se le asigna boundig-box.

* Objetivo de la deteccion: devolver un cls (en one-host vector),bounding box (Xmin,Ymin,Xmax,Ymax)
o las coordenadas del bounding box en pixeles (no recomendado)

* Para que la red de su resultado esperado la imagen es dividia en anchor boxes, entonces la red estima
el offset con respecto a cada anchor box.

* Una imagen de 640x480 es divida en 2x1 regiones (2 anchor boxes) lo que seria (Xmin,Ymin) y (Xmax - W/2,Ymax - H)
una segunda division seria 3x2 (Xmin - W/3,Ymin) y (Xmax - 2W/3,Ymax - H/2)

* Los anchor boxes que no contribuyan a la optimizacion deben ser suprimidos.

** La cantidad de anchor boxes que se necesitan depende de el tamaño de la imagen y el bounding box mas pequeño
del objeto a detectar. Para una imagen de 640x480 con una regilla de 40x30 de anchor boxes, el anchor box mas
pequeño cubre un espacio de 16x16 pixeles de la imagen de entrada. El numero total de bounding box seria 1608.

* El factor de escala desde el mas pequeño pude ser sumarizado de esta forma: s = [(1/40,1/30),(1/20,1/15),(1/10,1/8),
(1/5,1/4),(1/3,1/2),(1/2,1)] 11.2.1.

* El centroide de cada radio  de aspecto es el mismo como el original anchor box. a = [1,2,3,1/2,1/3] 11.2.2.

* Para cada radio de aspecto el correspondiente bounding box es: (W_i,H_i)= (W_sxj*RAIZ(a_i),H_syj*1/RAIZ(a_i)) 11.2.3
usando 5 diferentes radios de aspecto por anchor boxe, el numero total de anchor boxes es 8040.

* En SSD se recomienda un anchor boxe adicional con dimension (W_5,H_5)= (W*RAIZ(s_j*s_j+1),H*RAIZ(s_j*s_j+1))  11.2.4

* Concepto para considerar un anchor box como true bounding box. Intersection over Union (IoU): IoU = A(Inter)B/A(Union)B
el  ground true anchor boxe es A_j(gt)= max_j(IoU(B_i,A_j)) 11.3.2 

* La categoria y el offset de un anchor box positivo es determinada con respecto al ground true bounding box. La categoria 
de un anchor box positivo es la misma que la de un ground true bounding box. 

* Las funcion de perdida esta compuesta de categorical crossentropy para cls y L1 or  L2 para regresion L1 (MAE) y L2 (MSE) 
la perdida total es O = O + sigma*Gradiente(O). Como SSD es un tipo de entrenamiento supervisado lo sigueiente se cumple:
Y_label = etiqueta de clase,Y_gt = (X_bmin - X_amin,X_bmax - X_amax,Y_bmin - Y_amin,Y_bmax - Y_amax) 11.4.3. Se recomineda 
primero normalizar el valor de offset y el ground true bounding box: Y_box = ((X_bmin,Y_bmin),(X_bmax,Y_bmax)) -> (C_bx,C_by,W_b,H_b)
Y_anchor = ((X_amin,Y_amin),(X_amax,Y_amax)) -> (C_ax,C_ay,W_a,H_a). Para convertir se utiliza la siguiente ecuacion:
(C_bx,C_by) = (X_min + (X_max - X_min)/2,Y_min + (Y_max - Y_min)/2) 11.4.5 
(W_b,H_b) = (X_max - X_min,Y_max - Y_min) 11.4.6 
El ground true offset normalizado es Y_gt = (C_bx-C_ax)/W_a/std_x,(C_by-C_ay)/H_a/std_y,log(W_b/W_a)/std_w,log(H_b/H_a)/std_h) 11.4.8
valores recomendados para la std: std_x = std_y = 0.1 y std_w = std_h = 02 (para aliviar el desvanecimiento del gradiente) en otras pala-
bras el rango esperado de pixeles de error para x y y es de +-10%.

* Uso de una funcion de perdida mas robusta a los valores extremos (L1_smooth): L1_smooth = (std*u)^2/2, if |u| < 1/std^2 o |u| - 1/(2*std^2)

"""
#%% Packages
from __future__ import unicode_literals,absolute_import
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler
import layer_utils,os,skimage
import numpy as np
from skimage.io import imread
from data_generator import DataGenerator
from label_utils import build_label_dictionary
from boxes import show_boxes
from ssd_header import build_ssd
from loss_functions import focal_loss_categorical,smooth_l1_loss,l1_loss


#%% Class

"""
Descripcion de variables:
args: lista que contiene: 0:class threshold,1:soft nms,2:iou threshold,3:n layers,4:normalizado,5:batch size,6:epochs,
train_labels: csv de etiquetas de entrenamiento
"""

class SSD():
    def __init__(self,args:list,input_shape:tuple,loss_type:str='smooth_l1_loss',path:str='',train_labels:str='',test_label:str='') -> None:
        self.input_shape = input_shape # (480,640,3)
        self.ssd = None
        self.args = args
        self.train_labels = train_labels
        self.test_label = test_label
        self.train_generator = None
        self.loss_type = loss_type
        self.n_anchors = None
        self.feature_shape = None
        self.path = path
        self.save_dir = os.getcwd()
        self.build_model()


    def build_model(self) -> None:
        self.build_dictionary()
        self.n_anchors,self.feature_shape,self.ssd = build_ssd(self.input_shape,n_layers=self.args[3],
                                                                n_classes=self.n_classes)
        self.build_generator()
        
    
    def build_dictionary(self) -> None:
        path = os.path.join(self.path,self.train_labels)
        self.dictionary,self.classes = build_label_dictionary(path)
        self.n_classes =  len(self.classes)
        self.keys = np.array(list(self.dictionary.keys()))
    
    def build_generator(self) -> None:
        self.train_generator = DataGenerator(input_shape=self.input_shape,
                                            dictionary=self.dictionary,
                                            n_classes=self.n_classes,
                                            feature_shapes=self.feature_shape,
                                            n_anchors=self.n_anchors,
                                            batch_size=self.args[5],
                                            data_path=self.path,
                                            shuffle=True)

    def train(self) -> None:
        optimizer = Adam(lr=1e-3)
        if self.loss_type == 'focal_loss_categorical':
            loss = [focal_loss_categorical,smooth_l1_loss]

        elif self.loss_type == 'categorical_crossentropy':
            loss = ['categorical_crossentropy',smooth_l1_loss]

        else:
            loss = ['categorical_crossentropy',l1_loss]
        
        self.ssd.compile(optimizer=optimizer,loss=loss)

        self.model_name = "SSD_"+"{}.h5".format(self.loss_type)

        if not os.path.isdir(self.save_dir+'\\SSD_save'):
            os.mkdir(self.save_dir+'\\SSD_save')
        
        filepath = os.path.join(self.save_dir+'\\SSD_save',self.model_name)
        chexpoint = ModelCheckpoint(
            filepath=filepath,
            verbose=1,
            save_weights_only =True
        )
        scheduler = LearningRateScheduler(self.lr_scheduler)
        callbacks = [chexpoint,scheduler]

        self.ssd.fit(self.train_generator,use_multiprocessing=False,callbacks=callbacks,
                    epochs=self.args[6])

    def restore_weights(self,model:str) -> None:
        try:
            filename = self.save_dir+"\\SSD_save\\"+model
            self.ssd.load_weights(filename)
            print("[+]Weights loaded...")
        except:
            raise Exception("Not found directory")
        
    def detect_object(self,image) -> tuple:
        image = np.expand_dims(image,axis=0)
        classes,offsets = self.ssd.predict(image)
        image = np.squeeze(image,axis=0)
        classes = np.squeeze(classes)
        offsets = np.squeeze(offsets)
        return image,classes,offsets
    
    def evaluate(self,image_file=None,image=None) -> tuple:
        show = False
        if image is None:
            image = skimage.img_as_float(imread(image_file))
            show = True
        
        image,classes,offsets = self.detect_object(image)
        class_name,rects,_,_ = show_boxes(self.args,image,classes,offsets,self.feature_shape,show=show)

        return class_name,rects
    
    def evaluate_test(self) -> None:
        path = os.path.join(self.path,self.test_label)

        dictionary,_ = build_label_dictionary(path)
        keys = np.array(list(dictionary.keys()))
        s_precision = 0
        s_recall = 0
        s_iou = 0

        for key in keys:
            labels = np.array(dictionary[key])
            gt_boxes = labels[:,0:-1]
            gt_class_ids = labels[:,-1]
            image_file = os.path.join(self.path,key)
            image = skimage.img_as_float(imread(image_file))
            image,classes,offsets = self.detect_object(image)

            _,_,class_ids,boxes = show_boxes(self.args,image,classes,offsets,self.feature_shape,show=False)

            boxes = np.reshape(np.array(boxes),(-1,4))
            iou = layer_utils.iou(gt_boxes,boxes)
            if iou.size == 0:
                continue
            maxiou_class = np.argmax(iou,axis=1)
            tp = 0
            fp = 0
            s_image_iou = []
            for n in range(iou.shape[0]):
                if iou[n,maxiou_class[n]] > 0:
                    s_image_iou.append(iou[n,maxiou_class[n]])
                    if gt_class_ids[n] == class_ids[maxiou_class[n]]:
                        tp += 1
                    else: fp += 1
            
            fn = abs(len(gt_class_ids) - tp)
            s_iou += (np.sum(s_image_iou) / iou.shape[0])
            try:
                s_precision += (tp / (tp + fp))
                s_recall += (tp / (tp + fn))
            except ZeroDivisionError:
                continue
        
        n_test = len(keys)
        print("mIoU: {}".format(round(s_iou/n_test,3)))
        print("Precision: {}".format(round(s_precision/n_test,3)))
        print("Recall: {}".format(round(s_recall/n_test,3)))
    
    def print_summary(self) -> None:
        self.ssd.summary()
        plot_model(self.ssd,to_file=self.save_dir+"\\backbone.png",show_shapes=True,dpi=100)

    def lr_scheduler(self,epoch) -> float:
        lr = 1e-3
        if (epoch > 50): lr *= 1e-4
        elif (epoch > 100): lr *= 5e-4
        elif (epoch > 150): lr *= 5e-1

        print('Learning rate: ',lr)
        return lr 



#%% Main

parameters = [0.50,True,0.20,4,True,1,200]
losses_dict:dict = {
    0:'focal_loss_categorical',
    1:'categorical_crossentropy',
}
train_data = "C:\\Users\\MBI\\Documents\\Python_Scripts\\Datasets\\drinks"
test_label = 'labels_test.csv'
train_labels = "labels_train.csv"
input_shape = (480,640,3)
ssd = SSD(parameters,input_shape,loss_type=losses_dict[0],path=train_data,train_labels=train_labels,test_label=test_label)
#ssd.print_summary()

#%% Training

ssd.train()
# The training was stoped in 23 epoch whit mIoU: 38.9% Precesion: 73.2% Recall: 64.4%

#%% Evaluate

ssd.evaluate(train_data+"\\0010040.jpg")

#%% Load model

ssd.restore_weights('SSD_focal_loss_categorical.h5')

#%% Evaluate test

ssd.evaluate_test()
