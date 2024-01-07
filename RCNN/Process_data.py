"""
@author: MBI
Decripcion: Script para la preparacion de los datos.
"""
#=== Modulos ===#
import os,cv2
import pandas as pd
#=== Funciones ===#

anotation = "Airplanes_Annotations"
pathImage = 'Images'


def get_IOU(bb1,bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def processData():
    segmentation = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    for id, valor in enumerate(os.listdir(anotation)):
        if valor.startswith('airplane'):
            file = valor.split('.')[0] + '.jpg'
            print(file)
            image = cv2.imread(os.path.join(pathImage, file))
            ds = pd.read_csv(os.path.join(anotation, valor))
            list_values = []
            for row in ds.iterrows():
                x1 = int(row[1][0].split(' ')[0])
                y1 = int(row[1][0].split(' ')[1])
                x2 = int(row[1][0].split(' ')[2])
                y2 = int(row[1][0].split(' ')[3])
                list_values.append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})
            segmentation.setBaseImage(image)
            segmentation.switchToSelectiveSearchFast()
            results = segmentation.process()
            imageOut = image.copy()
            counter_true = 0
            counter_false = 0
            flag = 0
            bflag = 0
            fflag = 0

            for id, rec in enumerate(results):
                if id < 2000 and flag == 0:
                    for getvalue in list_values:
                        x, y, w, h = rec
                        iou = get_IOU(getvalue, {'x1': x, 'x2': x + w, 'y1': y, 'y2': y + h})
                        if counter_true < 30:
                            if iou > 0.7:
                                targetImage = imageOut[y:y + h, x:x + w]
                                image_resize = cv2.resize(targetImage, (224, 224), interpolation=cv2.INTER_AREA)
                                counter_true += 1
                                # Guardando en directorio
                                cv2.imwrite('./Airports_1/'+ valor.split('.')[0]+str(counter_true)+'.jpg',image_resize)

                        else:
                            fflag = 1

                        if counter_false < 30:
                            if iou < 0.3:
                                targetImage = imageOut[y:y + h, x:x + w]
                                image_resize = cv2.resize(targetImage, (224, 224), interpolation=cv2.INTER_AREA)
                                counter_false += 1
                                # Guardando en directorio
                                cv2.imwrite('./Airports_0/' + valor.split('.')[0] + str(counter_false) + '.jpg', image_resize)


                        else:
                            bflag = 1

                    if bflag == 1 and fflag == 1:
                        flag = 1

#=== Main() ===#
def main():
    if  not os.path.exists("dataset/Airports_1") and not os.path.exists('dataset/Airports_0'):
        os.makedirs(os.getcwd() + '\\'+ 'Airports_1')
        os.makedirs(os.getcwd()+ '\\'+ 'Airports_0')
    processData()

main()