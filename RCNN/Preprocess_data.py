"""
@author:MBI
Descripcion: Script para el preprocesamiento de los datos. Convierte de xml a csv.
"""
#===Modulos===#
import glob
from xml.etree import ElementTree as ET
from lxml import etree

#===Funciones===#

def xml2csv(path):
    for xml_file in glob.glob(path+'/*.xml'):
        tree = ET.parse(xml_file)
        doc = etree.parse(xml_file)
        count = doc.xpath("count(//object)") # cantidad de objetos en el xml
        root = tree.getroot()

        with open(str(xml_file)[0:-4]+'.csv','w+') as f:
            f.write(str(int(count)))

        for member in root.findall('object'): # Buscando las coordenadas
            valor = (
                member[4][0].text,
                member[4][1].text,
                member[4][2].text,
                member[4][3].text
            )
            coord = " ".join(valor)
            with open(str(xml_file)[0:-4]+'.csv','a') as f:
                f.write('\n')
                f.write(coord)

def convertion(path):
    xml2csv(path)
    print('Successfully converted xml to csv.')
