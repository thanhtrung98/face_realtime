import time
import cv2
import argparse
import face_model
from config import get_config
from matplotlib.image import imread
import numpy as np
import os
import shutil


# check folder
bank_path = '../embedding'
if os.path.exists(bank_path):
    shutil.rmtree(bank_path)
os.mkdir(bank_path)


if __name__ == '__main__':
    conf = get_config()
    model = face_model.FaceModel(conf)
    bank_path = '../facebank'
    face = imread(bank_path + '/Thi/2019-07-26-22-08-26.jpg')
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = np.transpose(face, (2,0,1))
    emb = model.get_feature(face)

    for folder in os.listdir(bank_path):
        for file in os.listdir(bank_path + '/' + folder):
            Linh = bank_path + '/' + folder + '/' + file
            face = imread(Linh)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = np.transpose(face, (2,0,1))
            emb = model.get_feature(face)
            f = open('../embedding/'+folder+'_'+file+".txt", "w")
            for i in emb:
                f.write(str(i) + '\n')
