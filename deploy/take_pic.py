import cv2
import argparse
from pathlib import Path
from datetime import datetime
import face_model
import numpy as np
import config
import os
import shutil
from utils import draw_box_name


parser = argparse.ArgumentParser(description='take a picture')
parser.add_argument('-name','-n', default='unknown', type=str,help='input the name of the recording person')
args = parser.parse_args()
conf = config.get_config()


# check folder
bank_path = '../facebank/' + args.name
if os.path.exists(bank_path):
    shutil.rmtree(bank_path)
os.mkdir(bank_path)


cap = cv2.VideoCapture(0)
model = face_model.FaceModel(conf)

while cap.isOpened():

    isSuccess,frame = cap.read()
    key = cv2.waitKey(1)&0xFF
    
    
    bbs, fcs = model.find_faces(frame, conf)
    if len(bbs)==1:
        frame = draw_box_name(bbs[0], "", frame)
        key = cv2.waitKey(1)&0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('t'):
            img = fcs[0][0]
            cv2.imwrite(bank_path+'/'+str('{}.jpg'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-"))), img)
            print('da chup')
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break

    cv2.imshow('camera', frame)

cap.release()
cv2.destroyAllWindows()
