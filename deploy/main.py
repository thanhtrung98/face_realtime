import time
import cv2
import argparse
import face_model
from utils import draw_box_name
from config import get_config
import os
import numpy as np

def upload():
    labels = []
    for file in os.listdir("../embedding"):
        a = "../embedding/"+file
        f = open(a,"r")
        label = []
        for i in f:
            label.append(float(i))
        labels.append([file.split('.')[0], label])  
    return labels

def get_name(emb, labels):
    min_dis = 100000
    name_res = "unknow"
    emb = np.array(emb)
    for name, emb2 in labels:
        emb2 = np.array(emb2)
        dist = np.sum(np.square(emb-emb2))
        if dist < min_dis:
            min_dis = dist
            if dist < 0.8:  #threshold
                name_res = name
    name_res = name_res.split('_')[0]
    return name_res +' score: '+ str( float(int(min_dis*100))/100)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face verify')
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")    
    parser.add_argument("-s", "--save", help="whether save",action="store_true")
    parser.add_argument("-c", "--score",default=True, help="whether show the confidence score",action="store_true")

    args = parser.parse_args()
    conf = get_config()
    model = face_model.FaceModel(conf)
    
    if args.update:
        print('facebank updated')
    else:
        print('facebank loaded')

    
    labels = upload()
    
    # inital camera
    cap = cv2.VideoCapture("test.mov")
    if args.save:
        video_writer = cv2.VideoWriter(str(conf.data_path/'recording.mov'), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),6, (1280,720))
    while cap.isOpened():
       # start_time = time.time()
        _, frame = cap.read()
        if frame is None:
            break
        start_time = time.time()
        bbs, fcs = model.find_faces(frame, conf)
        
        for bb, fc in zip(bbs, fcs):
           # start_time = time.time()
            emb = model.get_feature(fc)
       
            name = get_name(emb, labels)
            frame = draw_box_name(bb, name, frame)
        cv2.putText(frame,'FPS: ' + str(1.0 / (time.time() - start_time)),(50,50)
            , cv2.FONT_HERSHEY_SIMPLEX, 1, (0 ,0 ,255),2,cv2.LINE_AA)
        # cv2.imshow('face Capture', frame)
        print(name+' FPS: ' + str(1.0 / (time.time() - start_time)))
        # save video
        if args.save:
            video_writer.write(frame)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
        
    cap.release()
    if args.save:
        video_writer.release()

    cv2.destroyAllWindows()    
