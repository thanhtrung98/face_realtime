import numpy as np
from torchvision import transforms as trans
import torch
import cv2

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output


def draw_box_name(bbox,name,frame):
    frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
    frame = cv2.putText(frame,
                    name,
                    (bbox[0],bbox[1]), 
                    cv2.FONT_HERSHEY_TRIPLEX, 
                    1,
                    (100,255,0),
                    3,
                    cv2.LINE_AA)
    return frame
