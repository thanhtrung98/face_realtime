from easydict import EasyDict as edict
from pathlib import Path

def get_config():
    conf = edict()
    conf.facebank_path = Path('../facebank') #file data
    #conf.model_path = '../models/model-r100-ii/model,0' #link of models
    conf.model_path = '../models/model-y1-test2/model,0' #link of models
    conf.image_size = '112,112' # size of face boudingbox
    conf.embedding_size = 512 # size of vector of face
    conf.gpu = -1  # gpu id, < 0 means using CPU
    conf.flip = 0 # doi xung guong
    conf.det = 0 # mtcnn option, 1 means using R+O, 0 means using all net (P R O)
    conf.threshold = 0.8 #ver dist threshold
    conf.resize_ratio = 1 #
    return conf
