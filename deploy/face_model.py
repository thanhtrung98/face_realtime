import sys
import os
import numpy as np
import mxnet as mx
import cv2
import sklearn
from sklearn.decomposition import PCA
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
import face_preprocess
import torch



def do_flip(data):
  for idx in range(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (conf.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (conf.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model

class FaceModel:
  def __init__(self, conf):
    self.conf = conf
    if conf.gpu>=0:
      ctx = mx.gpu(conf.gpu)
    elif conf.gpu<0:
      ctx = mx.cpu()
    _vec = conf.image_size.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.model = None
    self.ga_model = None
    if len(conf.model_path)>0:
      self.model = get_model(ctx, image_size, conf.model_path, 'fc1')

    self.threshold = conf.threshold
    self.det_minsize = 50 #minimun size of bb of mtcnn
    self.det_threshold = [0.6,0.7,0.8] #bo fail bb
    #self.det_factor = 0.9
    self.image_size = image_size
    mtcnn_path = os.path.join('../models/mtcnn-model')
    if conf.det==0:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold)
    else:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
    self.detector = detector

  def find_faces(self, img, conf):
    faces = []
    boundb = []
    try:
      bounding_boxes, points= self.detector.detect_face(img, det_type = 0)
    except:
      return boundb, faces
    for bb, point in zip(bounding_boxes, points):
      bb=bb[0:4]
      bb = [int(i/conf.resize_ratio) for i in bb]
      point = np.array([int(i/conf.resize_ratio) for i in point])
      
      point=point.reshape((2,5)).T
      nimg = face_preprocess.preprocess(img, bb, point, image_size='112,112')
      nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
      aligned = np.transpose(nimg, (2,0,1))
      faces.append(aligned)
      boundb.append(bb)
    return boundb, faces

  def get_feature(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    embedding = self.model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding

  def infer(self, faces, target_embs, tta=False):
      '''
      faces : list of PIL Image
      target_embs : [n, 512] computed embeddings of faces in facebank
      names : recorded names of faces in facebank
      tta : test time augmentation (hfilp, that's all)
      '''
      embs = []
      # for img in faces: # for multi faces
      img = faces
      if tta:
          mirror = trans.functional.hflip(img)
          emb = torch.from_numpy(self.get_feature(img))
          emb_mirror = torch.from_numpy(self.get_feature(mirror))
          embs.append(l2_norm(emb + emb_mirror))
      else:
          embs.append(torch.from_numpy(self.get_feature(img)))
      source_embs = torch.cat(embs)
      diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
      dist = torch.sum(torch.pow(diff, 2), dim=1)
      minimum, min_idx = torch.min(dist, dim=1)
      min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
      return min_idx, minimum
