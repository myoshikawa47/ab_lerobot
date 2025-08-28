import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import ipdb

from eipl.utils import EarlyStopping, check_args, set_logdir, resize_img, normalization

class Processor:
    def __init__(self, img_bounds, vec_bounds, minmax):
        self.img_bounds = img_bounds
        self.vec_bounds = vec_bounds
        self.minmax = minmax
    
    def process_img(self, imgs, img_size):
        _imgs = normalization(imgs, self.img_bounds, self.minmax)
        _imgs = resize_img(_imgs, img_size)
        _imgs = _imgs.transpose(0, 1, 4, 2, 3)
        _imgs = np.clip(_imgs+0.2, 0, 1)
        
        return _imgs
    
    def process_vec(self, vecs):
        _vecs = normalization(vecs, self.vec_bounds, self.minmax)
        return _vecs
    
class Deprocessor:
    def __init__(self, img_bounds, vec_bounds, minmax, select_idxs=[0,2,4,6,8,10,12,14,16]):
        self.img_bounds = img_bounds
        self.vec_bounds = vec_bounds
        self.minmax = minmax
        self.select_idxs = select_idxs
    
    def deprocess_img(self, imgs_hat):
        _imgs_hat = normalization(imgs_hat, self.minmax, self.img_bounds)
        _imgs_hat = _imgs_hat.permute(0,1,3,4,2)
        _imgs_hat = _imgs_hat[self.select_idxs]
        pred_imgs = np.uint8(_imgs_hat.detach().clone().cpu().numpy())
        return pred_imgs
    
    def deprocess_feat(self, feat):
        feat_min = feat.flatten(-2,-1).min(dim=-1)[0].unflatten(-1, (-1,1,1))
        feat_max = feat.flatten(-2,-1).max(dim=-1)[0].unflatten(-1, (-1,1,1))
        _feat = (feat - feat_min) / (feat_max - feat_min)
        _feat = _feat * 255
        _feat = _feat.permute(0,1,3,4,2)
        _feat = _feat[self.select_idxs]
        norm_feat = np.uint8(_feat.detach().clone().cpu().numpy())
        return norm_feat
    
    def deprocess_vec(self, vecs_hat):
        vecs_hat = normalization(vecs_hat, self.minmax, self.vec_bounds)
        vecs_hat = vecs_hat[self.select_idxs]
        pred_vecs = vecs_hat.detach().clone().cpu().numpy()
        return pred_vecs
    
    def deprocess_key(self, key, img_size):   # 3,309,5,2
        batch, seq, _ = key.shape
        _key = key.reshape(batch, seq, -1, 2)
        _key = _key[self.select_idxs]
        x_key = torch.clip(_key[:,:,:,0].unsqueeze(-1)*img_size[1], 0.0, img_size[1])
        y_key = torch.clip(_key[:,:,:,1].unsqueeze(-1)*img_size[0], 0.0, img_size[0])
        _key = torch.cat([x_key, y_key], dim=-1)
        key = np.uint8(_key.detach().clone().cpu().numpy())
        return key