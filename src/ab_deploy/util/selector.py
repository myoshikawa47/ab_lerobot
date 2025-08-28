import torch
import torch.nn as nn

import numpy as np
import matplotlib as plt

import os
import sys
import ipdb

sys.path.append("../")
from model import SARNN, TopKSARNN, TopKDualSARNN, DualSARNN, HRNN, FasterHRNN
from model import Conv3dSARNN, FlowSARNN
from model import StackRNN, SAStackRNN, FoveaSAStackRNN
from model import VAE, NoGradVAE, LoRAVAE

from libs import fullBPTTtrainer4SARNN, fullBPTTtrainer4TopKSARNN, fullBPTTtrainer4TopKDualSARNN, fullBPTTtrainer4DualSARNN, fullBPTTtrainer4HRNN, fullBPTTtrainer4FasterHRNN
from libs import fullBPTTtrainer4Conv3dSARNN, fullBPTTtrainer4FlowSARNN
from libs import fullBPTTtrainer4StackRNN, fullBPTTtrainer4SAStackRNN, fullBPTTtrainer4FoveaSAStackRNN
from libs import fullBPTTtrainer4VAE, fullBPTTtrainer4NoGradVAE, fullBPTTtrainer4LoRAVAE

from inf import Inf4SARNN, Inf4TopKSARNN, Inf4TopKDualSARNN, Inf4DualSARNN, Inf4HRNN, Inf4StackRNN, Inf4SAStackRNN, Inf4FasterHRNN
from inf import Inf4Conv3dSARNN, Inf4FlowSARNN

class TrainSelector:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.loss_weight_dict = None
        
    def select_model(self):
        # define model
        if self.args.model_name in ["sarnn",]:
            self.model = SARNN(
                union_dim=self.args.hid_dim,
                state_dim=self.args.vec_dim,
                key_dim=self.args.key_dim,
                heatmap_size=self.args.heatmap_size,
                temperature=self.args.temperature,
            )
        elif self.args.model_name in ["topksarnn",]:
            self.model = TopKSARNN(
                union_dim=self.args.hid_dim,
                state_dim=self.args.vec_dim,
                key_dim=self.args.key_dim,
                top_k=self.args.top_k,
                heatmap_size=self.args.heatmap_size,
                temperature=self.args.temperature,
            )
        elif self.args.model_name in ["dualsarnn",]:
            self.model = DualSARNN(
                union_dim=self.args.hid_dim,
                state_dim=self.args.vec_dim,
                key_dim=self.args.key_dim,
                heatmap_size=self.args.heatmap_size,
                temperature=self.args.temperature,
            )
        elif self.args.model_name in ["conv3dsarnn",]:
            self.model = Conv3dSARNN(
                union_dim=self.args.hid_dim,
                state_dim=self.args.vec_dim,
                key_dim=self.args.key_dim,
                heatmap_size=self.args.heatmap_size,
                temperature=self.args.temperature,
            )
        elif self.args.model_name in ["flowsarnn",]:
            self.model = FlowSARNN(
                union_dim=self.args.hid_dim,
                state_dim=self.args.vec_dim,
                key_dim=self.args.key_dim,
                heatmap_size=self.args.heatmap_size,
                temperature=self.args.temperature,
            )
        elif self.args.model_name in ["topkdualsarnn",]:
            self.model = TopKDualSARNN(
                union_dim=self.args.hid_dim,
                state_dim=self.args.vec_dim,
                key_dim=self.args.key_dim,
                top_k=self.args.top_k,
                heatmap_size=self.args.heatmap_size,
                temperature=self.args.temperature,
            )
        elif self.args.model_name in ["stackrnn"]:
            self.model = StackRNN(
                img_feat_dim=self.args.vec_dim,
                vec_dim=self.args.vec_dim,
                union1_dim=self.args.hid_dim,
                union2_dim=self.args.hid_dim,
                union3_dim=self.args.hid_dim,
                rnn_type=self.args.rnn_type
            )
        elif self.args.model_name in ["sastackrnn"]:
            self.model = SAStackRNN(
                key_dim=self.args.key_dim,
                vec_dim=self.args.vec_dim,
                union1_dim=self.args.hid_dim,
                union2_dim=self.args.hid_dim,
                union3_dim=self.args.hid_dim,
                rnn_type=self.args.rnn_type
            )
        elif self.args.model_name in ["foveasastackrnn"]:
            self.model = FoveaSAStackRNN(
                key_dim=self.args.key_dim,
                vec_dim=self.args.vec_dim,
                union1_dim=self.args.hid_dim,
                union2_dim=self.args.hid_dim,
                union3_dim=self.args.hid_dim,
                rnn_type=self.args.rnn_type
            )
        elif self.args.model_name in ["hrnn"]:
            self.model = HRNN(
                img_feat_dim=self.args.vec_dim,
                vec_dim=self.args.vec_dim,
                sensory_dim=self.args.hid_dim,
                union1_dim=self.args.hid_dim,
                union2_dim=self.args.hid_dim,
            )
        elif self.args.model_name in ["fasterhrnn"]:
            self.model = FasterHRNN(
                img_feat_dim=self.args.vec_dim,
                vec_dim=self.args.vec_dim,
                sensory_dim=int(self.args.hid_dim/2),
                union1_dim=self.args.hid_dim,
                union2_dim=self.args.hid_dim,
            )
        elif self.args.model_name in ["vae"]:
            self.model = VAE(
                img_feat_dim=self.args.vec_dim,
            )
        elif self.args.model_name in ["nogradvae"]:
            self.model = NoGradVAE(
                img_feat_dim=self.args.vec_dim,
            )
        elif self.args.model_name in ["loravae"]:
            self.model = LoRAVAE(
                img_feat_dim=self.args.vec_dim,
            )
        else:
            print(f"{self.args.model_name} is invalid model")
            exit()
        
        return self.model
    
    def select_loss_weight(self):
        if self.model == None:
            print(f"{self.args.model_name} is invalid model")
            exit()
        
        if self.args.model_name in ["sarnn",
                                    "topksarnn",
                                    "dualsarnn",
                                    "conv3dsarnn",
                                    "flowsarnn",
                                    "topkdualsarnn",
                                    "sastackrnn",
                                    "foveasastackrnn"]:
            self.loss_weight_dict = {"img": self.args.img_loss, "vec": self.args.vec_loss, "key": self.args.key_loss}
        elif self.args.model_name in ["stackrnn",
                                      "hrnn",
                                      "fasterhrnn"]:
            self.loss_weight_dict = {"img": self.args.img_loss, "vec": self.args.vec_loss}
        elif self.args.model_name in ["vae",
                                      "nogradvae",
                                      "loravae",]:
            self.loss_weight_dict = {"img": self.args.img_loss,}
        else:
            print(f"{self.args.model_name} is invalid model")
            exit()
        return self.loss_weight_dict
    
    def select_trainer(self, optimizer, device):
        if self.model == None or self.loss_weight_dict == None:
            print(f"{self.args.model_name} is invalid model")
            exit()
        
        if self.args.model_name in ["sarnn",]:
            trainer = fullBPTTtrainer4SARNN(
                self.model, 
                optimizer, 
                loss_weight_dict=self.loss_weight_dict, 
                device=device)
        elif self.args.model_name in ["topksarnn",]:
            trainer = fullBPTTtrainer4TopKSARNN(
                self.model, 
                optimizer, 
                loss_weight_dict=self.loss_weight_dict, 
                device=device)
        elif self.args.model_name in ["dualsarnn",]:
            trainer = fullBPTTtrainer4DualSARNN(
                self.model, 
                optimizer, 
                loss_weight_dict=self.loss_weight_dict, 
                device=device)
        elif self.args.model_name in ["conv3dsarnn",]:
            trainer = fullBPTTtrainer4Conv3dSARNN(
                self.model, 
                optimizer, 
                loss_weight_dict=self.loss_weight_dict, 
                device=device)
        elif self.args.model_name in ["flowsarnn",]:
            trainer = fullBPTTtrainer4FlowSARNN(
                self.model, 
                optimizer, 
                loss_weight_dict=self.loss_weight_dict, 
                device=device)
        elif self.args.model_name in ["topkdualsarnn",]:
            trainer = fullBPTTtrainer4TopKDualSARNN(
                self.model, 
                optimizer, 
                loss_weight_dict=self.loss_weight_dict, 
                device=device)
        elif self.args.model_name in ["stackrnn",]:
            trainer = fullBPTTtrainer4StackRNN(
                self.model, 
                optimizer, 
                loss_weight_dict=self.loss_weight_dict, 
                device=device)
        elif self.args.model_name in ["sastackrnn",]:
            trainer = fullBPTTtrainer4SAStackRNN(
                self.model, 
                optimizer, 
                loss_weight_dict=self.loss_weight_dict, 
                device=device)
        elif self.args.model_name in ["foveasastackrnn",]:
            trainer = fullBPTTtrainer4FoveaSAStackRNN(
                self.model, 
                optimizer, 
                loss_weight_dict=self.loss_weight_dict, 
                device=device)
        elif self.args.model_name in ["hrnn",]:
            trainer = fullBPTTtrainer4HRNN(
                self.model, 
                optimizer, 
                loss_weight_dict=self.loss_weight_dict, 
                device=device)
        elif self.args.model_name in ["fasterhrnn",]:
            trainer = fullBPTTtrainer4FasterHRNN(
                self.model, 
                optimizer, 
                loss_weight_dict=self.loss_weight_dict, 
                device=device)
        elif self.args.model_name in ["vae",]:
            trainer = fullBPTTtrainer4VAE(
                self.model, 
                optimizer, 
                loss_weight_dict=self.loss_weight_dict, 
                device=device)
        elif self.args.model_name in ["nogradvae",]:
            trainer = fullBPTTtrainer4NoGradVAE(
                self.model, 
                optimizer, 
                loss_weight_dict=self.loss_weight_dict, 
                device=device)
        elif self.args.model_name in ["loravae",]:
            trainer = fullBPTTtrainer4LoRAVAE(
                self.model, 
                optimizer, 
                loss_weight_dict=self.loss_weight_dict, 
                device=device)
        else:
            print(f"{self.args.model_name} is invalid model")
            exit()
        return trainer


class InfSelector:
    def __init__(self, params, device):
        self.params = params
        self.model_name = params["model"]["model_name"]
        self.device = device
        self.model = None
        self.loss_weight_dict = None
        
    def select_model(self):
        # define model
        if self.model_name in ["sarnn",]:
            self.model = SARNN(
                union_dim=self.params["model"]["hid_dim"],
                state_dim=self.params["model"]["vec_dim"],
                key_dim=self.params["model"]["key_dim"],
                heatmap_size=self.params["model"]["heatmap_size"],
                temperature=self.params["model"]["temperature"],
            ).to(self.device)
        elif self.model_name in ["topksarnn",]:
            self.model = TopKSARNN(
                union_dim=self.params["model"]["hid_dim"],
                state_dim=self.params["model"]["vec_dim"],
                key_dim=self.params["model"]["key_dim"],
                top_k=self.params["model"]["top_k"],
                heatmap_size=self.params["model"]["heatmap_size"],
                temperature=self.params["model"]["temperature"],
            ).to(self.device)
        elif self.model_name in ["dualsarnn",]:
            self.model = DualSARNN(
                union_dim=self.params["model"]["hid_dim"],
                state_dim=self.params["model"]["vec_dim"],
                key_dim=self.params["model"]["key_dim"],
                heatmap_size=self.params["model"]["heatmap_size"],
                temperature=self.params["model"]["temperature"],
            ).to(self.device)
        elif self.model_name in ["conv3dsarnn",]:
            self.model = Conv3dSARNN(
                union_dim=self.params["model"]["hid_dim"],
                state_dim=self.params["model"]["vec_dim"],
                key_dim=self.params["model"]["key_dim"],
                heatmap_size=self.params["model"]["heatmap_size"],
                temperature=self.params["model"]["temperature"],
            ).to(self.device)
        elif self.model_name in ["flowsarnn",]:
            self.model = FlowSARNN(
                union_dim=self.params["model"]["hid_dim"],
                state_dim=self.params["model"]["vec_dim"],
                key_dim=self.params["model"]["key_dim"],
                heatmap_size=self.params["model"]["heatmap_size"],
                temperature=self.params["model"]["temperature"],
            ).to(self.device)
        elif self.model_name in ["topkdualsarnn",]:
            self.model = TopKDualSARNN(
                union_dim=self.params["model"]["hid_dim"],
                state_dim=self.params["model"]["vec_dim"],
                key_dim=self.params["model"]["key_dim"],
                top_k=self.params["model"]["top_k"],
                heatmap_size=self.params["model"]["heatmap_size"],
                temperature=self.params["model"]["temperature"],
            ).to(self.device)
        elif self.model_name in ["stackrnn"]:
            self.model = StackRNN(
                img_feat_dim=self.args.vec_dim,
                vec_dim=self.args.vec_dim,
                union1_dim=self.params["model"]["hid_dim"],
                union2_dim=self.params["model"]["hid_dim"],
                union3_dim=self.params["model"]["hid_dim"],
            ).to(self.device)
        elif self.model_name in ["sastackrnn"]:
            self.model = SAStackRNN(
                key_dim=self.params["model"]["key_dim"],
                vec_dim=self.args.vec_dim,
                union1_dim=self.params["model"]["hid_dim"],
                union2_dim=self.params["model"]["hid_dim"],
                union3_dim=self.params["model"]["hid_dim"],
            ).to(self.device)
        elif self.model_name in ["hrnn"]:
            self.model = HRNN(
                img_feat_dim=self.args.vec_dim,
                vec_dim=self.args.vec_dim,
                sensory_dim=self.params["model"]["hid_dim"],
                union1_dim=self.params["model"]["hid_dim"],
                union2_dim=self.params["model"]["hid_dim"],
            ).to(self.device)
        elif self.model_name in ["fasterhrnn"]:
            self.model = FasterHRNN(
                img_feat_dim=self.args.vec_dim,
                vec_dim=self.args.vec_dim,
                sensory_dim=int(self.params["model"]["hid_dim"]/2),
                union1_dim=self.params["model"]["hid_dim"],
                union2_dim=self.params["model"]["hid_dim"],
            ).to(self.device)
        else:
            print(f"{self.model_name} is invalid model")
            exit()
        
        return self.model
    
    def select_loss_weight(self):
        if self.model == None:
            print(f"{self.model_name} is invalid model")
            exit()
        
        if self.model_name in ["sarnn",
                               "topksarnn",
                               "dualsarnn",
                               "conv3dsarnn",
                               "flowsarnn",
                               "topkdualsarnn",
                               "sastackrnn"]:
            self.loss_weight_dict = {"img": self.params["loss"]["img_loss"], "vec": self.params["loss"]["vec_loss"], "key": self.params["loss"]["key_loss"]}
        elif self.model_name in ["stackrnn","hrnn","fasterhrnn",]:
            self.loss_weight_dict = {"img": self.params["loss"]["img_loss"], "vec": self.params["loss"]["vec_loss"]}
        else:
            print(f"{self.model_name} is invalid model")
            exit()
        return self.loss_weight_dict
    
    def select_inf(self, open_ratio):
        if self.model == None or self.loss_weight_dict == None:
            print(f"{self.model_name} is invalid model")
            exit()
        if self.model_name in ["sarnn"]:
            inf = Inf4SARNN(self.model, open_ratio=open_ratio)
        elif self.model_name in ["topksarnn"]:
            inf = Inf4TopKSARNN(self.model, open_ratio=open_ratio)
        elif self.model_name in ["dualsarnn"]:
            inf = Inf4DualSARNN(self.model, open_ratio=open_ratio)
        elif self.model_name in ["conv3dsarnn"]:
            inf = Inf4Conv3dSARNN(self.model, open_ratio=open_ratio)
        elif self.model_name in ["flowsarnn"]:
            inf = Inf4FlowSARNN(self.model, open_ratio=open_ratio)
        elif self.model_name in ["topkdualsarnn"]:
            inf = Inf4TopKDualSARNN(self.model, open_ratio=open_ratio)
        elif self.model_name in ["hrnn"]:
            inf = Inf4HRNN(self.model, open_ratio=open_ratio)
        elif self.model_name in ["stackrnn"]:
            inf = Inf4StackRNN(self.model, open_ratio=open_ratio)
        elif self.model_name in ["sastackrnn"]:
            inf = Inf4SAStackRNN(self.model, open_ratio=open_ratio)
        elif self.model_name in ["fasterhrnn",]:
            inf = Inf4FasterHRNN(self.model, open_ratio=open_ratio)
            
        return inf
    




class RTSelector:
    def __init__(self, params, device):
        self.params = params
        self.model_name = params["model"]["model_name"]
        self.device = device
        self.model = None
        self.loss_weight_dict = None
        
    def select_model(self):
        # define model
        if self.model_name in ["dualsarnn",]:
            self.model = DualSARNN(
                union_dim=self.params["model"]["hid_dim"],
                state_dim=self.params["model"]["vec_dim"],
                key_dim=self.params["model"]["key_dim"],
                heatmap_size=self.params["model"]["heatmap_size"],
                temperature=self.params["model"]["temperature"],
            )
        else:
            print(f"{self.model_name} is invalid model at model")
            exit()
        
        return self.model