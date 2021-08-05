import torch.utils.data as data
from avi_r import AVIReader
from PIL import Image
import os
import torch
import numpy as np
from numpy.random import randint
import cv2
import json
import decord
from decord import VideoReader
from decord import cpu, gpu
import torchvision
import gc
import math
decord.bridge.set_bridge('torch')
from torchvision.io import read_video,read_video_timestamps

class VideoRecord_MEVA(object):
    def __init__(self, dic, name, root_path, enlarge_rate=0.13, str_path = "/mnt/data/PIP/pip_250k_stabilized"):
        self._data = dic
        # self._path = name.split("_")[0] + ".mp4"
        self._path = name.split("_")[0]+".avi"
        self._prop = name
        self._root_path = root_path
        self.enlarge_rate = enlarge_rate
        self.val = False
        self.str_path=str_path
        self.trans = torchvision.transforms.ToPILImage(mode='RGB')

    def frames(self):
        if self._data["pos"] == "STR":
            vr = VideoReader(os.path.join(self.str_path,self._prop.split("$")[0]))
            if len(vr) ==0:
                print(os.path.join(self.str_path,self._prop.split("$")[0]))
        else:
            cap = AVIReader(os.path.join(self._root_path,self._path))
        
        # cap size(height*width, 1072*1920)
        if self._data["annotations"]["start"]>self._data["annotations"]["end"]:
            start = self._data["annotations"]["end"]
            end = self._data["annotations"]["start"]
        else:
            start = self._data["annotations"]["start"]
            end = self._data["annotations"]["end"]
        if self._data["pos"] == "STR":
            height,width,_=vr[0].shape 
        else:
            cap.seek(start)
            height = cap.height
            width = cap.width
        # 1920*1072
        x0,y0,x1,y1 = self._data["annotations"]["bbox"]
        enlarge_x = (x1-x0)*self.enlarge_rate
        enlarge_y = (y1-y0)*self.enlarge_rate
        x0 = int(max(0,x0-enlarge_x))
        x1 = int(min(width,x1+enlarge_x))
        y0 = int(max(0,y0-enlarge_y))
        y1 = int(min(height,y1+enlarge_y))
        images = []
        idxes = range(start,end,1)
        if self._data["pos"] == "STR":
            for idx in idxes:
                if idx >= len(vr):
                    idx = len(vr)-1
                if self.val:                    
                    img = vr[idx][y0:y1,x0:x1]
                    images.append(img)
                    del img
                else:
                    try:
                        img = vr[idx][y0:y1,x0:x1].permute(2,0,1)
                    # print(img.size())
                    # print(len(vr))
                        images.append(self.trans(img).convert('RGB'))
                    except:
                        print(self._prop)
                        print(y0,y1,x0,x1)
                        print(idx)
                        img = vr[idx].permute(2,0,1)
                        images.append(self.trans(img).convert('RGB'))
            del(vr)
        else:
            for frame in cap.get_iter(end-start): 
                if self.val:     
                    img = torch.from_numpy(frame.numpy('rgb24')).float()[y0:y1,x0:x1]
                else:
                    img = Image.fromarray(frame.numpy('rgb24')[y0:y1,x0:x1])
                images.append(img)
            cap.close()
        if self._data["annotations"]["start"]>self._data["annotations"]["end"]:
            images.reverse()
        # print(len(images))
        return images

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._prop

    @property
    def num_frames(self):
        return self._data["annotations"]["end"]-self._data["annotations"]["start"]

    @property
    def label(self):
        lab = []
        for prob in self._data["annotations"]["conf"]:
            if prob > 0:
                lab.append(1)
            else:
                lab.append(0)
        
        return torch.Tensor(lab)   
        # return torch.Tensor(self._data["annotations"]["conf"])

    @property
    def start(self):
        return self._data["annotations"]["start"]