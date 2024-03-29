# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import os
import numpy as np
from PIL import Image
import cv2

from .base_dataset import BaseDataset

class Facial(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_classes=15,
                 multi_scale=False, 
                 flip=False, 
                 ignore_label=255, 
                 base_size=616, 
                 crop_size=(616, 616),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(Facial, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split('    ') for line in open(root+list_path)]

        self.files = self.read_files()


        self.ignore_label = ignore_label
        
        self.color_list = [[0, 0, 0],
                            [1, 1, 1],
                            [2, 2, 2],
                            [3, 3, 3],
                            [4, 4, 4],
                            [5, 5, 5],
                            [6, 6, 6],
                            [7, 7, 7],
                            [8, 8, 8],
                            [9, 9, 9],
                            [10, 10, 10],
                            [11, 11, 11],
                            [12, 12, 12],
                            [13, 13, 13],
                            [14, 14, 14]]
        
        self.class_weights = None
        
        self.bd_dilate_size = bd_dilate_size
    
    def read_files(self):
        files = []

        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })
            
        return files
        
    def color2label(self, color_map):
        label = np.ones(color_map.shape[:2])*self.ignore_label
        for i, v in enumerate(self.color_list):
            label[(color_map == v).sum(2)==3] = i

        return label.astype(np.uint8)
    
    def label2color(self, label):
        color_map = np.zeros(label.shape+(3,))
        for i, v in enumerate(self.color_list):
            color_map[label==i] = self.color_list[i]
            
        return color_map.astype(np.uint8)

    # def __getitem__(self, index):
    #     item = self.files[index]
    #     name = item["name"]
    #     image = Image.open(os.path.join(self.root,'facial',item["img"])).convert('RGB')
    #     image = np.array(image)
    #     size = image.shape

    #     color_map = Image.open(os.path.join(self.root,'facial',item["label"])).convert('RGB')
    #     color_map = np.array(color_map)
    #     label = self.color2label(color_map)

    #     image, label, edge = self.gen_sample(image, label, 
    #                             self.multi_scale, self.flip, edge_pad=False,
    #                             edge_size=self.bd_dilate_size, city=False)

    #     return image.copy(), label.copy(), edge.copy(), np.array(size), name
    
    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]



        image = cv2.cvtColor(cv2.imread(os.path.join(self.root,'facial',item["img"]), 1), cv2.COLOR_BGR2RGB)
        size = image.shape

        # color_map = Image.open(os.path.join(self.root,'facial',item["label"])).convert('RGB')
        # color_map = np.array(color_map)
        # label = self.color2label(color_map)

        label = cv2.imread(os.path.join(self.root,'facial',item["label"]), 0)
        label = label.astype(np.uint8)

        # if (mylabel-label).sum() != 0:
        #     print(item["label"])



        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_pad=False,
                                edge_size=self.bd_dilate_size, city=False)
        

        return image.copy(), label.copy(), edge.copy(), np.array(size), name


    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.label2color(preds[i])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
