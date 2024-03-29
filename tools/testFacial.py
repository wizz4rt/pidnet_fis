# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import glob
import argparse
import cv2
import os
import numpy as np
import _init_paths
import models
import torch
import torch.nn.functional as F
from PIL import Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = [(128, 64,128),
                (244, 35,232),
                ( 70, 70, 70),
                (102,102,156),
                (190,153,153),
                (153,153,153),
                (250,170, 30),
                (220,220,  0),
                (107,142, 35),
                (152,251,152),
                ( 70,130,180),
                (220, 20, 60),
                (255,  0,  0),
                (  0,  0,142),
                (  0,  0, 70),
                (  0, 60,100),
                (  0, 80,100),
                (  0,  0,230),
                (119, 11, 32)]

color_map_normal = [(189, 229, 246),   # background
                    (66, 191, 247),    # face skin
                    (255, 108, 62),    # nose
                    (158, 158, 158),   # glasses
                    (139, 195, 74),    # right eye
                    (76, 175, 80),     # left eye
                    (255, 235, 59),    # right eyebrow
                    (205, 220, 57),    # left eyebrow
                    (255, 152, 0),     # right ear
                    (255, 193, 7),     # left ear
                    (103, 58, 183),    # mouth
                    (146, 116, 106),   # beard
                    (238, 87, 138),    # hair
                    (0, 150, 136),     # hat
                    (50, 147, 192)]    # neck

color_map_noneck = [(189, 229, 246),   # background
                    (66, 191, 247),    # face skin
                    (255, 108, 62),    # nose
                    (158, 158, 158),   # glasses
                    (139, 195, 74),    # right eye
                    (76, 175, 80),     # left eye
                    (255, 235, 59),    # right eyebrow
                    (205, 220, 57),    # left eyebrow
                    (255, 152, 0),     # right ear
                    (255, 193, 7),     # left ear
                    (103, 58, 183),    # mouth
                    (146, 116, 106),   # beard
                    (238, 87, 138),    # hair
                    (0, 150, 136)]     # hat

color_map_noleftright = [(189, 229, 246),  # background
                        (66, 191, 247),    # face skin
                        (255, 108, 62),    # nose
                        (158, 158, 158),   # glasses
                        (76, 175, 80),     # left eye
                        (205, 220, 57),    # left eyebrow
                        (255, 193, 7),     # left ear
                        (103, 58, 183),    # mouth
                        (146, 116, 106),   # beard
                        (238, 87, 138),    # hair
                        (0, 150, 136),     # hat
                        (50, 147, 192)]    # neck

color_map = color_map_normal


def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    
    parser.add_argument('--a', help='pidnet-s, pidnet-m or pidnet-l', default='pidnet-s', type=str)
    parser.add_argument('--c', help='colormap', type=str, default='normal')
    parser.add_argument('--p', help='dir for pretrained model', default='../pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt', type=str)
    parser.add_argument('--r', help='root or dir for input images', default='./samples/', type=str)
    parser.add_argument('--t', help='the format of input images (.jpg, .png, ...)', default='.png', type=str)     
    parser.add_argument('--o', help='output dir', default='./samples/outputs/', type=str)
    parser.add_argument('--n', help='number of classes', default=15, type=int)

    args = parser.parse_args()

    return args

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    
    return model

if __name__ == '__main__':
    args = parse_args()
    images_list = glob.glob(args.r+'*'+args.t)
    sv_path = args.o

    if args.c == 'normal':
        color_map = color_map_normal
    elif args.c == 'noneck':
        color_map = color_map_noneck
    elif args.c == 'noleftright':
        color_map = color_map_noleftright
    
    model = models.pidnet.get_pred_model(args.a, args.n)
    model = load_pretrained(model, args.p).cuda()
    model.eval()
    with torch.no_grad():
        for img_path in images_list:
            img_name = img_path.split("/")[-1]
            img = cv2.imread(os.path.join(args.r, img_name),
                               cv2.IMREAD_COLOR)
            sv_img = np.zeros_like(img).astype(np.uint8)
            img = input_transform(img)
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            pred = model(img)
            pred = F.interpolate(pred, size=img.size()[-2:], 
                                 mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            
            for i, color in enumerate(color_map):
                for j in range(3):
                    sv_img[:,:,j][pred==i] = color_map[i][j]
            sv_img = Image.fromarray(sv_img)

            print('Hello')
            
            if not os.path.exists(sv_path):
                os.mkdir(sv_path)
            sv_img.save(sv_path+img_name)
            
            
            
        
        