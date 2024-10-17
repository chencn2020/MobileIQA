import torch
import torchvision

import cv2
import numpy as np
import os
# from models import monet as MoNet
from models import mobileIQA as MoNet
import argparse
from utils.dataset.process import ToTensor, Normalize
from tqdm import tqdm
import torch.utils.data as data
import torch

from PIL import Image

class UHDIQA(data.Dataset):
    def __init__(self, image_path, config):
        imgname = []
        
        for img_name in tqdm(sorted(os.listdir(image_path))):
            imgname.append(os.path.join(config.image_path, img_name))
        self.samples = imgname

    def __getitem__(self, index):
        img_path = self.samples[index]
        # print(img_path)
        full_sample, resized_sample = load_image(img_path)
        full_sample = transform(full_sample)
        resized_sample = transform(resized_sample)
        return (full_sample, resized_sample), os.path.basename(img_path)

    def __len__(self):
        length = len(self.samples)
        return length

load_image, transform = None, None
def load_image_monet(img_path):
    def resize(d_img, size=[1740, 1080]):
        d_img = cv2.resize(d_img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.resize(d_img, size, interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1)) # 3 1080 1740
        return d_img

    d_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    resize_512 = resize(d_img, (270, 435))
    full_img = resize(d_img, [224, 224])

    return full_img, resize_512

def transform_monet(img):
    transforms = torchvision.transforms.Compose([Normalize(0.5, 0.5), ToTensor()])
    return transforms(img)

####
def load_image_general(img_path):
    def resize(d_img, size=[1740, 1080]):
        size = list(size)
        # size.reverse()
        d_img = d_img.resize(size)
        return d_img

    d_img = Image.open(img_path).convert('RGB')
    resize_512 = resize(d_img, (448, 448))
    full_img = resize(d_img, [1914, 1188])
    
    # 1740*1080

    return full_img, resize_512

def transform_general(img):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))
    ])
    return transforms(img)

####
def load_img_ori(img_path):
    def resize(d_img, size = (1740, 1080)):
        d_img = cv2.resize(d_img, size, interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1)) # 3 1080 1740
        
        return d_img
    
    d_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    resize_512 = resize(d_img, (270, 435))
    full_img = resize(d_img)
    
    
    return full_img, resize_512
    
def transform_ori(img):
    transforms=torchvision.transforms.Compose([Normalize(0.5, 0.5), ToTensor()])
    return transforms(img)


# vitAttIQA
full_size = [int(1907), int(1231)]
print('full size', full_size)
def load_image_vitAttIQA(img_path):
    def resize(d_img, size = [1740, 1080]):
        size = list(size)
        # size.reverse()
        d_img = d_img.resize(size)
        return d_img
    
    # d_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    d_img = Image.open(img_path).convert('RGB')
    
    resize_512 = resize(d_img, (270, 435))
    full_img = resize(d_img, (1907, 1231))
    # full_img = resize(d_img, [int(1740), int(1080)])
    
    
    return full_img, resize_512
    
def transform_vitAttIQA(img):
    transforms=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
    return transforms(img)

# vitAttIQAMoNet
print('full size', full_size)
def load_image_vitAttIQAMoNet(img_path):
    def resize(d_img, size = [1740, 1080]):
        size = list(size)
        # size.reverse()
        d_img = d_img.resize(size)
        return d_img
    
    # d_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    d_img = Image.open(img_path).convert('RGB')
    
    resize_512 = resize(d_img, (270, 435))
    # full_img = resize(d_img, ([int(1956), int(1304)]))
    full_img = resize(d_img, (1907, 1231))
    # full_img = resize(d_img, ([1044, 648]))
    # full_img = resize(d_img, [int(1740), int(1080)])
    
    
    return full_img, resize_512
    
def transform_vitAttIQAMoNet(img):
    transforms=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
    return transforms(img)

# mobileMoNet
print('full size', full_size)
def load_image_mobileMoNet(img_path):
    def resize(d_img, size = [1740, 1080]):
        size = list(size)
        # size.reverse()
        d_img = d_img.resize(size)
        return d_img
    
    # d_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    d_img = Image.open(img_path).convert('RGB')
    
    resize_512 = resize(d_img, (270, 435))
    # full_img = resize(d_img, ([int(1956), int(1304)]))
    full_img = resize(d_img, (1600, 993))
    # full_img = resize(d_img, [int(1740), int(1080)])
    
    
    return full_img, resize_512
    
def transform_mobileMoNet(img):
    transforms=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
    return transforms(img)


# RESIZE
def load_image_RESIZE(img_path, full_size=[1907, 1231]):
    def resize_and_pad(d_img, output_size):
        img_ratio = d_img.width / d_img.height
        output_ratio = output_size[0] / output_size[1]

        if img_ratio >= output_ratio:
            scale = output_size[0] / d_img.width
        else:
            scale = output_size[1] / d_img.height
        new_width = int(d_img.width * scale)
        new_height = int(d_img.height * scale)
        resized_img = d_img.resize((new_width, new_height), Image.LANCZOS)
        full_img = Image.new('RGB', output_size, (0, 0, 0))
        upper = (output_size[1] - new_height) // 2
        left = (output_size[0] - new_width) // 2
        full_img.paste(resized_img, (left, upper))

        return full_img
    d_img = Image.open(img_path).convert('RGB')
    resize_512 = resize_and_pad(d_img, (1435, 1270))
    full_img = resize_and_pad(d_img, full_size)
    
    return full_img, resize_512

def inference(config):
    global load_image, transform
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

    model_type = config.model_type # mobileIQA, efficientIQA
    print('Load Model: ', model_type)
    if model_type == 'mobileIQA':
        from models import mobileIQA as MoNet
        model = MoNet.MoNet(config).cuda()
        load_image = load_image_general
        transform = transform_general

    elif model_type == 'efficientIQA':
        from models import efficientIQA as MoNet
        model = MoNet.MoNet().cuda()
        load_image = load_image_general
        transform = transform_general

    elif model_type == 'monet':
        from models import monet as MoNet
        model = MoNet.MoNet(config).cuda()
        load_image = load_image_monet
        transform = transform_monet
    elif model_type == 'vitAttIQAMoNet':
        from models import vitAttIQAMoNet as MoNet
        model = MoNet.MoNet().cuda()
        load_image = load_image_vitAttIQAMoNet
        transform = transform_vitAttIQAMoNet
    elif model_type == 'mobileMoNet':
        from models import mobileMoNet as MoNet
        model = MoNet.MoNet().cuda()
        load_image = load_image_mobileMoNet
        transform = transform_mobileMoNet
    
    elif model_type == 'vitAttIQA':
        from models import vitAttIQA as MoNet
        model = MoNet.MoNet().cuda()
        load_image = load_image_vitAttIQA
        transform = transform_vitAttIQA
        print(load_image_vitAttIQA)
        print(transform_vitAttIQA)
        
    elif model_type == 'mobileIQA_Distill':
        from models import mobileIQA_Distill as MoNet
        model = MoNet.MoNet().cuda()
        load_image = load_image_vitAttIQA
        transform = transform_vitAttIQA
        print(load_image_vitAttIQA)
        print(transform_vitAttIQA)
        
    elif model_type == 'vitAttIQAMoNet_LDA':
        from models import vitAttIQAMoNet_LDA as MoNet
        model = MoNet.MoNet().cuda()
        load_image = load_image_vitAttIQAMoNet
        transform = transform_vitAttIQAMoNet
        print(load_image_vitAttIQA)
        print(transform_vitAttIQA)
    
    print(transform)
    print(load_image)
    model.load_state_dict(torch.load(config.pkl_path))
    model.eval()
    data = UHDIQA(config.image_path, config)
    dataloader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=False, num_workers=12)
    
    res = []
    with torch.inference_mode():
        for img_tensor, img_name in tqdm(dataloader):
            full_img, resize_img = img_tensor
            full_img = full_img.cuda()
            resize_img = resize_img.cuda()
            print(full_img.shape)
            if model_type == 'efficientIQA':
                print(resize_img.shape)
                iq = model(full_img, resize_img)
            else:
                iq = model(full_img)
            
            iq = iq.cpu().detach().numpy().tolist()
            img_name = list(img_name)
            for i, j in zip(img_name, iq):
                res.append([i, j])

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', dest='gpu_id', type=str, default='1')
    parser.add_argument('--model_type', dest='model_type', type=str)

    # model related
    parser.add_argument('--backbone', dest='backbone', type=str, default='vit_base_patch8_224',
                        help='The backbone for MoNet.')
    parser.add_argument('--mal_num', dest='mal_num', type=int, default=3, help='The number of the MAL modules.')

    # testing related
    parser.add_argument('--pkl_path', dest='pkl_path', type=str, default='./koniq10k_570908/best_model.pkl',
                        help='The path of pretrained model.')
    parser.add_argument('--image_path', dest='image_path', type=str, default='/disk1/chenzewen/Competition/ECCV_2024_IQA/uhd_iqa/challenge/validation',
                        help='The path where the model and logs will be saved.')

    config = parser.parse_args()

    iq_score = inference(config)
    res_score = {}
    if isinstance(iq_score, list):
        print('Image Name\timage quality score')
        for (image_name, score) in iq_score:
            res_score[image_name] = score
    else:
        print('The image quality score is: %.4f' % iq_score)

    templete = '/disk1/chenzewen/Competition/ECCV_2024_IQA/uhd_iqa/validation shared.csv'
    res_file = open(templete, 'r').readlines()
    res_file[0] = res_file[0].strip()
    for idx in range(1, len(res_file)):
        img_name = eval(res_file[idx].strip().split()[0])
        # print(res_score)
        res_file[idx] = '{},{}'.format(img_name, res_score[img_name])
    
    
    csv_file_name = os.path.basename(config.pkl_path).replace('pkl', 'csv')
    save_txt = os.path.join(os.path.dirname(config.pkl_path), csv_file_name)
    with open(save_txt, 'w') as f:
        for i in res_file:
            f.writelines('{}\n'.format(i))
    
    print('cd {} && zip {} {}'.format(os.path.dirname(config.pkl_path), 'val_res.zip', csv_file_name))
    os.system('cd {} && zip {} {}'.format(os.path.dirname(config.pkl_path), 'val_res.zip', csv_file_name))

    print('save', save_txt.replace('csv', 'zip'))
