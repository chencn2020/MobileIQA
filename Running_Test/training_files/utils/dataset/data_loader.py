import torch
import torchvision

from utils.dataset import folders
import torchvision

import torchvision
import torch
import random
    
class Data_Loader(object):
    """Dataset class for IQA databases"""

    def __init__(self, config, path, img_indx, istrain=True):

        self.batch_size = config.batch_size
        self.istrain = istrain
        dataset = config.dataset

        if istrain:
            transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((512, 512)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                    std=(0.229, 0.224, 0.225))
                ])
        else:
            transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((512, 512)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                    std=(0.229, 0.224, 0.225))
                ])

        if dataset == 'livec':
            self.data = folders.LIVEC(root=path, index=img_indx, transform=transforms)
        elif dataset == 'koniq10k':
            self.data = folders.Koniq10k(root=path, index=img_indx, transform=transforms)
        elif dataset == 'bid':
            self.data = folders.BID(root=path, index=img_indx, transform=transforms)
        elif dataset == 'spaq':
            self.data = folders.SPAQ(root=path, index=img_indx, transform=transforms)
        elif dataset == 'uhdiqa':
            self.data = folders.UHDIQA(root=path, index=img_indx, transform=transforms)
        else:
            raise Exception("Only support livec, koniq10k, bid, spaq, and uhdiqa.")

    def get_data(self):
        dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=False, num_workers=8)
        return dataloader