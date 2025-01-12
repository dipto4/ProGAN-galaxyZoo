from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import random

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torchvision.transforms as T
from progan_modules import Generator, Discriminator

device = torch.device("cuda:%d"%(1))
input_code_size = 128#args.z_dim
channel = 128
#batch_size = args.batch_size
#n_critic = args.n_critic



generator = Generator(in_channel=channel, input_code_dim=input_code_size, pixel_norm=False, tanh=False).to(device)
#discriminator = Discriminator(feat_dim=args.channel).to(device)
g_running = Generator(in_channel=channel, input_code_dim=input_code_size, pixel_norm=False, tanh=False).to(device)


## you can directly load a pretrained model here
generator.load_state_dict(torch.load('/export/home/phys/diptajym/10-701/depth/Progressive-GAN-pytorch/trial_spiral_2023-04-13_1_52/checkpoint/250000_g.model'))
g_running.load_state_dict(torch.load('/export/home/phys/diptajym/10-701/depth/Progressive-GAN-pytorch/trial_spiral_2023-04-13_1_52/checkpoint/250000_g.model'))
#discriminator.load_state_dict(torch.load('checkpoint/150000_d.model'))
transform = T.ToPILImage()
#images = g_running(torch.randn(100, input_code_size).to(device), step=4, alpha=1.0).data.cpu()
#print(len(images))
#img = transform(images[0])
#img.show()
BATCHES=20
count = 0
for b in range(0,BATCHES):

    images = g_running(torch.randn(100, input_code_size).to(device), step=5, alpha=1.0).data.cpu()
    for i in range(0,len(images)):
        utils.save_image(images[i],'dumped_spiral/dumped_{:d}.png'.format(count),normalize=True,range=(-1, 1))
        print(count)
        count+=1


