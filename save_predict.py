from __future__ import print_function
#%matplotlib inline
import argparse
import skimage
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
from inpaint import Generator, Discriminator, generate_mask


# SAVE_PATH = "./predict/65299_globalMSE_noPenalty"
# SAVE_PATH = "./predict/65299_localMSE_1_Penalty"
# SAVE_PATH = "./predict/65299_globalMSE_1_Penalty"
SAVE_PATH = "./predict/78359_localMSE_noPenalty"


os.makedirs(SAVE_PATH, exist_ok=True)

# ======= Load pre-trained model ======== #
# PATH = './models/65299_globalMSE_noPenalty'
# PATH = './models/65299_localMSE_1_Penalty'
# PATH = './models/65299_globalMSE_1_Penalty'
PATH = './models/78359_localMSE_noPenalty'

ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
netG = Generator(ngpu).to(device)
netG.load_state_dict(torch.load(PATH + '/generator'))
netG.eval()

netD = Discriminator(ngpu).to(device)
netD.load_state_dict(torch.load(PATH + '/discriminator'))
netD.eval()

# ======== Load validate file ======== #
LOCAL_PATH = './validate'
dataset = dset.ImageFolder(root=LOCAL_PATH,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

idx = 0
for t in range(30):
    for i, data in enumerate(dataloader, 0):
        fixed_img = data[0].to(device)
        print(i, "iteration")
        if(fixed_img.shape[0]!=16):
            break
        while(1):
            mask1, _ = generate_mask(16, 3, 64, 64)
            if np.sum(mask1)/16 > 800:
                break
        fixed_mask = torch.Tensor(mask1).to(device)
        holed_img = fixed_img*(1-fixed_mask)
        fixed_input = torch.cat((holed_img, fixed_mask), dim=1)
        fake = netG(fixed_input).detach().cpu()
        input_fake = (fake * (fixed_mask.detach().cpu())) + holed_img.detach().cpu()
        for i in range(16):
            save_image(input_fake.data[i], SAVE_PATH + "/predicted_%d.png" % idx)
            idx += 1
	    



