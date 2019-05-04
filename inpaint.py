from __future__ import print_function
import argparse
import skimage
import os
import numpy as np
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

def generate_mask(b,c,w,h):
    '''
    Generate random rectangular mask based on the input
    '''
    #Generate random coordinates
    x, y = np.random.randint(0,w//2, size=1)[0], np.random.randint(0,h//2, size=1)[0]
    left_x, left_y = w-x, h-y
    
    #Generate random length
    x_len, y_len = np.random.randint(0,left_x,size=1)[0], np.random.randint(0,left_y,size=1)[0]
    
    #Make white box
    box = np.array([1]*x_len*y_len*1*1).reshape(1, 1, x_len,y_len)
    mask = np.zeros((b,1,w,h))

    #Replace original image with white box
    mask[:,:,x:x+x_len, y:y+y_len] = box

    return mask, [y,y+y_len,x,x+x_len]


def compute_gradient_penalty(D, real_samples, fake_samples_g, fake_samples_l):
    '''
    Calculate the gradient penalty for Generator if the gradient changes too rapidly. Smooth the training process
    '''
    # Random weight term for interpolation between real and fake samples

    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples_g)).requires_grad_(True)
    d_interpolates = D(interpolates, fake_samples_l)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    print("fake shape: ", fake.shape)
    print("d_interpolates shape: ", d_interpolates.shape)
    print("interpolates shape: ", interpolates.shape)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # First Layer 64x64x4
            nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = 5, stride = 1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Second Layer 64x64x64
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # Third Layer 32x32x128
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # Fourth Layer 32x32x128
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            # Fifth Layer 16x16x256
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            # Sixth Layer 16x16x256 dilated
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding=2, dilation = 2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            # Seventh Layer 16x16x256 dilated
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding=4, dilation = 4),
            nn.ReLU(),
            nn.BatchNorm2d(256),
           
            # Eighth Layer 16x16x256 dilated
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding=8, dilation = 8),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            # 9th Layer 16x16x256
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            # 10th Layer 16x16x256
            nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(in_channels = 32, out_channels = 3, kernel_size = 3, stride = 1, padding=1),
            nn.Sigmoid(),
        )
        
    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.global_dis = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 5, stride = 2, padding=2, bias=False), # floor(64-5+2*2)/2 + 1 = 32
            nn.ReLU(),

            # state size. 64 x 32 x 32
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # state size. 128 x 16 x 16
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 5, stride = 2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # state size. 256 x 8 x 8
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 5, stride = 2, padding=2, bias=False), # floor(8-5+2*2)/2 + 1 = 4
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        self.l1 = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.Sigmoid()
        )
        
        self.l2 = nn.Sequential(
            nn.Linear(256*4*4, 1024),
            nn.Sigmoid()
        )
        

        self.local_dis = nn.Sequential(

            # state size. 3 x 32 x 32
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 5, stride = 2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # state size. 64 x 16 x 16
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # state size. 128 x 8 x 8
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 5, stride = 2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.l3 = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input1, input2):
        gd = self.global_dis(input1)
        ld = self.local_dis(input2)

        # Linearlize
        gd_l = gd.view(-1, 512*4*4)
        gd_l= self.l1(gd_l)
        ld_l = ld.view(-1, 256*4*4)
        ld_l = self.l2(ld_l)
        
        x = torch.cat((gd_l, ld_l), dim=1)
        return self.l3(x)

if __name__ == "__main__":

    # minimum mask size 
    MIN_MASK_SIZE = 25

    # Local dict
    LOCAL_DICT = 'final_trial_1'

    # LOCAL == 1, Generator only use local MSE; otherwise use both global and local
    LOCAL = 1

    # The path for the training set
    dataroot = "./train"
    
    # The path for storing intermediate result (with training data)
    os.makedirs(LOCAL_DICT, exist_ok=True)

    # Add WGAN_GP penalty
    ADD_GP_PEN = 1

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 32

    # Spatial size of training images. All images will be resized to this
    # Size using a transformer.
    image_size = 64

    # Number of channels in the training images. For color images this is 3
    nc = 3

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = 64

    # Size of feature maps in discriminator
    ndf = 64

    # Number of training epochs
    num_epochs = 200

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    cuda_ava = True if torch.cuda.is_available() else False

    Tensor = torch.cuda.FloatTensor if cuda_ava else torch.FloatTensor
    print("cuda available: ", cuda_ava)
    print("add wgan_gp penalty coefficient: ", ADD_GP_PEN)

    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Save the 
    for i, data in enumerate(dataloader, 0):
        fixed_img = data[0].to(device)
        while(1):
            mask1, _ = generate_mask(batch_size, 3, 64, 64)
            if np.sum(mask1)/batch_size > 800:
                break
        fixed_mask = torch.Tensor(mask1).to(device)
        holed_img = fixed_img*(1-fixed_mask)
        fixed_input = torch.cat((holed_img, fixed_mask), dim=1)
        save_image(holed_img[:16], LOCAL_DICT + "/holed_img.png", nrow=4, normalize=True)
        save_image(fixed_img[:16], LOCAL_DICT + "/origin_img.png", nrow=4, normalize=True)
        break

    '''
    The first thing is that the BCE objective for the Generator can more accurately be stated as 
    "the images output by the generator should be assigned a high probability by the Discriminator."
    It's not BCE as you might see in a binary reconstruction loss, which would be BCE(G(Z),X) 
    where G(Z) is a generated image and X is a sample, it's BCE(D(G(Z)),1) where D(G(Z)) is the probability
    assigned to the generated image by the discriminator. Given a "perfect" generator which always has photorealistic
    outputs, the D(G(Z)) values should always be close to 1. Obviously in practice there's difficulties getting 
    this kind of convergence (the training is sort of inherently unstable) but that is the goal.

    The second is that in the standard GAN algorithm, the latent vector (the "random noise" which the generator 
    receives as input and has to turn into an image) is sampled independently of training data. 
    If you were to use the MSE between the outputs of the GAN and a single image, you might get some sort of result out, 
    but you'd effectively be saying "given this (random) Z, produce this specific X" and you'd be implicitly forcing the 
    generator to learn a nonsensical embedding of the image. If you think of the Z vector as a high-level description of 
    the image, it would be like showing it a dog three times and asking it to generate the same dog given three different 
    (and uncorrelated) descriptions of the dog. Compare this with something like a VAE which has an explicit inference 
    mechanism (the encoder network of the VAE infers Z values given an image sample) and then attempts to reconstruct a 
    given image using those inferred Zs. The GAN does not attempt to reconstruct an image, so in its vanilla form it 
    doesn't make sense to compare its outputs to a set of samples using MSE or MAE.
    '''
    criterion = nn.BCELoss()

    iters = 0
    # img_list = []
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    running_loss_g = 0.0
    running_loss_d = 0.0

    print("Starting Training Loop...")

    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            # Calculate real image for gradient penalty
            real = Variable(data[0].type(Tensor))

            # Load images from dataloader
            img_batch = data[0].to(device)
            real_batch_size = img_batch.shape[0]
            
            # generate related mask with batch size
            # Dimention batch_size x channels x width x height
            while(1):
                mask_batch, idx = generate_mask(real_batch_size, 3, 64, 64)
                if np.sum(mask_batch)/batch_size > MIN_MASK_SIZE:
                    break
            mask_batch = torch.Tensor(mask_batch).to(device)
            
            ############## Discriminator ###############   
            netD.zero_grad()
            
            label = torch.full((real_batch_size,), 1, device=device)
            
            # True image for global discriminator
            global_img_batch = img_batch
            
            # Create the mask with smaller size
            local_img_batch1 = global_img_batch[:, :, idx[0]:idx[1], idx[2]:idx[3]]
            
            # Resize
            local_img_batch = F.upsample(local_img_batch1, size=(32,32), mode='bilinear')
            
            output = netD(global_img_batch, local_img_batch).view(-1)
            
            errD_real = criterion(output, label)
            
            errD_real.backward()
                        
            # Get the noise output from generator
            real_batch_size = img_batch.shape[0]
            
            img_holed_batch = img_batch*(1 - mask_batch).to(device)
            train_batch = torch.cat((img_holed_batch, mask_batch), dim=1)

            # Generate the predict image batch
            predicted_img_batch = netG(train_batch)
            
            global_img_fake_batch = predicted_img_batch
            
            local_img_fake_batch1 = global_img_fake_batch[:, :, idx[0]:idx[1], idx[2]:idx[3]]

            local_img_fake_batch = F.upsample(local_img_fake_batch1, size=(32,32), mode='bilinear')
            
            label.fill_(0)
            
            output = netD(global_img_fake_batch.detach(), local_img_fake_batch.detach()).view(-1)

            if ADD_GP_PEN != 0:
                gradPenalty = compute_gradient_penalty(netD, real.data, global_img_fake_batch.data, local_img_fake_batch.data)
                errD_fake = criterion(output, label) + gradPenalty*ADD_GP_PEN
            else:
                errD_fake = criterion(output, label)
            
            errD_fake.backward()

            errD = errD_real + errD_fake
            
            optimizerD.step()
            
            running_loss_d += errD.item()
            if iters % 500 == 0:    # print every 2000 mini-batches
                print('discriminator loss: %.3f' %
                      (running_loss_d / 500))
                running_loss_d = 0.0
                
                
            ############## Generator ###############    
            netG.zero_grad()
            
            label.fill_(0)
            noise_global = torch.randn(real_batch_size, 3, 64, 64, device=device)
            noise_local = torch.randn(real_batch_size, 3, 32, 32, device=device)
            output = netD(noise_global, noise_local)
            errG = criterion(output, label)
            errG.backward(retain_graph=True)

            # Calculate the MSE (Mean Squared Error)
            diff_batch = predicted_img_batch - img_batch
            MSE = torch.sum((diff_batch**2)*mask_batch)/(real_batch_size * 64 * 64)

            # Determine to use overall MSE or just mask MSE criteria
            if LOCAL:
                MSE = torch.sum((diff_batch**2)*mask_batch)/(real_batch_size * 64 * 64)
            else:
                MES_local = torch.sum((diff_batch**2)*mask_batch)/(real_batch_size * 64 * 64)
                MSE_global = torch.sum((diff_batch**2))/(real_batch_size * 64 * 64)
                MSE = MSE_local + MES_global

            MSE.backward()
            loss = MSE + errG
            optimizerG.step()

            running_loss_g += loss.item()
            # print every 500 mini-batches
            if iters % 500 == 0:   
                print('generator loss: %.3f' %
                      (running_loss_g / 500))
                running_loss_g = 0.0

            # Save predicted image every 500 iterations
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_input).detach().cpu()
                    input_fake = (fake * (fixed_mask.detach().cpu())) + holed_img.detach().cpu()
                    save_image(input_fake.data[:16], LOCAL_DICT + "/%d.png" % iters, nrow=4, normalize=True)

            # Save model every 500 iterations, use validate.py to get validation result
            if (iters % 10000 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                os.makedirs('./models/%d_' % iters + LOCAL_DICT, exist_ok=True)
                torch.save(netG.state_dict(), './models/%d_' % iters + LOCAL_DICT + '/generator')
                torch.save(netD.state_dict(), './models/%d_' % iters + LOCAL_DICT + '/discriminator')
            iters += 1;


