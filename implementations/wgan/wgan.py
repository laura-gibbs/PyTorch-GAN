import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

sys.path.append('./')
from csdataset import CSDataset
from skimage.transform import resize

from PIL import Image
import matplotlib.pyplot as plt


os.makedirs("wgan/images", exist_ok=True)
os.makedirs("wgan/wgan_mdt_tiles_32", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=301, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=512, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=2, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


def bound_and_norm(tens):
    return tens * 2 - 1


def save_tile(t, fp):
    t = (t + 1) / 2 #  Scale from [-1, 1] to [0, 1]
    arr = t.cpu().numpy()
    arr = (arr[0,0] * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    im.save(fp)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
dataloader = torch.utils.data.DataLoader(
    CSDataset(
        # root_dir = '../../../MDT-Calculations/saved_tiles/training/tiles_32_global_geodetic_only/',
        root_dir = 'C:/Users/oa18724/Documents/Master_PhD_folder/MDT-Calculations/saved_tiles/training/rescaled_tiles/',
        transform=transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            transforms.Lambda(bound_and_norm),
            # transforms.Normalize([0.5], [0.5])
        ])
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):

    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )
        batches_done += 1
    if epoch % opt.sample_interval == 0:
        gen_imgs = F.interpolate(gen_imgs, (32,32), mode='bilinear')
        save_image(gen_imgs.data[:144], "wgan/images/%d.png" % epoch, nrow=12, normalize=True)
    if epoch == opt.n_epochs-1:
        print('running')
        with torch.no_grad():
            for j in range(1000):
                z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
                gen_imgs = generator(z)
                gen_imgs = F.interpolate(gen_imgs, (32,32), mode='bilinear')
                arrs = gen_imgs.cpu().numpy()
                for k in range(opt.batch_size):
                    save_tile(gen_imgs.data[k].unsqueeze(0), "wgan/wgan_mdt_tiles_32/%d_"% epoch + str(k) + '_' + str(j) + ".png")

