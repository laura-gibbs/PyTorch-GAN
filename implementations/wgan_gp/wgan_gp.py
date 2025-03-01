import argparse
import os
import numpy as np
import shutil
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

sys.path.append('./')
from csdataset import CSDataset
from skimage.transform import resize

from PIL import Image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=301, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=512, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=2, help="interval betwen image samples")
parser.add_argument("--tile_size", type=int, default=32, help="tile size to train network on")
parser.add_argument("--overlap", type=int, default=16, help="overlap of generated training tiles")
parser.add_argument("--cs", dest="cs", action="store_true", help="whether the tiles are currents")
opt = parser.parse_args()
print(opt)

tile_save_dir = f'wgan_gp/wgan_gp_{"cs" if opt.cs else "mdt"}_tiles_{opt.tile_size}'
tile_training_dir = f'C:/Users/oa18724/Documents/Master_PhD_folder/MDT-Calculations/saved_tiles/training/rescaled_tiles/{"cs" if opt.cs else "mdt"}-size{opt.tile_size}-overlap{opt.overlap}'

os.makedirs("wgan_gp/images", exist_ok=True)
if os.path.exists(tile_save_dir):
    shutil.rmtree(tile_save_dir)
os.makedirs(tile_save_dir, exist_ok=True)

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


# Loss weight for gradient penalty
lambda_gp = .1

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
        root_dir=tile_training_dir,
        transform=transforms.Compose([
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.ToTensor(),
            # transforms.Lambda(bound_and_norm),
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()) * 2 - 1),
            # transforms.Normalize([0.5], [0.5])
        ])
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
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
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done += opt.n_critic
    if epoch % opt.sample_interval == 0:
        gen_imgs = F.interpolate(fake_imgs, (opt.tile_size, opt.tile_size), mode='bilinear')
        gen_imgs = (gen_imgs - gen_imgs.min())/(gen_imgs.max() - gen_imgs.min())
        save_image(gen_imgs.data[:144], "wgan_gp/images/%d.png" % epoch, nrow=12, normalize=True)
    if epoch == opt.n_epochs-1:
        print('running')
        with torch.no_grad():
            for j in range(200):
                z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
                gen_imgs = generator(z)
                gen_imgs = F.interpolate(gen_imgs, (opt.tile_size, opt.tile_size), mode='bilinear')
                arrs = gen_imgs.cpu().numpy()
                for k in range(opt.batch_size):
                    save_tile(gen_imgs.data[k].unsqueeze(0), f"{tile_save_dir}/{epoch}_{k}_{j}.png")
