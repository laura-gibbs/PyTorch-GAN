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
import torch

sys.path.append('./')
from csdataset import CSDataset
from skimage.transform import resize

from PIL import Image
import matplotlib.pyplot as plt
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_epochs", type=int, default=301, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=512, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=2, help="interval between image sampling")
parser.add_argument("--tile_size", type=int, default=32, help="tile size to train network on")
parser.add_argument("--overlap", type=int, default=16, help="overlap of generated training tiles")
parser.add_argument("--cs", dest="cs", action="store_true", help="whether the tiles are currents")
opt = parser.parse_args()
print(opt)

tile_save_dir = f'dcgan/gan_{"cs" if opt.cs else "mdt"}_tiles_{opt.tile_size}'
tile_training_dir = f'C:/Users/oa18724/Documents/Master_PhD_folder/MDT-Calculations/saved_tiles/training/rescaled_tiles/{"cs" if opt.cs else "mdt"}-size{opt.tile_size}-overlap{opt.overlap}'

os.makedirs("dcgan/images", exist_ok=True)
if os.path.exists(tile_save_dir):
    shutil.rmtree(tile_save_dir)
os.makedirs(tile_save_dir)

cuda = True if torch.cuda.is_available() else False


def bound_and_norm(tens):
    # From [0, 1] to [-1, 1]
    return tens * 2 - 1


def save_tile(t, fp):
    t = (t + 1) / 2 #  Scale from [-1, 1] to [0, 1]
    arr = t.cpu().numpy()
    arr = (arr[0,0] * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    im.save(fp)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
dataloader = torch.utils.data.DataLoader(
    CSDataset(
        root_dir=tile_training_dir,
        transform=transforms.Compose([
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )
        # Add code to resize image up
        batches_done = epoch * len(dataloader) + i
    if epoch % opt.sample_interval == 0:
        gen_imgs = F.interpolate(gen_imgs, (opt.tile_size, opt.tile_size), mode='bilinear')
        gen_imgs = (gen_imgs - gen_imgs.min())/(gen_imgs.max() - gen_imgs.min())
        save_image(gen_imgs.data[:144], "dcgan/images/%d.png" % epoch, nrow=12, normalize=True)
    if epoch == opt.n_epochs-1:
        print('running')
        # save yaml and trained model
        with open(f'{tile_save_dir}/config.yml', 'w') as f:
            yaml.dump(opt, f)
        # save model
        # torch.save(model.state_dict(), os.path.join(
        #     save_dir,
        #     save_name
        # ))
        with torch.no_grad():
            for j in range(200):
                z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
                gen_imgs = generator(z)
                gen_imgs = F.interpolate(gen_imgs, (opt.tile_size, opt.tile_size), mode='bilinear')
                arrs = gen_imgs.cpu().numpy()
                for k in range(opt.batch_size):
                    save_tile(gen_imgs.data[k].unsqueeze(0), f"{tile_save_dir}/{epoch}_{k}_{j}.png")
