'''
Description: CS5340 - Hopfield Network
Name: Your Name, Your partner's name
Matric No.: Your matric number, Your partner's matric number
'''


import matplotlib
matplotlib.use('Agg')
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import random


def load_image(fname):
    img = Image.open(fname).resize((32, 32))
    img_gray = img.convert('L')
    img_eq = ImageOps.autocontrast(img_gray)
    img_eq = np.array(img_eq.getdata()).reshape((img_eq.size[1], -1))
    return img_eq


def binarize_image(img_eq):
    img_bin = np.copy(img_eq)
    img_bin[img_bin < 128] = -1
    img_bin[img_bin >= 128] = 1
    return img_bin


def add_corruption(img):
    img = img.reshape((32, 32))
    t = np.random.choice(3)
    if t == 0:
        i = np.random.randint(32)
        img[i:(i + 8)] = -1
    elif t == 1:
        i = np.random.randint(32)
        img[:, i:(i + 8)] = -1
    else:
        mask = np.sum([np.diag(-np.ones(32 - np.abs(i)), i)
                       for i in np.arange(-4, 5)], 0).astype(np.int)
        img[mask == -1] = -1
    return img.ravel().reshape(32,32)

def mat2vec(x):
    m = x.shape[0]*x.shape[1]
    tmp1 = np.zeros(m)

    c = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp1[c] = x[i,j]
            c +=1
    return tmp1

def learn_hebbian(imgs):
    img_size = np.prod(imgs[0].shape)
    ######################################################################
    ######################################################################
    weights = np.zeros((img_size, img_size))
    Wh = np.zeros((img_size, img_size))
    bias = np.zeros(img_size)
    for k in imgs:   
        simg = mat2vec(k)
        for i in range(img_size):
            for j in range(i,img_size):
                if i == j:
                    weights[i,j] = 0
                else:
                    weights[i,j] = simg[i] * simg[j]
                    weights[j,i] = weights[i,j]

        Wh+= weights 
    weights = Wh
    #######################################################################
    #######################################################################
    return weights, bias

def sigmoid(z):
    s = 1.0/(1.0 + np.exp(-1.0 * z))
    return s    

import torch
from os import system

def learn_maxpl(imgs):
    img_size = np.prod(imgs[0].shape)
    ######################################################################
    ######################################################################
    weights = np.zeros((img_size, img_size))
    bias = np.zeros(img_size)

    # weights = torch.zeros(img_size, img_size, dtype=torch.double, requires_grad=True)
    sig = torch.nn.Sigmoid() # initialize sigmoid layer
    loss = torch.nn.BCELoss() # initialize loss function

    epoch = 25
    for itr in range(epoch):
        system('cls')
        print("Epoch number:("+str(itr+1)+"/"+str(epoch)+")")
        for k in imgs:   
            simg = torch.from_numpy(mat2vec(k))
            for i in range(img_size):
                wgts = torch.from_numpy(weights[i][:])
                wgts.requires_grad_(True)
                loss_out = loss(sig(wgts.dot(simg)), simg) # forward pass
                loss_out.backward() # backward pass

                wgts_grad = wgts.grad.numpy()
                weights[i][:] -= 0.1*wgts_grad
                weights[:][i] -= 0.1*wgts_grad
                weights[i][i] = 0

    #######################################################################
    #######################################################################
    return weights, bias


def plot_results(imgs, cimgs, rimgs, fname='result.png'):
    '''
    This helper function can be used to visualize results.
    '''
    img_dim = 32
    assert imgs.shape[0] == cimgs.shape[0] == rimgs.shape[0]
    n_imgs = imgs.shape[0]
    fig, axn = plt.subplots(n_imgs, 3, figsize=[8, 8])
    for j in range(n_imgs):
        axn[j][0].axis('off')
        axn[j][0].imshow(imgs[j].reshape(img_dim, img_dim), cmap='Greys_r')
    axn[0, 0].set_title('True')
    for j in range(n_imgs):
        axn[j][1].axis('off')
        axn[j][1].imshow(cimgs[j].reshape(img_dim, img_dim), cmap='Greys_r')
    axn[0, 1].set_title('Corrupted')
    for j in range(n_imgs):
        axn[j][2].axis('off')
        axn[j][2].imshow(rimgs[j].reshape((img_dim, img_dim)), cmap='Greys_r')
    axn[0, 2].set_title('Recovered')
    fig.tight_layout()
    plt.savefig(fname)


def recover(cimgs, W, b):
    img_size = np.prod(cimgs[0].shape)
    ######################################################################
    ######################################################################
    rimgs = []
    for k in cimgs:
        simg = mat2vec(k)
        for s in range(10000):
            m = len(simg)
            i = random.randint(0,m-1)
            u = np.dot(W[i][:],simg) - 0.5

            if u>0:
                simg[i] = 1
            else:
                simg[i] = -1
        rimgs.append(simg.reshape(32,32))
    rimgs = np.asarray(rimgs)
    #######################################################################
    #######################################################################
    return rimgs


def main():
    # Load Images and Binarize
    ifiles = sorted(glob.glob('images/*'))
    timgs = [load_image(ifile) for ifile in ifiles]
    imgs = np.asarray([binarize_image(img) for img in timgs])

    # Add corruption
    cimgs = []
    for i, img in enumerate(imgs):
        cimgs.append(add_corruption(np.copy(imgs[i])))
    cimgs = np.asarray(cimgs)

    # Recover 1 -- Hebbian
    Wh, bh = learn_hebbian(imgs)
    rimgs_h = recover(cimgs, Wh, bh)
    np.save('hebbian.npy', rimgs_h)
    plot_results(imgs,cimgs,rimgs_h,'result1.png')
    # print(Wh)

    # Recover 2 -- Max Pseudo Likelihood
    Wmpl, bmpl = learn_maxpl(imgs)
    rimgs_mpl = recover(cimgs, Wmpl, bmpl)
    np.save('mpl.npy', rimgs_mpl)
    plot_results(imgs,cimgs,rimgs_mpl,'result2.png')
    # print(Wmpl)

if __name__ == '__main__':
    main()
