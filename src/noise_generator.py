#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import struct

def matlib_show(images,labels):
    img = torchvision.utils.make_grid(images)
    img = img.numpy().transpose(1,2,0)
    plt.title(labels)
    plt.imshow(img)

mnist_raw_dir = '../data/mnist/MNIST/raw'
mnist_processed_dir = '../data/mnist/MNIST/processed'
mnist_train_images = 'train-images-idx3-ubyte'
mnist_train_labels = 'train-labels-idx1-ubyte'
mnist_test_images = 't10k-images-idx3-ubyte'
mnist_test_labels = 't10k-labels-idx1-ubyte'
mnist_test = 'test.pt'
mnist_train_images_path = mnist_raw_dir + '/' + mnist_train_images
mnist_train_labels_path = mnist_raw_dir + '/' + mnist_train_labels
mnist_test_images_path = mnist_raw_dir + '/' + mnist_test_images
mnist_test_labels_path = mnist_raw_dir + '/' + mnist_test_labels
mnist_processed_test_path = mnist_processed_dir + '/' + mnist_test

if not os.path.exists(mnist_raw_dir):
    print('No MNIST dataset in: {}'.format(mnist_raw_dir))
if not os.path.exists(mnist_processed_dir):
    print('No MNIST processed dataset in {}'.format(mnist_processed_dir))

noise_mnist_dir = '../data/noise-mnist'
noise_mnist_root_dir = '../data/noise-mnist/MNIST'
noise_mnist_raw_dir = '../data/noise-mnist/MNIST/raw'
noise_mnist_processed_dir = '../data/noise-mnist/MNIST/processed'
noise_mnist_train_images = 'train-noise-images-idx3-ubyte'
noise_mnist_train_labels = 'train-noise-labels-idx1-ubyte'
noise_mnist_test_images = 't10k-images-idx3-ubyte'
noise_mnist_test_labels = 't10k-labels-idx1-ubyte'
noise_mnist_test = 'test.pt'
merged_images_path= noise_mnist_raw_dir + '/' + noise_mnist_train_images
merged_labels_path= noise_mnist_raw_dir + '/' + noise_mnist_train_labels
noise_mnist_test_images_path = noise_mnist_raw_dir + '/' + noise_mnist_test_images
noise_mnist_test_labels_path = noise_mnist_raw_dir + '/' + noise_mnist_test_labels
noise_mnist_processed_test_path = noise_mnist_processed_dir + '/' + noise_mnist_test

with open(mnist_train_images_path,'rb') as f:
    mnist_images_magic, mnist_images_size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    print('load dataset: {}, magic number:{}, size:{}, nrows:{}, ncols:{}.'
          .format(mnist_train_images_path,
                  mnist_images_magic,
                  mnist_images_size,
                  nrows,
                  ncols))
    mnist_images = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    mnist_images = mnist_images.reshape((mnist_images_size, 1, nrows, ncols))

with open(mnist_train_labels_path,'rb') as f:
    mnist_labels_magic, mnist_labels_size = struct.unpack(">II", f.read(8))
    print('load dataset: {}, magic number:{}, size:{}.'
          .format(mnist_train_labels_path,
                  mnist_labels_magic,
                  mnist_labels_size))
    mnist_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    mnist_labels = mnist_labels.reshape((mnist_labels_size))

## visiable result
# matlib_show(torch.from_numpy(mnist_images[:16,]),torch.from_numpy(mnist_labels[:16,]))

### noise generator
# configs:
# MNIST 28*28 Cifar 32*32
noise_size = 20000
dim = 1
height = 28
width = 28

noise_images = torch.randint(256, (noise_size, 1, height,width))
noise_labels = torch.randint(10, (noise_size,))

## visiable result
# matlib_show(noise_images[:16,],noise_labels[:16,])

# plt.hist(noise_labels.numpy(), 100)
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Noise Labels')
# plt.show()

# perpare ubyte dataset
# convert data type to numpy array
np_noise_images = noise_images.numpy()
np_noise_labels = noise_labels.numpy()
merged_images = np.vstack((mnist_images,np_noise_images))
merged_labels = np.hstack((mnist_labels,np_noise_labels))

print('save ubyte images...')

magic=mnist_images_magic
size=noise_size+mnist_images_size
if not os.path.exists(noise_mnist_dir):
    os.mkdir(noise_mnist_dir)
if not os.path.exists(noise_mnist_root_dir):
    os.mkdir(noise_mnist_root_dir)
if not os.path.exists(noise_mnist_raw_dir):
    os.mkdir(noise_mnist_raw_dir)
shutil.copyfile(mnist_test_images_path, noise_mnist_test_images_path)
shutil.copyfile(mnist_test_labels_path, noise_mnist_test_labels_path)
with open(merged_images_path,'wb') as f:
    f.write(struct.pack(">II", magic, size))
    f.write(struct.pack(">II", nrows, ncols))
    print('write IDX3 done!')
    print('save dataset: {} magic number:{} size:{} nrows:{} ncols:{}'.format(merged_images_path,magic,size,nrows,ncols))
    for num in range(size):
        if num % 10000 == 0 and num != 0:
            print('write {} done!'.format(num))
        for x in range(nrows):
            for y in range(ncols):
                f.write(struct.pack("B",merged_images[num][0][x][y]))
    print('write {} done!'.format(size))

print('save ubyte labels...')

magic=mnist_labels_magic
size=noise_size+mnist_labels_size
with open(merged_labels_path,'wb') as f:
    f.write(struct.pack(">II", magic, size))
    print('write IDX3 done!')
    print('save dataset: {} magic number:{} size:{}'.format(merged_labels_path,magic,size))
    for num in range(size):
        if num % 10000 == 0 and num != 0:
            print('write {} done!'.format(num))
        f.write(struct.pack("B",merged_labels[num]))
    print('write {} done!'.format(size))

## perpare training.pt
if not os.path.exists(noise_mnist_processed_dir):
    os.mkdir(noise_mnist_processed_dir)
shutil.copyfile(mnist_processed_test_path, noise_mnist_processed_test_path)
with open(merged_images_path,'rb') as f:
    noise_mnist_images_magic, noise_mnist_images_size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    print('load dataset: {}, magic number:{}, size:{}, nrows:{}, ncols:{}.'
          .format(merged_images_path,
                  noise_mnist_images_magic,
                  noise_mnist_images_size,
                  nrows,
                  ncols))
    noise_mnist_images = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    noise_mnist_images = noise_mnist_images.reshape((noise_mnist_images_size, 1, nrows, ncols))

with open(merged_labels_path,'rb') as f:
    noise_mnist_labels_magic, noise_mnist_labels_size = struct.unpack(">II", f.read(8))
    print('load dataset: {} magic number:{} size:{}.'
          .format(merged_labels_path,
                  noise_mnist_labels_magic,
                  noise_mnist_labels_size))
    noise_mnist_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    noise_mnist_labels = noise_mnist_labels.reshape((noise_mnist_labels_size))

th_train_images = torch.from_numpy(noise_mnist_images)
th_train_labels = torch.from_numpy(noise_mnist_labels)

# matlib_show(th_train_images[0],th_train_labels[0])
# matlib_show(th_train_images[mnist_images_size],th_train_labels[mnist_labels_size])

zipped = (th_train_images.reshape(80000,28,28),th_train_labels)
print(zipped[0].shape)
print(zipped[1].shape)
training_path= noise_mnist_processed_dir + '/training.pt'
torch.save(zipped,training_path)