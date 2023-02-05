'''
just some code to test data augmentation

for augmentation strategy i was looking at SOTA results on MNIST:

An Ensemble of Simple Convolutional Neural Network Models for MNIST Digit Recognition
https://paperswithcode.com/paper/an-ensemble-of-simple-convolutional-neural

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

batch_size = 512

if torch.cuda.is_available():
    device = 'cuda'
    print("using cuda: " + str(torch.cuda.get_device_name(0)))
    torch.backends.cuda.matmul.allow_tf32 = True
    print("using TF32: " + str(torch.backends.cuda.matmul.allow_tf32))
else:
    device = 'cpu'
    print("using cpu")




from torchvision import datasets, transforms


# convert PIL images to torch tensors 
to_tensor = transforms.ToTensor()

# data augmentation
augment = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomAffine(0, translate=(0.2, 0.2)),
    ])

def one_hot(labels, n_classes=10):
    return torch.eye(n_classes)[labels]


mnist_train=datasets.MNIST('data', train=True, download=True, transform=augment)
mnist_test=datasets.MNIST('data', train=False, download=True, transform=None)

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

print(len(mnist_train))
print(len(train_loader))


# regular loading
def get_batch(split, bs=batch_size, start_ix=0):
    # generate a batch of data of inputs x and targets y
    if split == 'train':
        data = mnist_train
        ix = torch.randint(len(data), size=(bs,))
        x = torch.stack([to_tensor(augment(data[i][0])).flatten() for i in ix])
    else: 
        data = mnist_test
        ix = torch.arange(start_ix, start_ix+bs)
        x = torch.stack([to_tensor(data[i][0]).flatten() for i in ix])
    y = torch.stack([one_hot(data[i][1], 10) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

mini_batch_size = 4
batch = get_batch('test', bs=mini_batch_size, start_ix=0)

img_b,label_b = batch

# check if augmentation works
import matplotlib.pyplot as plt
for i in range(mini_batch_size):
    # use image from mnist_train
    print('image: ' + str(label_b[i]))
    img = img_b[i].reshape(28,28).cpu()
    plt.imshow(img, cmap='gray')
    plt.show()



