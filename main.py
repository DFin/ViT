''' 
Implementing a Transformer model for image classification

Initially will just use MNIST handwritten digits and process the full image before implementing
the 16x16 pixel input as described in the paper.
 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


# Hyperparameters & Global variables
# ----------------

# global variables
n_classes = 10                  # number of classes (10 for MNIST)
img_size = 784                  # image size (28x28 for MNIST)
val_size = 10000                # take 10.000 images for the dev validation set

# model hyperparameters
batch_size = 512                # lower for smaller VRAM
max_iters = 10000               # maximum training iterations
eval_interval = 500             # steps after which eval set is evaluated
learning_rate = 3e-4            # learning rate
eval_iters = 200                # number of iterations for evaluation

n_head = 8                     # number of attention heads 
d_head = 32                     # dimension of each attention head
n_embd = n_head * d_head       # embedding dimension (using head dimension * number of heads)
#n_embd = img_size               # instead of embedding dimension, use image size
n_layers = 8                   # number of layers 
dropout = 0.1                   # dropout rate
use_GELU = True                 # if GELU (True) or ReLU and dropout (False) should be used	


# ----------------


# for reproducability:
# torch.manual_seed(1337)

# using cuda and Tensorcores if available
if torch.cuda.is_available():
    device = 'cuda'
    print("using cuda: " + str(torch.cuda.get_device_name(0)))
    torch.backends.cuda.matmul.allow_tf32 = True
    print("using TF32: " + str(torch.backends.cuda.matmul.allow_tf32))
else:
    device = 'cpu'
    print("using cpu")


# Training data
# --------------
# using MNIST handwritten digits
from torchvision import datasets

mnist_train=datasets.MNIST('data', train=True, download=True)
mnist_test=datasets.MNIST('data', train=False, download=False)

# Data preprocessing
# ------------------


# convert PIL images to torch tensors 
import torchvision.transforms as transforms
transform = transforms.ToTensor()
train_data = torch.stack([transform(mnist_train[i][0]).flatten() for i in range(len(mnist_train))])
val_data = train_data[len(mnist_train)-val_size:]
train_data = train_data[:len(mnist_train)-val_size]
test_data = torch.stack([transform(mnist_test[i][0]).flatten() for i in range(len(mnist_test))])

# convert labels to torch tensors with one-hot encoding
def one_hot(labels, n_classes):
    return torch.eye(n_classes)[labels]
train_labels = one_hot(torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))]), 10)
val_labels = train_labels[len(train_labels)-val_size:]
train_labels = train_labels[:len(train_labels)-val_size]
test_labels = one_hot(torch.tensor([mnist_test[i][1] for i in range(len(mnist_test))]), 10)

print("MNIST train set size: " + str(len(train_data)))
print("MNIST val set size: " + str(len(val_data)))
print("MNIST test set size: " + str(len(mnist_test)))


# # show an example image of the training data
# import matplotlib.pyplot as plt
# rnd = torch.randint(0, len(mnist_train), (1,)).item()
# print(f'showing image {rnd} with a: {(mnist_train[rnd][1])}')
# print(f'label as one-hot is: {train_labels[rnd].numpy() }')
# plt.imshow(train_data[rnd].reshape(28,28), cmap='gray')
# plt.show()


# Data batching
# -------------


# get a batch of data
def get_batch(split):
    # generate a batch of data of inputs x and targets y
    if split == 'train':
        data_x = train_data
        data_y = train_labels
    elif split == 'test':
        data_x = test_data
        data_y = test_labels
    else:
        data_x = val_data
        data_y = val_labels
    ix = torch.randint(len(data_x), size=(batch_size,))
    x = torch.stack([data_x[i] for i in ix])
    y = torch.stack([data_y[i] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y




# loss calcucation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Transformer Model
# --------------

class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.linear_q = nn.Linear(n_embd, head_size, bias=False)
        self.linear_k = nn.Linear(n_embd, head_size, bias=False)
        self.linear_v = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N = x.shape
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        # calculate attention
        attn = torch.einsum('bi,bj->bij', q, k) # (B, N) @ (B, N) -> (B, N, N)
        attn = attn * N ** -0.5
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # calculate output
        out = torch.einsum('bij,bj->bi', attn, v) # (B, N, N) @ (B, N) -> (B, N)
        return out
    
class MultiHead(nn.Module):
    """ Multi-head self-attention """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.linear = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.linear(out)
        out = self.dropout(out)
        return out
    
class FeedForward(nn.Module):
    """ Feed-forward layer of the transformer """
    def __init__(self, n_embd):
        super().__init__()
        if use_GELU:
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4*n_embd),
                nn.GELU(),
                nn.Linear(4*n_embd, n_embd)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4*n_embd),
                nn.ReLU(),
                nn.Linear(4*n_embd, n_embd),
                nn.Dropout(dropout)
            )
    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """ One block of the transformer """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.attn = MultiHead(n_head, head_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x



class Transformer(nn.Module):
    """ the full transformer model """
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(img_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.linear_f = nn.Linear(n_embd, n_classes)

    def forward(self, x, y=None):
        x = self.projection(x) # (B, img_size) -> (B, n_embd)
        x = self.blocks(x)  # (B, N, n_embd) -> (B, N, n_embd)
        x = self.ln_f(x)
        logits = self.linear_f(x) 

        if y is None:
            loss = None
        else:
            B, N = logits.shape
            logits = logits.view(B*N)
            y = y.view(B*N)
            loss = F.cross_entropy(logits, y)
            # normalize loss by batch size
            loss = loss / B
        return logits, loss


model = Transformer()
m = model.to(device)   

# print number of parameters
print(f'number of parameters: %.2fM' %((sum(p.numel() for p in m.parameters() if p.requires_grad))/1e6))




# classify a single image
def classify(img_num):
    print('------------------------------------')
    print(f'classifyig image {img_num} with a: {(mnist_test[img_num][1])}')
    #print(f'label as one-hot is: {test_labels[img_num].numpy() }')

    # add batch dimension
    x = test_data[img_num].unsqueeze(0).to(device)
    logits, _ = model(x)
    logits = F.softmax(logits, dim=-1)
    logits = logits.detach().cpu().tolist()[0]
    sorted_list = sorted(logits, reverse=True)
    p1 = sorted_list[0]
    p2 = sorted_list[1]
    p3 = sorted_list[2]
    id_1 = logits.index(p1)
    id_2 = logits.index(p2)
    id_3 = logits.index(p3)
    #print(f'label as one-hot is: {logits}')
    print(f'predicted labels are: \n{id_1} with probability: {p1:.3f}\n{id_2} with probability: {p2:.3f}\n{id_3} with probability: {p3:.3f}')


# evaluate the model with the full test set
# TODO: #1 add batch processing to speed up
def evaluate():
    print('------------------------------------')
    print('evaluating the model')
    print('------------------------------------')
    correct = 0
    for i in range(len(test_data)):
        x = test_data[i].unsqueeze(0).to(device)
        logits, _ = model(x)
        logits = F.softmax(logits, dim=-1)
        logits = logits.detach().cpu().tolist()[0]
        max_value = max(logits)
        index = logits.index(max_value)
        if index == mnist_test[i][1]:
            correct += 1
    print(f'correct: {correct} out of {len(test_data)}')
    print(f'accuracy: {correct/len(test_data):.3f}')

# Training
# --------------


# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# for tracking training time
start_t = time.time()
t = start_t

# training loop
for iter in range(max_iters):
    # every once in a while evaluate the loss on the train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        dt = time.time() - t
        total_t = time.time() - start_t
        t = time.time()
        print(f'iter: {iter} train loss: {losses["train"]:.3f} val loss: {losses["val"]:.3f} time: {time.strftime("%H:%M:%S", time.gmtime(dt))} total: {time.strftime("%H:%M:%S", time.gmtime(total_t))}')

    # smaple a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

total_duration = time.time() - start_t
print(f'time needed to train: {time.strftime("%H:%M:%S", time.gmtime(total_duration))}')

print('------------------------------------')
print('example classification')
print('------------------------------------')

for i in range(10):
    rnd = torch.randint(0, len(mnist_test), (1,)).item()
    classify(rnd)

print('------------------------------------')
print('final evaluation')
print('------------------------------------')
evaluate()

