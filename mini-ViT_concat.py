''' 
Basically the same as mini-ViT.py, but with a concatenation of the input 
with the output of the transformer blocks. Embedding dimension is 2ximg_size, 
but the last of the transformer feed forward has img_size dimension size instead
of n_embd to be able to concatenate the input with the output of the transformer blocks.

This is a seperate file to keep the the mini-ViT.py file clean and easy to read.

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


# Hyperparameters & Global variables
# ----------------
#
# global variables
n_classes = 10                  # number of classes (10 for MNIST)
img_size = 784                  # image size (28x28 for MNIST)

# model hyperparameters
batch_size = 512                # lower for smaller VRAM (512 needs around 22 GB VRAM)
max_iters = 10000                # maximum training iterations (very long training time, this can be lowered)
learning_rate = 1e-4            # learning rate
eval_interval = 500             # steps after which eval set is evaluated
eval_iters = 200                # number of samples taken for evaluation

n_head = 16                     # number of attention heads (16 because 16x98 = 1568 = 2ximg_size)
d_head = 98                     # dimension of each attention head (56 because 16x98 = 1568 = 2ximg_size)
n_embd = 2 * img_size           # for n_emb we use the concatenation of the input and the previous layer so 2ximg_size
n_layers = 16                   # number of layers 
dropout = 0.1                   # dropout rate
# ----------------

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
# get MNIST handwritten digits 
from torchvision import datasets

mnist_train=datasets.MNIST('data', train=True, download=True)
mnist_test=datasets.MNIST('data', train=False, download=True)

# ------------------
# Data preprocessing
# ------------------

# convert PIL images to torch tensors 
import torchvision.transforms as transforms
transform = transforms.ToTensor()
# one hot encoding of labels
def one_hot(labels, n_classes):
    return torch.eye(n_classes)[labels]


train_data = torch.stack([transform(mnist_train[i][0]).flatten() for i in range(len(mnist_train))])
test_data = torch.stack([transform(mnist_test[i][0]).flatten() for i in range(len(mnist_test))])

train_labels = one_hot(torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))]), 10)
test_labels = one_hot(torch.tensor([mnist_test[i][1] for i in range(len(mnist_test))]), 10)

    
 
print("MNIST train set size: " + str(len(train_data)))
print("MNIST test set size: " + str(len(test_data)))


# Data batching
def get_batch(split, bs=batch_size, rnd=True, start_ix=0):
    # generate a batch of data of inputs x and targets y
    if split == 'train':
        data_x = train_data
        data_y = train_labels
    else: 
        data_x = test_data
        data_y = test_labels
    if rnd:
        ix = torch.randint(len(data_x), size=(bs,))
    else:
        ix = torch.arange(start_ix, start_ix+bs)
    x = torch.stack([data_x[i] for i in ix])
    y = torch.stack([data_y[i] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# loss calcucation
@torch.no_grad() # no need to calculate gradients for evaluation
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _ , loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#----------------
#Transformer Model
#----------------
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
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd//2) # n_embd//2 because of the residual connection
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
        # we wont do residual connections here 
        # since we are concatenating the original input 
        #x = x + self.attn(self.ln1(x))
        #x = x + self.ff(self.ln2(x))
        x = self.attn(self.ln1(x))
        x = self.ff(self.ln2(x))
        return x

class Transformer(nn.Module):
    """ the full transformer model """
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(img_size, img_size)
        #self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layers)])
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layers)])
        

        self.block = Block(n_embd, n_head)
        self.ln_f = nn.LayerNorm(n_embd)
        self.linear_f = nn.Linear(n_embd, n_classes)

    def forward(self, x, y=None):
        
        x_orig = x # save original input for concatenation
        x = self.projection(x)  #(B, img_size) -> (B, img_size)
        x = torch.cat([x, x_orig], dim=-1) # (B, img_size) -> (B, 2*img_size=n_embd)

        for i in range(n_layers):
            x = self.blocks[i](x)  # (B, n_embd) -> (B, img_size)
            x = torch.cat([x, x_orig], dim=-1) # (B, img_size) -> (B, n_embd)

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
#print(model)

# --------------
# Training
# --------------

# optimizer using AdamW 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
train = True
if train == True:
    # get current time for tracking training time
    start_t = time.time()
    t = start_t
    for iter in range(max_iters):
        # every once in a while evaluate the loss on the train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            dt = time.time() - t
            total_t = time.time() - start_t
            t = time.time()
            print(f'iter: {iter} train loss: {losses["train"]:.3f} val loss: {losses["test"]:.3f} time: {time.strftime("%H:%M:%S", time.gmtime(dt))} total: {time.strftime("%H:%M:%S", time.gmtime(total_t))}')

        # smaple a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_duration = time.time() - start_t
    print(f'time needed to train: {time.strftime("%H:%M:%S", time.gmtime(total_duration))}')

# example classification of the test set
print('------------------------------------')
print('example classification')
print('------------------------------------')

# classify a single image of the test set
def classify(img_num):
    print('------------------------------------')
    print(f'classifyig test image {img_num} with a: {(mnist_test[img_num][1])}')
    x = test_data[img_num].unsqueeze(0).to(device) # adding batch dimension so it becomes (1, 784)
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
    print(f'predicted labels are: \n{id_1} with probability: {p1*100:.2f}%\n{id_2} with probability: {p2*100:.2f}%\n{id_3} with probability: {p3*100:.2f}%')


# classify 10 random images of the test set
for i in range(10):
    rnd = torch.randint(0, len(mnist_test), (1,)).item()
    classify(rnd)


# evaluate the model with the full test set
print('------------------------------------')
print('final evaluation')
print('------------------------------------')

def evaluate():
    def eval(x,y):
        logits, _ = model(x)
        logits = F.softmax(logits, dim=-1)
        max_value = torch.max(logits, dim=1, keepdim=True)  # get maximum value of each row 
        index = max_value[1] # get the index 
        result = y.gather(dim=1, index=index) # get the index of the correct label
        result = result.sum()  # since this will be 0 for incorrect predictions and 1 for correct predictions we can just sum up
        return result
    
    correct = 0
    # evaluate the model in batches 
    for i in range(len(test_data)//batch_size):
        x,y = get_batch('test', bs=batch_size, rnd=False, start_ix=i*batch_size)
        result = eval(x,y)
        correct += result

    # get the rest of the data that is not a multiple of batch_size
    rest_bs = len(test_data)%batch_size
    x,y = get_batch('test', bs=rest_bs, rnd=False, start_ix=len(test_data)-rest_bs)
    result = eval(x,y)
    correct += result
    
    print(f'correct: {int(correct)} out of {len(test_data)}')
    print(f'accuracy: {100*correct/len(test_data):.2f}%')

evaluate()

print('------------------------------------')
