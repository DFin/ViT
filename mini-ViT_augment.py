''' 
Testing out data augmentation. Also changed to dataloader with multiprocessing 
due to slow get_batch function in the mini-ViT.py script.

the augmentation strategy i was looking at SOTA results on MNIST:
An Ensemble of Simple Convolutional Neural Network Models for MNIST Digit Recognition
https://paperswithcode.com/paper/an-ensemble-of-simple-convolutional-neural


'''

from torch.multiprocessing import freeze_support

if __name__ == '__main__': # this is needed for multiprocessing
    freeze_support()

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
    batch_size = 2048               # lower for smaller VRAM (2048 needs around 20 GB VRAM)
    max_iters = 5000                  # maximum training iterations // compared to mini-ViT.py an iter goes through the whole dataset
    learning_rate = 3e-4            # learning rate
    eval_interval = 10               # steps after which eval set is evaluated
    #eval_iters = 100               # number of samples taken for evaluation

    n_head = 14                     # number of attention heads (14 because 14x56 = 784 = img_size)
    d_head = 56                     # dimension of each attention head (56 because 14x56 = 784 = img_size)
    #n_embd = n_head * d_head       # embedding dimension (using head dimension * number of heads)
    n_embd = img_size               # instead of embedding dimension, use image size
    n_layers = 16                   # number of layers 
    dropout = 0.1                   # dropout rate
    use_GELU = True                 # if GELU (True) or ReLU and dropout (False) should be used
    use_lr_exp_decay = True        # if learning rate should be exponentially decayed
    lr_exp_decay_rate_gamma = 0.999 # learning rate decay rate gamma	
    num_threads = 6                 # number of threads for data loading (set to 0 for no multithreading)
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





    # ------------------
    # Data preprocessing
    # ------------------

    # TODO: #5 use torch dataloader instead of get_batch function
    from torchvision import datasets, transforms
    #mport torchvision.transforms as transforms

    # convert PIL images to torch tensors 
    to_tensor = transforms.ToTensor()

    # data augmentation
    augment = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomAffine(0, translate=(0.2, 0.2)),
        transforms.ToTensor(),
        ])

    # one hot encoding of labels
    def one_hot(labels, n_classes):
        return torch.eye(n_classes)[labels]
    
    def one_hot_encode(batch, num_classes):
        one_hot = torch.zeros(batch.shape[0], num_classes, dtype=torch.float32)
        one_hot[torch.arange(batch.shape[0]), batch] = 1
        return one_hot

    # Training data
    # get MNIST handwritten digits 

    mnist_train=datasets.MNIST('data', train=True, download=True, transform=augment)
    mnist_test=datasets.MNIST('data', train=False, download=True, transform=to_tensor)

    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_threads)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=1000, shuffle=False, num_workers=num_threads)


    def get_batch(split, bs=batch_size, start_ix=0):
        # generate a batch of data of inputs x and targets y
        if split == 'train':
            data = mnist_train
            ix = torch.randint(len(data), size=(bs,))
            #x = torch.stack([to_tensor(augment(data[i][0])).flatten() for i in ix])
            x = torch.stack([(data[i][0]).flatten() for i in ix])

        else: 
            data = mnist_test
            ix = torch.arange(start_ix, start_ix+bs)
            x = torch.stack([(data[i][0]).flatten() for i in ix])
        #y = torch.stack([one_hot(data[i][1], 10) for i in ix])
        y = torch.stack([torch.tensor(data[i][1]) for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y



    # loss calcucation
    @torch.no_grad() # no need to calculate gradients for evaluation
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'test']:
            eval_iters = len(train_loader) if split == 'train' else len(test_loader)
            losses = torch.zeros(eval_iters)
            for k, (X,Y) in enumerate(train_loader if split == 'train' else test_loader):
                X, Y = X.to(device), Y.to(device, dtype=torch.long)
                X = X.view(X.shape[0], -1)
                #X, Y = get_batch(split)
                _ , loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    
    # evaluate full model
    def evaluate():
        def eval(x,y):
            logits, _ = model(x)
            logits = F.softmax(logits, dim=-1)
            max_value = torch.max(logits, dim=1, keepdim=True)  # get maximum value of each row 
            index = max_value[1] # get the index 
            y = one_hot_encode(y, num_classes=10).to(device) # one hot encode the labels
            #y = y.to(device)
            result = y.gather(dim=1, index=index) # get the index of the correct label
            result = result.sum()  # since this will be 0 for incorrect predictions and 1 for correct predictions we can just sum up
            return result
        
        correct = 0
        for _, (xb, yb) in enumerate(test_loader):
            xb, yb = xb.to(device), yb.to(device, dtype=torch.long)
            xb = xb.view(xb.shape[0], -1) # flatten the images (B, 1, 28, 28) -> (B, 784)
            result = eval(xb,yb)
            correct += result
        print(f'correct: {int(correct)}/{len(mnist_test)} - accuracy: {100*correct/len(mnist_test):.2f}%')

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

            if img_size != n_embd:
                x = self.projection(x)  #(B, img_size) -> (B, n_embd)
            x = self.blocks(x)   
            x = self.ln_f(x)
            logits = self.linear_f(x) 

            if y is None:
                loss = None
            else:
                #B, N = logits.shape
                #logits = logits.view(B*N)
                #y = y.view(B*N)
                loss = F.cross_entropy(logits, y)
                # normalize loss by batch size
                loss = loss #/ B
            return logits, loss

    model = Transformer()
    m = model.to(device)   


    # print number of parameters
    print(f'number of parameters: %.2fM' %((sum(p.numel() for p in m.parameters() if p.requires_grad))/1e6))


    # --------------
    # Training
    # --------------

    # optimizer using AdamW 
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if use_lr_exp_decay:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_exp_decay_rate_gamma)

    # training loop

    train = True
    if train == True:
        # get current time for tracking training time
        start_t = time.time()
        t = start_t
        print('----------------------------------------')

        for iter in range(max_iters):
            # every once in a while evaluate the loss on the train and val sets
            if iter % eval_interval == 0:# and iter > 0:
                losses = estimate_loss()
                dt = time.time() - t
                total_t = time.time() - start_t
                t = time.time()
                print(f'iter: {iter} lr: {optimizer.param_groups[0]["lr"]:.8f} train loss: {losses["train"]:.3f} val loss: {losses["test"]:.3f} time: {time.strftime("%H:%M:%S", time.gmtime(dt))} total: {time.strftime("%H:%M:%S", time.gmtime(total_t))}')
                evaluate()
                print('----------------------------------------')

            for batch_idx, (xb, yb) in enumerate(train_loader):
                #print(batch_idx)
                xb, yb = xb.to(device), yb.to(device, dtype=torch.long)
                xb = xb.view(xb.shape[0], -1) # flatten the images (B, 1, 28, 28) -> (B, 784
                logits, loss = model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            if use_lr_exp_decay:
                scheduler.step()

                
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
        #x = to_tensor(mnist_test[img_num][0]).flatten().unsqueeze(0).to(device) # adding batch dimension so it becomes (1, 784)
        x = mnist_test[img_num][0].flatten().unsqueeze(0).to(device) # adding batch dimension so it becomes (1, 1, 28, 28)
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
    evaluate()
    print('------------------------------------')
