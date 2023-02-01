import torch
import torch.nn as nn
import torch.nn.functional as F


# using cuda and Tensorcores if available
if torch.cuda.is_available():
    device = 'cuda'
    print("using cuda: " + str(torch.cuda.get_device_name(0)))
    torch.backends.cuda.matmul.allow_tf32 = True
    print("using TF32: " + str(torch.backends.cuda.matmul.allow_tf32))
else:
    device = 'cpu'
    print("using cpu")


rnd = torch.randint(0, 1000, (1,))
print(f'random number device: {rnd.device}')
rnd = rnd.to(device)
print(f'random number device: {rnd.device}')


q = torch.randn(1, 768)
k = torch.randn(1, 768)
v = torch.randn(1, 768)

attn = torch.einsum('bi,bj->bij', q, k)

print(attn.shape)


attn = F.softmax(attn, dim=-1)

# calculate output
out = torch.einsum('bij,bj->bi', attn, v) # (B, N, N) @ (B, N) -> (B, N)

print(out.shape)


