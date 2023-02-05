# ViT


 The intention for this repo is to build a full Vision Transformer in pytorch from 
 scratch based on the paper "An Image is Worth 16x16 Words: Transformers for Image 
 Recognition at Scale" (https://arxiv.org/abs/2010.11929).
 
 This is soley for my own education to learn about transformers in computer vision,
 but I share the repo in case someone else profits from this.

 # mini-ViT.py

mini-ViT.py is a ~350 line of code standalone implementation of a transformer for MNIST
handwriting recognotion. It uses 16 layers of Multi-head self attention with 14 heads 
and a head dimension of 56. The embedding size of 784 which is the flattened image size
of MNISTs 28x28 images. For this embedding size (=image size) there is no projection layer
needed and the images are directly passed to the first transformer block.

This network architecture is overkill with ~100M parameter, but it reaches ~98.5% accuracy
on the test set and doesn't overfit so ill just leave it at that. For anything serious this
should be drastically optimized and should work with a fraction of parameters.

# mini-ViT_concat.py

Experiment where the transformer is fed a concatination of the original (flattened) 
input image with the previous output (or with the output of a linear layer for the 
first transformer block) 

This is roughly the same as mini-ViT.py, but due to the concatenation with the the
original input the embedding size is 2ximg_size (2*784=1568) instead. To better fit the change
of the embedding size the amount of heads is increased to 16 and the dimension is increased
to 98 (16*98=1568). Additionally after the transformer block there is an additional linear 
projection layer with img_size dimension (784) instead of n_embd (1568) to be able to concatenate
the input with the output of the transformer blocks.

Results didnt improve though.

# mini-ViT_augment.py

This experiment is using the same architecture as mini-ViT but does additional data augmentation.
In addtion dataloader doing augmentation with multiple worker is used to keep the GPU busy. 

This reaches a decent 99.3% accuracy.



