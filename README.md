# CoMBInE
*Cosmic Microwave Inpainting Experiment*

CoMBInE is a machine learning program that trhrough inpainting aims to reconstruct patches of the Cosmic Microwave Background thermals maps, as new nowel approach to eliminate/limit the effect of the foreground.

In this program we used the innovative method developed by a group of researches from NVIDIA corporation which uses Partial convolutional layers in place of the traditional convolutional layers. Traditional CNN filter responses are conditioned on both valid pixels as well as the substitute values in the masked holes which may lead to color discrepancy and blurriness. The partial convolutions are constructed such that, given a binary mask, the results depend only on the non-hole regions at every layer. Given sufficient layers of successive updates and with the addition of the automatic mask update step, which removes any masking where the partial convolution was able to operate on an unmasked value, even the largest masked holes will eventually shrink away, leaving only valid responses in the feature map.
See paper for more details: "*Image Inpainting for Irregular Holes Using Partial Convolutions*", https://arxiv.org/abs/1804.07723. 


# Dependencies
* Python 3.6
* Keras 2.2.4
* Tensorflow 1.12

# Implementation details
Details of the implementation are in the [paper itself](https://arxiv.org/abs/1804.07723), here a short summary of the main features.

## Mask Creation
The masks we generate are of two kinds and they both used OpenCV to draw their shapes. The first method contained in the script *utils/Mask_generator.py* creates masks composed of ellipses, lines and circle of random size and position. The second method contained in the script *utils/Mask_generator_circle.py* creates circular masks with center and radius that can vary according to one's preference. For simplicity, in the analysis done so far, the circular masks were always centered at the center of the images.

## Partial Convolution Layer
The key element in this implementation is the partial convolutional layer. Basically, given the convolutional filter **W** and the corresponding bias *b*, the following partial convolution is applied instead of a normal convolution:

<img src='./data/images/eq1.PNG' />

where ⊙ is element-wise multiplication and **M** is a binary mask of 0s and 1s. Importantly, after each partial convolution, the mask is also updated, so that if the convolution was able to condition its output on at least one valid input, then the mask is removed at that location, i.e.

<img src='./data/images/eq2.PNG' />

The result of this is that with a sufficiently deep network, the mask will eventually be all ones (i.e. disappear)

## UNet Architecture
Specific details of the architecture can be found in the paper, but essentially it's based on a UNet-like structure, where all normal convolutional layers are replace with partial convolutional layers, such that in all cases the image is passed through the network alongside the mask. The following provides an overview of the architecture.
<img src='./data/images/architecture.png' />

## Loss Function(s)
The loss function used in the paper is kinda intense, and can be reviewed in the paper. In short it includes:

* Per-pixel losses both for maskes and un-masked regions
* Perceptual loss based on ImageNet pre-trained VGG-16 (*pool1, pool2 and pool3 layers*)
* Style loss on VGG-16 features both for predicted image and for computed image (non-hole pixel set to ground truth)
* Total variation loss for a 1-pixel dilation of the hole region

The weighting of all these loss terms are as follows:
<img src='./data/images/eq7.PNG' />

## Training Procedure
Network was trained on ImageNet with a batch size of 1, and each epoch was specified to be 10,000 batches long. Training was furthermore performed using the Adam optimizer in two stages since batch normalization presents an issue for the masked convolutions (since mean and variance is calculated for hole pixels).

**Stage 1**
Learning rate of 0.0001 for 50 epochs with batch normalization enabled in all layers

**Stage 2**
Learning rate of 0.00005 for 50 epochs where batch normalization in all encoding layers is disabled.

# How to use this repository


## Libraries
*Mask Generator (Random shapes)*

*Mask Generator 2 (Circle with various radius)*

*PConv UNet Architecture (128*128)*

*PConv UNet Architecture (512*512)*

## Training

## Analysis
## Pre-trained weights
The model must be always at first initialized with the VGG 16 weights from imagenet before loading the new weights obtained after the training, in order to make valid predictions.
* [Ported VGG 16 weights](https://drive.google.com/open?id=1HOzmKQFljTdKWftEP-kWD7p2paEaeHM0)



