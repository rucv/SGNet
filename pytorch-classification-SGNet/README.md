# Pytorch-cifar100 SGNet

SGNet on cifar100 using pytorch

## Requirements

This is my experiment eviroument, pytorch0.4 should also be fine
- python3.5
- pytorch1.0
- tensorflow1.5(optional)
- cuda8.0
- cudnnv5
- tensorboardX1.6(optional)


## Usage

### 1. enter directory

### 2. dataset 
I will use cifar100 dataset from torchvision since it's more convenient, but I also
kept the sample code for writing your own dataset module in dataset folder, as an
example for people don't know how to write it.

### 3. run tensorbard(optional)
Install tensorboardX (a tensorboard wrapper for pytorch)
```bash
$ pip install tensorboardX
$ mkdir runs
Run tensorboard
$ tensorboard --logdir='runs' --port=6006 --host='localhost'
```

### 4. train the model
Specify the net you want to train using arg -net

```bash
$ python train_major_enhance.py -net vgg_mm
```

sometimes, you might want to use warmup training by set ```-warm``` to 1 or 2, to prevent network
diverge during early training phase.

The supported net args are:
```
vgg_mm
```
Normally, the weights file with the best accuracy would be written to the disk with name suffix 'best'(default in checkpoint folder).


### 5. test the model
Test the model using test.py, calculating inference time
```bash
$ python test_sp_enhance.py -net vgg_mm -weights path_to_vgg16_weights_file
```