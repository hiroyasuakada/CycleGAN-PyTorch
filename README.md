# CycleGAN-PyTorch

A simplest PyTorch implementation of CycleGAN. 


# Requirements

Tested on ...

- Linux environment
- One or more NVIDIA GPUs
- NVIDIA drivers, CUDA 9.0 toolkit and cuDNN 7.5
- Python 3.6, PyTorch 1.1
- For docker user, please use the [provided Dockerfile](https://github.com/hiroyasuakada/CycleGAN-PyTorch/blob/main/docker_ITC/dockerfile). (highly recommended)

# Usage
## ① Train CycleGAN

### 1. Download this repository

        git clone https://github.com/hiroyasuakada/CycleGAN-PyTorch.git
        cd CycleGAN-PyTorch

### 2. Prepare dataset

        mkdir dataset
        cd dataset

Please put your dataset in `dataset` folder or please download public datasets from [here](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/).

| Example of folder relation | &nbsp;
| :--- | :----------
| dataset
| &boxur;&nbsp; hourse2zebra
| &ensp;&ensp; &boxur;&nbsp;  trainA | image domain A for training
| &ensp;&ensp; &boxur;&nbsp;  trainB | image domain B for training
| &ensp;&ensp; &boxur;&nbsp;  testA | image domain A for testing
| &ensp;&ensp; &boxur;&nbsp;  testB | image domain B for testing

and then move back to `CycleGAN-PyTorch` folder by `cd ..` command.


### 3. Train the model

        python train.py [DATASET NAME]
        
        # for example
        python train.py horse2zebra
        
This will create `logs` folder in which training details and generated images at each epoch during the training will be saved. 
        
If you have multiple GPUs, 

        python train.py horse2zebra --gpu_ids 0 1 --batch_size 4 

If you want to resume training from a certain epoch, (for example, epoch 25)

        python train.py house2zebra --load_epoch 25
        
For more information about training setting, please run `python train.py --help`.



### 3. Test the model

        python test.py [DATASET NAME] [LOAD EPOCH]
        
        # for example, if you want to test your model at epoch 25
        python test.py horse2zebra 25
        
This will create `generated_imgs` folder in which you can find generated images from your model.

For more information about testing setting, please run `python test.py --help`.
        
## Reference

- **pytorch-CycleGAN-and-pix2pix[official]**: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
