# Neural Network Distillation

This repo is about neural network knowledge distillation. We experimented with the teacher-student framework, tested the effect of temperature on the distillation performance, reversed distillation, and self distillation.

## Files

`MNIST_CODE` Folder

- The `Model.py` file contains the architectures of student model and teacher model used to generate the results on MNIST dataset. This includes the MLP models.
- The `ece661_final_project.ipynb` notebook contains all the code to explore knowledge on MNIST dataset, including the general knowledge distillation, different temperature and omit one digit.

`CIFAR10_CODE` Folder

- The `Model.py` file contains the architectures of student model and teacher model used to generate the results on CIFAR10 dataset. This includes the ResNet20 and ResNet50 model
- The `resnet20_resnet50.ipynb` notebook contains the code for training ResNet20 and ResNet50 seperately, and tested the test accuracy on CIFAR10 dataset.
- The `self_distillation.ipynb` explores self_distillation on CIFAR10 with ResNet20 and ResNet50.
- The `reverse_training.ipynb` explores reverse distillation on CIFAR10 with ResNet50.

## How to Run

- The models are included in models.py. MNIST folder incluces MLP models, and CIFAR10 folder inclues the ResNet models. To create a instance of these models, simple import the libary from a notebook and call the class.
- The rest of the files are Jupyter Notebooks. to use it, simply open up a python enviroment and run the cells.
