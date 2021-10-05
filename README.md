# Adacomp
A Zeroth-order Adaptive Learning Rate Method to Reduce Cost of Hyperparameters Tuning for Deep Learning

This provides the code, data, and experiments for article "A Zeroth-order Adaptive Learning Rate Method to Reduce Cost of Hyperparameters Tuning for Deep Learning", which has been submitted to journal Applied Sciences. The method, named Adacomp, adaptively adjusts the learning rate only based on values of loss function. From high abstract, Adacomp penalizes learning rate when loss value decreses and compensates learning rate in the contrast. 

# Aim
Anyone who is interested in Adacomp can reproduce the experimental results, or makes a further study based on the provided code. 

# Structure
Each folder contains files main_SGD, main_Momentum, main_Adagrad, main_RMSprop, main_Adadelta, main_Adam, main_Adamax, and main_Adacomp (ours).

## 1. Code for MNIST (10 classification, epochs = 10)
(1) This provides code for Figures 1, 2 (robustness to initial learning rate), Figure 3 (robustness to batch size and network initialization), and Table 1 (convergence speed and computational efficiency). The used network architecture is borrowed from https://github.com/pytorch/examples. 

(2) To obtain experimental data in Figures 1, 2, using parallelization method 

parallel --eta --jobs 4 python filename --lr ::: 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0 > log.txt

to reduce time. Or else, run them one by one. Here, --lr denotes learning rate, filename is selected from main_SGD, main_Momentum, main_Adagrad, main_RMSprop, main_Adadelta, main_Adam, main_Adamax, and main_Adacomp.

(3) To obtain experimental data in Figure 3, using 

parallel --eta --jobs 4 python filename --batch-size ::: 16, 32, 64, 128 > log.txt

parallel --eta --jobs 4 python filename --seed ::: 16, 32, 64, 128 > log.txt

to reproduce it. Here, batch-size denotes the size of training mini-batch, and seed denotes the network initialization. 

(4) Explanations about all parameters are referred to details of code.

(5) Experimental data have been rearranged in file "heat_map_mnist.m". One can directly run it to observe experimental results. 
 ## 2. Code for KMNIST (10 classification, epochs = 20)
(1) This is used for part of Table 2 and Figure 6, which show the robustness to learning rate when dataset changes. 

(2) The operation similar to Code for MNIST, except using the dataset KMNIST here.

(3) Experimental data have been rearranged in file "heat_map_mnist.m". One can directly run it to observe experimental results.
## 3. Code for Fashion-MNIST (10 classification, epochs=20)
(1) This is used for part of Table 2 and Figure 6, which show the robustness to learning rate when dataset changes. 

(2) The operation similar to Code for MNIST, except using the dataset KMNIST here.

## 4. Code for CIFAR-10 (10 classification, epochs=100)
(1) This is used for Figures 4, 5, which show the robustness comparison with respect to network architecture. 
In particular, 6 out of 18 network architectures, LeNet, VGG19, ResNet18, MobileNet, SENet18, and SimpleDLA, are borrowed from https://github.com/kuangliu/pytorch-cifar.

(2) To obtain experimental data in Figures 4 and 5, typing 

parallel --eta --jobs 4 python filename --lr ::: 0.005, 0.05, 0.6 > log.txt

(3) Experimental data have been rearranged in file "heat_map_cifar10". One can directly run it to observe experimental results.

## 5. Code for CIFAR-100 (100 classification, epochs=150)
(1) This is used for part of Table 2 and Figure 6, which show the robustness to learning rate when dataset changes.
Meanwhile, this is used to compare the computational efficiency. 
The used network architecture is borrowed from https://github.com/junyuseu/367pytorch-cifar-models.git.

(2) To obtain experimental data in Table 2 and Figure 6, typing 

parallel --eta --jobs 4 python main_Adam.py --lr ::: 0.001 0.01 0.1 1 > log.txt

(3) Experimental data have been rearranged in file "comparison.m". One can directly run it to observe experimental results.
 
