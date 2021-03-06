# Generative Adversarial Networks Bundle

Generative Adversarial Networks are a pair of models which behave as adversaries to each other in order to compete and learn from experience to generate artificial data. This data could be anything; audio, text, images or numerical data.

This module implements GANs using the [Generalized Neural Network bundle](https://github.com/hpcc-systems/GNN). It can be used in various ways according to how the input is given and the output desired. Typically, the program flow goes as follows: -
1) Read the dataset and convert into appropriate tensor using GNN.Tensor functions.
2) Define the Generator and Discriminator model obeying the rules of GNN interface.
3) Call GAN.train() with the parameters defined below to train the GANs for set number of Epochs and given batchSize
4) Use returned generator to predict using GNN functions and returned discriminator to distinguish fake and real data
5) Output the predicted values as required for understanding

Refer to Test/GANtest.ecl for a better understanding of the working. 

## How to use

Assuming that HPCC cluster is up and running in your computer: -
1) Install ML Core as an ECL bundle by running the below in your terminal or command prompt.

        ecl bundle install https://github.com/hpcc-systems/ML_Core.git

2) Install Generalised Neural Networks bundle by running the below in your terminal or command prompt.
        
        ecl bundle install https://github.com/hpcc-systems/GNN.git


3) To make sure and also install the required python3 dependencies, please run the Setup.ecl file by running the below command.
        
        ecl run thor Setup.ecl

4) Now that the dependencies have been taken care of, install the current bundle by cloning this repository and running the below command

        ecl bundle install GAN

5) You run it by executing the following command if HPCC systems platform is running on server: -

        ecl run thor <filename>

This should enable you to use the GAN train function given the dataset appropriately.

## Helpful Test files

1. **GANtest.ecl**

    This test file tries to generate MNIST dataset using mostly Dense layers. This is the most primitive kind of GAN and it shows that the GAN works. The main working file is this. 
    End of this file also has a statement which enables you to run predict as many times as required separately, without requiring you to run the whole program again. 

2. **PredictTest.ecl**

    This test file is used to predict the saved generator model with different noise types as many times as required. GANtest.ecl and DCGANtest.ecl save their models in a logical file to be able to use for predictions after. 

3. **DCGANtest.ecl**

    This test file is the exact same as simpleGANtest.ecl, but has different layers. They are mostly convolutional neural networks and provide a better output than simple GAN. This shows that the only difference for GANs is the layers and the train function need not be changed at all for different kinds of GANs.

4. **SaveTest.ecl**

    This test file is a sort of reference for how models are being saved. For any further type of GAN developed, SaveTest.ecl may be referred to see how the model is saved easily. 


## Outputs of various models tested

### Simple GAN output

#### Epoch 2000

![Coming soon](Images/GAN/2000/Epoch_2000.png)

#### Epoch 3000

![Coming soon](Images/GAN/3000/Epoch_3000.png)

#### Epoch 4000

![Coming soon](Images/GAN/4000/Epoch_4000.png)


### Deep Convolution GAN output

#### Epoch 2000

![Coming soon](Images/DCGAN/2000/Epoch_2000.png)

#### Epoch 3000

![Coming soon](Images/DCGAN/3000/Epoch_3000.png)

#### Epoch 4000

![Coming soon](Images/DCGAN/4000/Epoch_4000.png)
