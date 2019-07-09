# Tips and tricks to stabilize GANs 
While research in Generative Adversarial Networks (GANs) continues to improve the
fundamental stability of these models,
we use a bunch of tricks to train them and make them stable day to day.
Here are a summary of some of the tricks.
## 1. Downsample Using Strided Convolutions
The discriminator model is a standard convolutional neural network model that takes an image as input and must output a binary classification as to whether it is real or fake.
It is standard practice with deep convolutional networks to use pooling layers to downsample the input and feature maps with the depth of the network.
This is not recommended for the DCGAN, and instead, they recommend downsampling using  [strided convolutions](https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/).

This involves defining a convolutional layer as per normal, but instead of using the default two-dimensional stride of (1,1) to change it to (2,2). This has the effect of downsampling the input, specifically halving the width and height of the input, resulting in output feature maps with one quarter the area.

## 2. Upsample Using Strided Convolutions

The generator model must generate an output image given as input at a random point from the latent space.
The recommended approach for achieving this is to use a transpose convolutional layer with a strided convolution. This is a special type of layer that performs the convolution operation in reverse. Intuitively, this means that setting a stride of 2×2 will have the opposite effect, upsampling the input instead of downsampling it in the case of a normal convolutional layer.
By stacking a transpose convolutional layer with strided convolutions, the generator model is able to scale a given input to the desired output dimensions.

## 3. Use LeakyReLU

The  [rectified linear activation unit](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/), or ReLU for short, is a simple calculation that returns the value provided as input directly, or the value 0.0 if the input is 0.0 or less.
It has become a best practice when developing deep convolutional neural networks generally.
The best practice for GANs is to use a variation of the ReLU that allows some values less than zero and learns where the cut-off should be in each node. This is called the leaky rectified linear activation unit, or LeakyReLU for short.
A negative slope can be specified for the LeakyReLU and the default value of 0.2 is recommended.
Originally, ReLU was recommend for use in the generator model and LeakyReLU was recommended for use in the discriminator model, although more recently, the LeakyReLU is recommended in both models.

## 4. Use Batch Normalization

[Batch normalization](https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/)  standardizes the activations from a prior layer to have a zero mean and unit variance. This has the effect of stabilizing the training process.

Batch normalization is used after the activation of convolution and transpose convolutional layers in the discriminator and generator models respectively.
It is added to the model after the hidden layer, but before the activation, such as LeakyReLU.
When batchnorm is not an option use instance normalization (for each sample, subtract mean and divide by standard deviation).

## 5. Use Gaussian Weight Initialization

Before a neural network can be trained, the model weights (parameters) must be initialized to small random variables.

The best practice for DCAGAN models reported in the paper is to initialize all weights using a zero-centered Gaussian distribution (the normal or bell-shaped distribution) with a standard deviation of 0.02.
- Tom White's [Sampling Generative Networks](https://arxiv.org/abs/1609.04468) ref code https://github.com/dribnet/plat has more details


## 6. Use Adam Stochastic Gradient Descent

Stochastic gradient descent, or SGD for short, is the standard algorithm used to optimize the weights of convolutional neural network models.

There are many variants of the training algorithm. The best practice for training DCGAN models is to use the  [Adam version of stochastic gradient descent](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)  with the learning rate of 0.0002 and the beta1 momentum value of 0.5 instead of the default of 0.9.

The Adam optimization algorithm with this configuration is recommended when both optimizing the discriminator and generator models.


## 7. Scale Images to the Range [-1,1]

It is recommended to use the hyperbolic tangent activation function as the output from the generator model.

As such, it is also recommended that real images used to train the discriminator are  [scaled so that their pixel values](https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/)  are in the range [-1,1]. This is so that the discriminator will always receive images as input, real and fake, that have pixel values in the same range.

## 8. A modified loss function

In GAN papers, the loss function to optimize G is `min (log 1-D)`, but in practice folks practically use `max log D`
  - because the first formulation has vanishing gradients early on
  - Goodfellow et. al (2014)

In practice, works well:
  - Flip labels when training generator: real = fake, fake = real


## 9.Use Soft and Noisy Labels

- Label Smoothing, i.e. if you have two target labels: Real=1 and Fake=0, then for each incoming sample, if it is real, then replace the label with a random number between 0.7 and 1.2, and if it is a fake sample, replace it with 0.0 and 0.3 (for example).
  - Salimans et. al. 2016
- make the labels the noisy for the discriminator: occasionally flip the labels when training the discriminator


## 10. Separate Batches of Real and Fake Images

The discriminator model is trained using stochastic gradient descent with mini-batches.
The best practice is to update the discriminator with separate batches of real and fake images rather than combining real and fake images into a single batch.
It is common to use the class label 1 to represent real images and class label 0 to represent fake images when training the discriminator model.
These are called hard labels, as the label values are precise or crisp.
It is a good practice to use soft labels, such as values slightly more or less than 1.0 or slightly more than 0.0 for real and fake images respectively, where the variation for each image is random.

This is often referred to as label smoothing and can have a  [regularizing effect](https://machinelearningmastery.com/introduction-to-regularization-to-reduce-overfitting-and-improve-generalization-error/)  when training the model.

The labels used when training the discriminator model are always correct.
This means that fake images are always labeled with class 0 and real images are always labeled with class 1.
It is recommended to introduce some errors to these labels where some fake images are marked as real, and some real images are marked as fake.

If you are using separate batches to update the discriminator for real and fake images, this may mean randomly adding some fake images to the batch of real images, or randomly adding some real images to the batch of fake images.
## 11. Use stability tricks from RL

- Experience Replay
  - Keep a replay buffer of past generations and occassionally show them
  - Keep checkpoints from the past of G and D and occassionaly swap them out for a few iterations
- All stability tricks that work for deep deterministic policy gradients
- See Pfau & Vinyals (2016)

## 12. Track failures early

- D loss goes to 0: failure mode
- check norms of gradients: if they are over 100 things are screwing up
- when things are working, D loss has low variance and goes down over time vs having huge variance and spiking
- if loss of generator steadily decreases, then it's fooling D with garbage (says martin)

## 13. Dont balance loss via statistics (unless you have a good reason to)

- Dont try to find a (number of G / number of D) schedule to uncollapse training
- It's hard and we've all tried it.
- If you do try it, have a principled approach to it, rather than intuition

For example
```
while lossD > A:
  train D
while lossG > B:
  train G
```
## 14: If you have labels, use them

- if you have labels available, training the discriminator to also classify the samples: auxillary GANs

## 15: Add noise to inputs, decay over time

- Add some artificial noise to inputs to D (Arjovsky et. al., Huszar, 2016)
  - http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
  - https://openreview.net/forum?id=Hk4_qw5xe
- adding gaussian noise to every layer of generator (Zhao et. al. EBGAN)
  - Improved GANs: OpenAI code also has it (commented out)

## 16: [notsure] Train discriminator more (sometimes)

- especially when you have noise
- hard to find a schedule of number of D iterations vs G iterations

## 17: [notsure] Batch Discrimination

- Mixed results

## 18: Discrete variables in Conditional GANs

- Use an Embedding layer
- Add as additional channels to images
- Keep embedding dimensionality low and upsample to match image channel size

## 19: Use Dropouts in G in both train and test phase
- Provide noise in the form of dropout (50%).
- Apply on several layers of our generator at both training and test time
- https://arxiv.org/pdf/1611.07004v1.pdf

