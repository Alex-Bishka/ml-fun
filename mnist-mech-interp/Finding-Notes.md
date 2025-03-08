# Overview

## NNs

- we can make NNs more interpretable at intermediate layers (see deep supervision)
  - use auxilary loss at intermediate layers to guide them to more human interpretable features
  - this, ideally, makes the NN more interpretable to us, and makes it act in a more intuitive manner 
- we have two tests for deep supervision:
  1. intermediate layers learning edge detection (horizontal/vertical edges of digits as subfeatures)
  2. hierarchal intermediate layers learning key subfeatures in each corner of an image divided in fourth, and then key subfeatures in the top/bottom half (the next layer)
- we have some evidence that these features are better than random
  - this is mixed and it seems to depend on the original seed used to create the noisy subfeatures (can hit better subfeature learning via noise)
- we have some evidence that learning the human subfeatures makes the model more human like
  - see our final summed activations (the subfeature influenced networks all seem to be less confident on classifying random noise images as digits)
- performance from our deep supervision has minor-to-negligible performance loss to the identical network that does not utilize the auxilary loss from deep supervision

While not deep supervision:
- we have shown that an NN can solve for both the local minima and constraints place on intermediate features

What we need to do:
- better visuals for our 4->2->classify architecture
- more exploration of random subfeature starting states
  - getting a good sample here could be interesting for determining the importance of our network
- compress this work s.t. that the network has a reasonable intermediate size
  - the 784 layer "nodes" are great for visualizations, but not practical... at all!
  - can use the 16 nodes layers (3blue1brown's design) and utilize the 16 nodes to encode subfeatures at each layer
- explore alternative subfeatures
  - does it matter how we combine our fundamental subfeatures in intermedate layers
  - also just more layers to our hierarchy... a good simple next jump could be 16->4->2->classify
- explore what point of subfeatures is more important
  - i.e. if we blot out/leave some layers alone, which layers would be more beneficial to affect?
  - do we need every layer to have subfeatures/be interpretable?

## CNNs

- we have shown that we can influence CNNs to learn intermediate features at the convolutional layers
  - and this minorly affects performance

we need to go further here and replicate a lot of the NN work