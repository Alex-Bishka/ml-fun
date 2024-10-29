# ml-fun

## MNIST Mech Interp

### Idea

What goes on in the CNN on the hand-written digit classification problem, does not match our intuitions.

3Blue1Brown exemplifies this in his neural network series. Basically from his videos, we expect each layer of the network to be able to break down digits into smaller pieces, and the neurons that fire together should resemble the digit that is classfied.

However, as 3Blue1Brown depicts, the activations look like nonsense. There seems to be nothing human interpretable in the layers of the neural network classifying the digits. 

Given, the research in Towards Monosemanticity, perhaps this makes sense. Features do not really correspond to neurons, although this might occur by sheer dumb luck. So, it seems likely that the neural network constructed in 3Blue1Brown's video series, suffers from polysemanticity.

So, what if we could reduce said polysemanticity? What if the solution is similar to what happens in Towards Monosemanticity? What if a sparse autoencoder trained on the activations of the CNN provides us a dictionary learning model that can extract feature that are significantly more monosemantic than neurons?

Now, I think there are some intermediate steps before jumping to this. Like, potential architecture changes to promote monosemanticity, as this not a transformer, so our loss function might be adaptable to incentivize monosemanticity.

Also, there's just some manual investigation to be done.

However, given that Towards Monosemanticity shows that neurons are not likely to represent features in a monosemantic fashion, the dictionary learning problem feels more appealing to me. This also feels like it could scale better since you are not changing the architecture of performant models.

### Similar Research?

Seems interpretable CNN's have been built with SHAP and LIME. I'll need to do further research on both of these.

Geeks for Geeks actually has a [tutorial](https://www.geeksforgeeks.org/sparse-autoencoders-in-deep-learning/) on it

This paper might also have done it: [Lightweighted Sparse Autoencoder based on Explainable Contribution](https://openreview.net/pdf/3c579559b740a47e5da49b7f0001580426943611.pdf). Again SHAP appears, but this time in tandem with the sparse autoencoder (SHAP-SAE). They don't seem to use it as a dictionary learning problem though, but rather a way to remove unit/links of low importance, which seems to make compute more efficient.

Might also want to look into VQ-VAEs.