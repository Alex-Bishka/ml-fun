# SAE Experiments

## Overview

We are having difficulty encoding our activation vectors into sparse vectors that make any sense. It seems that an untouched NN is learning, seemingly, random patterns from the handwritten digits. Additionally, the patterns the NN is picking up on might be artifacts (e.g. color of pixel - see inversion tests). In hindsight, this might have been expected. Still, I was hoping that we'd be able to discover something useful, or be able to navigate to some useful subfeature. Essentially, the hope was that a "default" model would be able to help us get useful information from the hidden activation space.

Our saliency maps rarely show human interpretablitity alignment; the input pixels that most affect the outcome of the sparse vectors appear to be random, and barely in line with human interpretable (sub-)features. Feature attribution techniques on the sparse features are not as clear as hoped.

Moreover, when constructing a maximally sparse vector (e.g. a vector encoded s.t. only one of the sparse features is active), the input image tends to be random noise rather than a specific human interpretable sub feature of the digit. So, feature inversion appears to be a difficult task.

Also, polysemanticity appears rampant, as many nodes activate maximally for different types of digits. It probably doesn't help that the majority of our network suffers from dead nodes. Overall, extracting interpretable features from the current attempts appears challenging, especially in an automatic fashion.

## Future Directions

### Direction 1: Improve the Feature Space the SAE Learns

Currently, we are training the SAE on raw hidden activations. As mentioned above, these tend to be entangled and our SAE struggles in this setting. We might need to disentangle in some fashion first.

Some options:
 1. Train the SAE on PCA or ICA basis instead (perhaps the SAE will learn better and more clear sparse basis vectors)
 2. Use a structured bottleneck (use sparsity during training) - or what if we **penalize garbage pixels**?
 3. Inject locality or Edge Priors (edge detectors of convolutions before applying the SAE - hope is we recognize lines and arcs)
 4. Contrastive Learning on Activations before SAE (forces digits to cluster and SAE will liklely learn less junk codes)
 5. Flip the script - define interpretible features first and then suprvise hidden activations to match them, and see if the SAE learns a clearer basis
     - Could we even use SAEs during training to see if we are learning in the "right" direction?
     - Potentially a defilbrilator to our gradients/hidden activation vectors?

### Direction 2: Explore Interpretability Techniques to Understand SAE Outputs

We might actually be learning something meaningful, but polysemanticity is high and visualization is hard.

Some options:
 1. Train a Linear Probe or SVM on Sparse Features (will tell us if our sparse codes are actually predicitive)
    - **Update:** SAEs seem to be pretty solid at picking up information from hidden space to identify digit classes - we need disentanglement or some other work on the feature space. Adjusting the `HIDDEN_SIZE` and sparsity penalty do yield improvements in linear probe performance; however the feature attribution overlays still seem to largerly be affected by "junk" pixels.
 2. Cluster Sparse Codes Across Digits (perhaps average sparse codes by image serve as good candidates for auxilary loss targets)

### Thoughts

I really think idea 2.1 is strong for our experiments, as it can verify the SAE's learnings. Additionally, I really like 1.5, because it could serve as an automated way of deciding whether or not an "arbitrary" set of human interpretable sub features **IS ACTUALLY USEFUL**. These might be the first two routes I take, but particularly 2.1 for validation on current (and future) experiments.

Another potential route to explore: feature attribution with gradient clipping (**in our case makes little difference** - gradients are not exploding) and feature inversion with regualrization (**also not helping our current case**).