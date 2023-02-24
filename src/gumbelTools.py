import numpy as np

def truncated_gumbel(probability, truncation, rng):

    gumbel = rng.gumbel() + np.log(probability)
    return -np.log(np.exp(-gumbel) + np.exp(-truncation))

def topdown(probabilities, k, rng):

    topgumbel = rng.gumbel()  # + np.log(sum(probabilities)) -> always zero
    gumbels = []
    for i in range(len(probabilities)):
        if i == k:
            gumbel = topgumbel - np.log(probabilities[i])
        elif probabilities[i]!=0:
            gumbel = truncated_gumbel(probabilities[i], topgumbel, rng) - np.log(probabilities[i])
        else:
            gumbel = rng.gumbel() # When the probability is zero, sample an unconstrained Gumbel
        gumbels.append(gumbel)

    return gumbels

def _sample_gumbels(probabilities, actual, num_samples, rng):

    gumbels = []
    for _ in range(num_samples):
        gumbels.append(topdown(probabilities, actual, rng))

    # Sanity check
    for gum in gumbels:
        temp = gum + np.log(probabilities)
        assert np.argmax(temp)==actual, "Sampled gumbels don't match with realized argmax"
    
    return gumbels