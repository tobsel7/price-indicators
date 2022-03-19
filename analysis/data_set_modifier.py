import numpy as np


# the mean of a feature is usually not 1
# with this function the distribution of a value in a set of samples can be shifted, by removing some samples
def move_dataset_to_mean(samples, mean=1, feature="future_price"):
    # calculate the variables of the normal distribution
    mean = samples[feature].mean()
    std = samples[feature].std()
    # define the factor, such that keep_probability(0) is 1
    factor = np.exp(-(np.power(mean, 2) - 1) / (2 * np.power(std, 2)))
    # define the probability distribution
    keep_probability = lambda x: factor * np.exp((np.power(x - mean, 2) - np.power(x - 1, 2)) / (2 * np.power(std, 2)))
    # create a set of random numbers between 0 and 1
    random_sample = np.random.random_sample(len(samples))
    # define which samples will be kept
    keep = random_sample < keep_probability(samples[feature])
    return samples[keep]
