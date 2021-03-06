import types

import numpy as np


# the mean of a feature is usually not 1
# with this function the distribution of a value in a set of samples can be shifted, by removing some samples
def move_dataset_to_mean(samples, desired_mean=1, feature="future_price", transform=lambda x: x, start_of_deletions=np.min):
    # we want to shift a log normal distribution
    # to simplify this process we take the logarithm and treat the distribution as a normal distribution from now on
    y = transform(samples[feature])

    # calculate the characteristic values of the normal distribution
    mean = np.mean(y)
    std = np.std(y)

    # define a "start" of the exponential function by setting the point where the keep probability should be = 1
    start = start_of_deletions(y) if isinstance(start_of_deletions, types.LambdaType) else start_of_deletions

    # define the probability distribution which is used to shift the sample distribution
    keep_probability = lambda x: np.exp(
        (np.power(x - start - mean, 2) - np.power(x - start - desired_mean, 2)) / (2 * np.power(std, 2))
    )

    # create a set of random numbers between 0 and 1
    random_sample = np.random.random_sample(len(samples))
    # define which samples will be kept
    keep = random_sample < keep_probability(y)

    return samples[keep]
