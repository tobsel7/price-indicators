import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


# this function demonstrates the functionality used in data_set_modifier to change the mean of a normal distribution
# a log normal distribution can be shifted the same way, the log has to be taken before this process
def shift_normal(mean=1.1, new_mean=1, std=0.5):
    x = np.arange(-0, 3, .01)
    f = np.exp((np.power(x - mean, 2) - np.power(x - new_mean, 2)) / (2 * np.power(std, 2)))
    norm = stats.norm.pdf(x, mean, std)
    plt.title("The product of a normal distribution and an exponential function")
    plt.plot(x, norm)
    plt.plot(x, f)
    plt.plot(x, f * norm)
    plt.legend(["initial normal distribution", "exponential function", "shifted normal distribution"])
    plt.show()


# simple function used to plot a distribution of values using a histogram
def plot_distribution(y_values):
    n_bins = min(int(len(y_values) / 10), 100)
    plt.hist(y_values, bins=n_bins)
    plt.tight_layout()
    plt.show()

# show example functions used in the operation
shift_normal(1.2, 1)
