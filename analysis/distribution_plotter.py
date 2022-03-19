import matplotlib.pyplot as plt


def plot_distribution(y_values):
    n_bins = int(len(y_values) / 10)
    plt.hist(y_values, bins=n_bins)
    plt.tight_layout()
    plt.show()
    # We can set the number of bins with the *bins* keyword argument.

