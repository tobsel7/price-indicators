import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from analysis import data_set_modifier
from charts.data_handler import data_handler


# this function demonstrates the functionality used in data_set_modifier to change the mean of a normal distribution
# a log normal distribution can be shifted the same way, the log has to be taken before this process
def shift_normal(mean=1.1, new_mean=1, std=0.4):
    x = np.arange(0, 3, .01)
    factor = np.exp(-(np.power(mean, 2) - np.power(new_mean, 2)) / (2 * np.power(std, 2)))
    f = factor * np.exp((np.power(x - mean, 2) - np.power(x - new_mean, 2)) / (2 * np.power(std, 2)))
    norm = stats.norm.pdf(x, mean, std)
    plt.title("The product of a normal distribution and an exponential function")
    plt.plot(x, norm)
    plt.plot(x, f)
    plt.plot(x, f * norm)
    plt.legend(["initial normal distribution", "exponential function", "shifted normal distribution"])
    plt.show()


# simple function used to plot a distribution of values using a histogram
def plot_distribution(y_values, xlabel="future price in relation to current price", ylabel="number of occurrences"):
    size = len(y_values)
    n_bins = min(int(size / 10), 100)
    plt.hist(y_values, bins=n_bins)
    plt.axvline(y_values.mean(), color="red", linestyle="dashed")
    plt.tight_layout()
    plt.title("Distribution plot with sample size {}".format(size))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def show_shift_data_set(samples_per_chart=40, normalize=True, prediction_interval=300):
    # plot the distribution before and after shifting it
    samples = data_handler.generate_samples(samples_per_chart=samples_per_chart, normalize=normalize, prediction_interval=prediction_interval)
    samples_mean = np.mean(samples["future_price"])
    samples_std = np.std(samples["future_price"])
    print("original number of samples: {}, average: {}, std: {}".format(len(samples), samples_mean, samples_std))
    plot_distribution(np.log(samples["future_price"]), "future price in relation to current price on logarithmic scale")
    modified = data_set_modifier.move_dataset_to_mean(samples)
    modified_mean = np.mean(np.log(modified["future_price"]))
    modified_std = np.std(np.log(modified["future_price"]))
    print("modified number of samples: {}, average: {}, std: {}".format(len(modified), modified_mean, modified_std))
    plot_distribution(np.log(modified["future_price"]), "future price in relation to current price on logarithmic scale")

def test():
    # TODO create a correct function here
    samples = data_handler.generate_samples(samples_per_chart=100, normalize=True, prediction_interval=30)
    samples = samples.sort_values(by='future_price')
    samples = samples[samples.future_price < 2]
    ma_positive = samples[samples.ma50 > 0]
    ma_negative = samples[samples.ma50 <= 0]
    print(np.average(ma_positive["future_price"]))
    print(np.average(ma_negative["future_price"]))
    plot_distribution(samples["future_price"])
    for indicator in ["rsi", "ma20", "ma50", "ma100", "ma200", "ma_trend", "ma_trend_crossing"]:
        correlation = samples[indicator].corr(np.log(samples['future_price']))
        print((indicator + " correlation: {}").format(correlation))


def show_demos():
    # show example functions used in the operation
    shift_normal(1.2, 1)

    # show how the data set can be shifted
    show_shift_data_set(300, True, 100)


