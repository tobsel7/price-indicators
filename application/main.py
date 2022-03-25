# import the charts handler
import random

from charts.data_handler import data_handler
import numpy as np
from analysis import analysis, data_set_modifier
import matplotlib.pyplot as plt
import scipy.stats as stats
from analysis import math_demos
DOWNLOAD = True


# main program
def main():
    if DOWNLOAD:
        data_handler.download_and_persist_chart_data()


def test3():
    samples = data_handler.generate_samples(samples_per_chart=10, normalize=True, prediction_interval=30)
    samples = samples.sort_values(by='future_price')
    mean = np.log(samples["future_price"]).mean()
    std = np.log(samples["future_price"]).std()
    x = np.arange(0.1, 3, .01)
    f = np.exp((np.power(np.log(x) - mean, 2) - np.power(np.log(x), 2)) / (2 * np.power(std, 2)))
    factor = np.exp(-(np.power(mean, 2)) / (2 * np.power(std, 2)))
    norm = stats.norm.pdf(x, mean, std)
    print((f * factor * norm).mean())
    plt.plot(x, norm)
    plt.plot(x, f * factor)
    plt.plot(x, f * factor * norm)
    plt.ylabel('gaussian distribution')
    plt.show()


def test2():
    #analysis.analyze_indicator_correlation(100, 10, 365, 30)
    #test2()
    samples = data_handler.generate_samples(samples_per_chart=10, normalize=True, prediction_interval=365)
    #function_fitter.fit_normal_distribution(samples["future_price"], plot=True)
    #print(len(samples))
    #print(samples.mean())
    print(np.mean(np.log(samples["future_price"])))
    math_demos.plot_distribution(samples["future_price"])
    modified = data_set_modifier.move_dataset_to_mean(samples, 1)
    print(np.mean(np.log(modified["future_price"])))
    math_demos.plot_distribution(modified["future_price"])

# current test code
def test():
    samples = data_handler.generate_samples(samples_per_chart=100, normalize=True, prediction_interval=30)
    samples = samples.sort_values(by='future_price')
    samples = samples[samples.future_price < 2]
    ma_positive = samples[samples.ma50 > 0]
    ma_negative = samples[samples.ma50 <= 0]
    print(np.average(ma_positive["future_price"]))
    print(np.average(ma_negative["future_price"]))
    math_demos.plot_distribution(samples["future_price"])
    for indicator in ["rsi", "ma20", "ma50", "ma100", "ma200", "ma_trend", "ma_trend_crossing"]:
        correlation = samples[indicator].corr(samples['future_price'] - 1)
        print((indicator + " correlation: {}").format(correlation))


# entry point of the program
if __name__ == '__main__':
    main()
