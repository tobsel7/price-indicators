# import the data handler
from charts.data_handler import data_handler
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from analysis import math_demos
from charts.indicators import indicator_calculator

DOWNLOAD = False


# main program
def main():
    if DOWNLOAD:
        data_handler.download_and_persist_chart_data(show_downloads=True)
    #math_demos.show_demos()
    #test()
    math_demos.correlation_test("nasdaq", future_price_interval=60)


def test():
    data = data_handler.get_chart_data("AAPL")
    full_data = data.get_full_data(normalize=True)
    full_data.to_csv("test.csv")


def test3():
    sample = data_handler.get_chart_data("AAPL")
    pos = 1300
    interval = 100
    regression = indicator_calculator._regression_lines(sample.get_closes(), pos, interval)
    x = np.arange(pos - interval, pos, 1)
    applx = np.arange(pos - interval, pos, 1)
    y_regr = regression[0] + regression[1] * x
    plt.plot(applx, sample.get_closes()[pos-interval:pos])
    plt.plot(x, y_regr)
    plt.show()
    print(regression[0] + regression[1] * pos)
    print(sample.get_closes()[pos])
    print(indicator_calculator.trend_channel_position(sample.get_closes(), pos, interval, standardize=True))


def test2():
    samples = data_handler.generate_samples(samples_per_year=10, normalize=True, future_price_interval=30)
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


def test1():
    samples = data_handler.generate_samples(samples_per_chart=50, normalize=True, future_price_interval=30)
    samples = samples.sort_values(by='future_price')
    samples = samples[samples.future_price < 2]
    ma_positive = samples[samples.ma50 > 0]
    ma_negative = samples[samples.ma50 <= 0]
    print(np.average(ma_positive["future_price"]))
    print(np.average(ma_negative["future_price"]))
    math_demos.plot_distribution(samples["future_price"])
    for indicator in ["rsi", "ma20", "ma50", "ma100", "ma200", "ma_trend", "ma_trend_crossing50", "horizontal_trend_pos100", "trend_channel_pos100"]:
        correlation = samples[indicator].corr(samples["future_price"] - 1)
        print((indicator + " correlation: {}").format(correlation))


# entry point of the program
if __name__ == '__main__':
    main()



