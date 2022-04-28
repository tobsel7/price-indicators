# import the chart handler
from charts.api import data_handler
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from analysis import demos
from charts.indicators import formulas
from charts.indicators import utilities
from charts.data_sets import file_handler


# main program
def main():
    #data_handler.download_and_persist_chart_data(show_downloads=True)
    #math_demos.show_demos()
    data = data_handler.get_chart_data("JNJ")
    file_handler.data_set_to_feather(data.get_full_data(), "JNJ_full")
    #samples = data_handler.generate_samples(asset_list="nasdaq", samples_per_year=1, normalize=True, future_price_interval=100)
    #file_handler.data_set_to_feather(samples.reset_index(), "nasdaq_samples_normalized")
    demos.correlation_test("nasdaq", future_price_interval=20)


def generate_csv():
    symbol = input("Enter the stock ticker:\n")
    data = data_handler.get_chart_data(symbol)
    file_handler.chart_to_sheet(data)


def test3():
    sample = data_handler.get_chart_data("AAPL")
    pos = 800
    interval = 100
    initial_value, slope = utilities.regression_lines(sample.get_closes(), interval)
    print(initial_value)
    print(slope)
    x = np.arange(pos - interval, pos, 1)
    y_regr = initial_value[pos-interval] + slope[pos-interval] * x
    plt.plot(x, sample.get_closes()[pos-interval:pos])
    plt.plot(x, y_regr)
    plt.show()


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
    demos.plot_distribution(samples["future_price"])
    for indicator in ["rsi", "ma20", "ma50", "ma100", "ma200", "ma_trend", "ma_trend_crossing50", "horizontal_trend_pos100", "trend_channel_pos100"]:
        correlation = samples[indicator].corr(samples["future_price"] - 1)
        print((indicator + " correlation: {}").format(correlation))


# entry point of the program
if __name__ == '__main__':
    main()



