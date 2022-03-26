import numpy as np

from charts.data_handler import data_handler
import matplotlib.pyplot as plt


def _plot_correlations(intervals, correlations):
    number_of_indicators = len(correlations)
    fig, axs = plt.subplots(number_of_indicators // 2 + number_of_indicators % 2, 2)
    fig.text(0.5, 0.04, "prediction interval", ha="center")
    fig.text(0.04, 0.5, "correlation", va="center", rotation="vertical")
    for index, indicator in enumerate(correlations):
        axs[index // 2][index % 2].title.set_text(indicator)
        axs[index // 2][index % 2].plot(intervals, correlations[indicator])
    plt.show()


def analyze_indicator_correlation(samples_per_chart=10, prediction_interval_min=1, prediction_interval_max=365, prediction_interval_step_size=20):
    intervals = np.arange(prediction_interval_min, prediction_interval_max, prediction_interval_step_size)
    correlations = dict.fromkeys(data_handler.generate_sample("AAPL")["features"], np.array([]))
    correlations.pop("open")
    correlations.pop("close")
    correlations.pop("high")
    correlations.pop("low")
    for interval in intervals:
        samples = data_handler.generate_samples(samples_per_chart=samples_per_chart, normalize=True, prediction_interval=interval)
        for feature in correlations.keys():
            correlation = samples[feature].corr(samples['future_price'] - 1)
            correlations[feature] = np.append(correlations[feature], correlation)
    _plot_correlations(intervals, correlations)



