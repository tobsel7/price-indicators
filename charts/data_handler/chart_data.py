# imports for sample generation and faster array processing
import numpy as np
import pandas as pd
# functions for calculating the indicators
from charts.indicators import indicator_calculator

# set the minimum length of preceding values for the safe calculation of all indicators
MIN_PRECEDING_VALUES = 250
# create samples for the price in 1 year per default
DEFAULT_PREDICATION_INTERVAL = 365


# charts class allowing the storage of charts and random generation of samples
class ChartData:

    # initialize the class with the chart and meta charts
    def __init__(self, chart, meta):
        # safe chart charts
        self._open = np.array(chart["open"])
        self._close = np.array(chart["close"])
        self._high = np.array(chart["high"])
        self._low = np.array(chart["low"])
        self._volume = np.array(chart["volume"])

        # safe meta charts
        self._name = meta["symbol"]
        self._granularity = meta["dataGranularity"]
        self._type = meta["instrumentType"].lower()
        self._size = len(self._open)

    # represent the chart charts with a string
    def __repr__(self):
        return ("Chart charts for the asset {}\n" + "Last price {}\n" + "Number of data points {}") \
            .format(self._name, self._close[-1], self._size)

    # the length of this chart data is defined by the number of opens (equivalent to days)
    def __len__(self):
        return self._size

    # getters
    def get_opens(self):
        return self._open

    def get_open(self, index):
        return self._open[index]

    def get_closes(self):
        return self._close

    def get_close(self, index):
        return self._close[index]

    def get_lows(self):
        return self._low

    def get_low(self, index):
        return self._low[index]

    def get_highs(self):
        return self._high

    def get_high(self, index):
        return self._high[index]

    def get_volumes(self):
        return self._volume

    def get_volume(self, index):
        return self._volume[index]

    # calculate all the features for a certain position in the time series
    def get_features(self, index, normalize=True):
        # take all relevant feature stored in this file
        if normalize:
            # normalize the price features
            close = self.get_close(index)
            price_features = {
                "open": self.get_open(index) / close,
                "close": 1,
                "low": self.get_low(index) / close,
                "high": self.get_high(index) / close,
                "volume": 0 if close == 0 else self.get_volume(index) / close,
            }
        else:
            price_features = {
                "open": self.get_open(index),
                "close": self.get_close(index),
                "low": self.get_low(index),
                "high": self.get_high(index),
                "volume": self.get_volume(index),
            }

        # calculate all the indicators
        indicator_features = indicator_calculator.calculate_all_indicators(self, index, normalize)

        # combine price features and indicator features
        return {**price_features, **indicator_features}

    # check whether valid samples can be created
    def can_create_samples(self, prediction_interval=DEFAULT_PREDICATION_INTERVAL):
        return self._size > MIN_PRECEDING_VALUES + prediction_interval

    def get_random_samples(self, normalize=True, prediction_interval=DEFAULT_PREDICATION_INTERVAL, samples_per_year=10):
        # only allow creation of samples for large enough time series
        if not self.can_create_samples(prediction_interval):
            raise Exception("Can not create a sample, because the size of the charts is too small.")

        # get chart features which are part of the samples
        volumes = np.array(self.get_volumes())
        # shift the future_price data and append nan values
        future_prices = np.array(self.get_closes()[prediction_interval:])
        future_prices = np.append(future_prices, np.zeros(prediction_interval) + np.nan)

        # merge the all the columns
        samples = pd.DataFrame({"volume": volumes, "future_price": future_prices})
        indicators = indicator_calculator.calculate_all_indicators(self, normalize)
        samples = pd.concat([samples, indicators], axis=1)
        #print(samples)
        # select a few rows and return them
        number_of_samples = int(len(self) / 365.0 * samples_per_year)
        choices = np.random.randint(MIN_PRECEDING_VALUES, self._size - prediction_interval - 1, size=number_of_samples)
        #print(choices)
        #print(samples.iloc[choices])
        return samples.iloc[choices]

    # generate a random sample by selecting an index and calculating all the features for the position
    def get_random_sample(self, normalize=True, prediction_interval=DEFAULT_PREDICATION_INTERVAL):
        # only allow creation of samples for large enough time series
        if not self.can_create_samples(prediction_interval):
            raise Exception("Can not create a sample, because the size of the charts is too small.")

        # get a random position in the time series respecting the space left for the calculation of the indicators
        position = np.random.randint(MIN_PRECEDING_VALUES, self._size - prediction_interval - 1)

        # get the future price
        future_price = self.get_close(position + prediction_interval) / self.get_close(position) if normalize \
            else self.get_close(position + prediction_interval)

        # bundle sample in a dictionary
        return {
            "features": self.get_features(position, normalize),
            "predication_interval": prediction_interval,
            "prediction_position": position,
            "future_price": future_price
        }

