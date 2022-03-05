# imports for sample generation
import random
from data.indicators import indicator_calculator

# set the minimum length of preceding values for the safe calculation of all indicators
MIN_PRECEDING_VALUES = 250
# create samples for the price in 1 year per default
DEFAULT_PREDICATION_INTERVAL = 365


# data class allowing the storage of chart data and random generation
class ChartData:

    # initialize the class with the chart and meta data
    def __init__(self, chart, meta):
        # safe chart data
        self._open = chart["open"]
        self._close = chart["close"]
        self._high = chart["high"]
        self._low = chart["low"]
        self._volume = chart["volume"]

        # safe meta data
        self._name = meta["symbol"]
        self._granularity = meta["dataGranularity"]
        self._type = meta["instrumentType"].lower()
        self._size = len(self._close)

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
    def get_features(self, index, normalized=False):
        # take all relevant feature stored in this file
        if normalized:
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
        indicator_features = indicator_calculator.calculate_all_indicators(self, index, normalized)

        # combine price features and indicator features
        return {**price_features, **indicator_features}

    # generate a random sample by selecting an index and calculating all the features for the position
    def get_random_sample(self, normalized=False, prediction_interval=DEFAULT_PREDICATION_INTERVAL):
        # only allow creation of samples for large enough time series
        if self._size < MIN_PRECEDING_VALUES + prediction_interval:
            raise Exception("Can not create a sample, because the size of the data is too small.")

        # get a random position in the time series respecting the space left for the calculation of the indicators
        position = random.randint(MIN_PRECEDING_VALUES, self._size - prediction_interval - 1)

        # get the future price
        future_price = self.get_close(position + prediction_interval) / self.get_close(position) if normalized \
            else self.get_close(position + prediction_interval)

        # bundle sample in a dictionary
        return {
            "features": self.get_features(position, normalized),
            "predication_interval": prediction_interval,
            "prediction_position": position,
            "future_price": future_price
        }

    # represent the chart data with a string
    def __repr__(self):
        return ("Chart data for the asset {}\n" + "Last price {}\n" + "Number of data points {}")\
            .format(self._name, self._close[-1], self._size)
