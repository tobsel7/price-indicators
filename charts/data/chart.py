# imports for sample generation and fast array processing
import numpy as np

# functions for calculating the indicators
from charts.indicators import indicators

# create samples for the price in 1 year per default
DEFAULT_PREDICATION_INTERVAL = 365


# the chart class encapsulates price data for one given stock symbol
# numpy is used primarily because of numerical performance properties
# resulting samples are pandas dataframes
class Chart:
    # initialize the class with the chart and meta charts
    def __init__(self, chart, meta):
        # safe price data in numpy arrays
        self._open = np.array(chart["open"])
        self._close = np.array(chart["close"])
        self._high = np.array(chart["high"])
        self._low = np.array(chart["low"])
        self._volume = np.array(chart["volume"])

        # safe metadata about the chart
        self._name = meta["symbol"]
        self._granularity = meta["dataGranularity"]
        self._type = meta["instrumentType"].lower()

    # represent the chart charts with a string
    def __repr__(self):
        return ("Chart data for the asset {}\n" + "Last price {}\n" + "Number of data points {}") \
            .format(self._name, self._close[-1], len(self))

    # the length of this chart data is defined by the number of closes
    def __len__(self):
        return len(self._close)

    # getters
    def get_opens(self):
        return self._open

    def get_closes(self):
        return self._close

    def get_lows(self):
        return self._low

    def get_highs(self):
        return self._high

    def get_volumes(self):
        return self._volume

    # check whether valid samples can be created
    def can_create_samples(self, prediction_interval=DEFAULT_PREDICATION_INTERVAL):
        return len(self) > indicators.MIN_PRECEDING_VALUES + prediction_interval

    # generate a full dataset including indicators
    def get_full_data(self, normalize=True):
        # only allow creation of samples for large enough time series
        if not self.can_create_samples(0):
            raise Exception("Can not create a sample, because the size of the charts is too small.")

        # the current closing price is part of the full data
        closes = self.get_closes()
        if normalize:
            # find the first nonzero value and divide all values by this nonzero value
            closes = closes / closes[np.argmax(closes != 0)]

        # the daily volume is also included
        volumes = self.get_volumes()

        # get all indicators for the dataset
        full_data = indicators.all_indicators(self, normalize)

        # merge and return the columns
        full_data["volume"] = volumes.tolist()
        full_data["current_price"] = closes.tolist()
        return full_data

    # generate a list of samples for analysis/training
    def get_random_samples(self, future_price_interval=DEFAULT_PREDICATION_INTERVAL, samples_per_year=10,
                           normalize=True):
        # only allow creation of samples for large enough time series
        if not self.can_create_samples(future_price_interval):
            raise Exception("Can not create a sample, because the size of the charts is too small.")

        # get all data (price data and indicators) for this stock symbol
        full_data = self.get_full_data(normalize=normalize)

        # shift the current price according to the future_price interval to get a future price for the data set
        future_prices = np.zeros(len(self)) + np.nan
        future_prices[:-future_price_interval] = np.divide(self.get_closes()[future_price_interval:],
                                                           self.get_closes()[:-future_price_interval],
                                                           out=future_prices[:-future_price_interval],
                                                           where=self.get_closes()[:-future_price_interval] != 0)

        full_data["future_price"] = future_prices.tolist()

        # randomly select a few rows
        number_of_samples = int(len(self) / 365.0 * samples_per_year)
        choices = np.random.randint(indicators.MIN_PRECEDING_VALUES,
                                    len(self) - future_price_interval - 1,
                                    size=number_of_samples
                                    )

        # return the selected samples
        return full_data.iloc[choices]

