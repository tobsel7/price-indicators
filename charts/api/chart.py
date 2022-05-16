# import numpy for sample generation and fast array processing
import numpy as np

# custom error class
from charts.api.errors import PriceDataLengthError

# functions for calculating the indicators
from charts.indicators import indicators

# the default interval by which the price is shifted
from charts.parameters import FUTURE_INTERVAL, TRADING_DAYS_PER_YEAR, SAMPLES_PER_YEAR


# the chart class encapsulates price data for one given stock symbol
class Chart:
    # initialize the class with the chart and metadata
    def __init__(self, chart, meta):
        # safe price data in numpy arrays
        self._open = np.array(chart["open"], dtype=float)
        self._close = np.array(chart["close"], dtype=float)
        self._low = np.array(chart["low"], dtype=float)
        self._high = np.array(chart["high"], dtype=float)
        self._volume = np.array(chart["volume"], dtype=float)

        # some chart data contains 0 values
        # they are replaced with not a number, because they lead to numerical problems
        self._open[self._open == 0] = np.nan
        self._close[self._close == 0] = np.nan
        self._low[self._low == 0] = np.nan
        self._high[self._high == 0] = np.nan
        self._volume[self._volume == 0] = np.nan

        # safe metadata about the chart
        self._name = meta["symbol"]
        self._granularity = meta["dataGranularity"]
        self._type = meta["instrumentType"].lower()

    # represent the chart data as a string
    def __repr__(self):
        return ("Price data for the asset {}\n" + "Last price {}\n" + "Number of data points {}") \
            .format(self._name, self._close[-1], len(self))

    # the length of this chart is defined by the number of closes
    def __len__(self):
        return len(self._close)

    # getters
    def get_name(self):
        return self._name

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
    def can_create_samples(self, future_interval=FUTURE_INTERVAL):
        return len(self) > indicators.MIN_PRECEDING_VALUES + future_interval + 1

    # generate a full dataset including indicators
    def get_full_data(self, normalize=True):
        # only allow creation of samples for large enough time series
        if not self.can_create_samples(0):
            raise PriceDataLengthError()

        # the current closing price and volume are part of the full data
        closes = self.get_closes()
        volumes = self.get_volumes()

        if normalize:
            # find the first valid value and divide all values by this price/volume
            reference_value_closes = closes[np.argmax(closes != np.nan)]
            reference_value_volumes = volumes[np.argmax(volumes != np.nan)]
            # some charts do not have valid closes or volumes, no normalization can be done for nan prices/volumes
            if reference_value_closes != np.nan:
                closes = closes / reference_value_closes
            if reference_value_volumes != np.nan:
                volumes = volumes / reference_value_volumes

        # get all indicators for the dataset
        full_data = indicators.all_indicators(self, normalize)

        # merge and return the columns
        full_data["volume"] = volumes.tolist()
        full_data["current_price"] = closes.tolist()
        return full_data

    # help function shifting the price and volatility columns and adding the shifted columns as new features
    def _add_future_to_data(self, data, future_interval, normalize):
        if future_interval > 0:
            # shift the current price according to the future_price interval to get a future price for the data set
            future_prices = np.append(self.get_closes()[future_interval:], np.zeros(future_interval) + np.nan)
            if normalize:
                # normalize by dividing the future price by the current price according to the price shift
                future_prices[:-future_interval] = future_prices[:-future_interval] / self.get_closes()[:-future_interval]
            # add the future price to the data frame
            data["future_price"] = future_prices.tolist()

            # find all volatility columns in the data frame
            current_volatility_columns = [column for column in data if column.startswith("volatility")]
            future_volatility_columns = ["future_" + column for column in current_volatility_columns]
            # add a future volatility columns by taking the volatility columns and shifting them by the future interval
            data[future_volatility_columns] = data[current_volatility_columns].shift(-future_interval)
        # return the complete data frame
        return data

    # generate a list of samples for analysis/training
    def get_random_samples(self, future_interval=FUTURE_INTERVAL, samples_per_year=SAMPLES_PER_YEAR, normalize=True):
        # only allow creation of samples for large enough time series
        if not self.can_create_samples(future_interval):
            raise PriceDataLengthError()

        # get all data (price data and indicators) for this stock symbol
        full_data = self.get_full_data(normalize=normalize)

        # add columns for shifted features (future values) to the data frame
        full_data = self._add_future_to_data(full_data, future_interval, normalize)

        # select all potential valid samples
        potential_samples = full_data[~full_data.isnull().any(axis=1)]

        # sample using a fraction samples_per_year / trading days per year (253 in the US)
        return potential_samples.sample(frac=samples_per_year / TRADING_DAYS_PER_YEAR)


