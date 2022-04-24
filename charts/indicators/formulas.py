import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from . import utilities


# internal function for all weighted moving averages
def _moving_average(closes, window, standardize=True):
    # get the window length
    window_size = len(window)
    # moving average is the convolution of a rectangular window with the closing prices
    ma = np.convolve(closes[:-1], window, mode="valid")

    # before moving averages go to zero, they are set to the current closing price
    # as a result, the indicator will be neutral (=1)
    ma = np.where(ma == 0, closes[len(window):], ma)

    # clip the values to a range between 0 and 2 before standardizing
    # TODO Deal with eventual 0 closing prices
    #ma = utilities.standardize_indicator(np.clip(ma / closes[window_size:], 0, 2), 0, 2) if standardize else ma
    # the ma is not well-defined before enough price data exists
    return np.append(np.zeros(window_size) + np.nan, ma)


# the mean price over a defined interval
def standard_moving_average(closes, interval=50, standardize=True):
    # the window function is a rectangular, this represents the equal weighted sum
    window = np.ones(interval, dtype=float) / interval
    return _moving_average(closes, window, standardize=standardize)


def weighted_moving_average(closes, interval=50, standardize=True):
    # the weights linearly increase, older prices have lower weights than more recent prices
    window = np.arange(0, 2 / interval, interval, dtype=float)
    return _moving_average(closes, window, standardize=standardize)


# a modified type of moving averages reacting quickly to recent price changes
def exponential_moving_average(closes, interval=50, smoothing=2, standardize=True):
    # initialize empty array
    ema = np.empty_like(closes)
    # define the weight of the newest price
    weight = smoothing / (interval + 1)
    # the first valid moving average is the standard moving average
    ema[interval] = np.mean(closes[:interval])

    # iteratively calculate the exponential moving average based on the previous moving average
    for i in range(interval + 1, len(closes)):
        ema[i] = closes[i - 1] * weight + ema[i - 1] * (1 - weight)

    # before moving averages go to zero, they are set to the current closing price
    # as a result, the indicator will be neutral (=1)
    # TODO Deal with eventual 0 closing prices
    # ema = np.where(ema == 0, closes, ema)

    # the moving average is not well-defined until enough price data exists
    ema[:interval] = np.nan
    # set a maximum of 2 for the ma, such that it can be standardized
    return utilities.standardize_indicator(np.clip(ema / closes, 0, 2), 0, 2) if standardize else ema


# the ma convergence divergence indicator is a comparison between to moving averages
def ma_convergence_divergence(closes, short_ma_length=12, long_ma_length=26, standardize=True):
    # get the tow moving averages
    short_ma = exponential_moving_average(closes=closes, interval=short_ma_length, standardize=False)
    long_ma = exponential_moving_average(closes=closes, interval=long_ma_length, standardize=False)
    # compare the moving averages
    macd = short_ma - long_ma
    return macd if standardize else utilities.standardize_indicator(macd, np.nanmax(macd), np.nanmin(macd))


def macd_cross(closes, short_ma_length=12, long_ma_length=26, signal_line_interval=9):
    macd = ma_convergence_divergence(closes, short_ma_length, long_ma_length, standardize=False)[long_ma_length:]
    macd_smoothed = exponential_moving_average(macd, interval=signal_line_interval, standardize=False)
    macd_signal_diff = macd - macd_smoothed
    macd_crossings = np.diff(np.sign(macd_signal_diff)) / 2
    window = np.arange(signal_line_interval, 0, -1) / signal_line_interval

    # it is assumed that no crossing has happened in previous, unknown data
    macd_crossings[:signal_line_interval] = np.nan
    # the difference operation leads to one more missing value for the first element
    macd_crossings = np.append(np.zeros(long_ma_length + 1) + np.nan, macd_crossings)
    return np.clip(np.convolve(macd_crossings, window, mode="full"), -1, 1)[:len(closes)]


# the indicator compares the average upward daily move with the average downward daily move
def relative_strength(closes, interval=14, standardize=True):
    # calculate the daily moves between closing prices
    moves = np.diff(closes)
    # define a sliding window
    sliding_window = sliding_window_view(moves, window_shape=interval)

    # Empty sliding windows will lead to runtime warnings
    # We accept this, because the result of an empty slice is nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # calculate the average of the upward movements for each time window
        average_ups = np.nanmean(np.where(sliding_window > 0, sliding_window, np.nan), axis=1)
        # calculate the average of the downward movements for each time window
        average_downs = np.nanmean(np.where(sliding_window < 0, sliding_window, np.nan), axis=1)

    # the strength is the relation of the two averages, set it to 1 in the case of division by zero
    strength = np.divide(average_ups, -average_downs,
                         out=np.ones_like(average_ups),
                         where=average_downs != 0)

    # rsi formula, result is between 0 and 100
    rsi = 100 - 100 / (1 + strength)
    # the rsi is not defined for the first closes
    rsi = np.append(np.zeros(interval) + np.nan, rsi)
    return utilities.standardize_indicator(rsi, 0, 100) if standardize else rsi


# an indicator trying to identify an upward or downward trend based on the macd indicator
def ma_trend(closes, short_ma_length=50, long_ma_length=200):
    # calculate the macd, no standardization is needed
    macd = ma_convergence_divergence(closes,
                                     short_ma_length=short_ma_length,
                                     long_ma_length=long_ma_length,
                                     standardize=False)
    # the trend is positive, if the short ma is higher than the long ma
    return np.sign(macd)


# an indicator trying to find crossings in the ma trend
def ma_crossing(closes, short_ma_length=50, long_ma_length=200, interval=50):
    # calculate the moving average trend
    trend = ma_trend(closes, short_ma_length=short_ma_length, long_ma_length=long_ma_length)

    # identify ma crossings using the difference of ma trends
    # is zero everywhere, except the trend flipping from -1 to 1 or vice-versa
    # note that in rare cases the ma trend can be 0 (two moving averages having the exact same value)
    # this indicator would interpret this partly as a crossing, even though it did not really happen
    ma_crossings = np.diff(trend) / 2
    window = np.arange(interval, 0, -1) / interval

    # it is assumed that no crossing has happened in previous, unknown data
    ma_crossings[:long_ma_length] = np.nan
    # the difference operation leads to one more missing value for the first element
    ma_crossings = np.append(np.array([np.nan]), ma_crossings)
    return np.clip(np.convolve(ma_crossings, window, mode="full"), -1, 1)[:len(closes)]


def _average_true_range(closes, lows, highs, interval=14):
    # initialize the parameters for the true range
    daily_range = highs - lows
    diff_high_close = np.abs(highs[1:] - closes[:-1])
    diff_close_low = np.abs(lows[1:] - closes[:-1])
    # calculate the true range for every time period
    true_range = np.amax(np.array([daily_range[:-1], diff_high_close, diff_close_low]), axis=0)
    # get the moving average over the true range
    return exponential_moving_average(true_range, interval=interval, standardize=False)


# average directional movement index uses differences between consecutive lows and consecutive highs to define a trend
def average_directional_movement(closes, lows, highs, interval=14, standardize=True):
    # calculate average true range
    average_true_range = _average_true_range(closes, lows, highs, interval=interval)

    ups = np.maximum(np.diff(highs), 0)
    downs = np.minimum(np.diff(lows), 0)
    smoothed_ups = exponential_moving_average(ups, interval=interval, standardize=False)
    smoothed_downs = exponential_moving_average(downs, interval=interval, standardize=False)
    dm_plus = smoothed_ups / average_true_range
    dm_minus = smoothed_downs / average_true_range
    directional_movements = np.divide(
        dm_plus - dm_minus,
        dm_plus + dm_minus,
        out=np.zeros_like(dm_minus),
        where=dm_plus + dm_minus != 0
    ) * 100

    return np.append(np.array([np.nan]), directional_movements)


def aaron(lows, highs, interval=25, standardize=True):
    window_lows = sliding_window_view(lows, window_shape=interval)
    window_highs = sliding_window_view(highs, window_shape=interval)
    aaron_down = 100 * (np.argmin(window_lows, axis=1) + 1) / interval
    aaron_down = np.append(np.zeros(interval) + np.nan, aaron_down)
    aaron_up = 100 * (np.argmax(window_highs, axis=1) + 1) / interval
    aaron_up = np.append(np.zeros(interval) + np.nan, aaron_up)
    aaron_diff = aaron_up - aaron_down
    oscillator = utilities.standardize_indicator(aaron_diff, -100, 100) if standardize else aaron_diff
    return aaron_down[:-1], aaron_up[:-1], oscillator[:-1]


def horizontal_channel_position(closes, interval=100, standardize=True):
    # create a window view sliding over the closes
    sliding_window = sliding_window_view(closes, window_shape=interval)
    # calculate the min for each window (range of closes)
    channel_low = np.min(sliding_window, axis=1)
    # calculate the max for each window (range of closes)
    channel_high = np.max(sliding_window, axis=1)

    # the min and max value is only defined if enough previous price exist
    channel_low = np.append(np.zeros(interval) + np.nan, channel_low)
    channel_high = np.append(np.zeros(interval) + np.nan, channel_high)

    if standardize:
        channel_low = channel_low[:-1] / closes
        channel_high = channel_high[:-1] / closes
    else:
        channel_low = channel_low[:-1]
        channel_high = channel_high[:-1]
    return channel_low, channel_high


def trend_channel_position(closes, interval=100, standardize=True):
    # calculate the parameters of the trend line for each time interval
    initial_value, slope = utilities.regression_lines(closes, interval)
    # construct the regression trendline
    trendlines = initial_value + slope * np.arange(interval - 1, len(closes) - 1, 1)

    # calculate the maximal deviated from the trend line for each time period
    sliding_window_time = sliding_window_view(np.arange(len(closes) - 1), window_shape=interval)
    max_distances = np.max(np.abs(closes[interval - 1:- 1] - (initial_value + slope * sliding_window_time.T)), axis=0)

    trendlines = np.append(np.zeros(interval) + np.nan, trendlines)
    max_distances = np.append(np.zeros(interval) + np.nan, max_distances)
    return utilities.construct_lower_upper_lines(closes, trendlines, max_distances, standardize=standardize)


def bollinger_bands(closes, interval=20, deviations=2, standardize=True):
    ma = standard_moving_average(closes, interval=interval, standardize=False)
    diff_sqared = np.power(closes - ma, 2)
    window = np.ones(interval, dtype=float) / interval
    std = np.sqrt(np.convolve(diff_sqared[:-1], window, mode="valid"))
    std = np.append(np.zeros(interval) + np.nan, std)
    distances = deviations * std
    return utilities.construct_lower_upper_lines(closes, ma, distances, standardize=standardize)



