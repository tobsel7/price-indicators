import warnings

import numpy as np
import pandas as pd

# the number of preceding days needed to calculate all the indicators
MIN_PRECEDING_VALUES = 250


# implements linear regression for a polynomial of degree 1
# used formula: A.T * A * x = A.T * b
# where A is the regression matrix and b the price data (goal)
def _regression_lines(closes, interval):
    # initialize the regression matrix for the complete price data interval
    regression_matrix = np.vstack([np.ones_like(closes), np.arange(len(closes))]).T
    # move a sliding window over the closes to repeatedly generate vectors holding a range of closing prices
    sliding_window_closes = np.lib.stride_tricks.sliding_window_view(closes, window_shape=interval)
    # move a sliding window over the regression matrix to generate multiple parts of the complete regression matrix
    sliding_window_time = np.lib.stride_tricks.sliding_window_view(regression_matrix, window_shape=(interval, 2))
    # ignore the redundant dimension created by the stride tricks library
    sliding_window_time = sliding_window_time[:, 0, :, :]

    # Use the inner product between a closing price window and a part of the regression matrix
    destination_vectors = np.einsum("ijk,ij->ik", sliding_window_time, sliding_window_closes)

    # Take the outer product of each regression matrix interval
    covariance_matrices = np.einsum("ijk,ijl->ikl", sliding_window_time, sliding_window_time)

    # evaluate the regression parameters for all time intervals
    result = np.einsum("ijk,ij->ik", np.linalg.inv(covariance_matrices), destination_vectors)
    # throw away the regression result for the last data point
    result = result[:-1, :]
    # split ub the two result columns in two variables for better transparency
    return result[:, 0], result[:, 1]


# takes an indicator and its range and maps it to a value between -1 and 1
def _standardize(indicator, indicator_min=0, indicator_max=100):
    # calculate the range of the input indicator
    indicator_range = indicator_max - indicator_min
    # calculate the middle of the input indicator
    indicator_middle = (indicator_max + indicator_min) / 2
    # shift and scale the indicator
    return 2 * (indicator - indicator_middle) / indicator_range


# a help function used to assign a relative position when compared with a range between a low and high
def _relative_position(closes, lows, highs, standardize=True):
    relative_position = 1 - np.divide(highs - closes, highs - lows, out=np.zeros_like(closes), where=highs - lows != 0)
    return _standardize(relative_position, 0, 1) if standardize else relative_position


# this transformation amplifies an indicator, if it moves above an inflection point by applying a logistic function
def _transform_logistic(indicator, inflection_point=0.4, base=10 ** 6):
    return np.sign(indicator) / (1 + base ** (-np.abs(indicator) + inflection_point))


# this transformation tries to extract only extreme values by mapping all a certain value of an indicator to 0, 1 or -1
def _transform_threshold(indicator, threshold):
    # find all positions where the threshold is being surpassed
    threshold_surpassed = np.abs(indicator) > threshold
    # map the indicator to its sign (-1 or 1), if the threshold is surpassed, else 0
    threshold_indicator = np.where(threshold_surpassed, np.sign(indicator), np.zeros_like(indicator))
    # deal with nan values, they should not be mapped to any number, because of the threshold transformation
    return np.where(np.isnan(indicator), indicator, threshold_indicator)


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
    ma = _standardize(np.clip(closes[window_size:] / ma, 0, 2), 0, 2) if standardize else ma
    # the ma is not well-defined before enough price data exists
    return np.append(np.zeros(window_size) + np.nan, ma)


# the mean price over a defined interval
def standard_moving_average(closes, interval=50, standardize=True):
    # the window function is a rectangular, this represents the equal weighted sum
    window = np.ones(interval, dtype=float) / interval
    return _moving_average(closes, window, standardize=standardize)


# a modified type of moving averages giving higher weights to more recent values
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
    ema = np.where(ema == 0, closes, ema)

    # the moving average is not well-defined until enough price data exists
    ema[:interval] = np.nan
    # set a maximum of 2 for the ma, such that it can be standardized
    return _standardize(np.clip(closes / ema, 0, 2), 0, 2) if standardize else ema


# the ma convergence divergence indicator is simply the difference between two moving averages
def ma_convergence_divergence(closes, short_ma_length=12, long_ma_length=26, standardize=True):
    short_ma = standard_moving_average(closes=closes, interval=short_ma_length, standardize=False)
    long_ma = standard_moving_average(closes=closes, interval=long_ma_length, standardize=False)
    macd = short_ma - long_ma
    # get the largest deviation from zero in the ma convergence divergence indicator
    macd_range_max = np.nanmax(np.abs(macd))
    return _standardize(macd, -macd_range_max, macd_range_max) if standardize else macd


# the indicator compares the average upward daily move with the average downward daily move
def relative_strength(closes, interval=14, standardize=True):
    # calculate the daily moves between closing prices
    moves = np.diff(closes)
    # define a sliding window
    sliding_window = np.lib.stride_tricks.sliding_window_view(moves, window_shape=interval)

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
    return _standardize(rsi, 0, 100) if standardize else rsi


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
    daily_range = np.abs(highs - lows)
    diff_high_close = np.abs(highs[1:] - closes[:-1])
    diff_close_low = np.abs(lows[1:] - closes[:-1])
    # calculate the true range for every time period
    true_range = np.amax(np.array([daily_range[1:], diff_high_close, diff_close_low]), axis=0)
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


def aaron_oscillator(lows, highs, interval=25, standardize=True):
    window_lows = np.lib.stride_tricks.sliding_window_view(lows, window_shape=interval)
    window_highs = np.lib.stride_tricks.sliding_window_view(highs, window_shape=interval)
    aaron_down = 100 * (np.argmin(window_lows, axis=1) + 1) / interval
    aaron_up = 100 * (np.argmax(window_highs, axis=1) + 1) / interval
    aaron_diff = aaron_up - aaron_down
    aaron_diff = np.append(np.zeros(interval) + np.nan, aaron_diff[:-1])
    return _standardize(aaron_diff, -100, 100) if standardize else aaron_diff


def horizontal_channel_position(closes, interval=100, standardize=True):
    # create a window view sliding over the closes
    sliding_window = np.lib.stride_tricks.sliding_window_view(closes, window_shape=interval)
    # calculate the min for each window (range of closes)
    range_min = np.min(sliding_window, axis=1)
    # calculate the max for each window (range of closes)
    range_max = np.max(sliding_window, axis=1)

    # the min and max value is only defined if enough previous price exist
    range_min = np.append(np.zeros(interval - 1) + np.nan, range_min)
    range_max = np.append(np.zeros(interval - 1) + np.nan, range_max)

    # retrieve the relative position between the horizontal channel defined by the lowest and highest value
    return _relative_position(closes, range_min, range_max, standardize)


def trend_channel_position(closes, interval=100, standardize=True):
    # calculate the parameters of the trend line for each time interval
    initial_value, slope = _regression_lines(closes, interval)
    # construct the regression trendline
    trendline = initial_value + slope * np.arange(interval - 1, len(closes) - 1, 1)

    # calculate the maximal deviated from the trend line for each time period
    sliding_window_time = np.lib.stride_tricks.sliding_window_view(np.arange(len(closes) - 1), window_shape=interval)
    max_distance = np.max(np.abs(closes[interval - 1:- 1] - (initial_value + slope * sliding_window_time.T)), axis=0)

    # return the relative position in this channel of lines
    trendline_pos = _relative_position(
        closes[interval - 1:-1],
        trendline - max_distance,
        trendline + max_distance,
        standardize)

    # for the first closes no relative position should be defined
    return np.append(np.zeros(interval) * np.nan, trendline_pos)


def calculate_all_indicators(chart_data, standardize=True):
    # retrieve data necessary for index calculations
    closes = chart_data.get_closes()
    lows = chart_data.get_lows()
    highs = chart_data.get_highs()

    # standard moving averages
    sma10 = standard_moving_average(closes, interval=10, standardize=standardize)
    sma20 = standard_moving_average(closes, interval=20, standardize=standardize)
    sma50 = standard_moving_average(closes, interval=50, standardize=standardize)
    sma100 = standard_moving_average(closes, interval=100, standardize=standardize)
    sma200 = standard_moving_average(closes, interval=200, standardize=standardize)

    # exponential moving averages
    ema10 = exponential_moving_average(closes, interval=10, smoothing=2, standardize=standardize)
    ema20 = exponential_moving_average(closes, interval=20, smoothing=2, standardize=standardize)
    ema50 = exponential_moving_average(closes, interval=50, smoothing=2, standardize=standardize)
    ema100 = exponential_moving_average(closes, interval=100, smoothing=2, standardize=standardize)
    ema200 = exponential_moving_average(closes, interval=200, smoothing=2, standardize=standardize)

    # macd
    macd_12_26 = ma_convergence_divergence(closes, short_ma_length=12, long_ma_length=26, standardize=standardize)

    # average directional movement index
    #adm = average_directional_movement(closes, lows, highs, interval=14)

    # trend
    cross50_200 = ma_crossing(closes, short_ma_length=50, long_ma_length=200, interval=50)
    trend20_50 = ma_trend(closes, short_ma_length=20, long_ma_length=50)
    trend50_200 = ma_trend(closes, short_ma_length=50, long_ma_length=200)
    horizontal_trend_pos100 = horizontal_channel_position(closes, interval=100, standardize=standardize)
    trend_channel_pos100 = trend_channel_position(closes, interval=100, standardize=standardize)
    aaron25 = aaron_oscillator(lows, highs, interval=25, standardize=standardize)

    # rsi
    rsi_standardized = relative_strength(closes, interval=50, standardize=True)
    rsi = rsi_standardized if standardize else relative_strength(closes, interval=50, standardize=standardize)
    rsi_logistic = _transform_logistic(rsi_standardized, inflection_point=0.4, base=10**6)
    rsi_threshold = _transform_threshold(rsi_standardized, threshold=0.4)

    return pd.DataFrame(
        {
            "sma10": sma10,
            "sma20": sma20,
            "sma50": sma50,
            "sma100": sma100,
            "sma200": sma200,
            "ema10": ema10,
            "ema20": ema20,
            "ema50": ema50,
            "ema100": ema100,
            "ema200": ema200,
            "macd12_26": macd_12_26,
            "ma_crossing50_200": cross50_200,
            "ma_trend20_50": trend20_50,
            "ma_trend50_200": trend50_200,
            "aaron25": aaron25,
            #"adm": adm,
            "rsi": rsi,
            "rsi_logistic": rsi_logistic,
            "rsi_threshold": rsi_threshold,
            "horizontal_trend_pos100": horizontal_trend_pos100,
            "trend_channel_pos100": trend_channel_pos100
        }
    )
