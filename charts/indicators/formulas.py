# some operations are numerically unstable, they are accepted because they return nan in this case
import warnings

# numpy for numerical functions
import numpy as np

# specifically import a sliding view function, because its often used
# the sliding view essentially allows convolution-like operations, but with nonlinear functions
from numpy.lib.stride_tricks import sliding_window_view

# indicator utility functions
from charts.indicators import utilities


# general definition of the moving average
# the price data is convolved with an arbitrary window function, the sum of the window impulses must be 1
def _moving_average(closes, window):
    # get the window length
    window_size = len(window)
    # moving average is the convolution of a rectangular window with the closing prices
    ma = np.convolve(closes, window, mode="valid")
    # the ma is not well-defined before enough price data exists
    return np.append(np.zeros(window_size - 1, dtype=np.float64) + np.nan, ma)


# calculate the average of the difference between a mean of an indicator and an indicator
# the square root of _average_deviations represents the l2 norm or standard deviation
# the _average deviations with the norm function np.abs represents the l1 norm, or manhattan distance
def _average_deviations(indicator, indicator_means, interval=100, norm_function=np.square):
    # apply a function to get the l1, l2 norms etc.
    differences_norm = norm_function(indicator - indicator_means)
    # average the deviations using convolution
    window = np.ones(interval, dtype=np.float64) / interval
    summed_deviations = np.convolve(differences_norm, window, mode="valid")
    summed_deviations = np.append(np.zeros(interval - 1) + np.nan, summed_deviations)
    return summed_deviations


# the standard deviation or l2 norm using convolution
def standard_deviation(indicator, indicator_mean, interval=100):
    return np.sqrt(_average_deviations(indicator, indicator_mean, interval=interval, norm_function=np.square))


# the mean price over a defined interval
def standard_moving_average(closes, interval=50):
    # the window function is a rectangular, this represents the equal weighted sum
    window = np.ones(interval, dtype=np.float64) / interval
    return _moving_average(closes, window)


# the weighted mean price over a defined interval
def linear_weighted_moving_average(closes, interval=50):
    # the weights linearly increase, older prices have lower weights than recent prices
    window = 2 * np.arange(1, interval + 1, dtype=np.float64) / (interval * (interval + 1))
    return _moving_average(closes, window)


# a modified type of moving averages reacting quickly to recent price changes
def exponential_moving_average(closes, interval=50, smoothing=2):
    # initialize empty array
    ema = np.zeros_like(closes) + np.nan
    # define the weight of the most recent price
    weight = smoothing / (interval + 1)
    # the first valid moving average is the standard moving average
    ema[interval - 1] = np.mean(closes[:interval])

    # iteratively calculate the exponential moving average based on the previous moving average
    for i in range(interval, len(closes)):
        ema[i] = closes[i] * weight + ema[i - 1] * (1 - weight)

    return ema


# a comparison (difference) between a fast and slow moving average
def ma_convergence_divergence(short_ma, long_ma, long_ma_interval, signal_line_length=9):
    # compare the exponential moving averages
    macd = short_ma - long_ma
    macd_signal = exponential_moving_average(macd[long_ma_interval - 1:], signal_line_length)
    macd_signal = np.append(np.zeros(long_ma_interval - 1, dtype=np.float64) + np.nan, macd_signal)
    # return the macd line and the signal line
    return macd, macd_signal


# the indicator compares the average upward daily move with the average downward daily move
def relative_strength(closes, interval=14):
    # calculate the daily moves between closing prices
    moves = np.diff(closes)
    # define a sliding window
    sliding_window_moves = sliding_window_view(moves, window_shape=interval)

    # Empty sliding windows will lead to runtime warnings
    # This is tolerated, because the result of an empty slice is correctly labeled as "not a number"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # calculate the average of the upward movements for each time window
        average_ups = np.nanmean(np.where(sliding_window_moves > 0, sliding_window_moves, np.nan), axis=1)
        # calculate the average of the downward movements for each time window
        average_downs = np.nanmean(np.where(sliding_window_moves < 0, sliding_window_moves, np.nan), axis=1)

    # the strength is the relation of the two averages
    strength = average_ups / -average_downs

    # rsi formula, result is between 0 and 100
    rsi = 100 - 100 / (1 + strength)
    # the rsi is not defined for the first closes
    rsi = np.append(np.zeros(interval, dtype=np.float64) + np.nan, rsi)
    return rsi


# an indicator trying to identify an upward or downward trend based two moving averages
def ma_trend(short_ma, long_ma):
    # the trend is positive, if the short ma is higher than the long ma, else negative
    return np.sign(short_ma - long_ma)


# an indicator identifying changes (crossings) in the moving average trend
def crossing(fast_indicator, slow_indicator, interval=50):
    # identify crossings between two lines
    # the result is zero everywhere, except the one line moves above the other
    # note that the result can be 0 (baseline and signal line being equal)
    # this implementation would interpret this partly as a crossing, even though it did not really happen

    crossings = np.diff(np.sign(fast_indicator - slow_indicator)) / 2
    # convolve the crossings with a linearly decreasing window function
    # this way a recent crossing receives a high value, while some past crossing is less relevant
    window = np.arange(interval, 0, -1) / interval

    # it is assumed that no crossing has happened in previous, unknown data
    # the difference operation leads to one more missing value for the first element
    crossings = np.append(np.array([np.nan]), crossings)

    return np.convolve(crossings, window, mode="full")[:len(slow_indicator)]


# the average true range is the maximum value between three different ranges
def _average_true_range(closes, lows, highs, interval=14):
    # initialize the parameters for the true range
    daily_range = highs - lows
    diff_high_close = np.abs(highs[1:] - closes[:-1])
    diff_close_low = np.abs(lows[1:] - closes[:-1])
    # calculate the true range for every time period
    true_range = np.amax(np.array([daily_range[1:], diff_high_close, diff_close_low]), axis=0)
    # get the moving average over the true range
    return exponential_moving_average(true_range, interval=interval)


# average directional movement index uses differences between consecutive lows and consecutive highs to define a trend
def average_directional_movement(closes, lows, highs, interval=14):
    # calculate average true range
    average_true_range = _average_true_range(closes, lows, highs, interval=interval)
    # get the upward and downward movements
    ups = np.maximum(np.diff(highs), 0)
    downs = np.minimum(np.diff(lows), 0)
    # smooth the movements using the exponential moving average
    smoothed_ups = exponential_moving_average(ups, interval=interval)
    smoothed_downs = exponential_moving_average(downs, interval=interval)

    # calculate the indicator parameters
    dm_plus = np.divide(smoothed_ups, average_true_range,
                        out=np.zeros_like(average_true_range) + np.nan, where=average_true_range != 0)
    dm_minus = np.divide(smoothed_downs, average_true_range,
                         out=np.zeros_like(average_true_range) + np.nan, where=average_true_range != 0)
    # use the directional movement formula
    directional_movements = np.divide(
        np.abs(dm_plus - dm_minus),
        dm_plus + dm_minus,
        out=np.zeros_like(downs),
        where=dm_plus + dm_minus != 0
    )
    smoothed_directional_movements = exponential_moving_average(directional_movements[interval:], interval=interval)
    # the first element is not defined
    return np.append(np.zeros(interval + 1, dtype=np.float64) + np.nan, smoothed_directional_movements)


# the aaron indicator is the time passed since the last high/low mapped to a value between 0 and 100
def aaron(lows, highs, interval=25):
    # construct windows for each time interval for the lows and highs
    window_lows = sliding_window_view(lows, window_shape=interval + 1)
    window_highs = sliding_window_view(highs, window_shape=interval + 1)
    # get the position of the lowest/highest value in each time period
    aaron_down = 100 * (np.argmin(window_lows, axis=1)) / interval
    aaron_up = 100 * (np.argmax(window_highs, axis=1)) / interval
    # the indicator is not defined for the first data points
    aaron_down = np.append(np.zeros(interval, dtype=np.float64) + np.nan, aaron_down)
    aaron_up = np.append(np.zeros(interval, dtype=np.float64) + np.nan, aaron_up)
    # construct an oscillator using the two indicators
    oscillator = aaron_up - aaron_down
    # return all three indicators
    return aaron_down, aaron_up, oscillator


# the highest and lowest close in some time interval define a range containing all recent prices
def horizontal_channel(closes, interval=100):
    # create a window view sliding over the closes
    sliding_window = sliding_window_view(closes, window_shape=interval)
    # calculate the min for each time interval
    channel_low = np.min(sliding_window, axis=1)
    # calculate the max for each time interval
    channel_high = np.max(sliding_window, axis=1)

    # the min and max value is only defined if enough previous price exist
    channel_low = np.append(np.zeros(interval - 1, dtype=np.float64) + np.nan, channel_low)
    channel_high = np.append(np.zeros(interval - 1, dtype=np.float64) + np.nan, channel_high)

    return channel_low, channel_high


# a channel constructed using regression
# the channel lines are defined as regression trend +- the maximal absolute from this line
def trend_channel(closes, interval=100):
    # calculate the parameters of the trend line for each time interval
    initial_value, slope = utilities.regression_lines(closes, interval)
    # construct the regression trendline
    trendlines = initial_value + slope * np.arange(interval - 1, len(closes), 1)

    # calculate the maximal deviated from the trend line for each time period
    sliding_window_time = sliding_window_view(np.arange(len(closes)), window_shape=interval)
    max_distances = np.max(np.abs(closes[interval - 1:] - (initial_value + slope * sliding_window_time.T)), axis=0)

    # no valid channel can be constructed for the first data points
    trendlines = np.append(np.zeros(interval - 1, dtype=np.float64) + np.nan, trendlines)
    max_distances = np.append(np.zeros(interval - 1, dtype=np.float64) + np.nan, max_distances)
    # construct the lower and upper channel line
    return utilities.construct_lower_upper_lines(trendlines, max_distances)


# the commodity channel index is a momentum indicator measuring deviations of the typical price from mean prices
def commodity_channel(lows, highs, closes, interval=20):
    # calculate the typical average price using the standard moving average function and the three prices
    summed_prices = lows + highs + closes
    typical_price = standard_moving_average(summed_prices, interval) / 3
    ma = standard_moving_average(closes, interval=interval)

    # calculate the mean deviation of the typical price from the mean price
    mean_deviation = _average_deviations(typical_price, ma, interval=interval, norm_function=np.abs)

    # return a ration of deviations from mean and the standard deviation
    cci = np.divide(typical_price - ma, .015 * mean_deviation,
                    out=np.zeros_like(closes, dtype=np.float64),
                    where=mean_deviation != 0)
    return cci


# chande momentum compares the sum of upward movements to the sum of downward movements
def chande_momentum(closes, interval=50):
    # calculate the daily moves between closing prices
    moves = np.diff(closes)
    # define a sliding window
    sliding_window_moves = sliding_window_view(moves, window_shape=interval)

    # Empty sliding windows will lead to runtime warnings
    # This is tolerated, because the result of an empty slice is correctly labeled as "not a number"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # calculate the average of the upward movements for each time window
        sum_ups = np.sum(np.where(sliding_window_moves > 0, sliding_window_moves, 0), axis=1)
        # calculate the average of the downward movements for each time window
        sum_downs = -np.sum(np.where(sliding_window_moves < 0, sliding_window_moves, 0), axis=1)
        sum_moves = sum_downs + sum_ups

    # the momentum is based on the difference between the total upward and downward movements
    momentum = np.divide(sum_ups - sum_downs, sum_moves, out=np.ones_like(sum_moves), where=sum_moves != 0)
    momentum = np.append(np.zeros(interval, dtype=np.float64) + np.nan, momentum)

    return momentum * 100


# a simple momentum indicator comparing a past closing price with the current price
def rate_of_change(closes, interval=100):
    # for the last elements, no future data will exist
    past = np.append(np.zeros(interval) + np.nan, closes[:-interval])
    # compare the shifted closing prices
    roc = np.divide(closes - past, past, out=np.ones_like(closes), where=past != 0)
    # the ratio is multiplied by 100 in the indicator formula
    return 100 * roc


# lines placed around a moving average using the current standard deviation
# the lines contract when the standard deviation declines and expand when the standard deviation increases
def bollinger_bands(closes, interval=20, deviations=2):
    # get the moving average
    ma = standard_moving_average(closes, interval=interval)
    std = standard_deviation(closes, ma, interval=interval)

    # place a line above and below the moving average with the distance of a specified number of standard deviations
    distances = deviations * std
    return utilities.construct_lower_upper_lines(ma, distances)
