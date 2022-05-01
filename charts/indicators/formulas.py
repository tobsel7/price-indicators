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
    ma = np.convolve(closes[:-1], window, mode="valid")
    # the ma is not well-defined before enough price data exists
    return np.append(np.zeros(window_size) + np.nan, ma)


# calculate the average of the difference between a mean of an indicator and an indicator
# the square root of _average_deviations represents the l2 norm or standard deviation
# the _average deviations with the norm function np.abs represents the l1 norm, or manhattan distance
def _average_deviations(indicator, indicator_means, interval=100, norm_function=np.square):
    # apply a function to get the l1, l2 norms etc.
    differences_norm = norm_function(indicator - indicator_means)
    # average the deviations using convolution
    window = np.ones(interval, dtype=float) / interval
    summed_deviations = np.convolve(differences_norm[:-1], window, mode="valid")
    summed_deviations = np.append(np.zeros(interval) + np.nan, summed_deviations)
    return summed_deviations


# the standard deviation or l2 norm using convolution
def standard_deviation(indicator, indicator_mean, interval=100):
    return np.sqrt(_average_deviations(indicator, indicator_mean, interval=interval, norm_function=np.square))


# the mean price over a defined interval
def standard_moving_average(closes, interval=50):
    # the window function is a rectangular, this represents the equal weighted sum
    window = np.ones(interval, dtype=float) / interval
    return _moving_average(closes, window)


# the weighted mean price over a defined interval
def weighted_moving_average(closes, interval=50):
    # the weights linearly increase, older prices have lower weights than recent prices
    window = np.arange(0, 2 / interval, interval, dtype=float)
    return _moving_average(closes, window)


# a modified type of moving averages reacting quickly to recent price changes
def exponential_moving_average(closes, interval=50, smoothing=2):
    # initialize empty array
    ema = np.empty_like(closes)
    # define the weight of the most recent price
    weight = smoothing / (interval + 1)
    # the first valid moving average is the standard moving average
    ema[interval] = np.mean(closes[:interval])

    # iteratively calculate the exponential moving average based on the previous moving average
    for i in range(interval + 1, len(closes)):
        ema[i] = closes[i - 1] * weight + ema[i - 1] * (1 - weight)

    # the moving average is not well-defined until enough price data exists
    ema[:interval] = np.nan
    return ema


# a comparison (difference) between a fast and slow moving average
def ma_convergence_divergence(closes, short_ma_length=12, long_ma_length=26):
    # get the tow moving averages
    short_ma = exponential_moving_average(closes=closes, interval=short_ma_length)
    long_ma = exponential_moving_average(closes=closes, interval=long_ma_length)
    # compare the moving averages
    return short_ma - long_ma


# the difference between two moving averages is mapped to -1 and 1, signalling moves of one ma moving above the other
def macd_cross(closes, short_ma_length=12, long_ma_length=26, signal_line_interval=9):
    # get the difference between to moving averages
    macd = ma_convergence_divergence(closes, short_ma_length, long_ma_length)[long_ma_length:]
    # smooth the macd
    macd_smoothed = exponential_moving_average(macd, interval=signal_line_interval)
    # difference between the averaged/smoothed macd and the current macd
    macd_signal_diff = macd - macd_smoothed
    # map the difference to crossings of the smoothed and current macd lines
    macd_crossings = np.diff(np.sign(macd_signal_diff)) / 2

    # it is assumed that no crossing has happened in previous, unknown data
    macd_crossings[:signal_line_interval] = np.nan
    # the difference operation leads to one more missing value for the first element
    macd_crossings = np.append(np.zeros(long_ma_length + 1) + np.nan, macd_crossings)

    # convolve the crossings with a window function
    window = np.arange(signal_line_interval, 0, -1) / signal_line_interval
    return np.clip(np.convolve(macd_crossings, window, mode="full"), -1, 1)[:len(closes)]


# the indicator compares the average upward daily move with the average downward daily move
def relative_strength(closes, interval=14):
    # calculate the daily moves between closing prices
    moves = np.diff(closes)
    # define a sliding window
    sliding_window = sliding_window_view(moves, window_shape=interval)

    # Empty sliding windows will lead to runtime warnings
    # This is tolerated, because the result of an empty slice is correctly labeled as "not a number"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # calculate the average of the upward movements for each time window
        average_ups = np.nanmean(np.where(sliding_window > 0, sliding_window, np.nan), axis=1)
        # calculate the average of the downward movements for each time window
        average_downs = np.nanmean(np.where(sliding_window < 0, sliding_window, np.nan), axis=1)

    # the strength is the relation of the two averages, set it to 1 in the case of division by zero
    strength = average_ups / -average_downs

    # rsi formula, result is between 0 and 100
    rsi = 100 - 100 / (1 + strength)
    # the rsi is not defined for the first closes
    rsi = np.append(np.zeros(interval) + np.nan, rsi)
    return rsi


# an indicator trying to identify an upward or downward trend based on the macd indicator
def ma_trend(closes, short_ma_length=50, long_ma_length=200):
    macd = ma_convergence_divergence(closes, short_ma_length=short_ma_length, long_ma_length=long_ma_length)
    # the trend is positive, if the short ma is higher than the long ma, else negative
    return np.sign(macd)


# an indicator identifying changes (crossings) in the moving average trend
def ma_crossing(closes, short_ma_length=50, long_ma_length=200, interval=50):
    # calculate the moving average trend
    trend = ma_trend(closes, short_ma_length=short_ma_length, long_ma_length=long_ma_length)

    # identify ma crossings using the difference of ma trends
    # is zero everywhere, except the trend flipping from -1 to 1 or vice-versa
    # note that the ma trend can be 0 (two moving averages having the exact same value)
    # this implementation would interpret this partly as a crossing, even though it did not really happen
    ma_crossings = np.diff(trend) / 2
    window = np.arange(interval, 0, -1) / interval

    # it is assumed that no crossing has happened in previous, unknown data
    ma_crossings[:long_ma_length] = np.nan
    # the difference operation leads to one more missing value for the first element
    ma_crossings = np.append(np.array([np.nan]), ma_crossings)
    return np.clip(np.convolve(ma_crossings, window, mode="full"), -1, 1)[:len(closes)]


# the average true range is the maximum value between three different ranges
def _average_true_range(closes, lows, highs, interval=14):
    # initialize the parameters for the true range
    daily_range = highs - lows
    diff_high_close = np.abs(highs[1:] - closes[:-1])
    diff_close_low = np.abs(lows[1:] - closes[:-1])
    # calculate the true range for every time period
    true_range = np.amax(np.array([daily_range[:-1], diff_high_close, diff_close_low]), axis=0)
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
    return np.append(np.zeros(interval + 1) + np.nan, smoothed_directional_movements)


# the aaron indicator is the time passed since the last high/low mapped to a value between 0 and 100
def aaron(lows, highs, interval=25):
    # construct windows for each time interval for the lows and highs
    window_lows = sliding_window_view(lows, window_shape=interval)
    window_highs = sliding_window_view(highs, window_shape=interval)
    # get the position of the lowest/highest value in each time period
    aaron_down = 100 * (np.argmin(window_lows, axis=1) + 1) / interval
    aaron_up = 100 * (np.argmax(window_highs, axis=1) + 1) / interval
    # the indicator is not defined for the first data points
    aaron_down = np.append(np.zeros(interval) + np.nan, aaron_down)
    aaron_up = np.append(np.zeros(interval) + np.nan, aaron_up)
    # construct an oscillator using the two indicators
    oscillator = aaron_up - aaron_down
    # return all three indicators, but omit the last one
    # the last indicators at the last position is for tomorrow
    return aaron_down[:-1], aaron_up[:-1], oscillator[:-1]


# the highest and lowest close in some time interval define a range containing all recent prices
def horizontal_channel_position(closes, interval=100, ):
    # create a window view sliding over the closes
    sliding_window = sliding_window_view(closes, window_shape=interval)
    # calculate the min for each time interval
    channel_low = np.min(sliding_window, axis=1)
    # calculate the max for each time interval
    channel_high = np.max(sliding_window, axis=1)

    # the min and max value is only defined if enough previous price exist
    channel_low = np.append(np.zeros(interval) + np.nan, channel_low)
    channel_high = np.append(np.zeros(interval) + np.nan, channel_high)

    # the last indicators at the last position is for tomorrow
    return channel_low[:-1], channel_high[:-1]


# a channel constructed using regression
# the channel lines are defined as regression trend +- the maximal absolute from this line
def trend_channel_position(closes, interval=100):
    # calculate the parameters of the trend line for each time interval
    initial_value, slope = utilities.regression_lines(closes, interval)
    # construct the regression trendline
    trendlines = initial_value + slope * np.arange(interval - 1, len(closes) - 1, 1)

    # calculate the maximal deviated from the trend line for each time period
    sliding_window_time = sliding_window_view(np.arange(len(closes) - 1), window_shape=interval)
    max_distances = np.max(np.abs(closes[interval - 1:- 1] - (initial_value + slope * sliding_window_time.T)), axis=0)

    # no valid channel can be constructed for the first data points
    trendlines = np.append(np.zeros(interval) + np.nan, trendlines)
    max_distances = np.append(np.zeros(interval) + np.nan, max_distances)
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
                    out=np.zeros_like(closes),
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
    momentum = np.append(np.zeros(interval) + np.nan, momentum)

    return momentum * 100


# a simple momentum indicator comparing a past closing price with the current price
def rate_of_change(closes, interval=100):
    # for the first elements no past data will exist
    current = np.append(np.zeros(interval), closes[:-interval])
    # for the last elements, no future data will exist
    past = np.append(closes[interval:], np.zeros(interval) + np.nan)
    # compare the shifted closing prices
    roc = np.divide(current - past, past, out=np.ones_like(closes), where=past != 0)
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
