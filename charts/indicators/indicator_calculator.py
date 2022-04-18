import numpy as np
import pandas as pd


# implements linear regression for a polynomial of degree 1
# used formula: A.T * A * x = A.T * b
def _regression_line(closes, index, interval):
    b = np.array(closes[index - interval:index])
    A = np.vstack([np.ones(interval), np.array(range(index - interval, index))]).transpose()
    return np.linalg.solve(A.T @ A, A.T @ b)


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
    return np.where(threshold_surpassed, np.sign(indicator), np.zeros_like(indicator))


# internal function for all weighted moving averages
def _moving_average(closes, window, standardize=True):
    # get the window length
    window_size = len(window)
    # moving average is the convolution of a rectangular window with the closing prices
    ma = np.convolve(closes[:-1], window, mode="valid")

    # clip the values to a range between 0 and 2 before standardizing
    ma = _standardize(np.clip(closes[window_size:] / ma, 0, 2), 0, 2) if standardize else ma
    # the ma is not defined in the beginning of the time series
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

    # the moving average is no well-defined for the first closes
    ema[:interval] = np.nan
    # set a maximum of 2 for the ma, such that it can be standardized
    return _standardize(np.clip(closes / ema, 0, 2), 0, 2) if standardize else ema


# the ma convergence divergence indicator is simply the difference between two moving averages
def ma_convergence_divergence(closes, short_ma_length=12, long_ma_length=26, standardize=True):
    short_ma = standard_moving_average(closes=closes, interval=short_ma_length, standardize=standardize)
    long_ma = standard_moving_average(closes=closes, interval=long_ma_length, standardize=standardize)
    return short_ma - long_ma


# the indicator compares the average upward daily move with the average downward daily move
def relative_strength(closes, interval=14, standardize=True):
    # calculate the daily moves between closing prices
    moves = np.diff(closes)
    # define a sliding window
    sliding_window = np.lib.stride_tricks.sliding_window_view(moves, window_shape=interval)
    # calculate the average of the upward movements for each time window
    average_ups = np.nanmean(np.where(sliding_window > 0, sliding_window, np.nan), axis=1)
    # calculate the average of the downward movements for each time window
    average_downs = np.nanmean(np.where(sliding_window < 0, sliding_window, np.nan), axis=1)

    # the strength is the relation of the two averages, set it to 1 in the case of division by zero
    strength = 1 - np.divide(average_ups, -average_downs,
                             out=np.ones_like(average_ups),
                             where=average_ups / average_downs != 0)
    # rsi formula, result is between 0 and 100
    rsi = 100 - 100 / (1 + strength)
    # the rsi is not defined for the first closes
    rsi = np.append(np.zeros(interval), rsi)
    return _standardize(rsi, 0, 100) if standardize else rsi


# an indicator trying to identify an upward or downward trend by comparing moving averages with different window lengths
def ma_trend(closes, short_ma_length=50, long_ma_length=200):
    # initialize sum variables for fast moving average calculations
    macd = ma_convergence_divergence(closes, short_ma_length=short_ma_length, long_ma_length=long_ma_length)
    # the trend is positive, if the long ma is lower than the short ma
    return np.sign(macd)


# compare
def ma_crossing(closes, short_ma_length=50, long_ma_length=200, interval=50):
    # calculate the difference between two moving averages
    macd = ma_convergence_divergence(closes, short_ma_length=short_ma_length, long_ma_length=long_ma_length)
    # map the differences of the mas to crossings (the sign of the difference has flipped)
    ma_crossings = np.diff(np.sign(macd)) / 2
    window = np.arange(interval, 0, -1) / interval

    # it is assumed that no crossing has happened in previous, unknown data
    ma_crossings[:long_ma_length] = 0
    ma_crossings = np.append(np.array([0]), ma_crossings)
    return np.clip(np.convolve(ma_crossings, window, mode="full"), -1, 1)[:len(closes)]


def true_range(closes, lows, highs, index, interval=14):
    daily_range = highs[index - interval:index] - lows[index - interval:index]
    diff_high_close = highs[index - interval:index] - closes[index - interval - 1:index - 1]
    diff_close_low = closes[index - interval - 1:index - 1] - lows[index - interval:index]
    return np.amax(np.array([daily_range, diff_high_close, diff_close_low]))


# average directional movement index uses differences between consecutive lows and consecutive highs to define a trend
def average_directional_movement(closes, lows, highs, index, interval=14, standardize=True):
    # calculate average true range
    average_true_range = np.mean(true_range(closes, lows, highs, index, interval=interval))

    pushes_up = np.max(np.diff(highs[index - interval - 1:index]), 0)
    pushes_down = np.max(-np.diff(lows[index - interval - 1:index]), 0)
    directional_movements = (pushes_up - pushes_down) / (pushes_up + pushes_down)
    print(len(directional_movements))
    return standard_moving_average(directional_movements, interval - 1, interval=interval, standardize=standardize)


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


def trend_channel_position(closes, index, interval=100, standardize=True):
    # calculate the regression line for this period of time
    initial_value, slope = _regression_line(closes, index, interval)

    # take only data for this time period
    data_points = np.array(closes[index - interval:index])
    # calculate the maximal deviated from the trend line
    max_distance = np.max(np.abs(data_points - (initial_value + slope * np.arange(index - interval, index, 1))))

    # construct an upper and lower trend line where all price points lie between
    current_upper_line = initial_value + slope * index + max_distance
    current_lower_line = initial_value + slope * index - max_distance
    # return the relative position in this channel of lines
    return _relative_position(closes[index], current_lower_line, current_upper_line, standardize)


def calculate_all_indicators(chart_data, standardize=True):
    # retrieve data necessary for index calculations
    closes = chart_data.get_closes()
    lows = chart_data.get_lows()
    highs = chart_data.get_highs()

    # calculate all indicators
    # moving averages
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

    """# average directional movement index
    #adm = average_directional_movement(closes, lows, highs, index, interval=14)"""

    # trend
    cross50_200 = ma_crossing(closes, short_ma_length=50, long_ma_length=200, interval=50)
    trend = ma_trend(closes, short_ma_length=50, long_ma_length=200)
    horizontal_trend_pos100 = horizontal_channel_position(closes, interval=100, standardize=standardize)
    """trend_channel_pos100 = trend_channel_position(closes, index, interval=100, standardize=standardize)"""

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
            "ma_trend_crossing50-200": cross50_200,
            "ma_trend": trend,
            "rsi": rsi,
            "rsi_logistic": rsi_logistic,
            "rsi_threshold": rsi_threshold,
            "horizontal_trend_pos100": horizontal_trend_pos100,
            # "trend_channel_pos100": trend_channel_pos100
        }
    )
