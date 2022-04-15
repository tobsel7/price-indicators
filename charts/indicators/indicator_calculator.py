# import math libraries needed for indicator calculation
import numpy as np


# implements linear regression for a polynomial of degree 1
# used formula: A.T * A * x = A.T * b
def _regression_line(closes, index, interval):
    b = np.array(closes[index - interval:index])
    A = np.vstack([np.ones(interval), np.array(range(index - interval, index))]).transpose()
    regression_trend = np.linalg.solve(A.T @ A, A.T @ b)
    return regression_trend


# takes an indicator and its range and maps it to a value between -1 and 1
def _standardize(indicator, indicator_min=0, indicator_max=100):
    indicator_range = indicator_max - indicator_min
    indicator_middle = (indicator_max + indicator_min) / 2
    return 2 * (indicator - indicator_middle) / indicator_range


# a help function used to assign a relative position when compared with a range between a high and a low
# if standardized = False, returns a value between 0 and 1
# if standardized, the result is scaled and is between -1 and 1
def _relative_position(value, high, low, standardize=True):
    relative_position = 1 - ((high - value) / (high - low)) if high - low != 0 else 0
    return _standardize(relative_position, 0, 1) if standardize else relative_position


# this transformation amplifies an indicator, if it moves above an inflection point by applying a logistic function
def _transform_logistic(indicator, inflection_point=0.4, base=10**6):
    return np.sign(indicator) / (1 + base**(-np.abs(indicator) + inflection_point))


# this transformation tries to extract only extreme values by mapping all a certain value of an indicator to 0, 1 or -1
def _transform_threshold(indicator, threshold):
    threshold_surpassed = np.abs(indicator) > threshold
    # return -1 if indicator < -threshold, 1 if indicator > 1, else 0
    #return np.where(threshold_surpassed, np.sign(indicator), np.zeros(indicator))
    return np.sign(indicator) if threshold_surpassed else 0


# the moving average for a certain index is the mean price of the last x days
def moving_average(closes, index, interval=50, standardize=True):
    # moving average is just the average of the last x charts points
    ma = np.mean(closes[index-interval:index])
    return _standardize(np.clip(closes[index] / ma, 0, 2), 0, 2) if standardize else ma


# a modified type of moving averages giving higher weights to more recent values
def exponential_moving_average(closes, index, interval=50, smoothing=2, standardize=True):
    #weights = (smoothing / (1 + np.arange(interval) + 1))
    start = index - interval
    ema = 0
    for i in range(interval):
        weight = (smoothing / (1 + i + 1))
        # https://www.investopedia.com/terms/e/ema.asp#:~:text=Finally%2C%20the%20following%20formula%20is,)%20x%20(1%2Dmultiplier)
        ema = ema * (1 - weight) + closes[start + i] * weight
    #ema = closes[index-interval:index] * weights
    return _standardize(np.clip(closes[index] / ema, 0, 2), 0, 2) if standardize else ema


# TODO Which default values?
# the ma convergence divergence indicator is simply the difference between to moving averages
def ma_convergence_divergence(closes, index, short_ma_length=50, long_ma_length=200, standardize=True):
    short_ma = moving_average(closes=closes, index=index, interval=short_ma_length, standardize=standardize)
    long_ma = moving_average(closes=closes, index=index, interval=long_ma_length, standardize=standardize)
    return short_ma - long_ma


# the indicator compares the average upward daily move with the average downward daily move
def relative_strength(closes, index, interval=14, standardize=True):
    # calculate the daily moves between closing prices
    moves = np.diff(closes[index - interval - 1: index])
    # calculate the average upwards and downward movements
    ups = np.mean(moves[moves > 0])
    downs = -np.mean(moves[moves < 0])
    # the strength is the relation of the two averages, set it to 1 per default
    strength = 1 if downs == 0 else ups / downs
    # calculate rsi based on formula
    rsi = 100 - 100 / (1 + strength)
    return _standardize(rsi, 0, 100) if standardize else rsi


# an indicator trying to identify an upward or downward trend by comparing moving averages with different window lengths
def ma_trend(closes, index, short_ma_length=50, long_ma_length=200):
    # initialize sum variables for fast moving average calculations
    macd = ma_convergence_divergence(closes, index, short_ma_length=short_ma_length, long_ma_length=long_ma_length)
    # the trend is positive, if the long ma is lower than the short ma
    return np.sign(macd)


# a trend crossing is happening
def ma_trend_crossing(closes, index, short_ma_length=50, long_ma_length=200, interval=50):
    # define the windows for convolution
    short_window = np.ones(short_ma_length, dtype=int)
    long_window = np.ones(long_ma_length, dtype=int)

    # calculate moving averages
    ma_short = np.convolve(closes[index - short_ma_length - interval:index], short_window, "valid") / short_ma_length
    ma_long = np.convolve(closes[index - long_ma_length - interval:index], long_window, "valid") / long_ma_length

    # calculate the difference between the moving averages
    ma_diff = ma_short - ma_long
    # map the differences of the mas to crossings (the sign of the difference has flipped)
    ma_crossings = np.diff(np.sign(ma_diff))
    ma_crossings = ma_crossings[ma_crossings != 0]

    # return the last crossing
    return 0 if len(ma_crossings) == 0 else ma_crossings[-1]


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
    return moving_average(directional_movements, interval - 1, interval=interval, standardize=standardize)


def horizontal_channel_position(closes, index, interval=100, standardize=True):
    high = np.max(closes[index-interval:index])
    low = np.min(closes[index-interval:index])
    return _relative_position(closes[index], high, low, standardize)


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
    return _relative_position(closes[index], current_upper_line, current_lower_line, standardize)


def calculate_all_indicators(chart_data, index, standardize=True):
    # retrieve data necessary for index calculations
    closes = chart_data.get_closes()
    lows = chart_data.get_lows()
    highs = chart_data.get_highs()


    # calculate all indicators
    # moving averages
    ma10 = moving_average(closes, index, interval=10, standardize=standardize)
    ma20 = moving_average(closes, index, interval=20, standardize=standardize)
    ma50 = moving_average(closes, index, interval=50, standardize=standardize)
    ma100 = moving_average(closes, index, interval=100, standardize=standardize)
    ma200 = moving_average(closes, index, interval=200, standardize=standardize)

    # exponential moving averages
    ma10_exp = exponential_moving_average(closes, index, interval=10, smoothing=2, standardize=standardize)
    ma20_exp = exponential_moving_average(closes, index, interval=20, smoothing=2, standardize=standardize)
    ma50_exp = exponential_moving_average(closes, index, interval=50, smoothing=2, standardize=standardize)
    ma100_exp = exponential_moving_average(closes, index, interval=100, smoothing=2, standardize=standardize)
    ma200_exp = exponential_moving_average(closes, index, interval=200, smoothing=2, standardize=standardize)

    # macd
    macd_12_26 = ma_convergence_divergence(closes, index, short_ma_length=12, long_ma_length=26, standardize=standardize)

    # average directional movement index
    #adm = average_directional_movement(closes, lows, highs, index, interval=14)

    # trend
    cross50_200 = ma_trend_crossing(closes, index, short_ma_length=50, long_ma_length=200, interval=50)
    trend = ma_trend(closes, index, short_ma_length=50, long_ma_length=200)
    horizontal_trend_pos100 = horizontal_channel_position(closes, index, interval=100, standardize=standardize)
    trend_channel_pos100 = trend_channel_position(closes, index, interval=100, standardize=standardize)

    # rsi
    rsi_standardized = relative_strength(closes, index, interval=50, standardize=True)
    rsi = rsi_standardized if standardize else relative_strength(closes, index, interval=50, standardize=standardize)
    rsi_logistic = _transform_logistic(rsi_standardized, inflection_point=0.4, base=10**6)
    rsi_threshold = _transform_threshold(rsi_standardized, threshold=0.4)

    # return all indicators as dictionary
    return {
        "ma10": ma10,
        "ma20": ma20,
        "ma50": ma50,
        "ma100": ma100,
        "ma200": ma200,
        "ma10_exp": ma10_exp,
        "ma20_exp": ma20_exp,
        "ma50_exp": ma50_exp,
        "ma100_exp": ma100_exp,
        "ma200_exp": ma200_exp,
        "macd12_26": macd_12_26,
        "ma_trend_crossing50-200": cross50_200,
        "ma_trend": trend,
        "rsi": rsi,
        "rsi_logistic": rsi_logistic,
        "rsi_threshold": rsi_threshold,
        "horizontal_trend_pos100": horizontal_trend_pos100,
        "trend_channel_pos100": trend_channel_pos100
    }

