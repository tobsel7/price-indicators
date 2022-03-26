# import math libraries needed for indicator calculation
import numpy as np


# implements linear regression for a polynom of degree 1
# used formula: A.T * A * x = A.T * b
def _regression_line(closes, index, interval):
    b = np.array(closes[index - interval:index])
    A = np.vstack([np.ones(interval), np.array(range(index - interval, index))]).transpose()
    regression_trend = np.linalg.solve(A.T @ A, A.T @ b)
    return regression_trend


# a help function used to assign a relative position when compared with a range between a high and a low
# if standardized = False, returns a value between 0 and 1
# if standardized, the result is scaled and is between -1 and 1
def _relative_position(value, high, low, standardize=False):
    relative_position = 1 - ((high - value) / (high - low))
    return relative_position * 2 - 1 if standardize else relative_position


# calculate the basic moving average indicator
def moving_average(closes, index, interval=50, standardize=False):
    # moving average is just the average of the last x charts points
    ma = np.average(closes[index-interval:index])
    ma_standardize = np.clip(closes[index] / ma - 1, -1, 1)
    return ma_standardize if standardize else ma


def relative_strength(closes, index, interval=14, standardize=False):
    # initialize values to approximately zero for numerical stability
    ups = 10e-6
    downs = 10e-6
    # sum up all upward and downward moves
    for i in range(index - interval, index):
        move = closes[i] - closes[i - 1]
        if move > 0:
            # price has moved up
            ups += move
        else:
            # price has moved down
            downs -= move
    strength = ups / downs
    # calculate rsi based on formula
    rsi = (0.5 - 1 / (1 + strength)) * 2 if standardize else 100 - 100 / (1 + strength)
    return rsi


def ma_trend(closes, index, short_ma_length=50, long_ma_length=200, interval=50):
    # initialize sum variables for fast moving average calculations
    ma_short = np.sum(closes[index-interval-short_ma_length-1:index-interval-1]) / short_ma_length
    ma_long = np.sum(closes[index-interval-long_ma_length-1:index-interval-1]) / long_ma_length
    # define the trend
    trend = 1 if ma_short > ma_long else -1
    return trend


def ma_trend_crossing(closes, index, short_ma_length=50, long_ma_length=200, interval=50):
    # initialize sum variables for fast moving average calculations
    sum_short = np.sum(closes[index-short_ma_length:index])
    sum_long = np.sum(closes[index-long_ma_length:index])

    # initialize the ma variables
    ma_short = sum_short / short_ma_length
    ma_long = sum_long / long_ma_length
    curr_ma_diff = ma_short - ma_long

    # check whether a ma crossing has happened in the defined interval by going backwards
    for i in range(index - 1, index - 1 - interval, - 1):
        # recalculate ma_short
        sum_short -= closes[i]
        sum_short += closes[i-short_ma_length]
        ma_short = sum_short / short_ma_length

        # recalculate ma_long
        sum_long -= closes[i]
        sum_long += closes[i-long_ma_length]
        ma_long = sum_long / long_ma_length

        # calculate previous ma difference
        prev_ma_diff = ma_short - ma_long

        # check if the short ma was lower than the long ma at the last index
        if prev_ma_diff < 0 < curr_ma_diff:
            # the shorter ma is now above the longer ma
            return 1
        elif prev_ma_diff > 0 > curr_ma_diff:
            # the shorter ma is now below the longer ma
            return -1
        else:
            # no crossing found, update the current ma difference and continue
            curr_ma_diff = prev_ma_diff

    # no crossing found withing the interval
    return 0


def horizontal_channel_position(closes, index, interval=100, standardize=False):
    high = np.max(closes[index-interval:index])
    low = np.min(closes[index-interval:index])
    return _relative_position(closes[index], high, low, standardize)


def trend_channel_position(closes, index, interval=100, standardize=False):
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


def calculate_all_indicators(chart_data, index, standardize=False):
    # retrieve closes from chart
    closes = np.array(chart_data.get_closes())

    # calculate all indicators
    ma10 = moving_average(closes, index, interval=10, standardize=standardize)
    ma20 = moving_average(closes, index, interval=20, standardize=standardize)
    ma50 = moving_average(closes, index, interval=50, standardize=standardize)
    ma100 = moving_average(closes, index, interval=100, standardize=standardize)
    ma200 = moving_average(closes, index, interval=200, standardize=standardize)
    cross = ma_trend_crossing(closes, index, short_ma_length=50, long_ma_length=200, interval=50)
    trend = ma_trend(closes, index, short_ma_length=50, long_ma_length=200)
    rsi = relative_strength(closes, index, interval=50, standardize=standardize)
    horizontal_trend_pos100 = horizontal_channel_position(closes, index, interval=100, standardize=standardize)
    trend_channel_pos100 = trend_channel_position(closes, index, interval=100, standardize=standardize)

    # return all indicators as dictionary
    return {
        "ma10": ma10,
        "ma20": ma20,
        "ma50": ma50,
        "ma100": ma100,
        "ma200": ma200,
        "ma_trend_crossing50": cross,
        "ma_trend": trend,
        "rsi": rsi,
        "horizontal_trend_pos100": horizontal_trend_pos100,
        "trend_channel_pos100": trend_channel_pos100
    }
