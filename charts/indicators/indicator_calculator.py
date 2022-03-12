# import math libraries needed for indicator calculation
import numpy as np


# calculate the basic moving average indicator
def moving_average(closes, index, normalized=False, interval=50):
    # moving average is just the average of the last x charts points
    ma = np.average(closes[index-interval:index])
    ma_normalized = np.clip(ma / closes[index] - 1, -1, 1)
    return ma_normalized if normalized else ma


def relative_strength(closes, index, normalized=False, interval=14):
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
    rsi = (0.5 - 1 / (1 + strength)) * 2 if normalized else 100 - 100 / (1 + strength)
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
    sum_short = np.sum(closes[index-interval-short_ma_length-1:index-interval-1])
    sum_long = np.sum(closes[index-interval-long_ma_length-1:index-interval-1])
    # initialize the difference between the moving averages
    ma_diff = sum_long / long_ma_length - sum_short / short_ma_length
    # per default no crossing has happened
    crossing = 0
    for i in range(index - interval, index):
        # update the sums for the last closes
        sum_short += closes[i] - closes[i-short_ma_length]
        sum_long += closes[i] - closes[i-long_ma_length]
        # update the moving averages
        ma_short = sum_short / short_ma_length
        ma_long = sum_long / long_ma_length

        # check if the short ma was lower than the long ma at the last index or a crossing has happened previously
        if ma_diff < 0 > ma_long - ma_short:
            # the shorter ma is now above the longer ma
            crossing = 1
        if ma_diff > 0 < ma_long - ma_short:
            # the shorter ma is now below the longer ma
            crossing = -1

        # update the difference between the moving averages
        ma_diff = ma_long - ma_short
    return crossing


def calculate_all_indicators(chart_data, index, normalized=False):
    closes = np.array(chart_data.get_closes())
    # normalize the charts if specified
    if normalized:
        closes = closes / closes[index]

    ma20 = moving_average(closes, index, normalized=True, interval=20)
    ma50 = moving_average(closes, index, normalized=True, interval=50)
    ma100 = moving_average(closes, index, normalized=True, interval=100)
    ma200 = moving_average(closes, index, normalized=True, interval=200)
    # map boolean of crossing to 0 or 1
    cross = int(ma_trend_crossing(closes, index, short_ma_length=50, long_ma_length=200, interval=50))
    trend = ma_trend(closes, index, short_ma_length=50, long_ma_length=200)
    rsi = relative_strength(closes, index, normalized=True, interval=50)

    # return all indicators as dictionary
    return {
        "ma20": ma20,
        "ma50": ma50,
        "ma100": ma100,
        "ma200": ma200,
        "ma_trend_crossing": cross,
        "ma_trend": trend,
        "rsi": rsi,
    }
