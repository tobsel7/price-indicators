# import math libraries needed for indicator calculation
import numpy as np


# calculate the basic moving average indicator
def moving_average(closes, index, normalized=False, interval=50):
    # moving average is just the average of the last x data points
    ma = np.average(closes[index-interval:index])
    return ma / closes[index] if normalized else ma


def relative_strength(closes, index, normalized=False, interval=50):
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
    rsi = 1 - 1 / (1 + strength) if normalized else 100 - 100 / (1 + strength)
    return rsi


def golden_cross(closes, index, interval=50):
    # initialize sum variables for fast moving average calculations
    sum50 = np.sum(closes[index-interval-50-1:index-interval-1])
    sum200 = np.sum(closes[index-interval-200-1:index-interval-1])
    # initialize the difference between the moving averages
    ma_diff = sum200 / 200 - sum50 / 50
    # per default no crossing has happened
    crossing = False
    for i in range(index - interval, index):
        # update the sum over the last 50 and 200 data points
        sum50 += closes[i] - closes[i-50]
        sum200 += closes[i] - closes[i-200]
        # update the moving averages
        ma50 = sum50 / 50
        ma200 = sum200 / 200
        # check if the ma 50 was lower than the ma 200 at the last index or a crossing has happened previously
        if ma_diff < 0 or crossing:
            # a golden cross happened if the ma 50 is now higher than the ma200 and continues to stay higher
            crossing = ma50 > ma200
        # update the difference between the moving averages
        ma_diff = ma200 - ma50
    return crossing


def calculate_all_indicators(chart_data, index, normalized=False):
    closes = np.array(chart_data.get_closes())
    # normalize the data if specified
    if normalized:
        closes = closes / closes[index]

    ma50 = moving_average(closes, index, normalized=True, interval=50)
    ma100 = moving_average(closes, index, normalized=True, interval=100)
    ma200 = moving_average(closes, index, normalized=True, interval=200)
    # map boolean of crossing to 0 or 1
    cross = int(golden_cross(closes, index, interval=50))
    rsi = relative_strength(closes, index, normalized=True, interval=50)

    # return all indicators as dictionary
    return {
        "ma50": ma50,
        "ma100": ma100,
        "ma50": ma200,
        "golden_cross": cross,
        "rsi": rsi,
    }
