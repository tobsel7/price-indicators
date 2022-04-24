from . import formulas
from . import utilities

import pandas as pd
import numpy as np

STANDARD_MOVING_AVERAGE_INTERVALS = [10, 20, 50, 100, 200]
STANDARD_MOVING_AVERAGE_TREND_PARAMETERS = {
    # short_ma: long_ma
    20: 50,
    50: 200
}
MACD_SIGNAL_LINE_SMOOTHING = 9
EXPONENTIAL_MOVING_AVERAGE_INTERVALS = [10, 20, 50, 100, 200]
EXPONENTIAL_MOVING_AVERAGE_SMOOTHING = 2
BOLLINGER_BAND_PARAMETERS = {
    # interval: deviations
    20: 2,
    50: 2
}
RSI_INTERVALS = [50]
RSI_LOGISTIC_TRANSFORMATION_BASE = 10 ** 6
RSI_LOGISTIC_TRANSFORMATION_INFLECTION_POINT = 0.4
AARON_INTERVALS = [25, 50]
TREND_CHANNEL_INTERVALS = [100, 200]

# the number of preceding days needed to calculate all the indicators
MIN_PRECEDING_VALUES = max([max(STANDARD_MOVING_AVERAGE_INTERVALS),
                            max(STANDARD_MOVING_AVERAGE_TREND_PARAMETERS.keys()),
                            max(STANDARD_MOVING_AVERAGE_TREND_PARAMETERS.values()),
                            max(EXPONENTIAL_MOVING_AVERAGE_INTERVALS),
                            max(BOLLINGER_BAND_PARAMETERS.keys()),
                            max(RSI_INTERVALS),
                            max(AARON_INTERVALS),
                            max(TREND_CHANNEL_INTERVALS)
                            ])


def standard_moving_averages(closes, intervals=STANDARD_MOVING_AVERAGE_INTERVALS, standardize=True):
    smas = {}
    for interval in intervals:
        sma = formulas.standard_moving_average(closes,
                                               interval=interval,
                                               standardize=standardize)
        smas["sma{}".format(interval)] = sma
    return smas


def standard_moving_average_trends(closes, parameters=STANDARD_MOVING_AVERAGE_TREND_PARAMETERS, standardize=True):
    summary = {}
    for short, long in parameters.items():
        macd = formulas.ma_convergence_divergence(closes, short_ma_length=short, long_ma_length=long, standardize=standardize)
        macd_cross = formulas.macd_cross(closes, short_ma_length=short, long_ma_length=long)
        trend = formulas.ma_trend(closes, short_ma_length=short, long_ma_length=long)
        cross = formulas.ma_crossing(closes, short_ma_length=short, long_ma_length=long, interval=short)
        summary["macd{}_{}".format(short, long)] = macd
        summary["macd_cross{}_{}".format(short, long)] = macd_cross
        summary["trend{}_{}".format(short, long)] = trend
        summary["cross{}_{}".format(short, long)] = cross

    return summary


def exponential_moving_averages(closes, intervals=EXPONENTIAL_MOVING_AVERAGE_INTERVALS, standardize=True):
    smoothing = EXPONENTIAL_MOVING_AVERAGE_SMOOTHING
    emas = {}
    for interval in intervals:
        ema = formulas.exponential_moving_average(closes,
                                                  interval=interval,
                                                  smoothing=smoothing,
                                                  standardize=standardize)
        emas["ema{}".format(interval)] = ema
    return emas


def average_directional_movements(closes, lows, highs, standardize=True):
    summary = {}
    # summary["adm"] = formulas.average_directional_movement(closes, lows, highs, interval=14)
    return summary


def aaron(lows, highs, intervals=AARON_INTERVALS, standardize=True):
    summary = {}
    for interval in intervals:
        down, up, oscillator = formulas.aaron(lows, highs, interval=interval, standardize=standardize)
        summary["aaron_down{}".format(interval)] = down
        summary["aaron_up{}".format(interval)] = up
        summary["aaron_oscillator{}".format(interval)] = oscillator

    return summary


def bollinger_bands(closes, parameters=BOLLINGER_BAND_PARAMETERS, standardize=True):
    summary = {}
    for interval, deviations in parameters.items():
        lower, upper = formulas.bollinger_bands(closes,
                                                interval=interval,
                                                deviations=deviations,
                                                standardize=standardize)
        position = utilities.relative_position(np.ones_like(closes) if standardize else closes,
                                               lower,
                                               upper,
                                               standardize=standardize)

        summary["bollinger_lower{}_{}".format(interval, deviations)] = lower
        summary["bollinger_upper{}_{}".format(interval, deviations)] = upper
        summary["bollinger_position{}_{}".format(interval, deviations)] = position

    return summary


def relative_strength_index(closes,
                            intervals=RSI_INTERVALS,
                            standardize=True,
                            logistic_transformation_inflection_point=RSI_LOGISTIC_TRANSFORMATION_INFLECTION_POINT,
                            logistic_transformation_base=RSI_LOGISTIC_TRANSFORMATION_BASE
                            ):
    summary = {}
    for interval in intervals:
        rsi_standardized = formulas.relative_strength(closes, interval=interval, standardize=True)
        rsi = rsi_standardized
        rsi_logistic = utilities.transform_logistic(rsi_standardized,
                                                    inflection_point=logistic_transformation_inflection_point,
                                                    base=logistic_transformation_base)
        rsi_threshold = utilities.transform_threshold(rsi_standardized, threshold=0.4)

        if not standardize:
            rsi *= 100
            rsi_logistic *= 100
            rsi_threshold *= 100

        summary["rsi{}".format(interval)] = rsi
        summary["rsi_logistic{}".format(interval)] = rsi_logistic
        summary["rsi_threshold{}".format(interval)] = rsi_threshold

    return summary


def trend_channels(closes, intervals=TREND_CHANNEL_INTERVALS, standardize=True):
    summary = {}
    for interval in intervals:
        horizontal_lower, horizontal_upper = formulas.horizontal_channel_position(closes,
                                                                                  interval=interval,
                                                                                  standardize=standardize)
        regression_lower, regression_upper = formulas.trend_channel_position(closes,
                                                                             interval=interval,
                                                                             standardize=standardize)
        summary["horizontal_lower{}".format(interval)] = horizontal_lower
        summary["horizontal_upper{}".format(interval)] = horizontal_upper
        summary["regression_lower{}".format(interval)] = regression_lower
        summary["regression_upper{}".format(interval)] = regression_upper

    return summary


def all_indicators(chart_data, standardize=True):
    # retrieve data necessary for indicator calculations
    closes = chart_data.get_closes()
    lows = chart_data.get_lows()
    highs = chart_data.get_highs()

    # GET A SUMMARY FOR ALL GROUPS OF INDICATORS
    # standard moving averages
    sma = standard_moving_averages(closes, standardize=standardize)
    # standard moving average trends
    sma_trend = standard_moving_average_trends(closes)
    # exponential moving averages
    ema = exponential_moving_averages(closes, standardize=standardize)
    # average directional movement index
    adm = average_directional_movements(closes, lows, highs, standardize=standardize)
    # aaron
    aarn = aaron(lows, highs, standardize=standardize)
    # bollinger bands
    bollinger = bollinger_bands(closes, standardize=standardize)
    # rsi
    rsi = relative_strength_index(closes, standardize=standardize)
    # trend channels
    channels = trend_channels(closes, standardize=standardize)

    # combine all indicators and return them as a dataframe
    merged_summaries = {**sma, **sma_trend, **ema, **adm, **aarn, **bollinger, **rsi, **channels}

    #for name, indicator in merged_summaries.items():
        #print("{} length {}".format(name, len(indicator)))
    return pd.DataFrame(merged_summaries)
