# get indicator formulas and utilities
from charts.indicators import formulas
from charts.indicators import utilities

# libraries for datasets
import pandas as pd
import numpy as np

# import default parameters
from charts.parameters import VOLATILITY_INTERVALS, MOVING_AVERAGE_INTERVALS, MOVING_AVERAGE_TREND_PARAMETERS, \
    MACD_PARAMETERS, MACD_SIGNAL_LINE_INTERVAL, EXPONENTIAL_MOVING_AVERAGE_SMOOTHING, BOLLINGER_BAND_PARAMETERS, \
    RSI_INTERVALS, RSI_LOGISTIC_TRANSFORMATION_BASE, RSI_LOGISTIC_TRANSFORMATION_INFLECTION_POINT, \
    AVERAGE_DIRECTIONAL_MOVEMENT_INTERVALS, AARON_INTERVALS,COMMODITY_CHANNEL_INTERVALS, TREND_CHANNEL_INTERVALS, \
    CHANDE_MOMENTUM_INTERVALS, RATE_OF_CHANGE_INTERVALS


# the number of preceding days needed to calculate all the indicators is the maximum of all interval parameters
MIN_PRECEDING_VALUES = max([max(VOLATILITY_INTERVALS * 2),
                            max(MOVING_AVERAGE_INTERVALS),
                            max(MOVING_AVERAGE_TREND_PARAMETERS.keys()),
                            max(MOVING_AVERAGE_TREND_PARAMETERS.values()),
                            max(np.array(list(BOLLINGER_BAND_PARAMETERS.keys())) * 2),
                            max(RSI_INTERVALS),
                            max(AVERAGE_DIRECTIONAL_MOVEMENT_INTERVALS * 2),
                            max(AARON_INTERVALS),
                            max(COMMODITY_CHANNEL_INTERVALS),
                            max(TREND_CHANNEL_INTERVALS),
                            max(CHANDE_MOMENTUM_INTERVALS),
                            max(RATE_OF_CHANGE_INTERVALS)
                            ])


# in this section all indicator formulas are called and the results stored in dictionaries with labels
# for more detailed information about the indicator formulas, consult the formulas.py file

def volatility(closes, intervals=VOLATILITY_INTERVALS, normalize=True):
    volat = {}
    for interval in intervals:
        ma = formulas.standard_moving_average(closes, interval)
        std = formulas.standard_deviation(closes, ma, interval=interval)
        if normalize:
            std = std / closes
        volat["volatility{}".format(interval)] = std

    return volat


def moving_averages(closes, intervals=MOVING_AVERAGE_INTERVALS, normalize=True):
    # use a default value for the exponential smoothing weights
    smoothing = EXPONENTIAL_MOVING_AVERAGE_SMOOTHING
    mas = {}
    for interval in intervals:
        sma = formulas.standard_moving_average(closes, interval=interval)
        lwma = formulas.linear_weighted_moving_average(closes, interval)
        ema = formulas.exponential_moving_average(closes, interval=interval, smoothing=smoothing)
        if normalize:
            sma = utilities.normalize_indicator(sma / closes, 0, 2, clip=False)
            lwma = utilities.normalize_indicator(lwma / closes, 0, 2, clip=False)
            ema = utilities.normalize_indicator(ema / closes, 0, 2, clip=False)

        mas["sma{}".format(interval)] = sma
        mas["lwma{}".format(interval)] = lwma
        mas["ema{}".format(interval)] = ema

    return mas


def moving_average_trends(closes, parameters=MOVING_AVERAGE_TREND_PARAMETERS):
    summary = {}
    for short, long in parameters.items():
        # calculate the moving averages
        short_sma = formulas.standard_moving_average(closes=closes, interval=short)
        long_sma = formulas.standard_moving_average(closes=closes, interval=long)

        # get the trend and trend crossings
        ma_trend = formulas.ma_trend(short_sma, long_sma)
        ma_cross = formulas.crossing(short_sma, long_sma, interval=short)

        summary["ma_trend{}_{}".format(short, long)] = ma_trend
        summary["ma_cross{}_{}".format(short, long)] = ma_cross

    return summary


def ma_convergence_divergence(closes, parameters=MACD_PARAMETERS, normalize=True):
    summary = {}
    for short, long in parameters.items():
        # calculate the exponential moving averages
        short_ema = formulas.exponential_moving_average(closes=closes, interval=short)
        long_ema = formulas.exponential_moving_average(closes=closes, interval=long)

        # define the macd base and signal line crossings
        macd, macd_signal = formulas.ma_convergence_divergence(short_ema, long_ema, long,
                                                               signal_line_length=MACD_SIGNAL_LINE_INTERVAL)
        macd_cross = formulas.crossing(macd, macd_signal, interval=MACD_SIGNAL_LINE_INTERVAL)
        if normalize:
            # the macd base and signal line is normalized using the long moving average
            macd = utilities.normalize_indicator(macd / long_ema, -1, 1, clip=False)
            macd_signal = utilities.normalize_indicator(macd_signal / long_ema, -1, 1, clip=False)

        summary["macd{}_{}".format(short, long)] = macd
        summary["macd_signal{}_{}".format(short, long)] = macd_signal
        summary["macd_cross{}_{}".format(short, long)] = macd_cross

    return summary


def average_directional_movements(closes, lows, highs,
                                  intervals=AVERAGE_DIRECTIONAL_MOVEMENT_INTERVALS, normalize=True):
    summary = {}
    for interval in intervals:
        adx = formulas.average_directional_movement(closes, lows, highs, interval)
        if normalize:
            # set the maximum of the indicator to 200 and the minimum to 200 for the purpose of standardization
            adx = utilities.normalize_indicator(adx, -200, 200, clip=False)
        summary["adm{}".format(interval)] = adx

    return summary


def aaron(lows, highs, intervals=AARON_INTERVALS, normalize=True):
    summary = {}
    for interval in intervals:
        down, up, oscillator = formulas.aaron(lows, highs, interval=interval)
        if normalize:
            down /= 100
            up /= 100
            oscillator = utilities.normalize_indicator(oscillator, -100, 100, clip=False)

        summary["aaron_down{}".format(interval)] = down
        summary["aaron_up{}".format(interval)] = up
        summary["aaron_oscillator{}".format(interval)] = oscillator

    return summary


def bollinger_bands(closes, parameters=BOLLINGER_BAND_PARAMETERS, normalize=True):
    summary = {}
    for interval, deviations in parameters.items():
        lower, upper = formulas.bollinger_bands(closes, interval=interval, deviations=deviations)
        position = utilities.relative_position(closes, lower, upper, normalize=True, clip=False)
        position_threshold = utilities.transform_threshold(position, 1)
        if normalize:
            lower = utilities.normalize_indicator(lower / closes, 0, 2)
            upper = utilities.normalize_indicator(upper / closes, 0, 2)
        else:
            position = (position + 1) / 2

        summary["bollinger_lower{}_{}".format(interval, deviations)] = lower
        summary["bollinger_upper{}_{}".format(interval, deviations)] = upper
        summary["bollinger_position{}_{}".format(interval, deviations)] = position
        summary["bollinger_threshold{}_{}".format(interval, deviations)] = position_threshold

    return summary


def relative_strength_index(closes,
                            intervals=RSI_INTERVALS,
                            normalize=True,
                            logistic_transformation_inflection_point=RSI_LOGISTIC_TRANSFORMATION_INFLECTION_POINT,
                            logistic_transformation_base=RSI_LOGISTIC_TRANSFORMATION_BASE
                            ):
    summary = {}
    for interval in intervals:
        rsi = formulas.relative_strength(closes, interval=interval)
        rsi_normalized = utilities.normalize_indicator(rsi, 0, 100, clip=False)
        rsi_logistic = utilities.transform_logistic(rsi_normalized,
                                                    inflection_point=logistic_transformation_inflection_point,
                                                    base=logistic_transformation_base)
        rsi_threshold = utilities.transform_threshold(rsi_normalized, threshold=0.4)

        if normalize:
            rsi = rsi_normalized
        else:
            rsi_logistic *= 100

        summary["rsi{}".format(interval)] = rsi
        summary["rsi_logistic{}".format(interval)] = rsi_logistic
        summary["rsi_threshold{}".format(interval)] = rsi_threshold

    return summary


def commodity_channel(lows, highs, closes, intervals=COMMODITY_CHANNEL_INTERVALS, normalize=True):
    summary = {}
    for interval in intervals:
        cci = formulas.commodity_channel(lows, highs, closes, interval=interval)
        cci_threshold = utilities.transform_threshold(cci, 100)
        if normalize:
            cci = utilities.normalize_indicator(cci, -200, 200, clip=False)

        summary["cci{}".format(interval)] = cci
        summary["cci_threshold{}".format(interval)] = cci_threshold

    return summary


def chande_momentum(closes, intervals=CHANDE_MOMENTUM_INTERVALS, normalize=True):
    summary = {}
    for interval in intervals:
        chande = formulas.chande_momentum(closes, interval=interval)
        chande_threshold = utilities.transform_threshold(chande, 50)
        if normalize:
            chande = utilities.normalize_indicator(chande, -100, 100, clip=False)

        summary["chande{}".format(interval)] = chande
        summary["chande_threshold{}".format(interval)] = chande_threshold

    return summary


def rate_of_change(closes, intervals=RATE_OF_CHANGE_INTERVALS, normalize=True):
    summary = {}
    for interval in intervals:
        roc = formulas.rate_of_change(closes, interval=interval)

        if normalize:
            roc = utilities.normalize_indicator(roc, -100, 100, clip=False)

        summary["rate_of_change{}".format(interval)] = roc

    return summary


def trend_channels(closes, intervals=TREND_CHANNEL_INTERVALS, normalize=True):
    summary = {}
    for interval in intervals:
        horizontal_lower, horizontal_upper = formulas.horizontal_channel(closes, interval=interval)
        horizontal_position = utilities.relative_position(closes, horizontal_lower, horizontal_upper,
                                                          normalize=True)
        horizontal_threshold = utilities.transform_threshold(horizontal_position, 1)
        regression_lower, regression_upper = formulas.trend_channel(closes, interval=interval)
        regression_position = utilities.relative_position(closes, regression_lower, regression_upper,
                                                          normalize=True)
        regression_threshold = utilities.transform_threshold(regression_position, 1)
        if normalize:
            horizontal_lower = utilities.normalize_indicator(horizontal_lower / closes, 0, 2, clip=False)
            horizontal_upper = utilities.normalize_indicator(horizontal_upper / closes, 0, 2, clip=False)
            regression_lower = utilities.normalize_indicator(regression_lower / closes, 0, 2, clip=False)
            regression_upper = utilities.normalize_indicator(regression_upper / closes, 0, 2, clip=False)
        else:
            horizontal_position = (horizontal_position + 1) / 2
            regression_position = (regression_position + 1) / 2

        summary["horizontal_lower{}".format(interval)] = horizontal_lower
        summary["horizontal_upper{}".format(interval)] = horizontal_upper
        summary["horizontal_position{}".format(interval)] = horizontal_position
        summary["horizontal_threshold{}".format(interval)] = horizontal_threshold
        summary["regression_lower{}".format(interval)] = regression_lower
        summary["regression_upper{}".format(interval)] = regression_upper
        summary["regression_position{}".format(interval)] = regression_position
        summary["regression_threshold{}".format(interval)] = regression_threshold
    return summary


def all_indicators(chart_data, normalize=True):
    # retrieve data necessary for indicator calculations
    closes = chart_data.get_closes()
    lows = chart_data.get_lows()
    highs = chart_data.get_highs()

    # volatility
    volat = volatility(closes, normalize=normalize)
    # different types of moving averages
    ma = moving_averages(closes, normalize=normalize)
    # moving average trends
    ma_trend = moving_average_trends(closes)
    # macd
    macd = ma_convergence_divergence(closes, normalize=normalize)
    # average directional movement index6
    adm = average_directional_movements(closes, lows, highs, normalize=normalize)
    # aaron
    aarn = aaron(lows, highs, normalize=normalize)
    # bollinger bands
    bollinger = bollinger_bands(closes, normalize=normalize)
    # rsi
    rsi = relative_strength_index(closes, normalize=normalize)
    # trend channels
    channels = trend_channels(closes, normalize=normalize)
    # commodity channel index
    cci = commodity_channel(lows, highs, closes, normalize=normalize)
    # chande momentum
    chande = chande_momentum(closes, normalize=normalize)

    # rate of change momentum
    roc = rate_of_change(closes, normalize=normalize)

    # combine all indicators and return them as a dataframe
    merged_summaries = {**volat, **ma, **ma_trend, **macd, **adm, **aarn, **bollinger, **rsi, **channels, **cci,
                        **chande, **roc}

    return pd.DataFrame(merged_summaries)
