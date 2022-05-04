# numerical operations
import numpy as np


# implements linear regression for a polynomial of degree 1
# used formula: A.T * A * x = A.T * b
# where A is the regression matrix and b the price data (goal)
def regression_lines(closes, interval):
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
    # split ub the two result columns in two variables for better transparency
    return result[:, 0], result[:, 1]


# constructs two lines below and above a trendline using a distance array
def construct_lower_upper_lines(trendlines, distances):
    upper_line = trendlines + distances
    lower_line = trendlines - distances
    return lower_line, upper_line


# takes an indicator and its range and maps it to a value between -1 and 1
def normalize_indicator(indicator, indicator_min=0, indicator_max=100, clip=False):
    if clip:
        # if clip is set to true, the ranges of the indicator are enforced
        indicator = np.clip(indicator, indicator_min, indicator_max)
    # calculate the range of the input indicator
    indicator_range = indicator_max - indicator_min
    # calculate the middle of the input indicator
    indicator_middle = (indicator_max + indicator_min) / 2
    # shift and scale the indicator
    return 2 * (indicator - indicator_middle) / indicator_range


# a help function used to assign a relative position when compared with an upper bound and a lower bound
def relative_position(indicator, lows, highs, normalize=True, clip=False):
    position = np.divide(indicator - lows, highs - lows, out=np.zeros_like(indicator), where=highs - lows != 0)
    if clip:
        # if the position shall be clipped,
        # a move of the indicator outside the range does not increase the position beyond -1 or 1
        position = np.clip(position, 0, 1)
    return normalize_indicator(position, 0, 1) if normalize else position


# this transformation amplifies an indicator, if it moves above an inflection point by applying a logistic function
def transform_logistic(indicator, inflection_point=0.4, base=10**6):
    return np.sign(indicator) / (1 + np.power(base, -np.abs(indicator) + inflection_point))


# this transformation tries to extract only extreme values
def transform_threshold(indicator, threshold):
    # find all positions where the threshold is being surpassed
    threshold_surpassed = np.abs(indicator) > threshold
    # map the indicator to its sign (-1 or 1), if the threshold is surpassed, else 0
    threshold_indicator = np.where(threshold_surpassed, np.sign(indicator), np.zeros_like(indicator))
    # deal with nan values, they should not be mapped to any number, because of the threshold transformation
    return np.where(np.isnan(indicator), indicator, threshold_indicator)
