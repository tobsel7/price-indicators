# data sets are pandas data frames
import pandas as pd

# the data handler provides price data
from charts.api import data_handler

# default values for storage paths and an existing asset_list
from charts.parameters import ASSET_LIST, ASSET_LIST_PATH, SAMPLES_PER_YEAR


# here a dataset of samples is generated by going through all ticker symbols of a list
# and retrieving a number of random samples from the corresponding chart data object
def generate_samples(asset_list=ASSET_LIST, samples_per_year=SAMPLES_PER_YEAR, normalize=True, future_interval=0):
    # initialize an empty array where all samples will be stored
    samples = pd.DataFrame()

    # get all names and symbols from the list
    asset_symbols = list(pd.read_csv(ASSET_LIST_PATH.format(asset_list), usecols=["Ticker"])["Ticker"])

    # go through all symbols
    for symbol in asset_symbols:
        # take only symbols with persisted data
        if data_handler.chart_exists(symbol):
            # gather the chart object
            chart_data = data_handler.get_chart_data(symbol, auto_persist_on_load=False)
            # take only long enough charts to prevent errors
            if chart_data.can_create_samples(future_interval):
                new_samples = chart_data.get_random_samples(normalize=normalize,
                                                            future_interval=future_interval,
                                                            samples_per_year=samples_per_year
                                                            )
                samples = pd.concat([samples, new_samples])

    # return all the gathered samples
    return samples
