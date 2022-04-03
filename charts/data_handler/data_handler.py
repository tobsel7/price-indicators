# import pandas to deal with generated data sets
import pandas as pd

# import functions for loading data from an external api and storing it in a chart object
from charts.data_handler import price_loader, chart_data

# import custom errors
from charts.data_handler.errors import DelistedError, APILimitError, MalformedResponseError

# a list of asset names is used to gather stock data for many companies
ASSET_LIST_PATH = "./persisted_data/asset_lists/{}.csv"
ASSET_LIST = "all_stocks"


# top level function to get chart data for any given symbol
# this function will try to get the data from a locally stored file first and download data if necessary
def get_chart_data(symbol, auto_persist_on_load=True):
    # local charts exists -> just return charts object
    if price_loader.persisted_data_exists(symbol):
        return price_loader.get_persisted_data(symbol)
    else:
        # no charts exists -> download and return the chart charts
        chart, meta = price_loader.get_price(symbol)
        if auto_persist_on_load:
            # persist the newly loaded charts
            price_loader.persist_data(chart, meta)
        return chart_data.ChartData(chart, meta)


# download and persist charts for a list of assets defined in the asset_list csv file
def download_and_persist_chart_data(asset_list=ASSET_LIST, show_downloads=False):
    # get all names and symbols from the list
    asset_names = pd.read_csv(ASSET_LIST_PATH.format(asset_list), usecols=["Ticker", "Name"])

    list_changed = False
    # go through all symbols
    for asset_symbol in asset_names["Ticker"]:
        if not price_loader.persisted_data_exists(asset_symbol):
            try:
                # download and persist the charts for one symbol
                get_chart_data(asset_symbol, True)
                if show_downloads:
                    print("Successfully downloaded data for symbol {}.".format(asset_symbol))
            except MalformedResponseError:
                # no correct data could be retrieved from the api
                asset_names.drop(asset_names.index[(asset_names["Ticker"] == asset_symbol)], inplace=True)
                list_changed = True
                continue
            except DelistedError:
                # the asset has been delisted, continue with the next item of the list
                asset_names.drop(asset_names.index[(asset_names["Ticker"] == asset_symbol)], inplace=True)
                list_changed = True
                continue
            except APILimitError:
                # the api limit has been reached, stop downloading
                if list_changed:
                    asset_names.to_csv(ASSET_LIST_PATH.format(asset_list))
                return
            except Exception as error:
                # an unknown error has occurred
                # print the error and stop downloading
                print(error)
                return


def generate_samples(asset_list=ASSET_LIST, samples_per_year=20, normalize=True, prediction_interval=365):
    # initialize an empty array where all samples will be stored
    samples = pd.DataFrame()

    # get all names and symbols from the list
    asset_symbols = list(pd.read_csv(ASSET_LIST_PATH.format(asset_list), usecols=["Ticker"])["Ticker"])

    # go through all symbols
    for symbol in asset_symbols:
        # take only symbols with persisted data
        if price_loader.persisted_data_exists(symbol):
            # gather the chart object
            chart = get_chart_data(symbol, False)
            # take only long enough charts to prevent errors
            if chart.can_create_samples(prediction_interval):
                number_of_samples_gatherable = int(len(chart) / 365.0 * samples_per_year)
                # take some samples defined by samples_per_chart
                for i in range(number_of_samples_gatherable):
                    # let the chart data object generate a sample
                    sample = chart.get_random_sample(normalize, prediction_interval)
                    # get features from the sample
                    features = sample["features"]
                    # add the future price to the dictionary
                    features["future_price"] = sample["future_price"]
                    # store the sample into the data frame
                    samples = pd.concat([samples, pd.DataFrame([features])], ignore_index=True)

    # return all the gathered samples
    return samples


# generate a random sample for some stored chart data
def generate_sample(symbol, normalize=True, prediction_interval=365):
    chart = get_chart_data(symbol)
    return chart.get_random_sample(normalize, prediction_interval)
