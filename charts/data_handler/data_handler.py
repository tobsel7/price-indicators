# import basic libraries for interaction with files and json strings
import os
import json
import pandas as pd
from charts.data_handler import price_loader, chart_data

# import custom errors
from charts.data_handler.errors import DelistedError, APILimitError, MalformedResponseError

# default storage path
STORAGE_PATH = "./persisted_data/charts/{}.json"
ASSET_LIST_PATH = "./persisted_data/asset_lists/{}.csv"
ASSET_LIST = "all_stocks"


# internal function for persisting a chart charts dictionary
def _persist_data(chart, meta):
    # define file name
    file_name = STORAGE_PATH.format(meta["symbol"])
    # create json charts
    json_data = json.dumps({"chart": chart, "meta": meta})
    # save json charts to file
    with open(file_name, 'w') as outfile:
        json.dump(json_data, outfile)


# internal function to retrieve persisted chart
def _get_persisted_data(symbol):
    # define file name
    file_name = STORAGE_PATH.format(symbol)
    # get json charts
    with open(file_name) as json_file:
        data = json.loads(json.load(json_file))
        # return new ChartData using loaded charts
        return chart_data.ChartData(data["chart"], data["meta"])


# internal function to normalize a ticker symbol
def _normalize_symbol(symbol):
    # those characters should be changed
    characters_to_change = "/^."
    # this is the character they will be changed to
    share_type_delimiter = "-"
    for character in characters_to_change:
        symbol = symbol.replace(character, share_type_delimiter)
    return symbol


# help function checking chart data has been downloaded already
def persisted_data_exists(symbol):
    return os.path.isfile(STORAGE_PATH.format(symbol))


# top level function to get chart from local storage or external api
def get_chart_data(symbol, auto_persist_on_load=True):
    # local charts exists -> just return charts object
    if persisted_data_exists(symbol):
        return _get_persisted_data(symbol)
    else:
        # no charts exists -> download and return the chart charts
        chart, meta = price_loader.get_price(symbol)
        if auto_persist_on_load:
            # persist the newly loaded charts
            _persist_data(chart, meta)
        return chart_data.ChartData(chart, meta)


# download and persist charts for a list of assets defined in the asset_list csv files
def download_and_persist_chart_data(asset_list=ASSET_LIST):
    # get all names and symbols from the list
    asset_names = pd.read_csv(ASSET_LIST_PATH.format(asset_list), usecols=["Ticker", "Name"])
    # go through all symbols
    for asset_symbol in asset_names["Ticker"]:
        # the lists contain different types of delimiters
        # for example BRK.A shares are listed as BRK.A, BRK-A and BRK/A and should be BRK-A
        asset_symbol = _normalize_symbol(asset_symbol)
        try:
            # download and persist the charts for one symbl
            get_chart_data(asset_symbol, True)
        except MalformedResponseError:
            # no correct data could be retrieved from the api
            continue
        except DelistedError:
            # the asset has been delisted, continue with the next item of the list
            continue
        except APILimitError:
            # the api limit has been reached, stop downloading
            return
        except Exception as error:
            # an unknown error has occured
            # print the error and stop downloading
            print(error)
            return


# generate a specific number of samples for all persisted data
def generate_samples(asset_list=ASSET_LIST, samples_per_chart=20, normalize=False, prediction_interval=365):
    # initialize an empty array where all samples will be stored
    samples = pd.DataFrame()

    # get all names and symbols from the list
    asset_symbols = list(pd.read_csv(ASSET_LIST_PATH.format(asset_list), usecols=["Ticker"])["Name"])

    # go through all symbols
    for symbol in asset_symbols:
        # take only symbols with persisted data
        if persisted_data_exists(symbol):
            # gather the chart object
            chart = get_chart_data(symbol, False)
            # take only long enough charts to prevent errors
            if chart.can_create_samples(prediction_interval):
                # take some samples defined by samples_per_chart
                for i in range(samples_per_chart):
                    # let the chart data object generate a sample
                    sample = chart.get_random_sample(normalize, prediction_interval)
                    # get features from the sample
                    features = sample["features"]
                    # add the future price to the dictionary
                    features["future_price"] = sample["future_price"]
                    # store the sample into the data frame
                    samples = pd.concat([samples, pd.DataFrame([features])], ignore_index=True)

    return samples


# generate a random sample for some stored chart data
def generate_sample(symbol, normalize=False, prediction_interval=365):
    chart = get_chart_data(symbol)
    return chart.get_random_sample(normalize, prediction_interval)
