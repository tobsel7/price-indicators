# import basic libraries for interaction with files and json strings
import os
import json
import pandas as pd
from data.data_handler import chart_data, price_loader

# import custom errors
from data.data_handler.errors import DelistedError, APILimitError

# default storage path
STORAGE_PATH = "./chart_storage/{}.json"
ASSET_LIST_PATH = "./chart_storage/asset_lists/{}.csv"
ASSET_LISTS = ["sp500"]

# internal function for persisting a chart data dictionary
def _persist_data(chart, meta):
    # define file name
    file_name = STORAGE_PATH.format(meta["symbol"])
    # create json data
    json_data = json.dumps({"chart": chart, "meta": meta})
    # save json data to file
    with open(file_name, 'w') as outfile:
        json.dump(json_data, outfile)


# internal function to retrieve persisted chart data
def _get_persisted_data(symbol):
    # define file name
    file_name = STORAGE_PATH.format(symbol)
    # get json data
    with open(file_name) as json_file:
        data = json.loads(json.load(json_file))
        # return new ChartData using loaded data
        return chart_data.ChartData(data["chart"], data["meta"])


# top level function to get chart data from local storage or external api
def get_chart_data(symbol, auto_persist_on_load=True):
    # local data exists -> just return data object
    if os.path.isfile(STORAGE_PATH.format(symbol)):
        return _get_persisted_data(symbol)
    else:
        # no data exists -> download and return the chart data
        chart, meta = price_loader.get_price(symbol)
        if auto_persist_on_load:
            # persist the newly loaded data
            _persist_data(chart, meta)
        return chart_data.ChartData(chart, meta)


# download and persist chart data for a list of assets defined in the asset_list csv files
def download_and_persist_chart_data(asset_lists=ASSET_LISTS):
    # go through all asset lists
    for asset_list in asset_lists:
        # get all names and symbols from the list
        asset_names = pd.read_csv(ASSET_LIST_PATH.format(asset_list), usecols=["Symbol", "Name"])
        # go through all symbols
        for asset_symbol in asset_names["Symbol"]:
            try:
                # download and persist the data for one symbl
                get_chart_data(asset_symbol, True)
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

