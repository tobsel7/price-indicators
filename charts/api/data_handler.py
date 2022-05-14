# import pandas to deal with generated data sets
import pandas as pd

# import functions for loading data from an external api and storing it in a chart object
from charts.api import price_loader, chart

# import custom errors
from charts.api.errors import DelistedError, APILimitError, MalformedResponseError

# import default variables
from charts.parameters import ASSET_LIST, ASSET_LIST_PATH


# function used to check whether some chart data exists locally for a given ticker symbol
def chart_exists(symbol):
    return price_loader.persisted_data_exists(symbol)


# top level function to get chart dta for any given symbol
# this function will try to get the data from a locally stored file first and download data if necessary
def get_chart_data(symbol, auto_persist_on_load=True):
    # local charts exists -> just return charts object
    if price_loader.persisted_data_exists(symbol):
        return price_loader.get_persisted_data(symbol)
    else:
        # no chart exists -> download and return the chart data
        chart_data, meta = price_loader.get_price(symbol)
        if auto_persist_on_load:
            # persist the newly loaded charts
            price_loader.persist_data(chart_data, meta)
        return chart.Chart(chart_data, meta)


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
