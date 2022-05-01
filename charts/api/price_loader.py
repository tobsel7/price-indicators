# imports for calling the yahoo api
import requests

# import custom error classes
from charts.api.errors import DelistedError, APILimitError, MalformedResponseError

# import functionalities to store and process the loaded data
import json
import os
from charts.api import chart

# static variables necessary for the api
# API_KEY = "SECRET-API-KEY"
from charts.api.api_key import API_KEY

# default storage path and api url
from charts.config import API_JSON_PATH, API_URL


# if a stock has no recent data, none values will occur. they should be deleted before performing any analysis
def _clean_nones(value):
    """
    Taken from answer by user MatanRubin
    https://stackoverflow.com/questions/4255400/exclude-empty-null-values-from-json-serialization
    Recursively remove all None values from dictionaries and lists, and returns
    the result as a new dictionary or list.
    """
    if isinstance(value, list):
        return [_clean_nones(x) for x in value if x is not None]
    elif isinstance(value, dict):
        return {
            key: _clean_nones(val)
            for key, val in value.items()
            if val is not None
        }
    else:
        return value


# function for persisting a chart data dictionary
def persist_data(chart_data, meta):
    # define file name
    file_name = API_JSON_PATH.format(meta["symbol"])
    # create json charts
    json_data = json.dumps({"chart": chart_data, "meta": meta})
    # save json charts to file
    with open(file_name, 'w') as outfile:
        json.dump(json_data, outfile)


# function to retrieve persisted chart
def get_persisted_data(symbol):
    # define file name
    file_name = API_JSON_PATH.format(symbol)
    # get json charts
    with open(file_name) as json_file:
        data = json.loads(json.load(json_file))
        # return new ChartData using loaded charts
        return chart.Chart(data["chart"], data["meta"])


# help function checking chart data has been downloaded already
def persisted_data_exists(symbol):
    return os.path.isfile(API_JSON_PATH.format(symbol))


# basic function returning the price history of one or multiple symbols
def get_price(symbol):
    # append api key to the headers
    headers = {"x-api-key": API_KEY}
    # create query with the symbols
    url = API_URL + symbol
    # define params
    params = {
        "range": "10y",
        "interval": "1d"
    }
    # call api and retrieve response
    response = requests.request("GET", url, headers=headers, params=params)
    # check response
    if not response.ok:
        if response.status_code == 429:
            # the limit of daily errors has been reached
            raise APILimitError(symbol)
        else:
            # unknown error
            raise Exception("Could not load the charts from the api.")

    # get json
    response_json = response.json()

    # react to error messages from the api
    if response_json["chart"]["error"] is not None:
        if "delisted" in response_json["chart"]["error"]["description"]:
            # the asset has been delisted, so no recent chart charts exists
            raise DelistedError(response_json["chart"]["error"]["description"])
        else:
            # unknown error
            raise Exception(response_json["chart"]["error"]["description"])

    # react to malformed responses
    if "open" not in response_json["chart"]["result"][0]["indicators"]["quote"][0] \
            or len(response_json["chart"]["result"][0]["indicators"]["quote"][0]["open"]) <= 1:
        raise MalformedResponseError()

    # delete None values
    response_json = _clean_nones(response_json)

    # retrieve relevant data
    data = response_json["chart"]["result"][0]
    chart_data = data["indicators"]["quote"][0]
    meta = data["meta"]
    return chart_data, meta
