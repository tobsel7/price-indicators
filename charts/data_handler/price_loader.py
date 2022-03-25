# imports for calling the yahoo api
import requests

# import custom error classes
from charts.data_handler.errors import DelistedError, APILimitError, MalformedResponseError

# static variables necessary for the api
from .api_key import API_KEY
CHART_URL = "https://yfapi.net/v8/finance/chart/"


# basic function returning the price history of one or multiple symbols
def get_price(symbol):
    # append api key to the headers
    headers = {"x-api-key": API_KEY}
    # create query with the symbols
    url = CHART_URL + symbol
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
    if len(response_json["chart"]["result"][0]["indicators"]["quote"]) <= 1:
        raise MalformedResponseError()

    # retrieve relevant charts
    data = response_json["chart"]["result"][0]
    chart = data["indicators"]["quote"][0]
    meta = data["meta"]
    return chart, meta
