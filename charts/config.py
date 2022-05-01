# abstract path for all persisted data
STORAGE_PATH = "./persisted_data/{}"

# persisted api jsons
API_JSON_PATH = STORAGE_PATH.format("api_jsons/{}.json")

# API URL
API_URL = "https://yfapi.net/v8/finance/chart/"
# a list of asset names is used to gather stock data for many companies
ASSET_LIST_PATH = STORAGE_PATH.format("asset_lists/{}.csv")
ASSET_LIST = "all_stocks"

# the default future_price interval
FUTURE_INTERVAL = 365

