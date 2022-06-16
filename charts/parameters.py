# abstract path for all persisted data
STORAGE_PATH = "./persisted_data/{}"

# persisted api jsons
API_JSON_PATH = STORAGE_PATH.format("api_jsons/{}.json")

# API URL
API_URL = "https://yfapi.net/v8/finance/chart/"

# a list of asset names is used to gather stock data for many companies
ASSET_LIST_PATH = STORAGE_PATH.format("asset_lists/{}.csv")
ASSET_LIST = "all_stocks"

# a list of country names
COUNTRIES = ["spain", "brazil", "australia", "canada", "ireland", "usa", "germany", "latvia", "france", "denmark",
             "isreal", "iceland", "switzerland", "finland", "southkorea", "mexiko", "hongkong", "argentina",
             "italy", "russia", "thailand", "china", "lithuania", "turkey", "taiwan", "austria", "portugal",
             "india", "greece", "estonia", "singapore", "norway", "newzealand", "belgium", "qatar", "sweden",
             "uk", "malaysia", "venezuela", "indonesia", "netherlands"]

# some tickers used for demonstrations
DEFAULT_TICKERS = ["IBM", "AAPL", "VOE.VI"]

# the default future_price interval
FUTURE_INTERVAL = 365

# the number of days being trading in a year
TRADING_DAYS_PER_YEAR = 253

# default samples per year taken from some chart data
SAMPLES_PER_YEAR = 10

# default parameters for all indicators
# volatility intervals
VOLATILITY_INTERVALS = [10, 20, 50, 100, 200]

# common moving average intervals
MOVING_AVERAGE_INTERVALS = [10, 20, 50, 100, 200]
# moving averages that are compared with each other
MOVING_AVERAGE_TREND_PARAMETERS = {
    # short_ma: long_ma
    20: 50,
    50: 200
}

# exponential moving averages that are compared in the macd indicator
MACD_PARAMETERS = {
    # short_ma: long_ma
    12: 26
}
# the macd indicator is smoothed to get its signal line
MACD_SIGNAL_LINE_INTERVAL = 9

# the smoothing parameter used for calculating the weights in the ema function
EXPONENTIAL_MOVING_AVERAGE_SMOOTHING = 2

# common intervals for constructing bollinger bands
BOLLINGER_BAND_PARAMETERS = {
    # interval: deviations
    20: 2,
    50: 2,
    100: 2,
    200: 2,
}

# relative strength intervals
RSI_INTERVALS = [4, 7, 14, 20]
# custom values for the logistic indicator transformation
RSI_LOGISTIC_TRANSFORMATION_BASE = 10 ** 6
RSI_LOGISTIC_TRANSFORMATION_INFLECTION_POINT = 0.4

# average direction movement intervals
AVERAGE_DIRECTIONAL_MOVEMENT_INTERVALS = [7, 14, 30]

# intervals for the aaron indicators
AARON_INTERVALS = [15, 25, 40]

# time intervals used when calculating the commodity channel index
COMMODITY_CHANNEL_INTERVALS = [20, 50, 100]

# time intervals for constructing linear trend channels
TREND_CHANNEL_INTERVALS = [20, 50, 100, 200]

# time intervals for chande momentum indicator
CHANDE_MOMENTUM_INTERVALS = [20, 50, 100]

# time interval for the rate of chance indicator
RATE_OF_CHANGE_INTERVALS = [20, 50, 100]
