# import the chart handler
from charts.api import data_handler
from charts.data_sets import files

# import the api key file in order to change it on demand
import charts.api.api_key as api_key

# countries with existing asset lists and some default ticker symbols for demonstrations
from charts.parameters import COUNTRIES, DEFAULT_TICKERS


# main cli program used to generate data sets
def main():
    print("Welcome to this data set tool!\n"
          "List the commands by typing 'help'.\n"
          "Exit by pressing -Enter- without any text.")

    # set up missing folders before creating any data sets
    files.setup()

    # define commands which are not empty
    commands = ["start"]
    while len(commands[0]) > 0:
        # get the commands separated by a space
        commands = input().split(" ")
        # filter the commands using the first word
        if commands[0] == "download" and len(commands) > 1:
            download(commands[1])
        elif commands[0] == "set" and len(commands) > 2 and commands[1] == "key":
            set_api_key(commands[2])
        elif commands[0] == "create" and len(commands) > 1:
            source = commands[1]
            parameter = commands[2] if len(commands) > 2 else ""
            # create data sets
            create(source, parameter)
        elif commands[0] == "info" and len(commands) > 1:
            info(commands[1])
        elif commands[0] == "show" and len(commands) > 1:
            show(commands[1])
        elif commands[0] == "help":
            # display help
            help_text()
        elif commands[0] == "" or commands[0] == "exit":
            print("Exiting program.")
            break
        else:
            print("Unknown command. Try again!")


# display the implemented commands
def help_text():
    print("--Commands:\n"
          "set key <API key> -> Change the used API key by providing a new one.\n"
          "download <stock ticker> -> Download and persist price data for one ticker symbol from yahoo finance.\n"
          "download <asset list> -> Download and persist price data from multiple assets from yahoo finance.\n"
          "create default -> Creates a few default data sets.\n"
          "create countries -> Creates a data set with default parameters for some countries with stored stock data.\n"
          "create <stock ticker> -> Creates a data set from one stock ticker\n"
          "create <asset list> -> Creates samples from a list of assets\n"
          "create <asset_list> all -> Persists every stock chart with all its indicators found in a list of assets.\n"
          "info <stock ticker> -> Displays some basic information about a stock ticker.\n"
          "info <asset list> -> Displays basic information about an asset list.\n"
          "show tickers -> Display all stock tickers which have stored chart data on this machine.\n"
          "show lists -> Display all the defined asset lists.\n"
          "show key -> Display the currently used API key.\n"
          "\n--Storage:\n"
          "Generated files are stored in the folder persisted_data split up according to the chosen data type.")


# display basic information about some stock data or asset list
def info(source):
    if data_handler.chart_exists(source.upper()):
        print(data_handler.get_chart_data(source))
    else:
        asset_list = data_handler.get_asset_list(source)
        print("Set of assets {}\nNumber of tickers: {}".format(source, len(asset_list)))
        # optionally search for a stock ticker in the list
        search = input("Enter the name of an asset, if you want to check whether this list contains the ticker."
                       "\n Note that the search is case-sensitive."
                       "\nTo return to creating data sets press -Enter-.\n")

        while len(search) > 0:
            matches = asset_list[asset_list["Name"].str.contains(search)]
            print(matches)
            search = input("Another search? To return press -Enter-.\n")


# show all persisted stock tickers or asset lists
def show(selection):
    if selection == "tickers":
        print(*files.get_persisted_stock_names(), sep="\n")
    elif selection == "lists":
        print(*files.get_asset_list_names(), sep="\n")
    elif selection == "key":
        print("API key: {}".format(api_key.API_KEY))


# download price data
def download(source):
    if source in files.get_asset_list_names():
        print("Downloading data for all tickers from found in the list {}.".format(source))
        data_handler.download_and_persist_chart_data(source, show_downloads=True)
    else:
        print("Downloading price data from ticker {}.".format(source))
        data_handler.get_chart_data(source, auto_persist_on_load=True)
        print("Successfully downloaded {} price data.".format(source))


# change the used api key
def set_api_key(key):
    api_key.API_KEY = key
    print("API key set to {}.".format(api_key.API_KEY))


# create data sets from some source defined by its name
def create(source, parameter=None):
    try:
        if source == "default":
            create_default_sets()
        elif source == "countries":
            create_country_sets()
        elif data_handler.chart_exists(source.upper()):
            create_file_from_ticker(source.upper())
        else:
            if parameter == "all":
                create_all_from_asset_list(source.lower())
            elif parameter == "prices":
                create_all_prices_from_asset_list(source.lower())
            else:
                create_file_from_asset_list(source.lower())

        print("Successfully created a data set using data from {}".format(source))
    except Exception as error:
        # an unknown error has occurred
        # print the error and stop downloading
        print(error)
        return


# for simplicity this function can be called
# it creates a few data sets and does not require any user input/configuration
def create_default_sets():
    for ticker in DEFAULT_TICKERS:
        print("Creating csv files from the {} chart data.".format(ticker))
        files.create_file_from_ticker(ticker, normalize=True, data_format="csv")
        files.create_file_from_ticker(ticker, normalize=True, data_format="feather")
        files.create_file_from_ticker(ticker, normalize=False, data_format="csv")
        files.create_file_from_ticker(ticker, normalize=False, data_format="feather")
        print("Finished creating {} data sets.".format(ticker))

    # create a small data set from the nasdaq stock listings
    print("Creating data set using sampled data from stocks in the nasdaq100 index.")
    files.create_random_data_set("nasdaq100", 10, normalize=True, future_interval=30, data_format="feather")
    files.create_random_data_set("nasdaq100", 30, normalize=False, future_interval=30, data_format="feather")
    print("Created nasdaq100 data set.")


# create data sets for each country with stored stock data
def create_country_sets():
    future_price_input = int(input("Enter the time interval between the current and future price in days.\n"))
    # go through all country
    for country in COUNTRIES:
        print("Creating feather file for the {} data set.".format(country))
        # create a data set with fixed parameters for simplicity
        files.create_random_data_set(country, 10, True, future_price_input, "feather")
        print("Created {} file.".format(country))


# lower level function creating a data set from one single price chart
def create_file_from_ticker(source):
    normalize = bool(input("Should the data set be normalized?"
                           "\n[Y(es), N(o)]: ").lower() == "y")
    file_format = input("Enter the wanted file format.\n"
                        "Options: csv, feather, hdf, gbq, excel, (-Enter- for default option)\n").lower()
    if file_format == "":
        file_format = "csv"

    files.create_file_from_ticker(source, normalize=normalize, data_format=file_format)


# creates a data set from a list of assets by sampling from all assets and merging all the samples'
# normalization is done per default
def create_file_from_asset_list(source):
    samples_per_year = int(input("Enter the samples per year taking from each chart in this asset list.\n"))
    future_price_input = input("Enter the time interval between the current and future price in days.\n").split(" ")
    future_price_intervals = list(map(int, future_price_input))
    file_format = input("Enter the wanted file format.\n"
                        "Options: csv, feather, hdf, gbq, excel, (-Enter- for default option)\n").lower()
    if file_format == "":
        file_format = "feather"

    for interval in future_price_intervals:
        print("Creating data set for interval {}.\nPlease Wait...".format(interval))
        files.create_random_data_set(source, samples_per_year=samples_per_year, normalize=True,
                                     future_interval=interval, data_format=file_format)


# accumulate all prices into one data set
def create_all_prices_from_asset_list(source):
    files.create_price_data_from_all(source)


# creates a data set from each asset stored in a list of assets
def create_all_from_asset_list(source):
    normalize = bool(input("Should the data sets be normalized?"
                           "\n[Y(es), N(o)]: ").lower() == "y")

    file_format = input("Enter the wanted file format.\n"
                        "Options: csv, feather, hdf, gbq, excel, (-Enter- for default option)\n").lower()

    if file_format == "":
        file_format = "feather"

    for ticker in data_handler.get_asset_list(source)["Ticker"]:
        print("Creating data set from {}.".format(ticker))
        files.create_file_from_ticker(ticker, normalize, data_format=file_format)


# entry point of the program
if __name__ == '__main__':
    main()



