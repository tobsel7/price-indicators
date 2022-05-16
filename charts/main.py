# import the chart handler
from charts.api import data_handler
from charts.data_sets import files


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
        if commands[0] == "create" and len(commands) > 1:
            # create data sets
            create(commands[1])
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
          "create default -> Creates a few default data sets.\n"
          "create countries -> Creates a data set with default parameters for some countries with stored stock data."
          "create <stock ticker> -> Creates a data set from one stock ticker\n"
          "create sample <asset list> -> Creates samples from a list of assets\n"
          "create all <asset_list> -> Persists every stock chart with all its indicators.\n"
          "info <stock ticker> -> Displays some basic information about a stock ticker.\n"
          "info <asset list> -> Displays basic information about an asset list.\n"
          "show tickers -> Display all stock tickers which have stored chart data on this machine.\n"
          "show lists -> Display all the defined asset lists.\n"
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
    else:
        print(*files.get_asset_list_names(), sep="\n")


# create data sets from some source defined by its name
def create(source):
    try:
        if source == "default":
            create_default_sets()
        elif source == "countries":
            create_country_sets()
        elif data_handler.chart_exists(source.upper()):
            create_file_from_ticker(source.upper())
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
    # store the IBM data set as a csv and feather
    print("Creating csv and feather files for the IBM chart data.")
    ibm_original = data_handler.get_chart_data("IBM").get_full_data(False)
    ibm_normalized = data_handler.get_chart_data("IBM").get_full_data(True)
    files.persist_data(ibm_original, name="IBM_original", data_format="csv")
    files.persist_data(ibm_normalized, name="IBM_normalized", data_format="csv")
    files.persist_data(ibm_original, name="IBM_original", data_format="feather")
    files.persist_data(ibm_normalized, name="IBM_normalized", data_format="feather")
    print("Created IBM data sets.")

    # store the JNJ data set as a csv and feather
    print("Creating csv and feather files for the JNJ chart data.")
    ibm_original = data_handler.get_chart_data("JNJ").get_full_data(False)
    ibm_normalized = data_handler.get_chart_data("JNJ").get_full_data(True)
    files.persist_data(ibm_original, name="JNJ_original", data_format="csv")
    files.persist_data(ibm_normalized, name="JNJ_normalized", data_format="csv")
    files.persist_data(ibm_original, name="JNJ_original", data_format="feather")
    files.persist_data(ibm_normalized, name="JNJ_normalized", data_format="feather")
    print("Created JNJ files.")

    # create a small data set from the nasdaq stock listings
    print("Creating data set using sampled data from stocks in the nasdaq100 index.")
    files.create_random_data_set("nasdaq100", 10, normalize=True, future_interval=30, data_format="feather")
    print("Created nasdaq100 data set.")


# create data sets for each country with stored stock data
def create_country_sets():
    countries = ["spain", "brazil", "australia", "canada", "ireland", "usa", "germany", "latvia", "france", "denmark",
                 "isreal", "iceland", "switzerland", "finland", "southkorea", "mexiko", "hongkong", "argentina",
                 "italy", "russia", "thailand", "china", "lithuania", "turkey", "taiwan", "austria", "portugal",
                 "india", "greece", "estonia", "singapore", "norway", "newzealand", "belgium", "qatar", "sweden",
                 "uk", "malaysia", "venezuela", "indonesia", "netherlands"]

    # go through all country
    for country in countries:
        print("Creating feather file for the {} data set.".format(country))
        # create a data set with fixed parameters for simplicity
        files.create_random_data_set(country, 10, True, 30, "feather")
        print("Created {} file.".format(country))


# lower level function creating a data set from one single price chart
def create_file_from_ticker(source):
    normalize = bool(input("Should the data set be normalized?"
                           "\n[Y(es), N(o)]: ").lower() == "y")
    data = data_handler.get_chart_data(source).get_full_data(normalize=normalize)
    file_format = input("Enter the wanted file format.\n"
                        "Options: csv, feather, hdf, gbq, excel, (-Enter- for default option)\n").lower()
    if file_format == "":
        file_format = "csv"

    file_name = source + "_normalized" if normalize else source + "_original"

    files.persist_data(data, file_name, file_format)


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


# entry point of the program
if __name__ == '__main__':
    main()



