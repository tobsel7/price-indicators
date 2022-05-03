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
    print("Commands:\n"
          "create default -> Creates a few default data sets."
          "create <asset list> -> Creates a data set from one stock ticker\n"
          "create <stock ticker> -> Creates a data set using an existing list of stock tickers.\n"
          "info <stock ticker> -> Displays some basic information about a stock ticker.\n"
          "Storage:\n"
          "Generated files are stored in the folder persisted_data.")


# display basic information about some stock data
def info(ticker):
    print(data_handler.get_chart_data(ticker))


# create data sets from some source defined by its name
def create(source):
    #try:
        if source == "default":
            create_default_sets()
        elif data_handler.chart_exists(source.upper()):
            create_file_from_ticker(source.upper())
        else:
            create_file_from_asset_list(source.lower())

        print("Successfully created a data set using data from {}".format(source))
    #except Exception as error:
        # an unknown error has occurred
        # print the error and stop downloading
        #print(error)
        #return


# for simplicity this function can be called
# it creates a few data sets and does not require any user input/configuration
def create_default_sets():
    files.persist_data(data_handler.get_chart_data("IBM").get_full_data(False), name="IBM_original", data_format="csv")
    files.persist_data(data_handler.get_chart_data("IBM").get_full_data(True), name="IBM_normalized", data_format="csv")
    print("Created IBM data sets.")
    files.persist_data(data_handler.get_chart_data("JNJ").get_full_data(False), name="JNJ_original", data_format="csv")
    files.persist_data(data_handler.get_chart_data("JNJ").get_full_data(True), name="JNJ_normalized", data_format="csv")
    print("Created JNJ data sets.")
    print("Created data set from stocks listed on the NASDAQ exchange. This will take some time...")
    files.create_random_data_set("nasdaq", 1, True, 50, "feather")


# lower level function creating a data set from one single price chart
def create_file_from_ticker(source):
    normalize = bool(input("Should the data set be normalized?"
                           "\n[Y(es), N(o)]: ").lower() == "y")
    data = data_handler.get_chart_data(source).get_full_data(normalize=normalize)
    file_format = input("Enter the wanted file format.\n"
                        "Options: csv, feather, hdf, gbq, excel\n").lower()
    file_name = source + "_normalized" if normalize else source + "_original"

    files.persist_data(data, file_name, file_format)


# creates a data set from a list of assets by sampling from all assets and merging all the samples
# normalization is done per default
def create_file_from_asset_list(source):
    samples_per_year = int(input("Enter the samples per year taking from each chart in this asset list.\n"))
    future_price_input = input("Enter the time interval between the current and future price in days.\n").split(" ")
    future_price_intervals = list(map(int, future_price_input))
    for interval in future_price_intervals:
        print("Creating data set for interval {}.\nPlease Wait...".format(interval))
        files.create_random_data_set(source, samples_per_year=samples_per_year, normalize=True,
                                     future_interval=interval)


# entry point of the program
if __name__ == '__main__':
    main()



