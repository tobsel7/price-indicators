# import the chart handler
from charts.api import data_handler
from charts.data_sets import files


# display the implemented commands
def help_text():
    print("Commands:\n"
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
    try:
        if data_handler.chart_exists(source.upper()):
            source = source.upper()
            normalize = bool(input("Should the data set be normalized?"
                                   "\n[Y(es), N(o)]: ").lower() == "y")
            data = data_handler.get_chart_data(source).get_full_data(normalize=normalize)
            file_format = input("Enter the wanted file format.\n"
                                "Options: csv, feather, hdf, gbq, xls\n").lower()
            file_name = source + "_normalized" if normalize else source + "_original"

            files.persist_data(data, file_name, file_format)
        else:
            source = source.lower()
            samples_per_year = int(input("Enter the samples per year taking from each chart in this asset list.\n"))
            normalize = bool(input("Should the data set be normalized?"
                                   "\n[Y(es), N(o)]: ").lower() == "y")
            future_price_interval = int(input("Enter the time interval between the current and future price "
                                              "in days.\n"))
            print("Processing...")
            files.create_random_data_set(source, samples_per_year=samples_per_year, normalize=normalize,
                                         future_price_interval=future_price_interval)
        print("Successfully created a data set using data from {}".format(source))
    except Exception as error:
        # an unknown error has occurred
        # print the error and stop downloading
        print(error)
        return


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


# entry point of the program
if __name__ == '__main__':
    main()



