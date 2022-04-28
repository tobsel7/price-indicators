import pandas as pd

# storage path
STORAGE_PATH = "./persisted_data/{}"


# a general function storing a data frame in various data formats
def _persist_data(data, file_name=None, data_format="csv"):
    if file_name is None:
        # set the name of the file to the dataframe name, if no file name is given
        file_name = data.name

    # store the data frame in a common data type
    if data_format == "csv":
        data.to_csv(STORAGE_PATH.format("sheets/" + file_name + ".csv"))
    elif data_format == "feather":
        data.to_feather(STORAGE_PATH.format("feather/" + file_name + ".feather"))
    elif data_format == "hdf":
        data_format.to_hdf(STORAGE_PATH.format("hdf/" + file_name + ".hdf"))
    elif data_format == "gbq":
        data.to_gbq(STORAGE_PATH.format("gbq/" + file_name + ".gbq"))
    elif data_format == "excel":
        data.to_excel(STORAGE_PATH.format("sheets/" + file_name + ".xls"))
    else:
        raise Exception("The file format {} is unknown.".format(data_format))


# chart data is stored as a .csv per default
def chart_to_sheet(chart, normalize=False):
    # get the chart data including the indicators
    full_data = chart.get_full_data(normalize=normalize)

    # store the data
    name = chart.get_name()
    file_name = name + "_normalized" if normalize else name + "_original"
    full_data.to_csv(STORAGE_PATH.format("sheets/" + file_name + ".csv"))


# any pandas data frame is stored as a feather file per default
def data_set_to_feather(data, name=None):
    _persist_data(data, file_name=name, data_format="feather")


def get_data_set(name):
    return pd.read_feather(STORAGE_PATH.format("data_sets/" + name + ".feather"))
