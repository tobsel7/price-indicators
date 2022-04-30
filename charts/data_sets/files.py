# storage path
from charts.config import STORAGE_PATH, ASSET_LIST
# a generator of samples
from charts.data_sets import generator


# a general function storing a data frame in various data formats
def persist_data(data, name=None, data_format="csv"):
    data = data.reset_index()
    if name is None:
        # set the name of the file to the dataframe name, if no file name is given
        name = data.name
    # store the data frame in a common file type
    if data_format == "csv":
        data.to_csv(STORAGE_PATH.format("sheets/" + name + ".csv"))
    elif data_format == "feather":
        data.to_feather(STORAGE_PATH.format("feather/" + name + ".feather"))
    elif data_format == "hdf":
        data_format.to_hdf(STORAGE_PATH.format("hdf/" + name + ".hdf"))
    elif data_format == "gbq":
        data.to_gbq(STORAGE_PATH.format("gbq/" + name + ".gbq"))
    elif data_format == "excel":
        data.to_excel(STORAGE_PATH.format("sheets/" + name + ".xls"))
    else:
        raise Exception("The file format {} is unknown.".format(data_format))


# chart data is stored as a .csv per default
def chart_to_sheet(chart, normalize=False):
    # get the chart data including the indicators
    full_data = chart.get_full_data(normalize=normalize)

    # store the data
    name = chart.get_name()
    file_name = name + "_normalized" if normalize else name + "_original"
    persist_data(full_data, name=file_name, data_format="csv")


# any pandas data frame is stored as a feather file per default
def data_set_to_feather(data, name=None):
    persist_data(data, name=name, data_format="feather")


# generate random samples from a list of assets and store them in an efficient format
def create_random_data_set(asset_list=ASSET_LIST, samples_per_year=20, normalize=True, future_price_interval=0):
    data = generator.generate_samples(asset_list=asset_list, samples_per_year=samples_per_year,
                                      normalize=normalize, future_price_interval=future_price_interval)
    normalize_string = "normalized" if normalize else "original"
    data_set_to_feather(data, "{}{}spy_{}shift_{}".format(asset_list, samples_per_year,
                                                          future_price_interval,
                                                          normalize_string))
