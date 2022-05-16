# subdirectories must be created if they do not exist
import os
# storage path
from charts.parameters import STORAGE_PATH, ASSET_LIST, SAMPLES_PER_YEAR
# a generator of samples
from charts.data_sets import generator


# this setup must be called once to ensure all directories exist before creating datasets
def setup():
    for directory in ["sheets", "feather", "hdf", "gbq"]:
        path = STORAGE_PATH.format(directory)
        # create an empty folder if it does not exist
        if not os.path.exists(path):
            os.mkdir(path)


# retrieve all stored asset lists
def get_asset_list_names():
    files = os.listdir(STORAGE_PATH.format("asset_lists"))
    return [file.split(".")[0] for file in files if file.endswith(".csv")]


# retrieve all stored stocks
def get_persisted_stock_names():
    files = os.listdir(STORAGE_PATH.format("api_jsons"))
    return [file.split(".")[0] for file in files if file.endswith(".json")]


# a general function storing a data frame in various data formats
def persist_data(data, name=None, data_format="csv"):
    data = data.reset_index()
    if name is None:
        # set the name of the file to the dataframe name, if no file name is given
        name = data.name
    # store the data frame in a common file type
    if data_format == "csv":
        data.to_csv(STORAGE_PATH.format("sheets/" + name + ".csv"), index=False)
    elif data_format == "feather":
        data.to_feather(STORAGE_PATH.format("feather/" + name + ".feather"))
    elif data_format == "hdf":
        data_format.to_hdf(STORAGE_PATH.format("hdf/" + name + ".hdf"))
    elif data_format == "gbq":
        data.to_gbq(STORAGE_PATH.format("gbq/" + name + ".gbq"))
    elif data_format == "excel":
        data.to_excel(STORAGE_PATH.format("sheets/" + name + ".xls"), index=False)
    else:
        raise Exception("The file format {} is unknown.".format(data_format))


# generate random samples from a list of assets and store them in an efficient format
def create_random_data_set(asset_list=ASSET_LIST, samples_per_year=SAMPLES_PER_YEAR, normalize=True, future_interval=0,
                           data_format="feather"):
    data = generator.generate_samples(asset_list=asset_list, samples_per_year=samples_per_year,
                                      normalize=normalize, future_interval=future_interval)
    normalize_string = "normalized" if normalize else "original"
    name = "{}_{}spy_{}shift_{}".format(asset_list, samples_per_year, future_interval, normalize_string)
    persist_data(data, name, data_format=data_format)
