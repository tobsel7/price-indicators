# price-indicators
This project is split up into two, in principal, separate applications:
- charts
- analysis

## charts
The charts application is used to **download, process, store price chart data and calculate technical indicators**. With the implemented command line interface, various data sets are generated.

## How to run:
### Option 1 (recommended):
It is recommended to open the whole project (folder price-indicators) in a **modern Python development environment**. The development happened in **PyCharm** but other IDEs should work as well, asuming that they can **configure the python path correctly** so that the individual modules can import functions from each other.

The application is started by running the file *charts/main.py* using a **Python 3** interpreter. 

### Option 2:
The **PYTHONPATH is configured** to be the project directory. This way, the program can be started using the following commands. Note that here an example path is used. Change it to the location of the project on your machine.

export PYTHONPATH=/home/tobias/PycharmProjects/price-indicators

cd price-indicators/charts

python3 main.py

### CLI interaction:
Using keybord inputs, data sets can then be generated interactively. Enter help first, to see the list of available commands.

Here an example cli interaction:

Welcome to this data set tool!

List the commands by typing 'help'.

Exit by pressing -Enter- without any text.

set key SECRET-YAHOO-API-KEY

API key set to SECRET-YAHOO-API-KEY.

download aapl

Downloading price data from ticker aapl.

Successfully downloaded aapl price data.

info aapl

Price data for the asset AAPL

Last price 138.27000427246094

Number of data points 2516

create aapl

Should the data set be normalized?

[Y(es), N(o)]: N

Enter the wanted file format.

Options: csv, feather, hdf, gbq, excel, (-Enter- for default option)

csv

Successfully created a data set using data from aapl

Exiting program.


## analysis
The analysis folder is a **collections of interactive python worksheets**. Using the previously generated data sets, they create visualizations of various aspects from the datasets.

## How to run:
Open the worksheets using jupyter notebook. 

# Storage:
All data is persisted in the folder *persisted_data*. It is split up into file formats, although for the purpose of this project, mainly the *.feather* type is used to minimize the needed storage. Files of the type *.csv* and *.xls* files are stored in *persisted_data/sheets*.

A few preprocesed data sets are available via this link:
https://u.pcloud.link/publink/show?code=kZDdJLVZP91Xe0vdAI5AtxQB69I4whcdBUL7

# Error Handling
If errors arise, it is most likely due to missing libraries (numpy, pandas, PyTorch, sklearn, seaborn, matplotlib, sci-kit etc.) or data sets.

In case of missing libraries, install the respective libaries using commands like:

pip install numpy 

pip install pandas 

etc.

In case of missing data sets, generate them using the **create** command when running the cli application (*charts/main.py*)

Other reason for program crashes are incorrect configurations, wrong current working directory or unset yahoo finance api key.

When running the python tool the current working directory must be **price-indicators/charts**.

The api key can be set dynamically in the cli session using the command **set key SECRET-YAHOO-API-KEY** or by writing it down in the file price-indicators/charts/api/api_key.py




