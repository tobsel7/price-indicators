# import the data handler
from data.data_handler import data_handler


# main program
def main():
    test()


# current test code
def test():
    result = data_handler.get_chart_data("ADM")
    print(result)
    sample = data_handler.get_chart_data("IBM").get_random_sample(True, 10)
    print(sample)
    data_handler.download_and_persist_chart_data()


# entry point of the program
if __name__ == '__main__':
    test()
