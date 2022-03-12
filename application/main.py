# import the charts handler
from charts.data_handler import data_handler

DOWNLOAD = False


# main program
def main():
    if DOWNLOAD:
        data_handler.download_and_persist_chart_data()
    test()


# current test code
def test():
    samples = data_handler.generate_samples(samples_per_chart=10, normalize=True)
    samples = samples.sort_values(by='future_price')
    print(sum(samples["future_price"]) / 1490)
    for indicator in ["rsi", "ma20", "ma50", "ma100", "ma200", "ma_trend", "ma_trend_crossing"]:
        correlation = samples[indicator].corr(samples['future_price'])
        print((indicator + " correlation: {}").format(correlation))


# entry point of the program
if __name__ == '__main__':
    main()
