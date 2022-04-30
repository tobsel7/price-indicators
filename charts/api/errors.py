# custom error used to notify the program that the api does not have charts for the given symbol
class DelistedError(Exception):
    def __init__(self, symbol, message="The asset charts is unavailable, because it has been delisted."):
        # store the symbol which has been delisted in the error class
        self.symbol = symbol
        # define custom error message
        error_message = "An error happened while downloading charts for the symbol {}. See detailed error below. \n{}"\
            .format(symbol, message)
        super().__init__(error_message)


# custom error used to notify that no calls can be made to the api due to rate limiting
class APILimitError(Exception):
    def __init__(self, message="The api limit of allowed calls for today has been reached."):
        super().__init__(message)


# custom error signaling some response from the api which can not be processed
class MalformedResponseError(Exception):
    def __init__(self, message="The api response can not be processed, because some fields are missing."):
        super().__init__(message)


# custom error signalling too short price data to create valid samples
class PriceDataLengthError(Exception):
    def __init__(self, message="The length of the known price data is not long enough to create valid samples.\n"
                               "Use a stock ticker with enough price data or decrease the future price interval."):
        super().__init__(message)
