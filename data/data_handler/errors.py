# custom error used to notify the program that the api does not have data for the given symbol
class DelistedError(Exception):
    def __init__(self, symbol, message="The asset data is unavailable, because it has been delisted."):
        # store the symbol which has been delisted in the error class
        self.symbol = symbol
        # define custom error message
        error_message = "An error happened while downloading data for the symbol {}. See detailed error below. \n{}"\
            .format(symbol, message)
        super().__init__(error_message)


# customer error used to notify that no calls can be made to the api due to rate limiting
class APILimitError(Exception):
    def __init__(self, message="The api limit of allowed calls for today has been reached."):
        super().__init__(message)
