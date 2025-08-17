from typing import Union

import requests


class UnauthorizedError(Exception):
    """Exception raised when the API returns a 401 Unauthorized response."""

    def __init__(self, orig_exception: Union[requests.exceptions.HTTPError, str]):
        self.orig_exception = orig_exception
        super().__init__(str(orig_exception))


class NotFoundError(Exception):
    """Exception raised when the API returns a 404 Not Found response or indicates a resource was not found."""

    def __init__(self, orig_exception: Union[requests.exceptions.HTTPError, str]):
        self.orig_exception = orig_exception
        super().__init__(str(orig_exception))
