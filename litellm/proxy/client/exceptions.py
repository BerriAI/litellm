from typing import Union

import requests

from litellm.litellm_core_utils.secret_redaction import redact_string


def _redact_orig_exception(
    orig_exception: Union[requests.exceptions.HTTPError, str],
) -> Union[requests.exceptions.HTTPError, str]:
    if isinstance(orig_exception, requests.exceptions.HTTPError):
        return requests.exceptions.HTTPError(redact_string(str(orig_exception)), response=orig_exception.response)
    return redact_string(str(orig_exception))


class UnauthorizedError(Exception):
    """Exception raised when the API returns a 401 Unauthorized response."""

    def __init__(self, orig_exception: Union[requests.exceptions.HTTPError, str]):
        self.orig_exception = _redact_orig_exception(orig_exception)
        super().__init__(str(self.orig_exception))


class NotFoundError(Exception):
    """Exception raised when the API returns a 404 Not Found response or indicates a resource was not found."""

    def __init__(self, orig_exception: Union[requests.exceptions.HTTPError, str]):
        self.orig_exception = _redact_orig_exception(orig_exception)
        super().__init__(str(self.orig_exception))
