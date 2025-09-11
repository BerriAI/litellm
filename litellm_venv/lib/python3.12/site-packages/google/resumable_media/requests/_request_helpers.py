# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared utilities used by both downloads and uploads.

This utilities are explicitly catered to ``requests``-like transports.
"""

import http.client
import requests.exceptions
import urllib3.exceptions  # type: ignore

import time

from google.resumable_media import common
from google.resumable_media import _helpers

_DEFAULT_RETRY_STRATEGY = common.RetryStrategy()
_SINGLE_GET_CHUNK_SIZE = 8192
# The number of seconds to wait to establish a connection
# (connect() call on socket). Avoid setting this to a multiple of 3 to not
# Align with TCP Retransmission timing. (typically 2.5-3s)
_DEFAULT_CONNECT_TIMEOUT = 61
# The number of seconds to wait between bytes sent from the server.
_DEFAULT_READ_TIMEOUT = 60

_CONNECTION_ERROR_CLASSES = (
    http.client.BadStatusLine,
    http.client.IncompleteRead,
    http.client.ResponseNotReady,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.Timeout,
    urllib3.exceptions.PoolError,
    urllib3.exceptions.ProtocolError,
    urllib3.exceptions.SSLError,
    urllib3.exceptions.TimeoutError,
    ConnectionError,  # Python 3.x only, superclass of ConnectionResetError.
)


class RequestsMixin(object):
    """Mix-in class implementing ``requests``-specific behavior.

    These are methods that are more general purpose, with implementations
    specific to the types defined in ``requests``.
    """

    @staticmethod
    def _get_status_code(response):
        """Access the status code from an HTTP response.

        Args:
            response (~requests.Response): The HTTP response object.

        Returns:
            int: The status code.
        """
        return response.status_code

    @staticmethod
    def _get_headers(response):
        """Access the headers from an HTTP response.

        Args:
            response (~requests.Response): The HTTP response object.

        Returns:
            ~requests.structures.CaseInsensitiveDict: The header mapping (keys
            are case-insensitive).
        """
        return response.headers

    @staticmethod
    def _get_body(response):
        """Access the response body from an HTTP response.

        Args:
            response (~requests.Response): The HTTP response object.

        Returns:
            bytes: The body of the ``response``.
        """
        return response.content


class RawRequestsMixin(RequestsMixin):
    @staticmethod
    def _get_body(response):
        """Access the response body from an HTTP response.

        Args:
            response (~requests.Response): The HTTP response object.

        Returns:
            bytes: The body of the ``response``.
        """
        if response._content is False:
            response._content = b"".join(
                response.raw.stream(_SINGLE_GET_CHUNK_SIZE, decode_content=False)
            )
            response._content_consumed = True
        return response._content


def wait_and_retry(func, get_status_code, retry_strategy):
    """Attempts to retry a call to ``func`` until success.

    Expects ``func`` to return an HTTP response and uses ``get_status_code``
    to check if the response is retry-able.

    ``func`` is expected to raise a failure status code as a
    common.InvalidResponse, at which point this method will check the code
    against the common.RETRIABLE list of retriable status codes.

    Will retry until :meth:`~.RetryStrategy.retry_allowed` (on the current
    ``retry_strategy``) returns :data:`False`. Uses
    :func:`_helpers.calculate_retry_wait` to double the wait time (with jitter)
    after each attempt.

    Args:
        func (Callable): A callable that takes no arguments and produces
            an HTTP response which will be checked as retry-able.
        get_status_code (Callable[Any, int]): Helper to get a status code
            from a response.
        retry_strategy (~google.resumable_media.common.RetryStrategy): The
            strategy to use if the request fails and must be retried.

    Returns:
        object: The return value of ``func``.
    """
    total_sleep = 0.0
    num_retries = 0
    # base_wait will be multiplied by the multiplier on the first retry.
    base_wait = float(retry_strategy.initial_delay) / retry_strategy.multiplier

    # Set the retriable_exception_type if possible. We expect requests to be
    # present here and the transport to be using requests.exceptions errors,
    # but due to loose coupling with the transport layer we can't guarantee it.

    while True:  # return on success or when retries exhausted.
        error = None
        try:
            response = func()
        except _CONNECTION_ERROR_CLASSES as e:
            error = e  # Fall through to retry, if there are retries left.
        except common.InvalidResponse as e:
            # An InvalidResponse is only retriable if its status code matches.
            # The `process_response()` method on a Download or Upload method
            # will convert the status code into an exception.
            if get_status_code(e.response) in common.RETRYABLE:
                error = e  # Fall through to retry, if there are retries left.
            else:
                raise  # If the status code is not retriable, raise w/o retry.
        else:
            return response

        base_wait, wait_time = _helpers.calculate_retry_wait(
            base_wait, retry_strategy.max_sleep, retry_strategy.multiplier
        )
        num_retries += 1
        total_sleep += wait_time

        # Check if (another) retry is allowed. If retries are exhausted and
        # no acceptable response was received, raise the retriable error.
        if not retry_strategy.retry_allowed(total_sleep, num_retries):
            raise error

        time.sleep(wait_time)
