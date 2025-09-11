# Copyright 2020 Google Inc.
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

"""Shared utilities used by both downloads and uploads."""

import logging
import random
import time


from google.resumable_media import common


RANGE_HEADER = "range"
CONTENT_RANGE_HEADER = "content-range"

_SLOW_CRC32C_WARNING = (
    "Currently using crcmod in pure python form. This is a slow "
    "implementation. Python 3 has a faster implementation, `google-crc32c`, "
    "which will be used if it is installed."
)
_HASH_HEADER = "x-goog-hash"
_MISSING_CHECKSUM = """\
No {checksum_type} checksum was returned from the service while downloading {}
(which happens for composite objects), so client-side content integrity
checking is not being performed."""
_LOGGER = logging.getLogger(__name__)


def do_nothing():
    """Simple default callback."""


def header_required(response, name, get_headers, callback=do_nothing):
    """Checks that a specific header is in a headers dictionary.

    Args:
        response (object): An HTTP response object, expected to have a
            ``headers`` attribute that is a ``Mapping[str, str]``.
        name (str): The name of a required header.
        get_headers (Callable[Any, Mapping[str, str]]): Helper to get headers
            from an HTTP response.
        callback (Optional[Callable]): A callback that takes no arguments,
            to be executed when an exception is being raised.

    Returns:
        str: The desired header.

    Raises:
        ~google.resumable_media.common.InvalidResponse: If the header
            is missing.
    """
    headers = get_headers(response)
    if name not in headers:
        callback()
        raise common.InvalidResponse(
            response, "Response headers must contain header", name
        )

    return headers[name]


def require_status_code(response, status_codes, get_status_code, callback=do_nothing):
    """Require a response has a status code among a list.

    Args:
        response (object): The HTTP response object.
        status_codes (tuple): The acceptable status codes.
        get_status_code (Callable[Any, int]): Helper to get a status code
            from a response.
        callback (Optional[Callable]): A callback that takes no arguments,
            to be executed when an exception is being raised.

    Returns:
        int: The status code.

    Raises:
        ~google.resumable_media.common.InvalidResponse: If the status code
            is not one of the values in ``status_codes``.
    """
    status_code = get_status_code(response)
    if status_code not in status_codes:
        callback()
        raise common.InvalidResponse(
            response,
            "Request failed with status code",
            status_code,
            "Expected one of",
            *status_codes
        )
    return status_code


def calculate_retry_wait(base_wait, max_sleep):
    """Calculate the amount of time to wait before a retry attempt.

    Wait time grows exponentially with the number of attempts, until
    ``max_sleep``.

    A random amount of jitter (between 0 and 1 seconds) is added to spread out
    retry attempts from different clients.

    Args:
        base_wait (float): The "base" wait time (i.e. without any jitter)
            that will be doubled until it reaches the maximum sleep.
        max_sleep (float): Maximum value that a sleep time is allowed to be.

    Returns:
        Tuple[float, float]: The new base wait time as well as the wait time
        to be applied (with a random amount of jitter between 0 and 1 seconds
        added).
    """
    new_base_wait = 2.0 * base_wait
    if new_base_wait > max_sleep:
        new_base_wait = max_sleep

    jitter_ms = random.randint(0, 1000)
    return new_base_wait, new_base_wait + 0.001 * jitter_ms


async def wait_and_retry(func, get_status_code, retry_strategy):
    """Attempts to retry a call to ``func`` until success.

    Expects ``func`` to return an HTTP response and uses ``get_status_code``
    to check if the response is retry-able.

    Will retry until :meth:`~.RetryStrategy.retry_allowed` (on the current
    ``retry_strategy``) returns :data:`False`. Uses
    :func:`calculate_retry_wait` to double the wait time (with jitter) after
    each attempt.

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
    base_wait = 0.5  # When doubled will give 1.0

    while True:  # return on success or when retries exhausted.
        error = None
        try:
            response = await func()
        except ConnectionError as e:
            error = e
        else:
            if get_status_code(response) not in common.RETRYABLE:
                return response

        if not retry_strategy.retry_allowed(total_sleep, num_retries):
            # Retries are exhausted and no acceptable response was received. Raise the
            # retriable_error or return the unacceptable response.
            if error:
                raise error

            return response

        base_wait, wait_time = calculate_retry_wait(base_wait, retry_strategy.max_sleep)

        num_retries += 1
        total_sleep += wait_time
        time.sleep(wait_time)


class _DoNothingHash(object):
    """Do-nothing hash object.

    Intended as a stand-in for ``hashlib.md5`` or a crc32c checksum
    implementation in cases where it isn't necessary to compute the hash.
    """

    def update(self, unused_chunk):
        """Do-nothing ``update`` method.

        Intended to match the interface of ``hashlib.md5`` and other checksums.
        Args:
            unused_chunk (bytes): A chunk of data.
        """
