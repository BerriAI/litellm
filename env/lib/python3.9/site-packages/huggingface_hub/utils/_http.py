# coding=utf-8
# Copyright 2022-present, the HuggingFace Inc. team.
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
"""Contains utilities to handle HTTP requests in Huggingface Hub."""
import io
import os
import threading
import time
import uuid
from functools import lru_cache
from http import HTTPStatus
from typing import Callable, Tuple, Type, Union

import requests
from requests import Response
from requests.adapters import HTTPAdapter
from requests.exceptions import ProxyError, Timeout
from requests.models import PreparedRequest

from . import logging
from ._typing import HTTP_METHOD_T


logger = logging.get_logger(__name__)

# Both headers are used by the Hub to debug failed requests.
# `X_AMZN_TRACE_ID` is better as it also works to debug on Cloudfront and ALB.
# If `X_AMZN_TRACE_ID` is set, the Hub will use it as well.
X_AMZN_TRACE_ID = "X-Amzn-Trace-Id"
X_REQUEST_ID = "x-request-id"


class UniqueRequestIdAdapter(HTTPAdapter):
    X_AMZN_TRACE_ID = "X-Amzn-Trace-Id"

    def add_headers(self, request, **kwargs):
        super().add_headers(request, **kwargs)

        # Add random request ID => easier for server-side debug
        if X_AMZN_TRACE_ID not in request.headers:
            request.headers[X_AMZN_TRACE_ID] = request.headers.get(X_REQUEST_ID) or str(uuid.uuid4())

        # Add debug log
        has_token = str(request.headers.get("authorization", "")).startswith("Bearer hf_")
        logger.debug(
            f"Request {request.headers[X_AMZN_TRACE_ID]}: {request.method} {request.url} (authenticated: {has_token})"
        )

    def send(self, request: PreparedRequest, *args, **kwargs) -> Response:
        """Catch any RequestException to append request id to the error message for debugging."""
        try:
            return super().send(request, *args, **kwargs)
        except requests.RequestException as e:
            request_id = request.headers.get(X_AMZN_TRACE_ID)
            if request_id is not None:
                # Taken from https://stackoverflow.com/a/58270258
                e.args = (*e.args, f"(Request ID: {request_id})")
            raise


def _default_backend_factory() -> requests.Session:
    session = requests.Session()
    session.mount("http://", UniqueRequestIdAdapter())
    session.mount("https://", UniqueRequestIdAdapter())
    return session


BACKEND_FACTORY_T = Callable[[], requests.Session]
_GLOBAL_BACKEND_FACTORY: BACKEND_FACTORY_T = _default_backend_factory


def configure_http_backend(backend_factory: BACKEND_FACTORY_T = _default_backend_factory) -> None:
    """
    Configure the HTTP backend by providing a `backend_factory`. Any HTTP calls made by `huggingface_hub` will use a
    Session object instantiated by this factory. This can be useful if you are running your scripts in a specific
    environment requiring custom configuration (e.g. custom proxy or certifications).

    Use [`get_session`] to get a configured Session. Since `requests.Session` is not guaranteed to be thread-safe,
    `huggingface_hub` creates 1 Session instance per thread. They are all instantiated using the same `backend_factory`
    set in [`configure_http_backend`]. A LRU cache is used to cache the created sessions (and connections) between
    calls. Max size is 128 to avoid memory leaks if thousands of threads are spawned.

    See [this issue](https://github.com/psf/requests/issues/2766) to know more about thread-safety in `requests`.

    Example:
    ```py
    import requests
    from huggingface_hub import configure_http_backend, get_session

    # Create a factory function that returns a Session with configured proxies
    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.proxies = {"http": "http://10.10.1.10:3128", "https": "https://10.10.1.11:1080"}
        return session

    # Set it as the default session factory
    configure_http_backend(backend_factory=backend_factory)

    # In practice, this is mostly done internally in `huggingface_hub`
    session = get_session()
    ```
    """
    global _GLOBAL_BACKEND_FACTORY
    _GLOBAL_BACKEND_FACTORY = backend_factory
    _get_session_from_cache.cache_clear()


def get_session() -> requests.Session:
    """
    Get a `requests.Session` object, using the session factory from the user.

    Use [`get_session`] to get a configured Session. Since `requests.Session` is not guaranteed to be thread-safe,
    `huggingface_hub` creates 1 Session instance per thread. They are all instantiated using the same `backend_factory`
    set in [`configure_http_backend`]. A LRU cache is used to cache the created sessions (and connections) between
    calls. Max size is 128 to avoid memory leaks if thousands of threads are spawned.

    See [this issue](https://github.com/psf/requests/issues/2766) to know more about thread-safety in `requests`.

    Example:
    ```py
    import requests
    from huggingface_hub import configure_http_backend, get_session

    # Create a factory function that returns a Session with configured proxies
    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.proxies = {"http": "http://10.10.1.10:3128", "https": "https://10.10.1.11:1080"}
        return session

    # Set it as the default session factory
    configure_http_backend(backend_factory=backend_factory)

    # In practice, this is mostly done internally in `huggingface_hub`
    session = get_session()
    ```
    """
    return _get_session_from_cache(process_id=os.getpid(), thread_id=threading.get_ident())


@lru_cache(maxsize=128)  # default value for Python>=3.8. Let's keep the same for Python3.7
def _get_session_from_cache(process_id: int, thread_id: int) -> requests.Session:
    """
    Create a new session per thread using global factory. Using LRU cache (maxsize 128) to avoid memory leaks when
    using thousands of threads. Cache is cleared when `configure_http_backend` is called.
    """
    return _GLOBAL_BACKEND_FACTORY()


def http_backoff(
    method: HTTP_METHOD_T,
    url: str,
    *,
    max_retries: int = 5,
    base_wait_time: float = 1,
    max_wait_time: float = 8,
    retry_on_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = (
        Timeout,
        ProxyError,
    ),
    retry_on_status_codes: Union[int, Tuple[int, ...]] = HTTPStatus.SERVICE_UNAVAILABLE,
    **kwargs,
) -> Response:
    """Wrapper around requests to retry calls on an endpoint, with exponential backoff.

    Endpoint call is retried on exceptions (ex: connection timeout, proxy error,...)
    and/or on specific status codes (ex: service unavailable). If the call failed more
    than `max_retries`, the exception is thrown or `raise_for_status` is called on the
    response object.

    Re-implement mechanisms from the `backoff` library to avoid adding an external
    dependencies to `hugging_face_hub`. See https://github.com/litl/backoff.

    Args:
        method (`Literal["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"]`):
            HTTP method to perform.
        url (`str`):
            The URL of the resource to fetch.
        max_retries (`int`, *optional*, defaults to `5`):
            Maximum number of retries, defaults to 5 (no retries).
        base_wait_time (`float`, *optional*, defaults to `1`):
            Duration (in seconds) to wait before retrying the first time.
            Wait time between retries then grows exponentially, capped by
            `max_wait_time`.
        max_wait_time (`float`, *optional*, defaults to `8`):
            Maximum duration (in seconds) to wait before retrying.
        retry_on_exceptions (`Type[Exception]` or `Tuple[Type[Exception]]`, *optional*, defaults to `(Timeout, ProxyError,)`):
            Define which exceptions must be caught to retry the request. Can be a single
            type or a tuple of types.
            By default, retry on `Timeout` and `ProxyError`.
        retry_on_status_codes (`int` or `Tuple[int]`, *optional*, defaults to `503`):
            Define on which status codes the request must be retried. By default, only
            HTTP 503 Service Unavailable is retried.
        **kwargs (`dict`, *optional*):
            kwargs to pass to `requests.request`.

    Example:
    ```
    >>> from huggingface_hub.utils import http_backoff

    # Same usage as "requests.request".
    >>> response = http_backoff("GET", "https://www.google.com")
    >>> response.raise_for_status()

    # If you expect a Gateway Timeout from time to time
    >>> http_backoff("PUT", upload_url, data=data, retry_on_status_codes=504)
    >>> response.raise_for_status()
    ```

    <Tip warning={true}>

    When using `requests` it is possible to stream data by passing an iterator to the
    `data` argument. On http backoff this is a problem as the iterator is not reset
    after a failed call. This issue is mitigated for file objects or any IO streams
    by saving the initial position of the cursor (with `data.tell()`) and resetting the
    cursor between each call (with `data.seek()`). For arbitrary iterators, http backoff
    will fail. If this is a hard constraint for you, please let us know by opening an
    issue on [Github](https://github.com/huggingface/huggingface_hub).

    </Tip>
    """
    if isinstance(retry_on_exceptions, type):  # Tuple from single exception type
        retry_on_exceptions = (retry_on_exceptions,)

    if isinstance(retry_on_status_codes, int):  # Tuple from single status code
        retry_on_status_codes = (retry_on_status_codes,)

    nb_tries = 0
    sleep_time = base_wait_time

    # If `data` is used and is a file object (or any IO), it will be consumed on the
    # first HTTP request. We need to save the initial position so that the full content
    # of the file is re-sent on http backoff. See warning tip in docstring.
    io_obj_initial_pos = None
    if "data" in kwargs and isinstance(kwargs["data"], io.IOBase):
        io_obj_initial_pos = kwargs["data"].tell()

    session = get_session()
    while True:
        nb_tries += 1
        try:
            # If `data` is used and is a file object (or any IO), set back cursor to
            # initial position.
            if io_obj_initial_pos is not None:
                kwargs["data"].seek(io_obj_initial_pos)

            # Perform request and return if status_code is not in the retry list.
            response = session.request(method=method, url=url, **kwargs)
            if response.status_code not in retry_on_status_codes:
                return response

            # Wrong status code returned (HTTP 503 for instance)
            logger.warning(f"HTTP Error {response.status_code} thrown while requesting {method} {url}")
            if nb_tries > max_retries:
                response.raise_for_status()  # Will raise uncaught exception
                # We return response to avoid infinite loop in the corner case where the
                # user ask for retry on a status code that doesn't raise_for_status.
                return response

        except retry_on_exceptions as err:
            logger.warning(f"'{err}' thrown while requesting {method} {url}")

            if nb_tries > max_retries:
                raise err

        # Sleep for X seconds
        logger.warning(f"Retrying in {sleep_time}s [Retry {nb_tries}/{max_retries}].")
        time.sleep(sleep_time)

        # Update sleep time for next retry
        sleep_time = min(max_wait_time, sleep_time * 2)  # Exponential backoff
