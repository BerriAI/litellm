# Copyright 2014 Google LLC
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
"""Batch updates / deletes of storage buckets / blobs.

A batch request is a single standard HTTP request containing multiple Cloud Storage JSON API calls.
Within this main HTTP request, there are multiple parts which each contain a nested HTTP request.
The body of each part is itself a complete HTTP request, with its own verb, URL, headers, and body.

Note that Cloud Storage does not support batch operations for uploading or downloading.
Additionally, the current batch design does not support library methods whose return values
depend on the response payload. See more details in the [Sending Batch Requests official guide](https://cloud.google.com/storage/docs/batch).

Examples of situations when you might want to use the Batch module:
``blob.patch()``
``blob.update()``
``blob.delete()``
``bucket.delete_blob()``
``bucket.patch()``
``bucket.update()``
"""
from email.encoders import encode_noop
from email.generator import Generator
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.parser import Parser
import io
import json

import requests

from google.cloud import _helpers
from google.cloud import exceptions
from google.cloud.storage._http import Connection
from google.cloud.storage.constants import _DEFAULT_TIMEOUT


class MIMEApplicationHTTP(MIMEApplication):
    """MIME type for ``application/http``.

    Constructs payload from headers and body

    :type method: str
    :param method: HTTP method

    :type uri: str
    :param uri: URI for HTTP request

    :type headers:  dict
    :param headers: HTTP headers

    :type body: str
    :param body: (Optional) HTTP payload

    """

    def __init__(self, method, uri, headers, body):
        if isinstance(body, dict):
            body = json.dumps(body)
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = len(body)
        if body is None:
            body = ""
        lines = [f"{method} {uri} HTTP/1.1"]
        lines.extend([f"{key}: {value}" for key, value in sorted(headers.items())])
        lines.append("")
        lines.append(body)
        payload = "\r\n".join(lines)
        super().__init__(payload, "http", encode_noop)


class _FutureDict(object):
    """Class to hold a future value for a deferred request.

    Used by for requests that get sent in a :class:`Batch`.
    """

    @staticmethod
    def get(key, default=None):
        """Stand-in for dict.get.

        :type key: object
        :param key: Hashable dictionary key.

        :type default: object
        :param default: Fallback value to dict.get.

        :raises: :class:`KeyError` always since the future is intended to fail
                 as a dictionary.
        """
        raise KeyError(f"Cannot get({key!r}, default={default!r}) on a future")

    def __getitem__(self, key):
        """Stand-in for dict[key].

        :type key: object
        :param key: Hashable dictionary key.

        :raises: :class:`KeyError` always since the future is intended to fail
                 as a dictionary.
        """
        raise KeyError(f"Cannot get item {key!r} from a future")

    def __setitem__(self, key, value):
        """Stand-in for dict[key] = value.

        :type key: object
        :param key: Hashable dictionary key.

        :type value: object
        :param value: Dictionary value.

        :raises: :class:`KeyError` always since the future is intended to fail
                 as a dictionary.
        """
        raise KeyError(f"Cannot set {key!r} -> {value!r} on a future")


class _FutureResponse(requests.Response):
    """Reponse that returns a placeholder dictionary for a batched requests."""

    def __init__(self, future_dict):
        super(_FutureResponse, self).__init__()
        self._future_dict = future_dict
        self.status_code = 204

    def json(self):
        return self._future_dict

    @property
    def content(self):
        return self._future_dict


class Batch(Connection):
    """Proxy an underlying connection, batching up change operations.

    .. warning::

        Cloud Storage does not support batch operations for uploading or downloading.
        Additionally, the current batch design does not support library methods whose
        return values depend on the response payload.

    :type client: :class:`google.cloud.storage.client.Client`
    :param client: The client to use for making connections.

    :type raise_exception: bool
    :param raise_exception:
        (Optional) Defaults to True. If True, instead of adding exceptions
        to the list of return responses, the final exception will be raised.
        Note that exceptions are unwrapped after all operations are complete
        in success or failure, and only the last exception is raised.
    """

    _MAX_BATCH_SIZE = 1000

    def __init__(self, client, raise_exception=True):
        api_endpoint = client._connection.API_BASE_URL
        client_info = client._connection._client_info
        super(Batch, self).__init__(
            client, client_info=client_info, api_endpoint=api_endpoint
        )
        self._requests = []
        self._target_objects = []
        self._responses = []
        self._raise_exception = raise_exception

    def _do_request(
        self, method, url, headers, data, target_object, timeout=_DEFAULT_TIMEOUT
    ):
        """Override Connection:  defer actual HTTP request.

        Only allow up to ``_MAX_BATCH_SIZE`` requests to be deferred.

        :type method: str
        :param method: The HTTP method to use in the request.

        :type url: str
        :param url: The URL to send the request to.

        :type headers: dict
        :param headers: A dictionary of HTTP headers to send with the request.

        :type data: str
        :param data: The data to send as the body of the request.

        :type target_object: object
        :param target_object:
            (Optional) This allows us to enable custom behavior in our batch
            connection. Here we defer an HTTP request and complete
            initialization of the object at a later time.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :rtype: tuple of ``response`` (a dictionary of sorts)
                and ``content`` (a string).
        :returns: The HTTP response object and the content of the response.
        """
        if len(self._requests) >= self._MAX_BATCH_SIZE:
            raise ValueError(
                "Too many deferred requests (max %d)" % self._MAX_BATCH_SIZE
            )
        self._requests.append((method, url, headers, data, timeout))
        result = _FutureDict()
        self._target_objects.append(target_object)
        if target_object is not None:
            target_object._properties = result
        return _FutureResponse(result)

    def _prepare_batch_request(self):
        """Prepares headers and body for a batch request.

        :rtype: tuple (dict, str)
        :returns: The pair of headers and body of the batch request to be sent.
        :raises: :class:`ValueError` if no requests have been deferred.
        """
        if len(self._requests) == 0:
            raise ValueError("No deferred requests")

        multi = MIMEMultipart()

        # Use timeout of last request, default to _DEFAULT_TIMEOUT
        timeout = _DEFAULT_TIMEOUT
        for method, uri, headers, body, _timeout in self._requests:
            subrequest = MIMEApplicationHTTP(method, uri, headers, body)
            multi.attach(subrequest)
            timeout = _timeout

        buf = io.StringIO()
        generator = Generator(buf, False, 0)
        generator.flatten(multi)
        payload = buf.getvalue()

        # Strip off redundant header text
        _, body = payload.split("\n\n", 1)
        return dict(multi._headers), body, timeout

    def _finish_futures(self, responses, raise_exception=True):
        """Apply all the batch responses to the futures created.

        :type responses: list of (headers, payload) tuples.
        :param responses: List of headers and payloads from each response in
                          the batch.

        :type raise_exception: bool
        :param raise_exception:
            (Optional) Defaults to True. If True, instead of adding exceptions
            to the list of return responses, the final exception will be raised.
            Note that exceptions are unwrapped after all operations are complete
            in success or failure, and only the last exception is raised.

        :raises: :class:`ValueError` if no requests have been deferred.
        """
        # If a bad status occurs, we track it, but don't raise an exception
        # until all futures have been populated.
        # If raise_exception=False, we add exceptions to the list of responses.
        exception_args = None

        if len(self._target_objects) != len(responses):  # pragma: NO COVER
            raise ValueError("Expected a response for every request.")

        for target_object, subresponse in zip(self._target_objects, responses):
            # For backwards compatibility, only the final exception will be raised.
            # Set raise_exception=False to include all exceptions to the list of return responses.
            if not 200 <= subresponse.status_code < 300 and raise_exception:
                exception_args = exception_args or subresponse
            elif target_object is not None:
                try:
                    target_object._properties = subresponse.json()
                except ValueError:
                    target_object._properties = subresponse.content

        if exception_args is not None:
            raise exceptions.from_http_response(exception_args)

    def finish(self, raise_exception=True):
        """Submit a single `multipart/mixed` request with deferred requests.

        :type raise_exception: bool
        :param raise_exception:
            (Optional) Defaults to True. If True, instead of adding exceptions
            to the list of return responses, the final exception will be raised.
            Note that exceptions are unwrapped after all operations are complete
            in success or failure, and only the last exception is raised.

        :rtype: list of tuples
        :returns: one ``(headers, payload)`` tuple per deferred request.
        """
        headers, body, timeout = self._prepare_batch_request()

        url = f"{self.API_BASE_URL}/batch/storage/v1"

        # Use the private ``_base_connection`` rather than the property
        # ``_connection``, since the property may be this
        # current batch.
        response = self._client._base_connection._make_request(
            "POST", url, data=body, headers=headers, timeout=timeout
        )

        # Raise exception if the top-level batch request fails
        if not 200 <= response.status_code < 300:
            raise exceptions.from_http_response(response)

        responses = list(_unpack_batch_response(response))
        self._finish_futures(responses, raise_exception=raise_exception)
        self._responses = responses
        return responses

    def current(self):
        """Return the topmost batch, or None."""
        return self._client.current_batch

    def __enter__(self):
        self._client._push_batch(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                self.finish(raise_exception=self._raise_exception)
        finally:
            self._client._pop_batch()


def _generate_faux_mime_message(parser, response):
    """Convert response, content -> (multipart) email.message.

    Helper for _unpack_batch_response.
    """
    # We coerce to bytes to get consistent concat across
    # Py2 and Py3. Percent formatting is insufficient since
    # it includes the b in Py3.
    content_type = _helpers._to_bytes(response.headers.get("content-type", ""))

    faux_message = b"".join(
        [b"Content-Type: ", content_type, b"\nMIME-Version: 1.0\n\n", response.content]
    )

    return parser.parsestr(faux_message.decode("utf-8"))


def _unpack_batch_response(response):
    """Convert requests.Response -> [(headers, payload)].

    Creates a generator of tuples of emulating the responses to
    :meth:`requests.Session.request`.

    :type response: :class:`requests.Response`
    :param response: HTTP response / headers from a request.
    """
    parser = Parser()
    message = _generate_faux_mime_message(parser, response)

    if not isinstance(message._payload, list):  # pragma: NO COVER
        raise ValueError("Bad response:  not multi-part")

    for subrequest in message._payload:
        status_line, rest = subrequest._payload.split("\n", 1)
        _, status, _ = status_line.split(" ", 2)
        sub_message = parser.parsestr(rest)
        payload = sub_message._payload
        msg_headers = dict(sub_message._headers)
        content_id = msg_headers.get("Content-ID")

        subresponse = requests.Response()
        subresponse.request = requests.Request(
            method="BATCH", url=f"contentid://{content_id}"
        ).prepare()
        subresponse.status_code = int(status)
        subresponse.headers.update(msg_headers)
        subresponse._content = payload.encode("utf-8")

        yield subresponse
