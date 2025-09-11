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

"""Virtual bases classes for downloading media from Google APIs."""


import http.client
import re

from google.resumable_media import _helpers
from google.resumable_media import common


_CONTENT_RANGE_RE = re.compile(
    r"bytes (?P<start_byte>\d+)-(?P<end_byte>\d+)/(?P<total_bytes>\d+)",
    flags=re.IGNORECASE,
)
_ACCEPTABLE_STATUS_CODES = (http.client.OK, http.client.PARTIAL_CONTENT)
_GET = "GET"
_ZERO_CONTENT_RANGE_HEADER = "bytes */0"


class DownloadBase(object):
    """Base class for download helpers.

    Defines core shared behavior across different download types.

    Args:
        media_url (str): The URL containing the media to be downloaded.
        stream (IO[bytes]): A write-able stream (i.e. file-like object) that
            the downloaded resource can be written to.
        start (int): The first byte in a range to be downloaded.
        end (int): The last byte in a range to be downloaded.
        headers (Optional[Mapping[str, str]]): Extra headers that should
            be sent with the request, e.g. headers for encrypted data.

    Attributes:
        media_url (str): The URL containing the media to be downloaded.
        start (Optional[int]): The first byte in a range to be downloaded.
        end (Optional[int]): The last byte in a range to be downloaded.
    """

    def __init__(self, media_url, stream=None, start=None, end=None, headers=None):
        self.media_url = media_url
        self._stream = stream
        self.start = start
        self.end = end
        if headers is None:
            headers = {}
        self._headers = headers
        self._finished = False
        self._retry_strategy = common.RetryStrategy()

    @property
    def finished(self):
        """bool: Flag indicating if the download has completed."""
        return self._finished

    @staticmethod
    def _get_status_code(response):
        """Access the status code from an HTTP response.

        Args:
            response (object): The HTTP response object.

        Raises:
            NotImplementedError: Always, since virtual.
        """
        raise NotImplementedError("This implementation is virtual.")

    @staticmethod
    def _get_headers(response):
        """Access the headers from an HTTP response.

        Args:
            response (object): The HTTP response object.

        Raises:
            NotImplementedError: Always, since virtual.
        """
        raise NotImplementedError("This implementation is virtual.")

    @staticmethod
    def _get_body(response):
        """Access the response body from an HTTP response.

        Args:
            response (object): The HTTP response object.

        Raises:
            NotImplementedError: Always, since virtual.
        """
        raise NotImplementedError("This implementation is virtual.")


class Download(DownloadBase):
    """Helper to manage downloading a resource from a Google API.

    "Slices" of the resource can be retrieved by specifying a range
    with ``start`` and / or ``end``. However, in typical usage, neither
    ``start`` nor ``end`` is expected to be provided.

    Args:
        media_url (str): The URL containing the media to be downloaded.
        stream (IO[bytes]): A write-able stream (i.e. file-like object) that
            the downloaded resource can be written to.
        start (int): The first byte in a range to be downloaded. If not
            provided, but ``end`` is provided, will download from the
            beginning to ``end`` of the media.
        end (int): The last byte in a range to be downloaded. If not
            provided, but ``start`` is provided, will download from the
            ``start`` to the end of the media.
        headers (Optional[Mapping[str, str]]): Extra headers that should
            be sent with the request, e.g. headers for encrypted data.
        checksum Optional([str]): The type of checksum to compute to verify
            the integrity of the object. The response headers must contain
            a checksum of the requested type. If the headers lack an
            appropriate checksum (for instance in the case of transcoded or
            ranged downloads where the remote service does not know the
            correct checksum) an INFO-level log will be emitted. Supported
            values are "md5", "crc32c" and None.
    """

    def __init__(
        self, media_url, stream=None, start=None, end=None, headers=None, checksum="md5"
    ):
        super(Download, self).__init__(
            media_url, stream=stream, start=start, end=end, headers=headers
        )
        self.checksum = checksum
        self._bytes_downloaded = 0
        self._expected_checksum = None
        self._checksum_object = None
        self._object_generation = None

    def _prepare_request(self):
        """Prepare the contents of an HTTP request.

        This is everything that must be done before a request that doesn't
        require network I/O (or other I/O). This is based on the `sans-I/O`_
        philosophy.

        Returns:
            Tuple[str, str, NoneType, Mapping[str, str]]: The quadruple

              * HTTP verb for the request (always GET)
              * the URL for the request
              * the body of the request (always :data:`None`)
              * headers for the request

        Raises:
            ValueError: If the current :class:`Download` has already
                finished.

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """
        if self.finished:
            raise ValueError("A download can only be used once.")

        add_bytes_range(self.start, self.end, self._headers)
        return _GET, self.media_url, None, self._headers

    def _process_response(self, response):
        """Process the response from an HTTP request.

        This is everything that must be done after a request that doesn't
        require network I/O (or other I/O). This is based on the `sans-I/O`_
        philosophy.

        Args:
            response (object): The HTTP response object.

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """
        # Tombstone the current Download so it cannot be used again.
        self._finished = True
        _helpers.require_status_code(
            response, _ACCEPTABLE_STATUS_CODES, self._get_status_code
        )

    def consume(self, transport, timeout=None):
        """Consume the resource to be downloaded.

        If a ``stream`` is attached to this download, then the downloaded
        resource will be written to the stream.

        Args:
            transport (object): An object which can make authenticated
                requests.
            timeout (Optional[Union[float, Tuple[float, float]]]):
                The number of seconds to wait for the server response.
                Depending on the retry strategy, a request may be repeated
                several times using the same timeout each time.

                Can also be passed as a tuple (connect_timeout, read_timeout).
                See :meth:`requests.Session.request` documentation for details.

        Raises:
            NotImplementedError: Always, since virtual.
        """
        raise NotImplementedError("This implementation is virtual.")


class ChunkedDownload(DownloadBase):
    """Download a resource in chunks from a Google API.

    Args:
        media_url (str): The URL containing the media to be downloaded.
        chunk_size (int): The number of bytes to be retrieved in each
            request.
        stream (IO[bytes]): A write-able stream (i.e. file-like object) that
            will be used to concatenate chunks of the resource as they are
            downloaded.
        start (int): The first byte in a range to be downloaded. If not
            provided, defaults to ``0``.
        end (int): The last byte in a range to be downloaded. If not
            provided, will download to the end of the media.
        headers (Optional[Mapping[str, str]]): Extra headers that should
            be sent with each request, e.g. headers for data encryption
            key headers.

    Attributes:
        media_url (str): The URL containing the media to be downloaded.
        start (Optional[int]): The first byte in a range to be downloaded.
        end (Optional[int]): The last byte in a range to be downloaded.
        chunk_size (int): The number of bytes to be retrieved in each request.

    Raises:
        ValueError: If ``start`` is negative.
    """

    def __init__(self, media_url, chunk_size, stream, start=0, end=None, headers=None):
        if start < 0:
            raise ValueError(
                "On a chunked download the starting " "value cannot be negative."
            )
        super(ChunkedDownload, self).__init__(
            media_url, stream=stream, start=start, end=end, headers=headers
        )
        self.chunk_size = chunk_size
        self._bytes_downloaded = 0
        self._total_bytes = None
        self._invalid = False

    @property
    def bytes_downloaded(self):
        """int: Number of bytes that have been downloaded."""
        return self._bytes_downloaded

    @property
    def total_bytes(self):
        """Optional[int]: The total number of bytes to be downloaded."""
        return self._total_bytes

    @property
    def invalid(self):
        """bool: Indicates if the download is in an invalid state.

        This will occur if a call to :meth:`consume_next_chunk` fails.
        """
        return self._invalid

    def _get_byte_range(self):
        """Determines the byte range for the next request.

        Returns:
            Tuple[int, int]: The pair of begin and end byte for the next
            chunked request.
        """
        curr_start = self.start + self.bytes_downloaded
        curr_end = curr_start + self.chunk_size - 1
        # Make sure ``curr_end`` does not exceed ``end``.
        if self.end is not None:
            curr_end = min(curr_end, self.end)
        # Make sure ``curr_end`` does not exceed ``total_bytes - 1``.
        if self.total_bytes is not None:
            curr_end = min(curr_end, self.total_bytes - 1)
        return curr_start, curr_end

    def _prepare_request(self):
        """Prepare the contents of an HTTP request.

        This is everything that must be done before a request that doesn't
        require network I/O (or other I/O). This is based on the `sans-I/O`_
        philosophy.

        .. note:

            This method will be used multiple times, so ``headers`` will
            be mutated in between requests. However, we don't make a copy
            since the same keys are being updated.

        Returns:
            Tuple[str, str, NoneType, Mapping[str, str]]: The quadruple

              * HTTP verb for the request (always GET)
              * the URL for the request
              * the body of the request (always :data:`None`)
              * headers for the request

        Raises:
            ValueError: If the current download has finished.
            ValueError: If the current download is invalid.

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """
        if self.finished:
            raise ValueError("Download has finished.")
        if self.invalid:
            raise ValueError("Download is invalid and cannot be re-used.")

        curr_start, curr_end = self._get_byte_range()
        add_bytes_range(curr_start, curr_end, self._headers)
        return _GET, self.media_url, None, self._headers

    def _make_invalid(self):
        """Simple setter for ``invalid``.

        This is intended to be passed along as a callback to helpers that
        raise an exception so they can mark this instance as invalid before
        raising.
        """
        self._invalid = True

    def _process_response(self, response):
        """Process the response from an HTTP request.

        This is everything that must be done after a request that doesn't
        require network I/O. This is based on the `sans-I/O`_ philosophy.

        For the time being, this **does require** some form of I/O to write
        a chunk to ``stream``. However, this will (almost) certainly not be
        network I/O.

        Updates the current state after consuming a chunk. First,
        increments ``bytes_downloaded`` by the number of bytes in the
        ``content-length`` header.

        If ``total_bytes`` is already set, this assumes (but does not check)
        that we already have the correct value and doesn't bother to check
        that it agrees with the headers.

        We expect the **total** length to be in the ``content-range`` header,
        but this header is only present on requests which sent the ``range``
        header. This response header should be of the form
        ``bytes {start}-{end}/{total}`` and ``{end} - {start} + 1``
        should be the same as the ``Content-Length``.

        Args:
            response (object): The HTTP response object (need headers).

        Raises:
            ~google.resumable_media.common.InvalidResponse: If the number
                of bytes in the body doesn't match the content length header.

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """
        # Verify the response before updating the current instance.
        if _check_for_zero_content_range(
            response, self._get_status_code, self._get_headers
        ):
            self._finished = True
            return

        _helpers.require_status_code(
            response,
            _ACCEPTABLE_STATUS_CODES,
            self._get_status_code,
            callback=self._make_invalid,
        )
        headers = self._get_headers(response)
        response_body = self._get_body(response)

        start_byte, end_byte, total_bytes = get_range_info(
            response, self._get_headers, callback=self._make_invalid
        )

        transfer_encoding = headers.get("transfer-encoding")

        if transfer_encoding is None:
            content_length = _helpers.header_required(
                response,
                "content-length",
                self._get_headers,
                callback=self._make_invalid,
            )
            num_bytes = int(content_length)
            if len(response_body) != num_bytes:
                self._make_invalid()
                raise common.InvalidResponse(
                    response,
                    "Response is different size than content-length",
                    "Expected",
                    num_bytes,
                    "Received",
                    len(response_body),
                )
        else:
            # 'content-length' header not allowed with chunked encoding.
            num_bytes = end_byte - start_byte + 1

        # First update ``bytes_downloaded``.
        self._bytes_downloaded += num_bytes
        # If the end byte is past ``end`` or ``total_bytes - 1`` we are done.
        if self.end is not None and end_byte >= self.end:
            self._finished = True
        elif end_byte >= total_bytes - 1:
            self._finished = True
        # NOTE: We only use ``total_bytes`` if not already known.
        if self.total_bytes is None:
            self._total_bytes = total_bytes
        # Write the response body to the stream.
        self._stream.write(response_body)

    def consume_next_chunk(self, transport, timeout=None):
        """Consume the next chunk of the resource to be downloaded.

        Args:
            transport (object): An object which can make authenticated
                requests.
            timeout (Optional[Union[float, Tuple[float, float]]]):
                The number of seconds to wait for the server response.
                Depending on the retry strategy, a request may be repeated
                several times using the same timeout each time.

                Can also be passed as a tuple (connect_timeout, read_timeout).
                See :meth:`requests.Session.request` documentation for details.

        Raises:
            NotImplementedError: Always, since virtual.
        """
        raise NotImplementedError("This implementation is virtual.")


def add_bytes_range(start, end, headers):
    """Add a bytes range to a header dictionary.

    Some possible inputs and the corresponding bytes ranges::

       >>> headers = {}
       >>> add_bytes_range(None, None, headers)
       >>> headers
       {}
       >>> add_bytes_range(500, 999, headers)
       >>> headers['range']
       'bytes=500-999'
       >>> add_bytes_range(None, 499, headers)
       >>> headers['range']
       'bytes=0-499'
       >>> add_bytes_range(-500, None, headers)
       >>> headers['range']
       'bytes=-500'
       >>> add_bytes_range(9500, None, headers)
       >>> headers['range']
       'bytes=9500-'

    Args:
        start (Optional[int]): The first byte in a range. Can be zero,
            positive, negative or :data:`None`.
        end (Optional[int]): The last byte in a range. Assumed to be
            positive.
        headers (Mapping[str, str]): A headers mapping which can have the
            bytes range added if at least one of ``start`` or ``end``
            is not :data:`None`.
    """
    if start is None:
        if end is None:
            # No range to add.
            return
        else:
            # NOTE: This assumes ``end`` is non-negative.
            bytes_range = "0-{:d}".format(end)
    else:
        if end is None:
            if start < 0:
                bytes_range = "{:d}".format(start)
            else:
                bytes_range = "{:d}-".format(start)
        else:
            # NOTE: This is invalid if ``start < 0``.
            bytes_range = "{:d}-{:d}".format(start, end)

    headers[_helpers.RANGE_HEADER] = "bytes=" + bytes_range


def get_range_info(response, get_headers, callback=_helpers.do_nothing):
    """Get the start, end and total bytes from a content range header.

    Args:
        response (object): An HTTP response object.
        get_headers (Callable[Any, Mapping[str, str]]): Helper to get headers
            from an HTTP response.
        callback (Optional[Callable]): A callback that takes no arguments,
            to be executed when an exception is being raised.

    Returns:
        Tuple[int, int, int]: The start byte, end byte and total bytes.

    Raises:
        ~google.resumable_media.common.InvalidResponse: If the
            ``Content-Range`` header is not of the form
            ``bytes {start}-{end}/{total}``.
    """
    content_range = _helpers.header_required(
        response, _helpers.CONTENT_RANGE_HEADER, get_headers, callback=callback
    )
    match = _CONTENT_RANGE_RE.match(content_range)
    if match is None:
        callback()
        raise common.InvalidResponse(
            response,
            "Unexpected content-range header",
            content_range,
            'Expected to be of the form "bytes {start}-{end}/{total}"',
        )

    return (
        int(match.group("start_byte")),
        int(match.group("end_byte")),
        int(match.group("total_bytes")),
    )


def _check_for_zero_content_range(response, get_status_code, get_headers):
    """Validate if response status code is 416 and content range is zero.

    This is the special case for handling zero bytes files.

    Args:
        response (object): An HTTP response object.
        get_status_code (Callable[Any, int]): Helper to get a status code
            from a response.
        get_headers (Callable[Any, Mapping[str, str]]): Helper to get headers
            from an HTTP response.

    Returns:
        bool: True if content range total bytes is zero, false otherwise.
    """
    if get_status_code(response) == http.client.REQUESTED_RANGE_NOT_SATISFIABLE:
        content_range = _helpers.header_required(
            response,
            _helpers.CONTENT_RANGE_HEADER,
            get_headers,
            callback=_helpers.do_nothing,
        )
        if content_range == _ZERO_CONTENT_RANGE_HEADER:
            return True
    return False
