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

"""Virtual bases classes for uploading media via Google APIs.

Supported here are:

* simple (media) uploads
* multipart uploads that contain both metadata and a small file as payload
* resumable uploads (with metadata as well)
"""

import http.client
import json
import os
import random
import sys

from google import _async_resumable_media
from google._async_resumable_media import _helpers
from google.resumable_media import _helpers as sync_helpers
from google.resumable_media import _upload as sync_upload
from google.resumable_media import common


from google.resumable_media._upload import (
    _CONTENT_TYPE_HEADER,
    _CONTENT_RANGE_TEMPLATE,
    _RANGE_UNKNOWN_TEMPLATE,
    _EMPTY_RANGE_TEMPLATE,
    _BOUNDARY_FORMAT,
    _MULTIPART_SEP,
    _CRLF,
    _MULTIPART_BEGIN,
    _RELATED_HEADER,
    _BYTES_RANGE_RE,
    _STREAM_ERROR_TEMPLATE,
    _POST,
    _PUT,
    _UPLOAD_CHECKSUM_MISMATCH_MESSAGE,
    _UPLOAD_METADATA_NO_APPROPRIATE_CHECKSUM_MESSAGE,
)


class UploadBase(object):
    """Base class for upload helpers.

    Defines core shared behavior across different upload types.

    Args:
        upload_url (str): The URL where the content will be uploaded.
        headers (Optional[Mapping[str, str]]): Extra headers that should
            be sent with the request, e.g. headers for encrypted data.

    Attributes:
        upload_url (str): The URL where the content will be uploaded.
    """

    def __init__(self, upload_url, headers=None):
        self.upload_url = upload_url
        if headers is None:
            headers = {}
        self._headers = headers
        self._finished = False
        self._retry_strategy = common.RetryStrategy()

    @property
    def finished(self):
        """bool: Flag indicating if the upload has completed."""
        return self._finished

    def _process_response(self, response):
        """Process the response from an HTTP request.

        This is everything that must be done after a request that doesn't
        require network I/O (or other I/O). This is based on the `sans-I/O`_
        philosophy.

        Args:
            response (object): The HTTP response object.

        Raises:
            ~google.resumable_media.common.InvalidResponse: If the status
                code is not 200.

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """
        # Tombstone the current upload so it cannot be used again (in either
        # failure or success).
        self._finished = True
        _helpers.require_status_code(response, (http.client.OK,), self._get_status_code)

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


class SimpleUpload(UploadBase):
    """Upload a resource to a Google API.

    A **simple** media upload sends no metadata and completes the upload
    in a single request.

    Args:
        upload_url (str): The URL where the content will be uploaded.
        headers (Optional[Mapping[str, str]]): Extra headers that should
            be sent with the request, e.g. headers for encrypted data.

    Attributes:
        upload_url (str): The URL where the content will be uploaded.
    """

    def _prepare_request(self, data, content_type):
        """Prepare the contents of an HTTP request.

        This is everything that must be done before a request that doesn't
        require network I/O (or other I/O). This is based on the `sans-I/O`_
        philosophy.

        .. note:

            This method will be used only once, so ``headers`` will be
            mutated by having a new key added to it.

        Args:
            data (bytes): The resource content to be uploaded.
            content_type (str): The content type for the request.

        Returns:
            Tuple[str, str, bytes, Mapping[str, str]]: The quadruple

              * HTTP verb for the request (always POST)
              * the URL for the request
              * the body of the request
              * headers for the request

        Raises:
            ValueError: If the current upload has already finished.
            TypeError: If ``data`` isn't bytes.

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """
        if self.finished:
            raise ValueError("An upload can only be used once.")

        if not isinstance(data, bytes):
            raise TypeError("`data` must be bytes, received", type(data))
        self._headers[_CONTENT_TYPE_HEADER] = content_type
        return _POST, self.upload_url, data, self._headers

    def transmit(self, transport, data, content_type, timeout=None):
        """Transmit the resource to be uploaded.

        Args:
            transport (object): An object which can make authenticated
                requests.
            data (bytes): The resource content to be uploaded.
            content_type (str): The content type of the resource, e.g. a JPEG
                image has content type ``image/jpeg``.
            timeout (Optional[Union[float, aiohttp.ClientTimeout]]):
                The number of seconds to wait for the server response.
                Depending on the retry strategy, a request may be repeated
                several times using the same timeout each time.
                Can also be passed as an `aiohttp.ClientTimeout` object.

        Raises:
            NotImplementedError: Always, since virtual.
        """
        raise NotImplementedError("This implementation is virtual.")


class MultipartUpload(UploadBase):
    """Upload a resource with metadata to a Google API.

    A **multipart** upload sends both metadata and the resource in a single
    (multipart) request.

    Args:
        upload_url (str): The URL where the content will be uploaded.
        headers (Optional[Mapping[str, str]]): Extra headers that should
            be sent with the request, e.g. headers for encrypted data.
        checksum Optional([str]): The type of checksum to compute to verify
            the integrity of the object. The request metadata will be amended
            to include the computed value. Using this option will override a
            manually-set checksum value. Supported values are "md5", "crc32c"
            and None. The default is None.

    Attributes:
        upload_url (str): The URL where the content will be uploaded.
    """

    def __init__(self, upload_url, headers=None, checksum=None):
        super(MultipartUpload, self).__init__(upload_url, headers=headers)
        self._checksum_type = checksum

    def _prepare_request(self, data, metadata, content_type):
        """Prepare the contents of an HTTP request.

        This is everything that must be done before a request that doesn't
        require network I/O (or other I/O). This is based on the `sans-I/O`_
        philosophy.

        .. note:

            This method will be used only once, so ``headers`` will be
            mutated by having a new key added to it.

        Args:
            data (bytes): The resource content to be uploaded.
            metadata (Mapping[str, str]): The resource metadata, such as an
                ACL list.
            content_type (str): The content type of the resource, e.g. a JPEG
                image has content type ``image/jpeg``.

        Returns:
            Tuple[str, str, bytes, Mapping[str, str]]: The quadruple

              * HTTP verb for the request (always POST)
              * the URL for the request
              * the body of the request
              * headers for the request

        Raises:
            ValueError: If the current upload has already finished.
            TypeError: If ``data`` isn't bytes.

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """
        if self.finished:
            raise ValueError("An upload can only be used once.")

        if not isinstance(data, bytes):
            raise TypeError("`data` must be bytes, received", type(data))

        checksum_object = sync_helpers._get_checksum_object(self._checksum_type)

        if checksum_object is not None:
            checksum_object.update(data)
            actual_checksum = sync_helpers.prepare_checksum_digest(
                checksum_object.digest()
            )
            metadata_key = sync_helpers._get_metadata_key(self._checksum_type)
            metadata[metadata_key] = actual_checksum

        content, multipart_boundary = construct_multipart_request(
            data, metadata, content_type
        )
        multipart_content_type = _RELATED_HEADER + multipart_boundary + b'"'

        self._headers[_CONTENT_TYPE_HEADER] = multipart_content_type

        return _POST, self.upload_url, content, self._headers

    def transmit(self, transport, data, metadata, content_type, timeout=None):
        """Transmit the resource to be uploaded.

        Args:
            transport (object): An object which can make authenticated
                requests.
            data (bytes): The resource content to be uploaded.
            metadata (Mapping[str, str]): The resource metadata, such as an
                ACL list.
            content_type (str): The content type of the resource, e.g. a JPEG
                image has content type ``image/jpeg``.
            timeout (Optional[Union[float, aiohttp.ClientTimeout]]):
                The number of seconds to wait for the server response.
                Depending on the retry strategy, a request may be repeated
                several times using the same timeout each time.
                Can also be passed as an `aiohttp.ClientTimeout` object.

        Raises:
            NotImplementedError: Always, since virtual.
        """
        raise NotImplementedError("This implementation is virtual.")


class ResumableUpload(UploadBase, sync_upload.ResumableUpload):
    """Initiate and fulfill a resumable upload to a Google API.

    A **resumable** upload sends an initial request with the resource metadata
    and then gets assigned an upload ID / upload URL to send bytes to.
    Using the upload URL, the upload is then done in chunks (determined by
    the user) until all bytes have been uploaded.

    Args:
        upload_url (str): The URL where the resumable upload will be initiated.
        chunk_size (int): The size of each chunk used to upload the resource.
        headers (Optional[Mapping[str, str]]): Extra headers that should
            be sent with the :meth:`initiate` request, e.g. headers for
            encrypted data. These **will not** be sent with
            :meth:`transmit_next_chunk` or :meth:`recover` requests.
        checksum Optional([str]): The type of checksum to compute to verify
            the integrity of the object. After the upload is complete, the
            server-computed checksum of the resulting object will be read
            and google.resumable_media.common.DataCorruption will be raised on
            a mismatch. The corrupted file will not be deleted from the remote
            host automatically. Supported values are "md5", "crc32c" and None.
            The default is None.

    Attributes:
        upload_url (str): The URL where the content will be uploaded.

    Raises:
        ValueError: If ``chunk_size`` is not a multiple of
            :data:`.UPLOAD_CHUNK_SIZE`.
    """

    def __init__(self, upload_url, chunk_size, checksum=None, headers=None):
        super(ResumableUpload, self).__init__(upload_url, headers=headers)
        if chunk_size % _async_resumable_media.UPLOAD_CHUNK_SIZE != 0:
            raise ValueError(
                "{} KB must divide chunk size".format(
                    _async_resumable_media.UPLOAD_CHUNK_SIZE / 1024
                )
            )
        self._chunk_size = chunk_size
        self._stream = None
        self._content_type = None
        self._bytes_uploaded = 0
        self._bytes_checksummed = 0
        self._checksum_type = checksum
        self._checksum_object = None
        self._total_bytes = None
        self._resumable_url = None
        self._invalid = False

    @property
    def invalid(self):
        """bool: Indicates if the upload is in an invalid state.

        This will occur if a call to :meth:`transmit_next_chunk` fails.
        To recover from such a failure, call :meth:`recover`.
        """
        return self._invalid

    @property
    def chunk_size(self):
        """int: The size of each chunk used to upload the resource."""
        return self._chunk_size

    @property
    def resumable_url(self):
        """Optional[str]: The URL of the in-progress resumable upload."""
        return self._resumable_url

    @property
    def bytes_uploaded(self):
        """int: Number of bytes that have been uploaded."""
        return self._bytes_uploaded

    @property
    def total_bytes(self):
        """Optional[int]: The total number of bytes to be uploaded.

        If this upload is initiated (via :meth:`initiate`) with
        ``stream_final=True``, this value will be populated based on the size
        of the ``stream`` being uploaded. (By default ``stream_final=True``.)

        If this upload is initiated with ``stream_final=False``,
        :attr:`total_bytes` will be :data:`None` since it cannot be
        determined from the stream.
        """
        return self._total_bytes

    def _prepare_initiate_request(
        self, stream, metadata, content_type, total_bytes=None, stream_final=True
    ):
        """Prepare the contents of HTTP request to initiate upload.

        This is everything that must be done before a request that doesn't
        require network I/O (or other I/O). This is based on the `sans-I/O`_
        philosophy.

        Args:
            stream (IO[bytes]): The stream (i.e. file-like object) that will
                be uploaded. The stream **must** be at the beginning (i.e.
                ``stream.tell() == 0``).
            metadata (Mapping[str, str]): The resource metadata, such as an
                ACL list.
            content_type (str): The content type of the resource, e.g. a JPEG
                image has content type ``image/jpeg``.
            total_bytes (Optional[int]): The total number of bytes to be
                uploaded. If specified, the upload size **will not** be
                determined from the stream (even if ``stream_final=True``).
            stream_final (Optional[bool]): Indicates if the ``stream`` is
                "final" (i.e. no more bytes will be added to it). In this case
                we determine the upload size from the size of the stream. If
                ``total_bytes`` is passed, this argument will be ignored.

        Returns:
            Tuple[str, str, bytes, Mapping[str, str]]: The quadruple

              * HTTP verb for the request (always POST)
              * the URL for the request
              * the body of the request
              * headers for the request

        Raises:
            ValueError: If the current upload has already been initiated.
            ValueError: If ``stream`` is not at the beginning.

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """
        if self.resumable_url is not None:
            raise ValueError("This upload has already been initiated.")
        if stream.tell() != 0:
            raise ValueError("Stream must be at beginning.")

        self._stream = stream
        self._content_type = content_type
        headers = {
            _CONTENT_TYPE_HEADER: "application/json; charset=UTF-8",
            "x-upload-content-type": content_type,
        }
        # Set the total bytes if possible.
        if total_bytes is not None:
            self._total_bytes = total_bytes
        elif stream_final:
            self._total_bytes = get_total_bytes(stream)
        # Add the total bytes to the headers if set.
        if self._total_bytes is not None:
            content_length = "{:d}".format(self._total_bytes)
            headers["x-upload-content-length"] = content_length

        headers.update(self._headers)
        payload = json.dumps(metadata).encode("utf-8")
        return _POST, self.upload_url, payload, headers

    def _process_initiate_response(self, response):
        """Process the response from an HTTP request that initiated upload.

        This is everything that must be done after a request that doesn't
        require network I/O (or other I/O). This is based on the `sans-I/O`_
        philosophy.

        This method takes the URL from the ``Location`` header and stores it
        for future use. Within that URL, we assume the ``upload_id`` query
        parameter has been included, but we do not check.

        Args:
            response (object): The HTTP response object (need headers).

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """
        _helpers.require_status_code(
            response,
            (http.client.OK,),
            self._get_status_code,
            callback=self._make_invalid,
        )
        self._resumable_url = _helpers.header_required(
            response, "location", self._get_headers
        )

    def initiate(
        self,
        transport,
        stream,
        metadata,
        content_type,
        total_bytes=None,
        stream_final=True,
        timeout=None,
    ):
        """Initiate a resumable upload.

        By default, this method assumes your ``stream`` is in a "final"
        state ready to transmit. However, ``stream_final=False`` can be used
        to indicate that the size of the resource is not known. This can happen
        if bytes are being dynamically fed into ``stream``, e.g. if the stream
        is attached to application logs.

        If ``stream_final=False`` is used, :attr:`chunk_size` bytes will be
        read from the stream every time :meth:`transmit_next_chunk` is called.
        If one of those reads produces strictly fewer bites than the chunk
        size, the upload will be concluded.

        Args:
            transport (object): An object which can make authenticated
                requests.
            stream (IO[bytes]): The stream (i.e. file-like object) that will
                be uploaded. The stream **must** be at the beginning (i.e.
                ``stream.tell() == 0``).
            metadata (Mapping[str, str]): The resource metadata, such as an
                ACL list.
            content_type (str): The content type of the resource, e.g. a JPEG
                image has content type ``image/jpeg``.
            total_bytes (Optional[int]): The total number of bytes to be
                uploaded. If specified, the upload size **will not** be
                determined from the stream (even if ``stream_final=True``).
            stream_final (Optional[bool]): Indicates if the ``stream`` is
                "final" (i.e. no more bytes will be added to it). In this case
                we determine the upload size from the size of the stream. If
                ``total_bytes`` is passed, this argument will be ignored.
            timeout (Optional[Union[float, aiohttp.ClientTimeout]]):
                The number of seconds to wait for the server response.
                Depending on the retry strategy, a request may be repeated
                several times using the same timeout each time.
                Can also be passed as an `aiohttp.ClientTimeout` object.

        Raises:
            NotImplementedError: Always, since virtual.
        """
        raise NotImplementedError("This implementation is virtual.")

    def _prepare_request(self):
        """Prepare the contents of HTTP request to upload a chunk.

        This is everything that must be done before a request that doesn't
        require network I/O. This is based on the `sans-I/O`_ philosophy.

        For the time being, this **does require** some form of I/O to read
        a chunk from ``stream`` (via :func:`get_next_chunk`). However, this
        will (almost) certainly not be network I/O.

        Returns:
            Tuple[str, str, bytes, Mapping[str, str]]: The quadruple

              * HTTP verb for the request (always PUT)
              * the URL for the request
              * the body of the request
              * headers for the request

            The headers **do not** incorporate the ``_headers`` on the
            current instance.

        Raises:
            ValueError: If the current upload has finished.
            ValueError: If the current upload is in an invalid state.
            ValueError: If the current upload has not been initiated.
            ValueError: If the location in the stream (i.e. ``stream.tell()``)
                does not agree with ``bytes_uploaded``.

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """
        if self.finished:
            raise ValueError("Upload has finished.")
        if self.invalid:
            raise ValueError(
                "Upload is in an invalid state. To recover call `recover()`."
            )
        if self.resumable_url is None:
            raise ValueError(
                "This upload has not been initiated. Please call "
                "initiate() before beginning to transmit chunks."
            )

        start_byte, payload, content_range = get_next_chunk(
            self._stream, self._chunk_size, self._total_bytes
        )
        if start_byte != self.bytes_uploaded:
            msg = _STREAM_ERROR_TEMPLATE.format(start_byte, self.bytes_uploaded)
            raise ValueError(msg)

        self._update_checksum(start_byte, payload)

        headers = {
            _CONTENT_TYPE_HEADER: self._content_type,
            _helpers.CONTENT_RANGE_HEADER: content_range,
        }
        return _PUT, self.resumable_url, payload, headers

    def _make_invalid(self):
        """Simple setter for ``invalid``.

        This is intended to be passed along as a callback to helpers that
        raise an exception so they can mark this instance as invalid before
        raising.
        """
        self._invalid = True

    async def _process_resumable_response(self, response, bytes_sent):
        """Process the response from an HTTP request.

        This is everything that must be done after a request that doesn't
        require network I/O (or other I/O). This is based on the `sans-I/O`_
        philosophy.

        Args:
            response (object): The HTTP response object.
            bytes_sent (int): The number of bytes sent in the request that
                ``response`` was returned for.

        Raises:
            ~google.resumable_media.common.InvalidResponse: If the status
                code is 308 and the ``range`` header is not of the form
                ``bytes 0-{end}``.
            ~google.resumable_media.common.InvalidResponse: If the status
                code is not 200 or 308.

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """
        status_code = _helpers.require_status_code(
            response,
            (http.client.OK, http.client.PERMANENT_REDIRECT),
            self._get_status_code,
            callback=self._make_invalid,
        )
        if status_code == http.client.OK:
            # NOTE: We use the "local" information of ``bytes_sent`` to update
            #       ``bytes_uploaded``, but do not verify this against other
            #       state. However, there may be some other information:
            #
            #       * a ``size`` key in JSON response body
            #       * the ``total_bytes`` attribute (if set)
            #       * ``stream.tell()`` (relying on fact that ``initiate()``
            #         requires stream to be at the beginning)
            self._bytes_uploaded = self._bytes_uploaded + bytes_sent
            # Tombstone the current upload so it cannot be used again.
            self._finished = True
            # Validate the checksum. This can raise an exception on failure.
            await self._validate_checksum(response)
        else:
            bytes_range = _helpers.header_required(
                response,
                _helpers.RANGE_HEADER,
                self._get_headers,
                callback=self._make_invalid,
            )
            match = _BYTES_RANGE_RE.match(bytes_range)
            if match is None:
                self._make_invalid()
                raise common.InvalidResponse(
                    response,
                    'Unexpected "range" header',
                    bytes_range,
                    'Expected to be of the form "bytes=0-{end}"',
                )
            self._bytes_uploaded = int(match.group("end_byte")) + 1

    async def _validate_checksum(self, response):
        """Check the computed checksum, if any, against the response headers.
        Args:
            response (object): The HTTP response object.
        Raises:
            ~google.resumable_media.common.DataCorruption: If the checksum
            computed locally and the checksum reported by the remote host do
            not match.
        """
        if self._checksum_type is None:
            return
        metadata_key = sync_helpers._get_metadata_key(self._checksum_type)
        metadata = await response.json()
        remote_checksum = metadata.get(metadata_key)
        if remote_checksum is None:
            raise common.InvalidResponse(
                response,
                _UPLOAD_METADATA_NO_APPROPRIATE_CHECKSUM_MESSAGE.format(metadata_key),
                self._get_headers(response),
            )
        local_checksum = sync_helpers.prepare_checksum_digest(
            self._checksum_object.digest()
        )
        if local_checksum != remote_checksum:
            raise common.DataCorruption(
                response,
                _UPLOAD_CHECKSUM_MISMATCH_MESSAGE.format(
                    self._checksum_type.upper(), local_checksum, remote_checksum
                ),
            )

    def transmit_next_chunk(self, transport, timeout=None):
        """Transmit the next chunk of the resource to be uploaded.

        If the current upload was initiated with ``stream_final=False``,
        this method will dynamically determine if the upload has completed.
        The upload will be considered complete if the stream produces
        fewer than :attr:`chunk_size` bytes when a chunk is read from it.

        Args:
            transport (object): An object which can make authenticated
                requests.
            timeout (Optional[Union[float, aiohttp.ClientTimeout]]):
                The number of seconds to wait for the server response.
                Depending on the retry strategy, a request may be repeated
                several times using the same timeout each time.
                Can also be passed as an `aiohttp.ClientTimeout` object.

        Raises:
            NotImplementedError: Always, since virtual.
        """
        raise NotImplementedError("This implementation is virtual.")

    def _prepare_recover_request(self):
        """Prepare the contents of HTTP request to recover from failure.

        This is everything that must be done before a request that doesn't
        require network I/O. This is based on the `sans-I/O`_ philosophy.

        We assume that the :attr:`resumable_url` is set (i.e. the only way
        the upload can end up :attr:`invalid` is if it has been initiated.

        Returns:
            Tuple[str, str, NoneType, Mapping[str, str]]: The quadruple

              * HTTP verb for the request (always PUT)
              * the URL for the request
              * the body of the request (always :data:`None`)
              * headers for the request

            The headers **do not** incorporate the ``_headers`` on the
            current instance.

        Raises:
            ValueError: If the current upload is not in an invalid state.

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """
        if not self.invalid:
            raise ValueError("Upload is not in invalid state, no need to recover.")

        headers = {_helpers.CONTENT_RANGE_HEADER: "bytes */*"}
        return _PUT, self.resumable_url, None, headers

    def _process_recover_response(self, response):
        """Process the response from an HTTP request to recover from failure.

        This is everything that must be done after a request that doesn't
        require network I/O (or other I/O). This is based on the `sans-I/O`_
        philosophy.

        Args:
            response (object): The HTTP response object.

        Raises:
            ~google.resumable_media.common.InvalidResponse: If the status
                code is not 308.
            ~google.resumable_media.common.InvalidResponse: If the status
                code is 308 and the ``range`` header is not of the form
                ``bytes 0-{end}``.

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """
        _helpers.require_status_code(
            response,
            (http.client.PERMANENT_REDIRECT,),
            self._get_status_code,
        )
        headers = self._get_headers(response)
        if _helpers.RANGE_HEADER in headers:
            bytes_range = headers[_helpers.RANGE_HEADER]
            match = _BYTES_RANGE_RE.match(bytes_range)
            if match is None:
                raise common.InvalidResponse(
                    response,
                    'Unexpected "range" header',
                    bytes_range,
                    'Expected to be of the form "bytes=0-{end}"',
                )
            self._bytes_uploaded = int(match.group("end_byte")) + 1
        else:
            # In this case, the upload has not "begun".
            self._bytes_uploaded = 0

        self._stream.seek(self._bytes_uploaded)
        self._invalid = False

    def recover(self, transport):
        """Recover from a failure.

        This method should be used when a :class:`ResumableUpload` is in an
        :attr:`~ResumableUpload.invalid` state due to a request failure.

        This will verify the progress with the server and make sure the
        current upload is in a valid state before :meth:`transmit_next_chunk`
        can be used again.

        Args:
            transport (object): An object which can make authenticated
                requests.

        Raises:
            NotImplementedError: Always, since virtual.
        """
        raise NotImplementedError("This implementation is virtual.")


def get_boundary():
    """Get a random boundary for a multipart request.

    Returns:
        bytes: The boundary used to separate parts of a multipart request.
    """
    random_int = random.randrange(sys.maxsize)
    boundary = _BOUNDARY_FORMAT.format(random_int)
    # NOTE: Neither % formatting nor .format() are available for byte strings
    #       in Python 3.4, so we must use unicode strings as templates.
    return boundary.encode("utf-8")


def construct_multipart_request(data, metadata, content_type):
    """Construct a multipart request body.

    Args:
        data (bytes): The resource content (UTF-8 encoded as bytes)
            to be uploaded.
        metadata (Mapping[str, str]): The resource metadata, such as an
            ACL list.
        content_type (str): The content type of the resource, e.g. a JPEG
            image has content type ``image/jpeg``.

    Returns:
        Tuple[bytes, bytes]: The multipart request body and the boundary used
        between each part.
    """
    multipart_boundary = get_boundary()
    json_bytes = json.dumps(metadata).encode("utf-8")
    content_type = content_type.encode("utf-8")
    # Combine the two parts into a multipart payload.
    # NOTE: We'd prefer a bytes template but are restricted by Python 3.4.
    boundary_sep = _MULTIPART_SEP + multipart_boundary
    content = (
        boundary_sep
        + _MULTIPART_BEGIN
        + json_bytes
        + _CRLF
        + boundary_sep
        + _CRLF
        + b"content-type: "
        + content_type
        + _CRLF
        + _CRLF
        + data  # Empty line between headers and body.
        + _CRLF
        + boundary_sep
        + _MULTIPART_SEP
    )

    return content, multipart_boundary


def get_total_bytes(stream):
    """Determine the total number of bytes in a stream.

    Args:
       stream (IO[bytes]): The stream (i.e. file-like object).

    Returns:
        int: The number of bytes.
    """
    current_position = stream.tell()
    # NOTE: ``.seek()`` **should** return the same value that ``.tell()``
    #       returns, but in Python 2, ``file`` objects do not.
    stream.seek(0, os.SEEK_END)
    end_position = stream.tell()
    # Go back to the initial position.
    stream.seek(current_position)

    return end_position


def get_next_chunk(stream, chunk_size, total_bytes):
    """Get a chunk from an I/O stream.

    The ``stream`` may have fewer bytes remaining than ``chunk_size``
    so it may not always be the case that
    ``end_byte == start_byte + chunk_size - 1``.

    Args:
        stream (IO[bytes]): The stream (i.e. file-like object).
        chunk_size (int): The size of the chunk to be read from the ``stream``.
        total_bytes (Optional[int]): The (expected) total number of bytes
            in the ``stream``.

    Returns:
        Tuple[int, bytes, str]: Triple of:

          * the start byte index
          * the content in between the start and end bytes (inclusive)
          * content range header for the chunk (slice) that has been read

    Raises:
        ValueError: If ``total_bytes == 0`` but ``stream.read()`` yields
            non-empty content.
        ValueError: If there is no data left to consume. This corresponds
            exactly to the case ``end_byte < start_byte``, which can only
            occur if ``end_byte == start_byte - 1``.
    """
    start_byte = stream.tell()
    if total_bytes is not None and start_byte + chunk_size >= total_bytes > 0:
        payload = stream.read(total_bytes - start_byte)
    else:
        payload = stream.read(chunk_size)
    end_byte = stream.tell() - 1

    num_bytes_read = len(payload)
    if total_bytes is None:
        if num_bytes_read < chunk_size:
            # We now **KNOW** the total number of bytes.
            total_bytes = end_byte + 1
    elif total_bytes == 0:
        # NOTE: We also expect ``start_byte == 0`` here but don't check
        #       because ``_prepare_initiate_request()`` requires the
        #       stream to be at the beginning.
        if num_bytes_read != 0:
            raise ValueError(
                "Stream specified as empty, but produced non-empty content."
            )
    else:
        if num_bytes_read == 0:
            raise ValueError(
                "Stream is already exhausted. There is no content remaining."
            )

    content_range = get_content_range(start_byte, end_byte, total_bytes)
    return start_byte, payload, content_range


def get_content_range(start_byte, end_byte, total_bytes):
    """Convert start, end and total into content range header.

    If ``total_bytes`` is not known, uses "bytes {start}-{end}/*".
    If we are dealing with an empty range (i.e. ``end_byte < start_byte``)
    then "bytes */{total}" is used.

    This function **ASSUMES** that if the size is not known, the caller will
    not also pass an empty range.

    Args:
        start_byte (int): The start (inclusive) of the byte range.
        end_byte (int): The end (inclusive) of the byte range.
        total_bytes (Optional[int]): The number of bytes in the byte
            range (if known).

    Returns:
        str: The content range header.
    """
    if total_bytes is None:
        return _RANGE_UNKNOWN_TEMPLATE.format(start_byte, end_byte)
    elif end_byte < start_byte:
        return _EMPTY_RANGE_TEMPLATE.format(total_bytes)
    else:
        return _CONTENT_RANGE_TEMPLATE.format(start_byte, end_byte, total_bytes)
