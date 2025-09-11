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
import re
import sys
import urllib.parse

from google import resumable_media
from google.resumable_media import _helpers
from google.resumable_media import common

from xml.etree import ElementTree


_CONTENT_TYPE_HEADER = "content-type"
_CONTENT_RANGE_TEMPLATE = "bytes {:d}-{:d}/{:d}"
_RANGE_UNKNOWN_TEMPLATE = "bytes {:d}-{:d}/*"
_EMPTY_RANGE_TEMPLATE = "bytes */{:d}"
_BOUNDARY_WIDTH = len(str(sys.maxsize - 1))
_BOUNDARY_FORMAT = "==============={{:0{:d}d}}==".format(_BOUNDARY_WIDTH)
_MULTIPART_SEP = b"--"
_CRLF = b"\r\n"
_MULTIPART_BEGIN = b"\r\ncontent-type: application/json; charset=UTF-8\r\n\r\n"
_RELATED_HEADER = b'multipart/related; boundary="'
_BYTES_RANGE_RE = re.compile(r"bytes=0-(?P<end_byte>\d+)", flags=re.IGNORECASE)
_STREAM_ERROR_TEMPLATE = (
    "Bytes stream is in unexpected state. "
    "The local stream has had {:d} bytes read from it while "
    "{:d} bytes have already been updated (they should match)."
)
_STREAM_READ_PAST_TEMPLATE = (
    "{:d} bytes have been read from the stream, which exceeds "
    "the expected total {:d}."
)
_DELETE = "DELETE"
_POST = "POST"
_PUT = "PUT"
_UPLOAD_CHECKSUM_MISMATCH_MESSAGE = (
    "The computed ``{}`` checksum, ``{}``, and the checksum reported by the "
    "remote host, ``{}``, did not match."
)
_UPLOAD_METADATA_NO_APPROPRIATE_CHECKSUM_MESSAGE = (
    "Response metadata had no ``{}`` value; checksum could not be validated."
)
_UPLOAD_HEADER_NO_APPROPRIATE_CHECKSUM_MESSAGE = (
    "Response headers had no ``{}`` value; checksum could not be validated."
)
_MPU_INITIATE_QUERY = "?uploads"
_MPU_PART_QUERY_TEMPLATE = "?partNumber={part}&uploadId={upload_id}"
_S3_COMPAT_XML_NAMESPACE = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_UPLOAD_ID_NODE = "UploadId"
_MPU_FINAL_QUERY_TEMPLATE = "?uploadId={upload_id}"


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


class MultipartUpload(UploadBase):
    """Upload a resource with metadata to a Google API.

    A **multipart** upload sends both metadata and the resource in a single
    (multipart) request.

    Args:
        upload_url (str): The URL where the content will be uploaded.
        headers (Optional[Mapping[str, str]]): Extra headers that should
            be sent with the request, e.g. headers for encrypted data.
        checksum (Optional([str])): The type of checksum to compute to verify
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

        checksum_object = _helpers._get_checksum_object(self._checksum_type)
        if checksum_object is not None:
            checksum_object.update(data)
            actual_checksum = _helpers.prepare_checksum_digest(checksum_object.digest())
            metadata_key = _helpers._get_metadata_key(self._checksum_type)
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


class ResumableUpload(UploadBase):
    """Initiate and fulfill a resumable upload to a Google API.

    A **resumable** upload sends an initial request with the resource metadata
    and then gets assigned an upload ID / upload URL to send bytes to.
    Using the upload URL, the upload is then done in chunks (determined by
    the user) until all bytes have been uploaded.

    Args:
        upload_url (str): The URL where the resumable upload will be initiated.
        chunk_size (int): The size of each chunk used to upload the resource.
        headers (Optional[Mapping[str, str]]): Extra headers that should
            be sent with every request.
        checksum (Optional([str])): The type of checksum to compute to verify
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
        if chunk_size % resumable_media.UPLOAD_CHUNK_SIZE != 0:
            raise ValueError(
                "{} KB must divide chunk size".format(
                    resumable_media.UPLOAD_CHUNK_SIZE / 1024
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

        # Signed URL requires content type set directly - not through x-upload-content-type
        parse_result = urllib.parse.urlparse(self.upload_url)
        parsed_query = urllib.parse.parse_qs(parse_result.query)
        if "x-goog-signature" in parsed_query or "X-Goog-Signature" in parsed_query:
            # Deconstruct **self._headers first so that content type defined here takes priority
            headers = {**self._headers, _CONTENT_TYPE_HEADER: content_type}
        else:
            # Deconstruct **self._headers first so that content type defined here takes priority
            headers = {
                **self._headers,
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
            (http.client.OK, http.client.CREATED),
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

            The headers incorporate the ``_headers`` on the current instance.

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
            **self._headers,
            _CONTENT_TYPE_HEADER: self._content_type,
            _helpers.CONTENT_RANGE_HEADER: content_range,
        }
        return _PUT, self.resumable_url, payload, headers

    def _update_checksum(self, start_byte, payload):
        """Update the checksum with the payload if not already updated.

        Because error recovery can result in bytes being transmitted more than
        once, the checksum tracks the number of bytes checked in
        self._bytes_checksummed and skips bytes that have already been summed.
        """
        if not self._checksum_type:
            return

        if not self._checksum_object:
            self._checksum_object = _helpers._get_checksum_object(self._checksum_type)

        if start_byte < self._bytes_checksummed:
            offset = self._bytes_checksummed - start_byte
            data = payload[offset:]
        else:
            data = payload

        self._checksum_object.update(data)
        self._bytes_checksummed += len(data)

    def _make_invalid(self):
        """Simple setter for ``invalid``.

        This is intended to be passed along as a callback to helpers that
        raise an exception so they can mark this instance as invalid before
        raising.
        """
        self._invalid = True

    def _process_resumable_response(self, response, bytes_sent):
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
            self._validate_checksum(response)
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

    def _validate_checksum(self, response):
        """Check the computed checksum, if any, against the recieved metadata.

        Args:
            response (object): The HTTP response object.

        Raises:
            ~google.resumable_media.common.DataCorruption: If the checksum
            computed locally and the checksum reported by the remote host do
            not match.
        """
        if self._checksum_type is None:
            return
        metadata_key = _helpers._get_metadata_key(self._checksum_type)
        metadata = response.json()
        remote_checksum = metadata.get(metadata_key)
        if remote_checksum is None:
            raise common.InvalidResponse(
                response,
                _UPLOAD_METADATA_NO_APPROPRIATE_CHECKSUM_MESSAGE.format(metadata_key),
                self._get_headers(response),
            )
        local_checksum = _helpers.prepare_checksum_digest(
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

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """
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
            response, (http.client.PERMANENT_REDIRECT,), self._get_status_code
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


class XMLMPUContainer(UploadBase):
    """Initiate and close an upload using the XML MPU API.

    An XML MPU sends an initial request and then receives an upload ID.
    Using the upload ID, the upload is then done in numbered parts and the
    parts can be uploaded concurrently.

    In order to avoid concurrency issues with this container object, the
    uploading of individual parts is handled separately, by XMLMPUPart objects
    spawned from this container class. The XMLMPUPart objects are not
    necessarily in the same process as the container, so they do not update the
    container automatically.

    MPUs are sometimes referred to as "Multipart Uploads", which is ambiguous
    given the JSON multipart upload, so the abbreviation "MPU" will be used
    throughout.

    See: https://cloud.google.com/storage/docs/multipart-uploads

    Args:
        upload_url (str): The URL of the object (without query parameters). The
            initiate, PUT, and finalization requests will all use this URL, with
            varying query parameters.
        filename (str): The name (path) of the file to upload.
        headers (Optional[Mapping[str, str]]): Extra headers that should
            be sent with every request.

    Attributes:
        upload_url (str): The URL where the content will be uploaded.
        upload_id (Optional(str)): The ID of the upload from the initialization
            response.
    """

    def __init__(self, upload_url, filename, headers=None, upload_id=None):
        super().__init__(upload_url, headers=headers)
        self._filename = filename
        self._upload_id = upload_id
        self._parts = {}

    @property
    def upload_id(self):
        return self._upload_id

    def register_part(self, part_number, etag):
        """Register an uploaded part by part number and corresponding etag.

        XMLMPUPart objects represent individual parts, and their part number
        and etag can be registered to the container object with this method
        and therefore incorporated in the finalize() call to finish the upload.

        This method accepts part_number and etag, but not XMLMPUPart objects
        themselves, to reduce the complexity involved in running XMLMPUPart
        uploads in separate processes.

        Args:
            part_number (int): The part number. Parts are assembled into the
                final uploaded object with finalize() in order of their part
                numbers.
            etag (str): The etag included in the server response after upload.
        """
        self._parts[part_number] = etag

    def _prepare_initiate_request(self, content_type):
        """Prepare the contents of HTTP request to initiate upload.

        This is everything that must be done before a request that doesn't
        require network I/O (or other I/O). This is based on the `sans-I/O`_
        philosophy.

        Args:
            content_type (str): The content type of the resource, e.g. a JPEG
                image has content type ``image/jpeg``.

        Returns:
            Tuple[str, str, bytes, Mapping[str, str]]: The quadruple

              * HTTP verb for the request (always POST)
              * the URL for the request
              * the body of the request
              * headers for the request

        Raises:
            ValueError: If the current upload has already been initiated.

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """
        if self.upload_id is not None:
            raise ValueError("This upload has already been initiated.")

        initiate_url = self.upload_url + _MPU_INITIATE_QUERY

        headers = {
            **self._headers,
            _CONTENT_TYPE_HEADER: content_type,
        }
        return _POST, initiate_url, None, headers

    def _process_initiate_response(self, response):
        """Process the response from an HTTP request that initiated the upload.

        This is everything that must be done after a request that doesn't
        require network I/O (or other I/O). This is based on the `sans-I/O`_
        philosophy.

        This method takes the URL from the ``Location`` header and stores it
        for future use. Within that URL, we assume the ``upload_id`` query
        parameter has been included, but we do not check.

        Args:
            response (object): The HTTP response object.

        Raises:
            ~google.resumable_media.common.InvalidResponse: If the status
                code is not 200.

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """
        _helpers.require_status_code(response, (http.client.OK,), self._get_status_code)
        root = ElementTree.fromstring(response.text)
        self._upload_id = root.find(_S3_COMPAT_XML_NAMESPACE + _UPLOAD_ID_NODE).text

    def initiate(
        self,
        transport,
        content_type,
        timeout=None,
    ):
        """Initiate an MPU and record the upload ID.

        Args:
            transport (object): An object which can make authenticated
                requests.
            content_type (str): The content type of the resource, e.g. a JPEG
                image has content type ``image/jpeg``.
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

    def _prepare_finalize_request(self):
        """Prepare the contents of an HTTP request to finalize the upload.

        All of the parts must be registered before calling this method.

        Returns:
            Tuple[str, str, bytes, Mapping[str, str]]: The quadruple

              * HTTP verb for the request (always POST)
              * the URL for the request
              * the body of the request
              * headers for the request

        Raises:
            ValueError: If the upload has not been initiated.
        """
        if self.upload_id is None:
            raise ValueError("This upload has not yet been initiated.")

        final_query = _MPU_FINAL_QUERY_TEMPLATE.format(upload_id=self._upload_id)
        finalize_url = self.upload_url + final_query
        final_xml_root = ElementTree.Element("CompleteMultipartUpload")
        for part_number, etag in self._parts.items():
            part = ElementTree.SubElement(final_xml_root, "Part")  # put in a loop
            ElementTree.SubElement(part, "PartNumber").text = str(part_number)
            ElementTree.SubElement(part, "ETag").text = etag
        payload = ElementTree.tostring(final_xml_root)
        return _POST, finalize_url, payload, self._headers

    def _process_finalize_response(self, response):
        """Process the response from an HTTP request that finalized the upload.

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

        _helpers.require_status_code(response, (http.client.OK,), self._get_status_code)
        self._finished = True

    def finalize(
        self,
        transport,
        timeout=None,
    ):
        """Finalize an MPU request with all the parts.

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

    def _prepare_cancel_request(self):
        """Prepare the contents of an HTTP request to cancel the upload.

        Returns:
            Tuple[str, str, bytes, Mapping[str, str]]: The quadruple

              * HTTP verb for the request (always DELETE)
              * the URL for the request
              * the body of the request
              * headers for the request

        Raises:
            ValueError: If the upload has not been initiated.
        """
        if self.upload_id is None:
            raise ValueError("This upload has not yet been initiated.")

        cancel_query = _MPU_FINAL_QUERY_TEMPLATE.format(upload_id=self._upload_id)
        cancel_url = self.upload_url + cancel_query
        return _DELETE, cancel_url, None, self._headers

    def _process_cancel_response(self, response):
        """Process the response from an HTTP request that canceled the upload.

        This is everything that must be done after a request that doesn't
        require network I/O (or other I/O). This is based on the `sans-I/O`_
        philosophy.

        Args:
            response (object): The HTTP response object.

        Raises:
            ~google.resumable_media.common.InvalidResponse: If the status
                code is not 204.

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """

        _helpers.require_status_code(
            response, (http.client.NO_CONTENT,), self._get_status_code
        )

    def cancel(
        self,
        transport,
        timeout=None,
    ):
        """Cancel an MPU request and permanently delete any uploaded parts.

        This cannot be undone.

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


class XMLMPUPart(UploadBase):
    """Upload a single part of an existing XML MPU container.

    An XML MPU sends an initial request and then receives an upload ID.
    Using the upload ID, the upload is then done in numbered parts and the
    parts can be uploaded concurrently.

    In order to avoid concurrency issues with the container object, the
    uploading of individual parts is handled separately by multiple objects
    of this class. Once a part is uploaded, it can be registered with the
    container with `container.register_part(part.part_number, part.etag)`.

    MPUs are sometimes referred to as "Multipart Uploads", which is ambiguous
    given the JSON multipart upload, so the abbreviation "MPU" will be used
    throughout.

    See: https://cloud.google.com/storage/docs/multipart-uploads

    Args:
        upload_url (str): The URL of the object (without query parameters).
        upload_id (str): The ID of the upload from the initialization response.
        filename (str): The name (path) of the file to upload.
        start (int): The byte index of the beginning of the part.
        end (int): The byte index of the end of the part.
        part_number (int): The part number. Part numbers will be assembled in
            sequential order when the container is finalized.
        headers (Optional[Mapping[str, str]]): Extra headers that should
            be sent with every request.
        checksum (Optional([str])): The type of checksum to compute to verify
            the integrity of the object. The request headers will be amended
            to include the computed value. Supported values are "md5", "crc32c"
            and None. The default is None.

    Attributes:
        upload_url (str): The URL of the object (without query parameters).
        upload_id (str): The ID of the upload from the initialization response.
        filename (str): The name (path) of the file to upload.
        start (int): The byte index of the beginning of the part.
        end (int): The byte index of the end of the part.
        part_number (int): The part number. Part numbers will be assembled in
            sequential order when the container is finalized.
        etag (Optional(str)): The etag returned by the service after upload.
    """

    def __init__(
        self,
        upload_url,
        upload_id,
        filename,
        start,
        end,
        part_number,
        headers=None,
        checksum=None,
    ):
        super().__init__(upload_url, headers=headers)
        self._filename = filename
        self._start = start
        self._end = end
        self._upload_id = upload_id
        self._part_number = part_number
        self._etag = None
        self._checksum_type = checksum
        self._checksum_object = None

    @property
    def part_number(self):
        return self._part_number

    @property
    def upload_id(self):
        return self._upload_id

    @property
    def filename(self):
        return self._filename

    @property
    def etag(self):
        return self._etag

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    def _prepare_upload_request(self):
        """Prepare the contents of HTTP request to upload a part.

        This is everything that must be done before a request that doesn't
        require network I/O. This is based on the `sans-I/O`_ philosophy.

        For the time being, this **does require** some form of I/O to read
        a part from ``stream`` (via :func:`get_part_payload`). However, this
        will (almost) certainly not be network I/O.

        Returns:
            Tuple[str, str, bytes, Mapping[str, str]]: The quadruple

              * HTTP verb for the request (always PUT)
              * the URL for the request
              * the body of the request
              * headers for the request

            The headers incorporate the ``_headers`` on the current instance.

        Raises:
            ValueError: If the current upload has finished.

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """
        if self.finished:
            raise ValueError("This part has already been uploaded.")

        with open(self._filename, "br") as f:
            f.seek(self._start)
            payload = f.read(self._end - self._start)

        self._checksum_object = _helpers._get_checksum_object(self._checksum_type)
        if self._checksum_object is not None:
            self._checksum_object.update(payload)

        part_query = _MPU_PART_QUERY_TEMPLATE.format(
            part=self._part_number, upload_id=self._upload_id
        )
        upload_url = self.upload_url + part_query
        return _PUT, upload_url, payload, self._headers

    def _process_upload_response(self, response):
        """Process the response from an HTTP request.

        This is everything that must be done after a request that doesn't
        require network I/O (or other I/O). This is based on the `sans-I/O`_
        philosophy.

        Args:
            response (object): The HTTP response object.

        Raises:
            ~google.resumable_media.common.InvalidResponse: If the status
                code is not 200 or the response is missing data.

        .. _sans-I/O: https://sans-io.readthedocs.io/
        """
        _helpers.require_status_code(
            response,
            (http.client.OK,),
            self._get_status_code,
        )

        self._validate_checksum(response)

        etag = _helpers.header_required(response, "etag", self._get_headers)
        self._etag = etag
        self._finished = True

    def upload(
        self,
        transport,
        timeout=None,
    ):
        """Upload the part.

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

    def _validate_checksum(self, response):
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

        remote_checksum = _helpers._get_uploaded_checksum_from_headers(
            response, self._get_headers, self._checksum_type
        )

        if remote_checksum is None:
            metadata_key = _helpers._get_metadata_key(self._checksum_type)
            raise common.InvalidResponse(
                response,
                _UPLOAD_METADATA_NO_APPROPRIATE_CHECKSUM_MESSAGE.format(metadata_key),
                self._get_headers(response),
            )
        local_checksum = _helpers.prepare_checksum_digest(
            self._checksum_object.digest()
        )
        if local_checksum != remote_checksum:
            raise common.DataCorruption(
                response,
                _UPLOAD_CHECKSUM_MISMATCH_MESSAGE.format(
                    self._checksum_type.upper(), local_checksum, remote_checksum
                ),
            )


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
