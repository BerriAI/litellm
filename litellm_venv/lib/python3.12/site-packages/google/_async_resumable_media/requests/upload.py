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

"""Support for resumable uploads.

Also supported here are simple (media) uploads and multipart
uploads that contain both metadata and a small file as payload.
"""


from google._async_resumable_media import _upload
from google._async_resumable_media.requests import _request_helpers


class SimpleUpload(_request_helpers.RequestsMixin, _upload.SimpleUpload):
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

    async def transmit(
        self,
        transport,
        data,
        content_type,
        timeout=_request_helpers._DEFAULT_TIMEOUT,
    ):
        """Transmit the resource to be uploaded.

        Args:
            transport (~requests.Session): A ``requests`` object which can
                make authenticated requests.
            data (bytes): The resource content to be uploaded.
            content_type (str): The content type of the resource, e.g. a JPEG
                image has content type ``image/jpeg``.
            timeout (Optional[Union[float, aiohttp.ClientTimeout]]):
                The number of seconds to wait for the server response.
                Depending on the retry strategy, a request may be repeated
                several times using the same timeout each time.
                Can also be passed as an `aiohttp.ClientTimeout` object.

        Returns:
            ~requests.Response: The HTTP response returned by ``transport``.
        """
        method, url, payload, headers = self._prepare_request(data, content_type)

        response = await _request_helpers.http_request(
            transport,
            method,
            url,
            data=payload,
            headers=headers,
            retry_strategy=self._retry_strategy,
            timeout=timeout,
        )
        self._process_response(response)
        return response


class MultipartUpload(_request_helpers.RequestsMixin, _upload.MultipartUpload):
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
            manually-set checksum value. Supported values are "md5",
            "crc32c" and None. The default is None.

    Attributes:
        upload_url (str): The URL where the content will be uploaded.
    """

    async def transmit(
        self,
        transport,
        data,
        metadata,
        content_type,
        timeout=_request_helpers._DEFAULT_TIMEOUT,
    ):
        """Transmit the resource to be uploaded.

        Args:
            transport (~requests.Session): A ``requests`` object which can
                make authenticated requests.
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

        Returns:
            ~requests.Response: The HTTP response returned by ``transport``.
        """
        method, url, payload, headers = self._prepare_request(
            data, metadata, content_type
        )

        response = await _request_helpers.http_request(
            transport,
            method,
            url,
            data=payload,
            headers=headers,
            retry_strategy=self._retry_strategy,
            timeout=timeout,
        )
        self._process_response(response)
        return response


class ResumableUpload(_request_helpers.RequestsMixin, _upload.ResumableUpload):
    """Initiate and fulfill a resumable upload to a Google API.

    A **resumable** upload sends an initial request with the resource metadata
    and then gets assigned an upload ID / upload URL to send bytes to.
    Using the upload URL, the upload is then done in chunks (determined by
    the user) until all bytes have been uploaded.

    When constructing a resumable upload, only the resumable upload URL and
    the chunk size are required:

    .. testsetup:: resumable-constructor

       bucket = 'bucket-foo'

    .. doctest:: resumable-constructor

       >>> from google.resumable_media.requests import ResumableUpload
       >>>
       >>> url_template = (
       ...     'https://www.googleapis.com/upload/storage/v1/b/{bucket}/o?'
       ...     'uploadType=resumable')
       >>> upload_url = url_template.format(bucket=bucket)
       >>>
       >>> chunk_size = 3 * 1024 * 1024  # 3MB
       >>> upload = ResumableUpload(upload_url, chunk_size)

    When initiating an upload (via :meth:`initiate`), the caller is expected
    to pass the resource being uploaded as a file-like ``stream``. If the size
    of the resource is explicitly known, it can be passed in directly:

    .. testsetup:: resumable-explicit-size

       import os
       import tempfile

       import mock
       import requests
       import http.client

       from google.resumable_media.requests import ResumableUpload

       upload_url = 'http://test.invalid'
       chunk_size = 3 * 1024 * 1024  # 3MB
       upload = ResumableUpload(upload_url, chunk_size)

       file_desc, filename = tempfile.mkstemp()
       os.close(file_desc)

       data = b'some bytes!'
       with open(filename, 'wb') as file_obj:
           file_obj.write(data)

       fake_response = requests.Response()
       fake_response.status_code = int(http.client.OK)
       fake_response._content = b''
       resumable_url = 'http://test.invalid?upload_id=7up'
       fake_response.headers['location'] = resumable_url

       post_method = mock.Mock(return_value=fake_response, spec=[])
       transport = mock.Mock(request=post_method, spec=['request'])

    .. doctest:: resumable-explicit-size

       >>> import os
       >>>
       >>> upload.total_bytes is None
       True
       >>>
       >>> stream = open(filename, 'rb')
       >>> total_bytes = os.path.getsize(filename)
       >>> metadata = {'name': filename}
       >>> response = upload.initiate(
       ...     transport, stream, metadata, 'text/plain',
       ...     total_bytes=total_bytes)
       >>> response
       <Response [200]>
       >>>
       >>> upload.total_bytes == total_bytes
       True

    .. testcleanup:: resumable-explicit-size

       os.remove(filename)

    If the stream is in a "final" state (i.e. it won't have any more bytes
    written to it), the total number of bytes can be determined implicitly
    from the ``stream`` itself:

    .. testsetup:: resumable-implicit-size

       import io

       import mock
       import requests
       import http.client

       from google.resumable_media.requests import ResumableUpload

       upload_url = 'http://test.invalid'
       chunk_size = 3 * 1024 * 1024  # 3MB
       upload = ResumableUpload(upload_url, chunk_size)

       fake_response = requests.Response()
       fake_response.status_code = int(http.client.OK)
       fake_response._content = b''
       resumable_url = 'http://test.invalid?upload_id=7up'
       fake_response.headers['location'] = resumable_url

       post_method = mock.Mock(return_value=fake_response, spec=[])
       transport = mock.Mock(request=post_method, spec=['request'])

       data = b'some MOAR bytes!'
       metadata = {'name': 'some-file.jpg'}
       content_type = 'image/jpeg'

    .. doctest:: resumable-implicit-size

       >>> stream = io.BytesIO(data)
       >>> response = upload.initiate(
       ...     transport, stream, metadata, content_type)
       >>>
       >>> upload.total_bytes == len(data)
       True

    If the size of the resource is **unknown** when the upload is initiated,
    the ``stream_final`` argument can be used. This might occur if the
    resource is being dynamically created on the client (e.g. application
    logs). To use this argument:

    .. testsetup:: resumable-unknown-size

       import io

       import mock
       import requests
       import http.client

       from google.resumable_media.requests import ResumableUpload

       upload_url = 'http://test.invalid'
       chunk_size = 3 * 1024 * 1024  # 3MB
       upload = ResumableUpload(upload_url, chunk_size)

       fake_response = requests.Response()
       fake_response.status_code = int(http.client.OK)
       fake_response._content = b''
       resumable_url = 'http://test.invalid?upload_id=7up'
       fake_response.headers['location'] = resumable_url

       post_method = mock.Mock(return_value=fake_response, spec=[])
       transport = mock.Mock(request=post_method, spec=['request'])

       metadata = {'name': 'some-file.jpg'}
       content_type = 'application/octet-stream'

       stream = io.BytesIO(b'data')

    .. doctest:: resumable-unknown-size

       >>> response = upload.initiate(
       ...     transport, stream, metadata, content_type,
       ...     stream_final=False)
       >>>
       >>> upload.total_bytes is None
       True

    Args:
        upload_url (str): The URL where the resumable upload will be initiated.
        chunk_size (int): The size of each chunk used to upload the resource.
        headers (Optional[Mapping[str, str]]): Extra headers that should
            be sent with the :meth:`initiate` request, e.g. headers for
            encrypted data. These **will not** be sent with
            :meth:`transmit_next_chunk` or :meth:`recover` requests.
        checksum Optional([str]): The type of checksum to compute to verify
            the integrity of the object. After the upload is complete, the
            server-computed checksum of the resulting object will be checked
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

    async def initiate(
        self,
        transport,
        stream,
        metadata,
        content_type,
        total_bytes=None,
        stream_final=True,
        timeout=_request_helpers._DEFAULT_TIMEOUT,
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
            transport (~requests.Session): A ``requests`` object which can
                make authenticated requests.
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

        Returns:
            ~requests.Response: The HTTP response returned by ``transport``.
        """
        method, url, payload, headers = self._prepare_initiate_request(
            stream,
            metadata,
            content_type,
            total_bytes=total_bytes,
            stream_final=stream_final,
        )
        response = await _request_helpers.http_request(
            transport,
            method,
            url,
            data=payload,
            headers=headers,
            retry_strategy=self._retry_strategy,
            timeout=timeout,
        )
        self._process_initiate_response(response)
        return response

    async def transmit_next_chunk(
        self, transport, timeout=_request_helpers._DEFAULT_TIMEOUT
    ):
        """Transmit the next chunk of the resource to be uploaded.

        If the current upload was initiated with ``stream_final=False``,
        this method will dynamically determine if the upload has completed.
        The upload will be considered complete if the stream produces
        fewer than :attr:`chunk_size` bytes when a chunk is read from it.

        In the case of failure, an exception is thrown that preserves the
        failed response:

        .. testsetup:: bad-response

           import io

           import mock
           import requests
           import http.client

           from google import resumable_media
           import google.resumable_media.requests.upload as upload_mod

           transport = mock.Mock(spec=['request'])
           fake_response = requests.Response()
           fake_response.status_code = int(http.client.BAD_REQUEST)
           transport.request.return_value = fake_response

           upload_url = 'http://test.invalid'
           upload = upload_mod.ResumableUpload(
               upload_url, resumable_media.UPLOAD_CHUNK_SIZE)
           # Fake that the upload has been initiate()-d
           data = b'data is here'
           upload._stream = io.BytesIO(data)
           upload._total_bytes = len(data)
           upload._resumable_url = 'http://test.invalid?upload_id=nope'

        .. doctest:: bad-response
           :options: +NORMALIZE_WHITESPACE

           >>> error = None
           >>> try:
           ...     upload.transmit_next_chunk(transport)
           ... except resumable_media.InvalidResponse as caught_exc:
           ...     error = caught_exc
           ...
           >>> error
           InvalidResponse('Request failed with status code', 400,
                           'Expected one of', <HTTPStatus.OK: 200>, 308)
           >>> error.response
           <Response [400]>

        Args:
            transport (~requests.Session): A ``requests`` object which can
                make authenticated requests.
            timeout (Optional[Union[float, aiohttp.ClientTimeout]]):
                The number of seconds to wait for the server response.
                Depending on the retry strategy, a request may be repeated
                several times using the same timeout each time.
                Can also be passed as an `aiohttp.ClientTimeout` object.

        Returns:
            ~requests.Response: The HTTP response returned by ``transport``.

        Raises:
            ~google.resumable_media.common.InvalidResponse: If the status
                code is not 200 or 308.
            ~google.resumable_media.common.DataCorruption: If this is the final
                chunk, a checksum validation was requested, and the checksum
                does not match or is not available.
        """
        method, url, payload, headers = self._prepare_request()
        response = await _request_helpers.http_request(
            transport,
            method,
            url,
            data=payload,
            headers=headers,
            retry_strategy=self._retry_strategy,
            timeout=timeout,
        )
        await self._process_resumable_response(response, len(payload))
        return response

    async def recover(self, transport):
        """Recover from a failure.

        This method should be used when a :class:`ResumableUpload` is in an
        :attr:`~ResumableUpload.invalid` state due to a request failure.

        This will verify the progress with the server and make sure the
        current upload is in a valid state before :meth:`transmit_next_chunk`
        can be used again.

        Args:
            transport (~requests.Session): A ``requests`` object which can
                make authenticated requests.

        Returns:
            ~requests.Response: The HTTP response returned by ``transport``.
        """
        method, url, payload, headers = self._prepare_recover_request()
        # NOTE: We assume "payload is None" but pass it along anyway.
        response = await _request_helpers.http_request(
            transport,
            method,
            url,
            data=payload,
            headers=headers,
            retry_strategy=self._retry_strategy,
        )
        self._process_recover_response(response)
        return response
