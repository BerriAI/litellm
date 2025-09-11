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

"""``requests`` utilities for Google Media Downloads and Resumable Uploads.

This sub-package assumes callers will use the `requests`_ library
as transport and `google-auth`_ for sending authenticated HTTP traffic
with ``requests``.

.. _requests: http://docs.python-requests.org/
.. _google-auth: https://google-auth.readthedocs.io/

====================
Authorized Transport
====================

To use ``google-auth`` and ``requests`` to create an authorized transport
that has read-only access to Google Cloud Storage (GCS):

.. testsetup:: get-credentials

   import google.auth
   import google.auth.credentials as creds_mod
   import mock

   def mock_default(scopes=None):
       credentials = mock.Mock(spec=creds_mod.Credentials)
       return credentials, 'mock-project'

   # Patch the ``default`` function on the module.
   original_default = google.auth.default
   google.auth.default = mock_default

.. doctest:: get-credentials

   >>> import google.auth
   >>> import google.auth.transport.requests as tr_requests
   >>>
   >>> ro_scope = 'https://www.googleapis.com/auth/devstorage.read_only'
   >>> credentials, _ = google.auth.default(scopes=(ro_scope,))
   >>> transport = tr_requests.AuthorizedSession(credentials)
   >>> transport
   <google.auth.transport.requests.AuthorizedSession object at 0x...>

.. testcleanup:: get-credentials

   # Put back the correct ``default`` function on the module.
   google.auth.default = original_default

================
Simple Downloads
================

To download an object from Google Cloud Storage, construct the media URL
for the GCS object and download it with an authorized transport that has
access to the resource:

.. testsetup:: basic-download

   import mock
   import requests
   import http.client

   bucket = 'bucket-foo'
   blob_name = 'file.txt'

   fake_response = requests.Response()
   fake_response.status_code = int(http.client.OK)
   fake_response.headers['Content-Length'] = '1364156'
   fake_content = mock.MagicMock(spec=['__len__'])
   fake_content.__len__.return_value = 1364156
   fake_response._content = fake_content

   get_method = mock.Mock(return_value=fake_response, spec=[])
   transport = mock.Mock(request=get_method, spec=['request'])

.. doctest:: basic-download

   >>> from google.resumable_media.requests import Download
   >>>
   >>> url_template = (
   ...     'https://www.googleapis.com/download/storage/v1/b/'
   ...     '{bucket}/o/{blob_name}?alt=media')
   >>> media_url = url_template.format(
   ...     bucket=bucket, blob_name=blob_name)
   >>>
   >>> download = Download(media_url)
   >>> response = download.consume(transport)
   >>> download.finished
   True
   >>> response
   <Response [200]>
   >>> response.headers['Content-Length']
   '1364156'
   >>> len(response.content)
   1364156

To download only a portion of the bytes in the object,
specify ``start`` and ``end`` byte positions (both optional):

.. testsetup:: basic-download-with-slice

   import mock
   import requests
   import http.client

   from google.resumable_media.requests import Download

   media_url = 'http://test.invalid'
   start = 4096
   end = 8191
   slice_size = end - start + 1

   fake_response = requests.Response()
   fake_response.status_code = int(http.client.PARTIAL_CONTENT)
   fake_response.headers['Content-Length'] = '{:d}'.format(slice_size)
   content_range = 'bytes {:d}-{:d}/1364156'.format(start, end)
   fake_response.headers['Content-Range'] = content_range
   fake_content = mock.MagicMock(spec=['__len__'])
   fake_content.__len__.return_value = slice_size
   fake_response._content = fake_content

   get_method = mock.Mock(return_value=fake_response, spec=[])
   transport = mock.Mock(request=get_method, spec=['request'])

.. doctest:: basic-download-with-slice

   >>> download = Download(media_url, start=4096, end=8191)
   >>> response = download.consume(transport)
   >>> download.finished
   True
   >>> response
   <Response [206]>
   >>> response.headers['Content-Length']
   '4096'
   >>> response.headers['Content-Range']
   'bytes 4096-8191/1364156'
   >>> len(response.content)
   4096

=================
Chunked Downloads
=================

For very large objects or objects of unknown size, it may make more sense
to download the object in chunks rather than all at once. This can be done
to avoid dropped connections with a poor internet connection or can allow
multiple chunks to be downloaded in parallel to speed up the total
download.

A :class:`.ChunkedDownload` uses the same media URL and authorized
transport that a basic :class:`.Download` would use, but also
requires a chunk size and a write-able byte ``stream``. The chunk size is used
to determine how much of the resouce to consume with each request and the
stream is to allow the resource to be written out (e.g. to disk) without
having to fit in memory all at once.

.. testsetup:: chunked-download

   import io

   import mock
   import requests
   import http.client

   media_url = 'http://test.invalid'

   fifty_mb = 50 * 1024 * 1024
   one_gb = 1024 * 1024 * 1024
   fake_response = requests.Response()
   fake_response.status_code = int(http.client.PARTIAL_CONTENT)
   fake_response.headers['Content-Length'] = '{:d}'.format(fifty_mb)
   content_range = 'bytes 0-{:d}/{:d}'.format(fifty_mb - 1, one_gb)
   fake_response.headers['Content-Range'] = content_range
   fake_content_begin = b'The beginning of the chunk...'
   fake_content = fake_content_begin + b'1' * (fifty_mb - 29)
   fake_response._content = fake_content

   get_method = mock.Mock(return_value=fake_response, spec=[])
   transport = mock.Mock(request=get_method, spec=['request'])

.. doctest:: chunked-download

   >>> from google.resumable_media.requests import ChunkedDownload
   >>>
   >>> chunk_size = 50 * 1024 * 1024  # 50MB
   >>> stream = io.BytesIO()
   >>> download = ChunkedDownload(
   ...     media_url, chunk_size, stream)
   >>> # Check the state of the download before starting.
   >>> download.bytes_downloaded
   0
   >>> download.total_bytes is None
   True
   >>> response = download.consume_next_chunk(transport)
   >>> # Check the state of the download after consuming one chunk.
   >>> download.finished
   False
   >>> download.bytes_downloaded  # chunk_size
   52428800
   >>> download.total_bytes  # 1GB
   1073741824
   >>> response
   <Response [206]>
   >>> response.headers['Content-Length']
   '52428800'
   >>> response.headers['Content-Range']
   'bytes 0-52428799/1073741824'
   >>> len(response.content) == chunk_size
   True
   >>> stream.seek(0)
   0
   >>> stream.read(29)
   b'The beginning of the chunk...'

The download will change it's ``finished`` status to :data:`True`
once the final chunk is consumed. In some cases, the final chunk may
not be the same size as the other chunks:

.. testsetup:: chunked-download-end

   import mock
   import requests
   import http.client

   from google.resumable_media.requests import ChunkedDownload

   media_url = 'http://test.invalid'

   fifty_mb = 50 * 1024 * 1024
   one_gb = 1024 * 1024 * 1024
   stream = mock.Mock(spec=['write'])
   download = ChunkedDownload(media_url, fifty_mb, stream)
   download._bytes_downloaded = 20 * fifty_mb
   download._total_bytes = one_gb

   fake_response = requests.Response()
   fake_response.status_code = int(http.client.PARTIAL_CONTENT)
   slice_size = one_gb - 20 * fifty_mb
   fake_response.headers['Content-Length'] = '{:d}'.format(slice_size)
   content_range = 'bytes {:d}-{:d}/{:d}'.format(
       20 * fifty_mb, one_gb - 1, one_gb)
   fake_response.headers['Content-Range'] = content_range
   fake_content = mock.MagicMock(spec=['__len__'])
   fake_content.__len__.return_value = slice_size
   fake_response._content = fake_content

   get_method = mock.Mock(return_value=fake_response, spec=[])
   transport = mock.Mock(request=get_method, spec=['request'])

.. doctest:: chunked-download-end

   >>> # The state of the download in progress.
   >>> download.finished
   False
   >>> download.bytes_downloaded  # 20 chunks at 50MB
   1048576000
   >>> download.total_bytes  # 1GB
   1073741824
   >>> response = download.consume_next_chunk(transport)
   >>> # The state of the download after consuming the final chunk.
   >>> download.finished
   True
   >>> download.bytes_downloaded == download.total_bytes
   True
   >>> response
   <Response [206]>
   >>> response.headers['Content-Length']
   '25165824'
   >>> response.headers['Content-Range']
   'bytes 1048576000-1073741823/1073741824'
   >>> len(response.content) < download.chunk_size
   True

In addition, a :class:`.ChunkedDownload` can also take optional
``start`` and ``end`` byte positions.

Usually, no checksum is returned with a chunked download. Even if one is returned,
it is not validated. If you need to validate the checksum, you can do so
by buffering the chunks and validating the checksum against the completed download.

==============
Simple Uploads
==============

Among the three supported upload classes, the simplest is
:class:`.SimpleUpload`. A simple upload should be used when the resource
being uploaded is small and when there is no metadata (other than the name)
associated with the resource.

.. testsetup:: simple-upload

   import json

   import mock
   import requests
   import http.client

   bucket = 'some-bucket'
   blob_name = 'file.txt'

   fake_response = requests.Response()
   fake_response.status_code = int(http.client.OK)
   payload = {
       'bucket': bucket,
       'contentType': 'text/plain',
       'md5Hash': 'M0XLEsX9/sMdiI+4pB4CAQ==',
       'name': blob_name,
       'size': '27',
   }
   fake_response._content = json.dumps(payload).encode('utf-8')

   post_method = mock.Mock(return_value=fake_response, spec=[])
   transport = mock.Mock(request=post_method, spec=['request'])

.. doctest:: simple-upload
   :options: +NORMALIZE_WHITESPACE

   >>> from google.resumable_media.requests import SimpleUpload
   >>>
   >>> url_template = (
   ...     'https://www.googleapis.com/upload/storage/v1/b/{bucket}/o?'
   ...     'uploadType=media&'
   ...     'name={blob_name}')
   >>> upload_url = url_template.format(
   ...     bucket=bucket, blob_name=blob_name)
   >>>
   >>> upload = SimpleUpload(upload_url)
   >>> data = b'Some not too large content.'
   >>> content_type = 'text/plain'
   >>> response = upload.transmit(transport, data, content_type)
   >>> upload.finished
   True
   >>> response
   <Response [200]>
   >>> json_response = response.json()
   >>> json_response['bucket'] == bucket
   True
   >>> json_response['name'] == blob_name
   True
   >>> json_response['contentType'] == content_type
   True
   >>> json_response['md5Hash']
   'M0XLEsX9/sMdiI+4pB4CAQ=='
   >>> int(json_response['size']) == len(data)
   True

In the rare case that an upload fails, an :exc:`.InvalidResponse`
will be raised:

.. testsetup:: simple-upload-fail

   import time

   import mock
   import requests
   import http.client

   from google import resumable_media
   from google.resumable_media import _helpers
   from google.resumable_media.requests import SimpleUpload as constructor

   upload_url = 'http://test.invalid'
   data = b'Some not too large content.'
   content_type = 'text/plain'

   fake_response = requests.Response()
   fake_response.status_code = int(http.client.SERVICE_UNAVAILABLE)

   post_method = mock.Mock(return_value=fake_response, spec=[])
   transport = mock.Mock(request=post_method, spec=['request'])

   time_sleep = time.sleep
   def dont_sleep(seconds):
       raise RuntimeError('No sleep', seconds)

   def SimpleUpload(*args, **kwargs):
       upload = constructor(*args, **kwargs)
       # Mock the cumulative sleep to avoid retries (and `time.sleep()`).
       upload._retry_strategy = resumable_media.RetryStrategy(
           max_cumulative_retry=-1.0)
       return upload

   time.sleep = dont_sleep

.. doctest:: simple-upload-fail
   :options: +NORMALIZE_WHITESPACE

   >>> upload = SimpleUpload(upload_url)
   >>> error = None
   >>> try:
   ...     upload.transmit(transport, data, content_type)
   ... except resumable_media.InvalidResponse as caught_exc:
   ...     error = caught_exc
   ...
   >>> error
   InvalidResponse('Request failed with status code', 503,
                   'Expected one of', <HTTPStatus.OK: 200>)
   >>> error.response
   <Response [503]>
   >>>
   >>> upload.finished
   True

.. testcleanup:: simple-upload-fail

   # Put back the correct ``sleep`` function on the ``time`` module.
   time.sleep = time_sleep

Even in the case of failure, we see that the upload is
:attr:`~.SimpleUpload.finished`, i.e. it cannot be re-used.

=================
Multipart Uploads
=================

After the simple upload, the :class:`.MultipartUpload` can be used to
achieve essentially the same task. However, a multipart upload allows some
metadata about the resource to be sent along as well. (This is the "multi":
we send a first part with the metadata and a second part with the actual
bytes in the resource.)

Usage is similar to the simple upload, but :meth:`~.MultipartUpload.transmit`
accepts an extra required argument: ``metadata``.

.. testsetup:: multipart-upload

   import json

   import mock
   import requests
   import http.client

   bucket = 'some-bucket'
   blob_name = 'file.txt'
   data = b'Some not too large content.'
   content_type = 'text/plain'

   fake_response = requests.Response()
   fake_response.status_code = int(http.client.OK)
   payload = {
       'bucket': bucket,
       'name': blob_name,
       'metadata': {'color': 'grurple'},
   }
   fake_response._content = json.dumps(payload).encode('utf-8')

   post_method = mock.Mock(return_value=fake_response, spec=[])
   transport = mock.Mock(request=post_method, spec=['request'])

.. doctest:: multipart-upload

   >>> from google.resumable_media.requests import MultipartUpload
   >>>
   >>> url_template = (
   ...     'https://www.googleapis.com/upload/storage/v1/b/{bucket}/o?'
   ...     'uploadType=multipart')
   >>> upload_url = url_template.format(bucket=bucket)
   >>>
   >>> upload = MultipartUpload(upload_url)
   >>> metadata = {
   ...     'name': blob_name,
   ...     'metadata': {
   ...         'color': 'grurple',
   ...     },
   ... }
   >>> response = upload.transmit(transport, data, metadata, content_type)
   >>> upload.finished
   True
   >>> response
   <Response [200]>
   >>> json_response = response.json()
   >>> json_response['bucket'] == bucket
   True
   >>> json_response['name'] == blob_name
   True
   >>> json_response['metadata'] == metadata['metadata']
   True

As with the simple upload, in the case of failure an :exc:`.InvalidResponse`
is raised, enclosing the :attr:`~.InvalidResponse.response` that caused
the failure and the ``upload`` object cannot be re-used after a failure.

=================
Resumable Uploads
=================

A :class:`.ResumableUpload` deviates from the other two upload classes:
it transmits a resource over the course of multiple requests. This
is intended to be used in cases where:

* the size of the resource is not known (i.e. it is generated on the fly)
* requests must be short-lived
* the client has request **size** limitations
* the resource is too large to fit into memory

In general, a resource should be sent in a **single** request to avoid
latency and reduce QPS. See `GCS best practices`_ for more things to
consider when using a resumable upload.

.. _GCS best practices: https://cloud.google.com/storage/docs/\
                        best-practices#uploading

After creating a :class:`.ResumableUpload` instance, a
**resumable upload session** must be initiated to let the server know that
a series of chunked upload requests will be coming and to obtain an
``upload_id`` for the session. In contrast to the other two upload classes,
:meth:`~.ResumableUpload.initiate` takes a byte ``stream`` as input rather
than raw bytes as ``data``. This can be a file object, a :class:`~io.BytesIO`
object or any other stream implementing the same interface.

.. testsetup:: resumable-initiate

   import io

   import mock
   import requests
   import http.client

   bucket = 'some-bucket'
   blob_name = 'file.txt'
   data = b'Some resumable bytes.'
   content_type = 'text/plain'

   fake_response = requests.Response()
   fake_response.status_code = int(http.client.OK)
   fake_response._content = b''
   upload_id = 'ABCdef189XY_super_serious'
   resumable_url_template = (
       'https://www.googleapis.com/upload/storage/v1/b/{bucket}'
       '/o?uploadType=resumable&upload_id={upload_id}')
   resumable_url = resumable_url_template.format(
       bucket=bucket, upload_id=upload_id)
   fake_response.headers['location'] = resumable_url
   fake_response.headers['x-guploader-uploadid'] = upload_id

   post_method = mock.Mock(return_value=fake_response, spec=[])
   transport = mock.Mock(request=post_method, spec=['request'])

.. doctest:: resumable-initiate

   >>> from google.resumable_media.requests import ResumableUpload
   >>>
   >>> url_template = (
   ...     'https://www.googleapis.com/upload/storage/v1/b/{bucket}/o?'
   ...     'uploadType=resumable')
   >>> upload_url = url_template.format(bucket=bucket)
   >>>
   >>> chunk_size = 1024 * 1024  # 1MB
   >>> upload = ResumableUpload(upload_url, chunk_size)
   >>> stream = io.BytesIO(data)
   >>> # The upload doesn't know how "big" it is until seeing a stream.
   >>> upload.total_bytes is None
   True
   >>> metadata = {'name': blob_name}
   >>> response = upload.initiate(transport, stream, metadata, content_type)
   >>> response
   <Response [200]>
   >>> upload.resumable_url == response.headers['Location']
   True
   >>> upload.total_bytes == len(data)
   True
   >>> upload_id = response.headers['X-GUploader-UploadID']
   >>> upload_id
   'ABCdef189XY_super_serious'
   >>> upload.resumable_url == upload_url + '&upload_id=' + upload_id
   True

Once a :class:`.ResumableUpload` has been initiated, the resource is
transmitted in chunks until completion:

.. testsetup:: resumable-transmit

   import io
   import json

   import mock
   import requests
   import http.client

   from google import resumable_media
   import google.resumable_media.requests.upload as upload_mod

   data = b'01234567891'
   stream = io.BytesIO(data)
   # Create an "already initiated" upload.
   upload_url = 'http://test.invalid'
   chunk_size = 256 * 1024  # 256KB
   upload = upload_mod.ResumableUpload(upload_url, chunk_size)
   upload._resumable_url = 'http://test.invalid?upload_id=mocked'
   upload._stream = stream
   upload._content_type = 'text/plain'
   upload._total_bytes = len(data)

   # After-the-fact update the chunk size so that len(data)
   # is split into three.
   upload._chunk_size = 4
   # Make three fake responses.
   fake_response0 = requests.Response()
   fake_response0.status_code = http.client.PERMANENT_REDIRECT
   fake_response0.headers['range'] = 'bytes=0-3'

   fake_response1 = requests.Response()
   fake_response1.status_code = http.client.PERMANENT_REDIRECT
   fake_response1.headers['range'] = 'bytes=0-7'

   fake_response2 = requests.Response()
   fake_response2.status_code = int(http.client.OK)
   bucket = 'some-bucket'
   blob_name = 'file.txt'
   payload = {
       'bucket': bucket,
       'name': blob_name,
       'size': '{:d}'.format(len(data)),
   }
   fake_response2._content = json.dumps(payload).encode('utf-8')

   # Use the fake responses to mock a transport.
   responses = [fake_response0, fake_response1, fake_response2]
   put_method = mock.Mock(side_effect=responses, spec=[])
   transport = mock.Mock(request=put_method, spec=['request'])

.. doctest:: resumable-transmit

   >>> response0 = upload.transmit_next_chunk(transport)
   >>> response0
   <Response [308]>
   >>> upload.finished
   False
   >>> upload.bytes_uploaded == upload.chunk_size
   True
   >>>
   >>> response1 = upload.transmit_next_chunk(transport)
   >>> response1
   <Response [308]>
   >>> upload.finished
   False
   >>> upload.bytes_uploaded == 2 * upload.chunk_size
   True
   >>>
   >>> response2 = upload.transmit_next_chunk(transport)
   >>> response2
   <Response [200]>
   >>> upload.finished
   True
   >>> upload.bytes_uploaded == upload.total_bytes
   True
   >>> json_response = response2.json()
   >>> json_response['bucket'] == bucket
   True
   >>> json_response['name'] == blob_name
   True
"""
from google.resumable_media.requests.download import ChunkedDownload
from google.resumable_media.requests.download import Download
from google.resumable_media.requests.upload import MultipartUpload
from google.resumable_media.requests.download import RawChunkedDownload
from google.resumable_media.requests.download import RawDownload
from google.resumable_media.requests.upload import ResumableUpload
from google.resumable_media.requests.upload import SimpleUpload
from google.resumable_media.requests.upload import XMLMPUContainer
from google.resumable_media.requests.upload import XMLMPUPart

__all__ = [
    "ChunkedDownload",
    "Download",
    "MultipartUpload",
    "RawChunkedDownload",
    "RawDownload",
    "ResumableUpload",
    "SimpleUpload",
    "XMLMPUContainer",
    "XMLMPUPart",
]
