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

# pylint: disable=too-many-lines

"""Create / interact with Google Cloud Storage blobs.
"""

import base64
import copy
import hashlib
from io import BytesIO
from io import TextIOWrapper
import logging
import mimetypes
import os
import re
from email.parser import HeaderParser
from urllib.parse import parse_qsl
from urllib.parse import quote
from urllib.parse import urlencode
from urllib.parse import urlsplit
from urllib.parse import urlunsplit
import warnings

from google import resumable_media
from google.resumable_media.requests import ChunkedDownload
from google.resumable_media.requests import Download
from google.resumable_media.requests import RawDownload
from google.resumable_media.requests import RawChunkedDownload
from google.resumable_media.requests import MultipartUpload
from google.resumable_media.requests import ResumableUpload

from google.api_core.iam import Policy
from google.cloud import exceptions
from google.cloud._helpers import _bytes_to_unicode
from google.cloud._helpers import _datetime_to_rfc3339
from google.cloud._helpers import _rfc3339_nanos_to_datetime
from google.cloud._helpers import _to_bytes
from google.cloud.exceptions import NotFound
from google.cloud.storage._helpers import _add_etag_match_headers
from google.cloud.storage._helpers import _add_generation_match_parameters
from google.cloud.storage._helpers import _PropertyMixin
from google.cloud.storage._helpers import _scalar_property
from google.cloud.storage._helpers import _bucket_bound_hostname_url
from google.cloud.storage._helpers import _raise_if_more_than_one_set
from google.cloud.storage._helpers import _api_core_retry_to_resumable_media_retry
from google.cloud.storage._helpers import _get_default_headers
from google.cloud.storage._helpers import _get_default_storage_base_url
from google.cloud.storage._signing import generate_signed_url_v2
from google.cloud.storage._signing import generate_signed_url_v4
from google.cloud.storage._helpers import _NUM_RETRIES_MESSAGE
from google.cloud.storage._helpers import _API_VERSION
from google.cloud.storage._helpers import _virtual_hosted_style_base_url
from google.cloud.storage._opentelemetry_tracing import create_trace_span
from google.cloud.storage.acl import ACL
from google.cloud.storage.acl import ObjectACL
from google.cloud.storage.constants import _DEFAULT_TIMEOUT
from google.cloud.storage.constants import ARCHIVE_STORAGE_CLASS
from google.cloud.storage.constants import COLDLINE_STORAGE_CLASS
from google.cloud.storage.constants import MULTI_REGIONAL_LEGACY_STORAGE_CLASS
from google.cloud.storage.constants import NEARLINE_STORAGE_CLASS
from google.cloud.storage.constants import REGIONAL_LEGACY_STORAGE_CLASS
from google.cloud.storage.constants import STANDARD_STORAGE_CLASS
from google.cloud.storage.retry import ConditionalRetryPolicy
from google.cloud.storage.retry import DEFAULT_RETRY
from google.cloud.storage.retry import DEFAULT_RETRY_IF_ETAG_IN_JSON
from google.cloud.storage.retry import DEFAULT_RETRY_IF_GENERATION_SPECIFIED
from google.cloud.storage.retry import DEFAULT_RETRY_IF_METAGENERATION_SPECIFIED
from google.cloud.storage.fileio import BlobReader
from google.cloud.storage.fileio import BlobWriter


_DEFAULT_CONTENT_TYPE = "application/octet-stream"
_DOWNLOAD_URL_TEMPLATE = "{hostname}/download/storage/{api_version}{path}?alt=media"
_BASE_UPLOAD_TEMPLATE = (
    "{hostname}/upload/storage/{api_version}{bucket_path}/o?uploadType="
)
_MULTIPART_URL_TEMPLATE = _BASE_UPLOAD_TEMPLATE + "multipart"
_RESUMABLE_URL_TEMPLATE = _BASE_UPLOAD_TEMPLATE + "resumable"
# NOTE: "acl" is also writeable but we defer ACL management to
#       the classes in the google.cloud.storage.acl module.
_CONTENT_TYPE_FIELD = "contentType"
_WRITABLE_FIELDS = (
    "cacheControl",
    "contentDisposition",
    "contentEncoding",
    "contentLanguage",
    _CONTENT_TYPE_FIELD,
    "crc32c",
    "customTime",
    "md5Hash",
    "metadata",
    "name",
    "retention",
    "storageClass",
)
_READ_LESS_THAN_SIZE = (
    "Size {:d} was specified but the file-like object only had " "{:d} bytes remaining."
)
_CHUNKED_DOWNLOAD_CHECKSUM_MESSAGE = (
    "A checksum of type `{}` was requested, but checksumming is not available "
    "for downloads when chunk_size is set."
)
_COMPOSE_IF_GENERATION_LIST_DEPRECATED = (
    "'if_generation_match: type list' is deprecated and supported for "
    "backwards-compatability reasons only.  Use 'if_source_generation_match' "
    "instead' to match source objects' generations."
)
_COMPOSE_IF_GENERATION_LIST_AND_IF_SOURCE_GENERATION_ERROR = (
    "Use 'if_generation_match' to match the generation of the destination "
    "object by passing in a generation number, instead of a list. "
    "Use 'if_source_generation_match' to match source objects generations."
)
_COMPOSE_IF_METAGENERATION_LIST_DEPRECATED = (
    "'if_metageneration_match: type list' is deprecated and supported for "
    "backwards-compatability reasons only. Note that the metageneration to "
    "be matched is that of the destination blob. Please pass in a single "
    "value (type long)."
)
_COMPOSE_IF_SOURCE_GENERATION_MISMATCH_ERROR = (
    "'if_source_generation_match' length must be the same as 'sources' length"
)
_DOWNLOAD_AS_STRING_DEPRECATED = (
    "Blob.download_as_string() is deprecated and will be removed in future. "
    "Use Blob.download_as_bytes() instead."
)
_GS_URL_REGEX_PATTERN = re.compile(
    r"(?P<scheme>gs)://(?P<bucket_name>[a-z0-9_.-]+)/(?P<object_name>.+)"
)

_DEFAULT_CHUNKSIZE = 104857600  # 1024 * 1024 B * 100 = 100 MB
_MAX_MULTIPART_SIZE = 8388608  # 8 MB

_logger = logging.getLogger(__name__)


class Blob(_PropertyMixin):
    """A wrapper around Cloud Storage's concept of an ``Object``.

    :type name: str
    :param name: The name of the blob.  This corresponds to the unique path of
                 the object in the bucket. If bytes, will be converted to a
                 unicode object. Blob / object names can contain any sequence
                 of valid unicode characters, of length 1-1024 bytes when
                 UTF-8 encoded.

    :type bucket: :class:`google.cloud.storage.bucket.Bucket`
    :param bucket: The bucket to which this blob belongs.

    :type chunk_size: int
    :param chunk_size:
        (Optional) The size of a chunk of data whenever iterating (in bytes).
        This must be a multiple of 256 KB per the API specification. If not
        specified, the chunk_size of the blob itself is used. If that is not
        specified, a default value of 40 MB is used.

    :type encryption_key: bytes
    :param encryption_key:
        (Optional) 32 byte encryption key for customer-supplied encryption.
        See https://cloud.google.com/storage/docs/encryption#customer-supplied.

    :type kms_key_name: str
    :param kms_key_name:
        (Optional) Resource name of Cloud KMS key used to encrypt the blob's
        contents.

    :type generation: long
    :param generation:
        (Optional) If present, selects a specific revision of this object.
    """

    _chunk_size = None  # Default value for each instance.
    _CHUNK_SIZE_MULTIPLE = 256 * 1024
    """Number (256 KB, in bytes) that must divide the chunk size."""

    STORAGE_CLASSES = (
        STANDARD_STORAGE_CLASS,
        NEARLINE_STORAGE_CLASS,
        COLDLINE_STORAGE_CLASS,
        ARCHIVE_STORAGE_CLASS,
        MULTI_REGIONAL_LEGACY_STORAGE_CLASS,
        REGIONAL_LEGACY_STORAGE_CLASS,
    )
    """Allowed values for :attr:`storage_class`.

    See
    https://cloud.google.com/storage/docs/json_api/v1/objects#storageClass
    https://cloud.google.com/storage/docs/per-object-storage-class

    .. note::
       This list does not include 'DURABLE_REDUCED_AVAILABILITY', which
       is only documented for buckets (and deprecated).
    """

    def __init__(
        self,
        name,
        bucket,
        chunk_size=None,
        encryption_key=None,
        kms_key_name=None,
        generation=None,
    ):
        """
        property :attr:`name`
            Get the blob's name.
        """
        name = _bytes_to_unicode(name)
        super(Blob, self).__init__(name=name)

        self.chunk_size = chunk_size  # Check that setter accepts value.
        self._bucket = bucket
        self._acl = ObjectACL(self)
        _raise_if_more_than_one_set(
            encryption_key=encryption_key, kms_key_name=kms_key_name
        )

        self._encryption_key = encryption_key

        if kms_key_name is not None:
            self._properties["kmsKeyName"] = kms_key_name

        if generation is not None:
            self._properties["generation"] = generation

    @property
    def bucket(self):
        """Bucket which contains the object.

        :rtype: :class:`~google.cloud.storage.bucket.Bucket`
        :returns: The object's bucket.
        """
        return self._bucket

    @property
    def chunk_size(self):
        """Get the blob's default chunk size.

        :rtype: int or ``NoneType``
        :returns: The current blob's chunk size, if it is set.
        """
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, value):
        """Set the blob's default chunk size.

        :type value: int
        :param value: (Optional) The current blob's chunk size, if it is set.

        :raises: :class:`ValueError` if ``value`` is not ``None`` and is not a
                 multiple of 256 KB.
        """
        if value is not None and value > 0 and value % self._CHUNK_SIZE_MULTIPLE != 0:
            raise ValueError(
                "Chunk size must be a multiple of %d." % (self._CHUNK_SIZE_MULTIPLE,)
            )
        self._chunk_size = value

    @property
    def encryption_key(self):
        """Retrieve the customer-supplied encryption key for the object.

        :rtype: bytes or ``NoneType``
        :returns:
            The encryption key or ``None`` if no customer-supplied encryption key was used,
            or the blob's resource has not been loaded from the server.
        """
        return self._encryption_key

    @encryption_key.setter
    def encryption_key(self, value):
        """Set the blob's encryption key.

        See https://cloud.google.com/storage/docs/encryption#customer-supplied

        To perform a key rotation for an encrypted blob, use :meth:`rewrite`.
        See https://cloud.google.com/storage/docs/encryption/using-customer-supplied-keys?hl=ca#rotating

        :type value: bytes
        :param value: 32 byte encryption key for customer-supplied encryption.
        """
        self._encryption_key = value

    @staticmethod
    def path_helper(bucket_path, blob_name):
        """Relative URL path for a blob.

        :type bucket_path: str
        :param bucket_path: The URL path for a bucket.

        :type blob_name: str
        :param blob_name: The name of the blob.

        :rtype: str
        :returns: The relative URL path for ``blob_name``.
        """
        return bucket_path + "/o/" + _quote(blob_name)

    @property
    def acl(self):
        """Create our ACL on demand."""
        return self._acl

    def __repr__(self):
        if self.bucket:
            bucket_name = self.bucket.name
        else:
            bucket_name = None

        return f"<Blob: {bucket_name}, {self.name}, {self.generation}>"

    @property
    def path(self):
        """Getter property for the URL path to this Blob.

        :rtype: str
        :returns: The URL path to this Blob.
        """
        if not self.name:
            raise ValueError("Cannot determine path without a blob name.")

        return self.path_helper(self.bucket.path, self.name)

    @property
    def client(self):
        """The client bound to this blob."""
        return self.bucket.client

    @property
    def user_project(self):
        """Project ID billed for API requests made via this blob.

        Derived from bucket's value.

        :rtype: str
        """
        return self.bucket.user_project

    def _encryption_headers(self):
        """Return any encryption headers needed to fetch the object.

        :rtype: List(Tuple(str, str))
        :returns: a list of tuples to be passed as headers.
        """
        return _get_encryption_headers(self._encryption_key)

    @property
    def _query_params(self):
        """Default query parameters."""
        params = {}
        if self.generation is not None:
            params["generation"] = self.generation
        if self.user_project is not None:
            params["userProject"] = self.user_project
        return params

    @property
    def public_url(self):
        """The public URL for this blob.

        Use :meth:`make_public` to enable anonymous access via the returned
        URL.

        :rtype: `string`
        :returns: The public URL for this blob.
        """
        if self.client:
            endpoint = self.client.api_endpoint
        else:
            endpoint = _get_default_storage_base_url()
        return "{storage_base_url}/{bucket_name}/{quoted_name}".format(
            storage_base_url=endpoint,
            bucket_name=self.bucket.name,
            quoted_name=_quote(self.name, safe=b"/~"),
        )

    @classmethod
    def from_string(cls, uri, client=None):
        """Get a constructor for blob object by URI.

        .. code-block:: python

            from google.cloud import storage
            from google.cloud.storage.blob import Blob
            client = storage.Client()
            blob = Blob.from_string("gs://bucket/object", client=client)

        :type uri: str
        :param uri: The blob uri following a gs://bucket/object pattern.
          Both a bucket and object name is required to construct a blob object.

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use.  Application code should
            *always* pass ``client``.

        :rtype: :class:`google.cloud.storage.blob.Blob`
        :returns: The blob object created.
        """
        from google.cloud.storage.bucket import Bucket

        match = _GS_URL_REGEX_PATTERN.match(uri)
        if not match:
            raise ValueError("URI pattern must be gs://bucket/object")
        bucket = Bucket(client, name=match.group("bucket_name"))
        return cls(match.group("object_name"), bucket)

    def generate_signed_url(
        self,
        expiration=None,
        api_access_endpoint=None,
        method="GET",
        content_md5=None,
        content_type=None,
        response_disposition=None,
        response_type=None,
        generation=None,
        headers=None,
        query_parameters=None,
        client=None,
        credentials=None,
        version=None,
        service_account_email=None,
        access_token=None,
        virtual_hosted_style=False,
        bucket_bound_hostname=None,
        scheme="http",
    ):
        """Generates a signed URL for this blob.

        .. note::

            If you are on Google Compute Engine, you can't generate a signed
            URL using GCE service account.
            If you'd like to be able to generate a signed URL from GCE,
            you can use a standard service account from a JSON file rather
            than a GCE service account.

        If you have a blob that you want to allow access to for a set
        amount of time, you can use this method to generate a URL that
        is only valid within a certain time period.

        See a [code sample](https://cloud.google.com/storage/docs/samples/storage-generate-signed-url-v4#storage_generate_signed_url_v4-python).

        This is particularly useful if you don't want publicly
        accessible blobs, but don't want to require users to explicitly
        log in.

        If ``bucket_bound_hostname`` is set as an argument of :attr:`api_access_endpoint`,
        ``https`` works only if using a ``CDN``.

        :type expiration: Union[Integer, datetime.datetime, datetime.timedelta]
        :param expiration:
            Point in time when the signed URL should expire. If a ``datetime``
            instance is passed without an explicit ``tzinfo`` set,  it will be
            assumed to be ``UTC``.

        :type api_access_endpoint: str
        :param api_access_endpoint: (Optional) URI base, for instance
            "https://storage.googleapis.com". If not specified, the client's
            api_endpoint will be used. Incompatible with bucket_bound_hostname.

        :type method: str
        :param method: The HTTP verb that will be used when requesting the URL.

        :type content_md5: str
        :param content_md5:
            (Optional) The MD5 hash of the object referenced by ``resource``.

        :type content_type: str
        :param content_type:
            (Optional) The content type of the object referenced by
            ``resource``.

        :type response_disposition: str
        :param response_disposition:
            (Optional) Content disposition of responses to requests for the
            signed URL.  For example, to enable the signed URL to initiate a
            file of ``blog.png``, use the value ``'attachment;
            filename=blob.png'``.

        :type response_type: str
        :param response_type:
            (Optional) Content type of responses to requests for the signed
            URL. Ignored if content_type is set on object/blob metadata.

        :type generation: str
        :param generation:
            (Optional) A value that indicates which generation of the resource
            to fetch.

        :type headers: dict
        :param headers:
            (Optional) Additional HTTP headers to be included as part of the
            signed URLs. See:
            https://cloud.google.com/storage/docs/xml-api/reference-headers
            Requests using the signed URL *must* pass the specified header
            (name and value) with each request for the URL.

        :type query_parameters: dict
        :param query_parameters:
            (Optional) Additional query parameters to be included as part of the
            signed URLs. See:
            https://cloud.google.com/storage/docs/xml-api/reference-headers#query

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use.  If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type credentials: :class:`google.auth.credentials.Credentials`
        :param credentials:
            (Optional) The authorization credentials to attach to requests.
            These credentials identify this application to the service.  If
            none are specified, the client will attempt to ascertain the
            credentials from the environment.

        :type version: str
        :param version:
            (Optional) The version of signed credential to create.  Must be one
            of 'v2' | 'v4'.

        :type service_account_email: str
        :param service_account_email:
            (Optional) E-mail address of the service account.

        :type access_token: str
        :param access_token: (Optional) Access token for a service account.

        :type virtual_hosted_style: bool
        :param virtual_hosted_style:
            (Optional) If true, then construct the URL relative the bucket's
            virtual hostname, e.g., '<bucket-name>.storage.googleapis.com'.
            Incompatible with bucket_bound_hostname.

        :type bucket_bound_hostname: str
        :param bucket_bound_hostname:
            (Optional) If passed, then construct the URL relative to the bucket-bound hostname.
            Value can be a bare or with scheme, e.g., 'example.com' or 'http://example.com'.
            Incompatible with api_access_endpoint and virtual_hosted_style.
            See: https://cloud.google.com/storage/docs/request-endpoints#cname

        :type scheme: str
        :param scheme:
            (Optional) If ``bucket_bound_hostname`` is passed as a bare
            hostname, use this value as the scheme.  ``https`` will work only
            when using a CDN.  Defaults to ``"http"``.

        :raises: :exc:`ValueError` when version is invalid or mutually exclusive arguments are used.
        :raises: :exc:`TypeError` when expiration is not a valid type.
        :raises: :exc:`AttributeError` if credentials is not an instance
                of :class:`google.auth.credentials.Signing`.

        :rtype: str
        :returns: A signed URL you can use to access the resource
                  until expiration.
        """
        if version is None:
            version = "v2"
        elif version not in ("v2", "v4"):
            raise ValueError("'version' must be either 'v2' or 'v4'")

        if (
            api_access_endpoint is not None or virtual_hosted_style
        ) and bucket_bound_hostname:
            raise ValueError(
                "The bucket_bound_hostname argument is not compatible with "
                "either api_access_endpoint or virtual_hosted_style."
            )

        if api_access_endpoint is None:
            client = self._require_client(client)
            api_access_endpoint = client.api_endpoint

        quoted_name = _quote(self.name, safe=b"/~")

        # If you are on Google Compute Engine, you can't generate a signed URL
        # using GCE service account.
        # See https://github.com/googleapis/google-auth-library-python/issues/50
        if virtual_hosted_style:
            api_access_endpoint = _virtual_hosted_style_base_url(
                api_access_endpoint, self.bucket.name
            )
            resource = f"/{quoted_name}"
        elif bucket_bound_hostname:
            api_access_endpoint = _bucket_bound_hostname_url(
                bucket_bound_hostname, scheme
            )
            resource = f"/{quoted_name}"
        else:
            resource = f"/{self.bucket.name}/{quoted_name}"

        if credentials is None:
            client = self._require_client(client)  # May be redundant, but that's ok.
            credentials = client._credentials

        client = self._require_client(client)
        universe_domain = client.universe_domain

        if version == "v2":
            helper = generate_signed_url_v2
        else:
            helper = generate_signed_url_v4

        if self._encryption_key is not None:
            encryption_headers = _get_encryption_headers(self._encryption_key)
            if headers is None:
                headers = {}
            if version == "v2":
                # See: https://cloud.google.com/storage/docs/access-control/signed-urls-v2#about-canonical-extension-headers
                v2_copy_only = "X-Goog-Encryption-Algorithm"
                headers[v2_copy_only] = encryption_headers[v2_copy_only]
            else:
                headers.update(encryption_headers)

        return helper(
            credentials,
            resource=resource,
            expiration=expiration,
            api_access_endpoint=api_access_endpoint,
            method=method.upper(),
            content_md5=content_md5,
            content_type=content_type,
            response_type=response_type,
            response_disposition=response_disposition,
            generation=generation,
            headers=headers,
            query_parameters=query_parameters,
            service_account_email=service_account_email,
            access_token=access_token,
            universe_domain=universe_domain,
        )

    @create_trace_span(name="Storage.Blob.exists")
    def exists(
        self,
        client=None,
        if_etag_match=None,
        if_etag_not_match=None,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=DEFAULT_RETRY,
        soft_deleted=None,
    ):
        """Determines whether or not this blob exists.

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use.  If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type if_etag_match: Union[str, Set[str]]
        :param if_etag_match:
            (Optional) See :ref:`using-if-etag-match`

        :type if_etag_not_match: Union[str, Set[str]]
        :param if_etag_not_match:
            (Optional) See :ref:`using-if-etag-not-match`

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC. See: :ref:`configuring_retries`

        :type soft_deleted: bool
        :param soft_deleted:
            (Optional) If True, looks for a soft-deleted object. Will only return True
            if the object exists and is in a soft-deleted state.
            :attr:`generation` is required to be set on the blob if ``soft_deleted`` is set to True.
            See: https://cloud.google.com/storage/docs/soft-delete

        :rtype: bool
        :returns: True if the blob exists in Cloud Storage.
        """
        client = self._require_client(client)
        # We only need the status code (200 or not) so we seek to
        # minimize the returned payload.
        query_params = self._query_params
        query_params["fields"] = "name"
        if soft_deleted is not None:
            query_params["softDeleted"] = soft_deleted

        _add_generation_match_parameters(
            query_params,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
        )

        headers = {}
        _add_etag_match_headers(
            headers, if_etag_match=if_etag_match, if_etag_not_match=if_etag_not_match
        )

        try:
            # We intentionally pass `_target_object=None` since fields=name
            # would limit the local properties.
            client._get_resource(
                self.path,
                query_params=query_params,
                headers=headers,
                timeout=timeout,
                retry=retry,
                _target_object=None,
            )
        except NotFound:
            # NOTE: This will not fail immediately in a batch. However, when
            #       Batch.finish() is called, the resulting `NotFound` will be
            #       raised.
            return False
        return True

    @create_trace_span(name="Storage.Blob.delete")
    def delete(
        self,
        client=None,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=DEFAULT_RETRY_IF_GENERATION_SPECIFIED,
    ):
        """Deletes a blob from Cloud Storage.

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use. If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC.
            The default value is ``DEFAULT_RETRY_IF_GENERATION_SPECIFIED``, a conditional retry
            policy which will only enable retries if ``if_generation_match`` or ``generation``
            is set, in order to ensure requests are idempotent before retrying them.
            Change the value to ``DEFAULT_RETRY`` or another `google.api_core.retry.Retry` object
            to enable retries regardless of generation precondition setting.
            See [Configuring Retries](https://cloud.google.com/python/docs/reference/storage/latest/retry_timeout).

        :raises: :class:`google.cloud.exceptions.NotFound`
                 (propagated from
                 :meth:`google.cloud.storage.bucket.Bucket.delete_blob`).
        """
        self.bucket.delete_blob(
            self.name,
            client=client,
            generation=self.generation,
            timeout=timeout,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
            retry=retry,
        )

    def _get_transport(self, client):
        """Return the client's transport.

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use.  If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :rtype transport:
            :class:`~google.auth.transport.requests.AuthorizedSession`
        :returns: The transport (with credentials) that will
                  make authenticated requests.
        """
        client = self._require_client(client)
        return client._http

    def _get_download_url(
        self,
        client,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
    ):
        """Get the download URL for the current blob.

        If the ``media_link`` has been loaded, it will be used, otherwise
        the URL will be constructed from the current blob's path (and possibly
        generation) to avoid a round trip.

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client: The client to use.

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :rtype: str
        :returns: The download URL for the current blob.
        """
        name_value_pairs = []
        if self.media_link is None:
            hostname = _get_host_name(client._connection)
            base_url = _DOWNLOAD_URL_TEMPLATE.format(
                hostname=hostname, path=self.path, api_version=_API_VERSION
            )
            if self.generation is not None:
                name_value_pairs.append(("generation", f"{self.generation:d}"))
        else:
            base_url = self.media_link

        if self.user_project is not None:
            name_value_pairs.append(("userProject", self.user_project))

        _add_generation_match_parameters(
            name_value_pairs,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
        )
        return _add_query_parameters(base_url, name_value_pairs)

    def _extract_headers_from_download(self, response):
        """Extract headers from a non-chunked request's http object.

        This avoids the need to make a second request for commonly used
        headers.

        :type response:
            :class requests.models.Response
        :param response: The server response from downloading a non-chunked file
        """
        self._properties["contentEncoding"] = response.headers.get(
            "Content-Encoding", None
        )
        self._properties[_CONTENT_TYPE_FIELD] = response.headers.get(
            "Content-Type", None
        )
        self._properties["cacheControl"] = response.headers.get("Cache-Control", None)
        self._properties["storageClass"] = response.headers.get(
            "X-Goog-Storage-Class", None
        )
        self._properties["contentLanguage"] = response.headers.get(
            "Content-Language", None
        )
        self._properties["etag"] = response.headers.get("ETag", None)
        self._properties["generation"] = response.headers.get("X-goog-generation", None)
        self._properties["metageneration"] = response.headers.get(
            "X-goog-metageneration", None
        )
        #  'X-Goog-Hash': 'crc32c=4gcgLQ==,md5=CS9tHYTtyFntzj7B9nkkJQ==',
        x_goog_hash = response.headers.get("X-Goog-Hash", "")

        if x_goog_hash:
            digests = {}
            for encoded_digest in x_goog_hash.split(","):
                match = re.match(r"(crc32c|md5)=([\w\d/\+/]+={0,3})", encoded_digest)
                if match:
                    method, digest = match.groups()
                    digests[method] = digest

            self._properties["crc32c"] = digests.get("crc32c", None)
            self._properties["md5Hash"] = digests.get("md5", None)

    def _do_download(
        self,
        transport,
        file_obj,
        download_url,
        headers,
        start=None,
        end=None,
        raw_download=False,
        timeout=_DEFAULT_TIMEOUT,
        checksum="md5",
        retry=None,
    ):
        """Perform a download without any error handling.

        This is intended to be called by :meth:`_prep_and_do_download` so it can
        be wrapped with error handling / remapping.

        :type transport:
            :class:`~google.auth.transport.requests.AuthorizedSession`
        :param transport:
            The transport (with credentials) that will make authenticated
            requests.

        :type file_obj: file
        :param file_obj: A file handle to which to write the blob's data.

        :type download_url: str
        :param download_url: The URL where the media can be accessed.

        :type headers: dict
        :param headers: Headers to be sent with the request(s).

        :type start: int
        :param start: (Optional) The first byte in a range to be downloaded.

        :type end: int
        :param end: (Optional) The last byte in a range to be downloaded.

        :type raw_download: bool
        :param raw_download:
            (Optional) If true, download the object without any expansion.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type checksum: str
        :param checksum:
            (Optional) The type of checksum to compute to verify the integrity
            of the object. The response headers must contain a checksum of the
            requested type. If the headers lack an appropriate checksum (for
            instance in the case of transcoded or ranged downloads where the
            remote service does not know the correct checksum, including
            downloads where chunk_size is set) an INFO-level log will be
            emitted. Supported values are "md5", "crc32c" and None. The default
            is "md5".

        :type retry: google.api_core.retry.Retry
        :param retry: (Optional) How to retry the RPC. A None value will disable
            retries. A google.api_core.retry.Retry value will enable retries,
            and the object will configure backoff and timeout options. Custom
            predicates (customizable error codes) are not supported for media
            operations such as this one.

            This private method does not accept ConditionalRetryPolicy values
            because the information necessary to evaluate the policy is instead
            evaluated in blob._prep_and_do_download().

            See the retry.py source code and docstrings in this package
            (google.cloud.storage.retry) for information on retry types and how
            to configure them.
        """

        retry_strategy = _api_core_retry_to_resumable_media_retry(retry)

        extra_attributes = {
            "url.full": download_url,
            "download.chunk_size": f"{self.chunk_size}",
            "download.raw_download": raw_download,
            "upload.checksum": f"{checksum}",
        }
        args = {"timeout": timeout}

        if self.chunk_size is None:
            if raw_download:
                klass = RawDownload
                download_class = "RawDownload"
            else:
                klass = Download
                download_class = "Download"

            download = klass(
                download_url,
                stream=file_obj,
                headers=headers,
                start=start,
                end=end,
                checksum=checksum,
            )
            download._retry_strategy = retry_strategy
            with create_trace_span(
                name=f"Storage.{download_class}/consume",
                attributes=extra_attributes,
                api_request=args,
            ):
                response = download.consume(transport, timeout=timeout)
                self._extract_headers_from_download(response)
        else:
            if checksum:
                msg = _CHUNKED_DOWNLOAD_CHECKSUM_MESSAGE.format(checksum)
                _logger.info(msg)

            if raw_download:
                klass = RawChunkedDownload
                download_class = "RawChunkedDownload"
            else:
                klass = ChunkedDownload
                download_class = "ChunkedDownload"

            download = klass(
                download_url,
                self.chunk_size,
                file_obj,
                headers=headers,
                start=start if start else 0,
                end=end,
            )

            download._retry_strategy = retry_strategy
            with create_trace_span(
                name=f"Storage.{download_class}/consumeNextChunk",
                attributes=extra_attributes,
                api_request=args,
            ):
                while not download.finished:
                    download.consume_next_chunk(transport, timeout=timeout)

    @create_trace_span(name="Storage.Blob.downloadToFile")
    def download_to_file(
        self,
        file_obj,
        client=None,
        start=None,
        end=None,
        raw_download=False,
        if_etag_match=None,
        if_etag_not_match=None,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        timeout=_DEFAULT_TIMEOUT,
        checksum="md5",
        retry=DEFAULT_RETRY,
    ):
        """Download the contents of this blob into a file-like object.

        .. note::

           If the server-set property, :attr:`media_link`, is not yet
           initialized, makes an additional API request to load it.

        If the :attr:`chunk_size` of a current blob is `None`, will download data
        in single download request otherwise it will download the :attr:`chunk_size`
        of data in each request.

        For more fine-grained control over the download process, check out
        [`google-resumable-media`](https://googleapis.dev/python/google-resumable-media/latest/index.html).
        For example, this library allows downloading **parts** of a blob rather than the whole thing.

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        :type file_obj: file
        :param file_obj: A file handle to which to write the blob's data.

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use.  If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type start: int
        :param start: (Optional) The first byte in a range to be downloaded.

        :type end: int
        :param end: (Optional) The last byte in a range to be downloaded.

        :type raw_download: bool
        :param raw_download:
            (Optional) If true, download the object without any expansion.

        :type if_etag_match: Union[str, Set[str]]
        :param if_etag_match:
            (Optional) See :ref:`using-if-etag-match`

        :type if_etag_not_match: Union[str, Set[str]]
        :param if_etag_not_match:
            (Optional) See :ref:`using-if-etag-not-match`

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type checksum: str
        :param checksum:
            (Optional) The type of checksum to compute to verify the integrity
            of the object. The response headers must contain a checksum of the
            requested type. If the headers lack an appropriate checksum (for
            instance in the case of transcoded or ranged downloads where the
            remote service does not know the correct checksum, including
            downloads where chunk_size is set) an INFO-level log will be
            emitted. Supported values are "md5", "crc32c" and None. The default
            is "md5".

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry: (Optional) How to retry the RPC. A None value will disable
            retries. A google.api_core.retry.Retry value will enable retries,
            and the object will define retriable response codes and errors and
            configure backoff and timeout options.

            A google.cloud.storage.retry.ConditionalRetryPolicy value wraps a
            Retry object and activates it only if certain conditions are met.
            This class exists to provide safe defaults for RPC calls that are
            not technically safe to retry normally (due to potential data
            duplication or other side-effects) but become safe to retry if a
            condition such as if_metageneration_match is set.

            See the retry.py source code and docstrings in this package
            (google.cloud.storage.retry) for information on retry types and how
            to configure them.

            Media operations (downloads and uploads) do not support non-default
            predicates in a Retry object. The default will always be used. Other
            configuration changes for Retry objects such as delays and deadlines
            are respected.

        :raises: :class:`google.cloud.exceptions.NotFound`
        """

        self._prep_and_do_download(
            file_obj,
            client=client,
            start=start,
            end=end,
            raw_download=raw_download,
            if_etag_match=if_etag_match,
            if_etag_not_match=if_etag_not_match,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
            timeout=timeout,
            checksum=checksum,
            retry=retry,
        )

    def _handle_filename_and_download(self, filename, *args, **kwargs):
        """Download the contents of this blob into a named file.

        :type filename: str
        :param filename: A filename to be passed to ``open``.

        For *args and **kwargs, refer to the documentation for download_to_filename() for more information.
        """

        try:
            with open(filename, "wb") as file_obj:
                self._prep_and_do_download(
                    file_obj,
                    *args,
                    **kwargs,
                )

        except resumable_media.DataCorruption:
            # Delete the corrupt downloaded file.
            os.remove(filename)
            raise

        updated = self.updated
        if updated is not None:
            mtime = updated.timestamp()
            os.utime(file_obj.name, (mtime, mtime))

    @create_trace_span(name="Storage.Blob.downloadToFilename")
    def download_to_filename(
        self,
        filename,
        client=None,
        start=None,
        end=None,
        raw_download=False,
        if_etag_match=None,
        if_etag_not_match=None,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        timeout=_DEFAULT_TIMEOUT,
        checksum="md5",
        retry=DEFAULT_RETRY,
    ):
        """Download the contents of this blob into a named file.

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        See a [code sample](https://cloud.google.com/storage/docs/samples/storage-download-encrypted-file#storage_download_encrypted_file-python)
        to download a file with a [`customer-supplied encryption key`](https://cloud.google.com/storage/docs/encryption#customer-supplied).

        :type filename: str
        :param filename: A filename to be passed to ``open``.

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use. If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type start: int
        :param start: (Optional) The first byte in a range to be downloaded.

        :type end: int
        :param end: (Optional) The last byte in a range to be downloaded.

        :type raw_download: bool
        :param raw_download:
            (Optional) If true, download the object without any expansion.

        :type if_etag_match: Union[str, Set[str]]
        :param if_etag_match:
            (Optional) See :ref:`using-if-etag-match`

        :type if_etag_not_match: Union[str, Set[str]]
        :param if_etag_not_match:
            (Optional) See :ref:`using-if-etag-not-match`

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type checksum: str
        :param checksum:
            (Optional) The type of checksum to compute to verify the integrity
            of the object. The response headers must contain a checksum of the
            requested type. If the headers lack an appropriate checksum (for
            instance in the case of transcoded or ranged downloads where the
            remote service does not know the correct checksum, including
            downloads where chunk_size is set) an INFO-level log will be
            emitted. Supported values are "md5", "crc32c" and None. The default
            is "md5".

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry: (Optional) How to retry the RPC. A None value will disable
            retries. A google.api_core.retry.Retry value will enable retries,
            and the object will define retriable response codes and errors and
            configure backoff and timeout options.

            A google.cloud.storage.retry.ConditionalRetryPolicy value wraps a
            Retry object and activates it only if certain conditions are met.
            This class exists to provide safe defaults for RPC calls that are
            not technically safe to retry normally (due to potential data
            duplication or other side-effects) but become safe to retry if a
            condition such as if_metageneration_match is set.

            See the retry.py source code and docstrings in this package
            (google.cloud.storage.retry) for information on retry types and how
            to configure them.

            Media operations (downloads and uploads) do not support non-default
            predicates in a Retry object. The default will always be used. Other
            configuration changes for Retry objects such as delays and deadlines
            are respected.

        :raises: :class:`google.cloud.exceptions.NotFound`
        """

        self._handle_filename_and_download(
            filename,
            client=client,
            start=start,
            end=end,
            raw_download=raw_download,
            if_etag_match=if_etag_match,
            if_etag_not_match=if_etag_not_match,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
            timeout=timeout,
            checksum=checksum,
            retry=retry,
        )

    @create_trace_span(name="Storage.Blob.downloadAsBytes")
    def download_as_bytes(
        self,
        client=None,
        start=None,
        end=None,
        raw_download=False,
        if_etag_match=None,
        if_etag_not_match=None,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        timeout=_DEFAULT_TIMEOUT,
        checksum="md5",
        retry=DEFAULT_RETRY,
    ):
        """Download the contents of this blob as a bytes object.

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use. If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type start: int
        :param start: (Optional) The first byte in a range to be downloaded.

        :type end: int
        :param end: (Optional) The last byte in a range to be downloaded.

        :type raw_download: bool
        :param raw_download:
            (Optional) If true, download the object without any expansion.

        :type if_etag_match: Union[str, Set[str]]
        :param if_etag_match:
            (Optional) See :ref:`using-if-etag-match`

        :type if_etag_not_match: Union[str, Set[str]]
        :param if_etag_not_match:
            (Optional) See :ref:`using-if-etag-not-match`

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type checksum: str
        :param checksum:
            (Optional) The type of checksum to compute to verify the integrity
            of the object. The response headers must contain a checksum of the
            requested type. If the headers lack an appropriate checksum (for
            instance in the case of transcoded or ranged downloads where the
            remote service does not know the correct checksum, including
            downloads where chunk_size is set) an INFO-level log will be
            emitted. Supported values are "md5", "crc32c" and None. The default
            is "md5".

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry: (Optional) How to retry the RPC. A None value will disable
            retries. A google.api_core.retry.Retry value will enable retries,
            and the object will define retriable response codes and errors and
            configure backoff and timeout options.

            A google.cloud.storage.retry.ConditionalRetryPolicy value wraps a
            Retry object and activates it only if certain conditions are met.
            This class exists to provide safe defaults for RPC calls that are
            not technically safe to retry normally (due to potential data
            duplication or other side-effects) but become safe to retry if a
            condition such as if_metageneration_match is set.

            See the retry.py source code and docstrings in this package
            (google.cloud.storage.retry) for information on retry types and how
            to configure them.

            Media operations (downloads and uploads) do not support non-default
            predicates in a Retry object. The default will always be used. Other
            configuration changes for Retry objects such as delays and deadlines
            are respected.

        :rtype: bytes
        :returns: The data stored in this blob.

        :raises: :class:`google.cloud.exceptions.NotFound`
        """

        string_buffer = BytesIO()

        self._prep_and_do_download(
            string_buffer,
            client=client,
            start=start,
            end=end,
            raw_download=raw_download,
            if_etag_match=if_etag_match,
            if_etag_not_match=if_etag_not_match,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
            timeout=timeout,
            checksum=checksum,
            retry=retry,
        )
        return string_buffer.getvalue()

    @create_trace_span(name="Storage.Blob.downloadAsString")
    def download_as_string(
        self,
        client=None,
        start=None,
        end=None,
        raw_download=False,
        if_etag_match=None,
        if_etag_not_match=None,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=DEFAULT_RETRY,
    ):
        """(Deprecated) Download the contents of this blob as a bytes object.

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        .. note::
           Deprecated alias for :meth:`download_as_bytes`.

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use. If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type start: int
        :param start: (Optional) The first byte in a range to be downloaded.

        :type end: int
        :param end: (Optional) The last byte in a range to be downloaded.

        :type raw_download: bool
        :param raw_download:
            (Optional) If true, download the object without any expansion.

        :type if_etag_match: Union[str, Set[str]]
        :param if_etag_match:
            (Optional) See :ref:`using-if-etag-match`

        :type if_etag_not_match: Union[str, Set[str]]
        :param if_etag_not_match:
            (Optional) See :ref:`using-if-etag-not-match`

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry: (Optional) How to retry the RPC. A None value will disable
            retries. A google.api_core.retry.Retry value will enable retries,
            and the object will define retriable response codes and errors and
            configure backoff and timeout options.

            A google.cloud.storage.retry.ConditionalRetryPolicy value wraps a
            Retry object and activates it only if certain conditions are met.
            This class exists to provide safe defaults for RPC calls that are
            not technically safe to retry normally (due to potential data
            duplication or other side-effects) but become safe to retry if a
            condition such as if_metageneration_match is set.

            See the retry.py source code and docstrings in this package
            (google.cloud.storage.retry) for information on retry types and how
            to configure them.

            Media operations (downloads and uploads) do not support non-default
            predicates in a Retry object. The default will always be used. Other
            configuration changes for Retry objects such as delays and deadlines
            are respected.

        :rtype: bytes
        :returns: The data stored in this blob.

        :raises: :class:`google.cloud.exceptions.NotFound`
        """
        warnings.warn(
            _DOWNLOAD_AS_STRING_DEPRECATED, PendingDeprecationWarning, stacklevel=2
        )
        return self.download_as_bytes(
            client=client,
            start=start,
            end=end,
            raw_download=raw_download,
            if_etag_match=if_etag_match,
            if_etag_not_match=if_etag_not_match,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
            timeout=timeout,
            retry=retry,
        )

    @create_trace_span(name="Storage.Blob.downloadAsText")
    def download_as_text(
        self,
        client=None,
        start=None,
        end=None,
        raw_download=False,
        encoding=None,
        if_etag_match=None,
        if_etag_not_match=None,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=DEFAULT_RETRY,
    ):
        """Download the contents of this blob as text (*not* bytes).

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use. If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type start: int
        :param start: (Optional) The first byte in a range to be downloaded.

        :type end: int
        :param end: (Optional) The last byte in a range to be downloaded.

        :type raw_download: bool
        :param raw_download:
            (Optional) If true, download the object without any expansion.

        :type encoding: str
        :param encoding: (Optional) encoding to be used to decode the
            downloaded bytes.  Defaults to the ``charset`` param of
            attr:`content_type`, or else to "utf-8".

        :type if_etag_match: Union[str, Set[str]]
        :param if_etag_match:
            (Optional) See :ref:`using-if-etag-match`

        :type if_etag_not_match: Union[str, Set[str]]
        :param if_etag_not_match:
            (Optional) See :ref:`using-if-etag-not-match`

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry: (Optional) How to retry the RPC. A None value will disable
            retries. A google.api_core.retry.Retry value will enable retries,
            and the object will define retriable response codes and errors and
            configure backoff and timeout options.

            A google.cloud.storage.retry.ConditionalRetryPolicy value wraps a
            Retry object and activates it only if certain conditions are met.
            This class exists to provide safe defaults for RPC calls that are
            not technically safe to retry normally (due to potential data
            duplication or other side-effects) but become safe to retry if a
            condition such as if_metageneration_match is set.

            See the retry.py source code and docstrings in this package
            (google.cloud.storage.retry) for information on retry types and how
            to configure them.

            Media operations (downloads and uploads) do not support non-default
            predicates in a Retry object. The default will always be used. Other
            configuration changes for Retry objects such as delays and deadlines
            are respected.

        :rtype: text
        :returns: The data stored in this blob, decoded to text.
        """
        data = self.download_as_bytes(
            client=client,
            start=start,
            end=end,
            raw_download=raw_download,
            if_etag_match=if_etag_match,
            if_etag_not_match=if_etag_not_match,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
            timeout=timeout,
            retry=retry,
        )

        if encoding is not None:
            return data.decode(encoding)

        if self.content_type is not None:
            msg = HeaderParser().parsestr("Content-Type: " + self.content_type)
            params = dict(msg.get_params()[1:])
            if "charset" in params:
                return data.decode(params["charset"])

        return data.decode("utf-8")

    def _get_content_type(self, content_type, filename=None):
        """Determine the content type from the current object.

        The return value will be determined in order of precedence:

        - The value passed in to this method (if not :data:`None`)
        - The value stored on the current blob
        - The default value ('application/octet-stream')

        :type content_type: str
        :param content_type: (Optional) Type of content.

        :type filename: str
        :param filename:
            (Optional) The name of the file where the content is stored.

        :rtype: str
        :returns: Type of content gathered from the object.
        """
        if content_type is None:
            content_type = self.content_type

        if content_type is None and filename is not None:
            content_type, _ = mimetypes.guess_type(filename)

        if content_type is None:
            content_type = _DEFAULT_CONTENT_TYPE

        return content_type

    def _get_writable_metadata(self):
        """Get the object / blob metadata which is writable.

        This is intended to be used when creating a new object / blob.

        See the [`API reference docs`](https://cloud.google.com/storage/docs/json_api/v1/objects)
        for more information, the fields marked as writable are:

        * ``acl``
        * ``cacheControl``
        * ``contentDisposition``
        * ``contentEncoding``
        * ``contentLanguage``
        * ``contentType``
        * ``crc32c``
        * ``customTime``
        * ``md5Hash``
        * ``metadata``
        * ``name``
        * ``retention``
        * ``storageClass``

        For now, we don't support ``acl``, access control lists should be
        managed directly through :class:`ObjectACL` methods.
        """
        # NOTE: This assumes `self.name` is unicode.
        object_metadata = {"name": self.name}
        for key in self._changes:
            if key in _WRITABLE_FIELDS:
                object_metadata[key] = self._properties[key]

        return object_metadata

    def _get_upload_arguments(self, client, content_type, filename=None, command=None):
        """Get required arguments for performing an upload.

        The content type returned will be determined in order of precedence:

        - The value passed in to this method (if not :data:`None`)
        - The value stored on the current blob
        - The default value ('application/octet-stream')

        :type content_type: str
        :param content_type: Type of content being uploaded (or :data:`None`).

        :type command: str
        :param command:
            (Optional) Information about which interface for upload was used,
            to be included in the X-Goog-API-Client header. Please leave as None
            unless otherwise directed.

        :rtype: tuple
        :returns: A triple of

                  * A header dictionary
                  * An object metadata dictionary
                  * The ``content_type`` as a string (according to precedence)
        """
        content_type = self._get_content_type(content_type, filename=filename)
        # Add any client attached custom headers to the upload headers.
        headers = {
            **_get_default_headers(
                client._connection.user_agent, content_type, command=command
            ),
            **_get_encryption_headers(self._encryption_key),
            **client._extra_headers,
        }
        object_metadata = self._get_writable_metadata()
        return headers, object_metadata, content_type

    def _do_multipart_upload(
        self,
        client,
        stream,
        content_type,
        size,
        num_retries,
        predefined_acl,
        if_generation_match,
        if_generation_not_match,
        if_metageneration_match,
        if_metageneration_not_match,
        timeout=_DEFAULT_TIMEOUT,
        checksum=None,
        retry=None,
        command=None,
    ):
        """Perform a multipart upload.

        The content type of the upload will be determined in order
        of precedence:

        - The value passed in to this method (if not :data:`None`)
        - The value stored on the current blob
        - The default value ('application/octet-stream')

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use.  If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type stream: IO[bytes]
        :param stream: A bytes IO object open for reading.

        :type content_type: str
        :param content_type: Type of content being uploaded (or :data:`None`).

        :type size: int
        :param size:
            The number of bytes to be uploaded (which will be read from
            ``stream``). If not provided, the upload will be concluded once
            ``stream`` is exhausted (or :data:`None`).

        :type num_retries: int
        :param num_retries:
            Number of upload retries. By default, only uploads with
            if_generation_match set will be retried, as uploads without the
            argument are not guaranteed to be idempotent. Setting num_retries
            will override this default behavior and guarantee retries even when
            if_generation_match is not set.  (Deprecated: This argument
            will be removed in a future release.)

        :type predefined_acl: str
        :param predefined_acl: (Optional) Predefined access control list

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type checksum: str
        :param checksum:
            (Optional) The type of checksum to compute to verify
            the integrity of the object. The request metadata will be amended
            to include the computed value. Using this option will override a
            manually-set checksum value. Supported values are "md5",
            "crc32c" and None. The default is None.

        :type retry: google.api_core.retry.Retry
        :param retry: (Optional) How to retry the RPC. A None value will disable
            retries. A google.api_core.retry.Retry value will enable retries,
            and the object will configure backoff and timeout options. Custom
            predicates (customizable error codes) are not supported for media
            operations such as this one.

            This private method does not accept ConditionalRetryPolicy values
            because the information necessary to evaluate the policy is instead
            evaluated in blob._do_upload().

            See the retry.py source code and docstrings in this package
            (google.cloud.storage.retry) for information on retry types and how
            to configure them.

        :type command: str
        :param command:
            (Optional) Information about which interface for upload was used,
            to be included in the X-Goog-API-Client header. Please leave as None
            unless otherwise directed.

        :rtype: :class:`~requests.Response`
        :returns: The "200 OK" response object returned after the multipart
                  upload request.
        :raises: :exc:`ValueError` if ``size`` is not :data:`None` but the
                 ``stream`` has fewer than ``size`` bytes remaining.
        """
        if size is None:
            data = stream.read()
        else:
            data = stream.read(size)
            if len(data) < size:
                msg = _READ_LESS_THAN_SIZE.format(size, len(data))
                raise ValueError(msg)

        client = self._require_client(client)
        transport = self._get_transport(client)
        if "metadata" in self._properties and "metadata" not in self._changes:
            self._changes.add("metadata")
        info = self._get_upload_arguments(client, content_type, command=command)
        headers, object_metadata, content_type = info

        hostname = _get_host_name(client._connection)
        base_url = _MULTIPART_URL_TEMPLATE.format(
            hostname=hostname, bucket_path=self.bucket.path, api_version=_API_VERSION
        )
        name_value_pairs = []

        if self.user_project is not None:
            name_value_pairs.append(("userProject", self.user_project))

        # When a Customer Managed Encryption Key is used to encrypt Cloud Storage object
        # at rest, object resource metadata will store the version of the Key Management
        # Service cryptographic material. If a Blob instance with KMS Key metadata set is
        # used to upload a new version of the object then the existing kmsKeyName version
        # value can't be used in the upload request and the client instead ignores it.
        if (
            self.kms_key_name is not None
            and "cryptoKeyVersions" not in self.kms_key_name
        ):
            name_value_pairs.append(("kmsKeyName", self.kms_key_name))

        if predefined_acl is not None:
            name_value_pairs.append(("predefinedAcl", predefined_acl))

        if if_generation_match is not None:
            name_value_pairs.append(("ifGenerationMatch", if_generation_match))

        if if_generation_not_match is not None:
            name_value_pairs.append(("ifGenerationNotMatch", if_generation_not_match))

        if if_metageneration_match is not None:
            name_value_pairs.append(("ifMetagenerationMatch", if_metageneration_match))

        if if_metageneration_not_match is not None:
            name_value_pairs.append(
                ("ifMetaGenerationNotMatch", if_metageneration_not_match)
            )

        upload_url = _add_query_parameters(base_url, name_value_pairs)
        upload = MultipartUpload(upload_url, headers=headers, checksum=checksum)

        upload._retry_strategy = _api_core_retry_to_resumable_media_retry(
            retry, num_retries
        )

        extra_attributes = {
            "url.full": upload_url,
            "upload.checksum": f"{checksum}",
        }
        args = {"timeout": timeout}
        with create_trace_span(
            name="Storage.MultipartUpload/transmit",
            attributes=extra_attributes,
            client=client,
            api_request=args,
        ):
            response = upload.transmit(
                transport, data, object_metadata, content_type, timeout=timeout
            )

            return response

    def _initiate_resumable_upload(
        self,
        client,
        stream,
        content_type,
        size,
        num_retries,
        predefined_acl=None,
        extra_headers=None,
        chunk_size=None,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        timeout=_DEFAULT_TIMEOUT,
        checksum=None,
        retry=None,
        command=None,
    ):
        """Initiate a resumable upload.

        The content type of the upload will be determined in order
        of precedence:

        - The value passed in to this method (if not :data:`None`)
        - The value stored on the current blob
        - The default value ('application/octet-stream')

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use.  If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type stream: IO[bytes]
        :param stream: A bytes IO object open for reading.

        :type content_type: str
        :param content_type: Type of content being uploaded (or :data:`None`).

        :type size: int
        :param size:
            The number of bytes to be uploaded (which will be read from
            ``stream``). If not provided, the upload will be concluded once
            ``stream`` is exhausted (or :data:`None`).

        :type predefined_acl: str
        :param predefined_acl: (Optional) Predefined access control list

        :type num_retries: int
        :param num_retries:
            Number of upload retries. By default, only uploads with
            if_generation_match set will be retried, as uploads without the
            argument are not guaranteed to be idempotent. Setting num_retries
            will override this default behavior and guarantee retries even when
            if_generation_match is not set.  (Deprecated: This argument
            will be removed in a future release.)

        :type extra_headers: dict
        :param extra_headers:
            (Optional) Extra headers to add to standard headers.

        :type chunk_size: int
        :param chunk_size:
            (Optional) Chunk size to use when creating a
            :class:`~google.resumable_media.requests.ResumableUpload`.
            If not passed, will fall back to the chunk size on the
            current blob, if the chunk size of a current blob is also
            `None`, will set the default value.
            The default value of ``chunk_size`` is 100 MB.

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type checksum: str
        :param checksum:
            (Optional) The type of checksum to compute to verify
            the integrity of the object. After the upload is complete, the
            server-computed checksum of the resulting object will be checked
            and google.resumable_media.common.DataCorruption will be raised on
            a mismatch. On a validation failure, the client will attempt to
            delete the uploaded object automatically. Supported values
            are "md5", "crc32c" and None. The default is None.

        :type retry: google.api_core.retry.Retry
        :param retry: (Optional) How to retry the RPC. A None value will disable
            retries. A google.api_core.retry.Retry value will enable retries,
            and the object will configure backoff and timeout options. Custom
            predicates (customizable error codes) are not supported for media
            operations such as this one.

            This private method does not accept ConditionalRetryPolicy values
            because the information necessary to evaluate the policy is instead
            evaluated in blob._do_upload().

            See the retry.py source code and docstrings in this package
            (google.cloud.storage.retry) for information on retry types and how
            to configure them.

        :type command: str
        :param command:
            (Optional) Information about which interface for upload was used,
            to be included in the X-Goog-API-Client header. Please leave as None
            unless otherwise directed.

        :rtype: tuple
        :returns:
            Pair of

            * The :class:`~google.resumable_media.requests.ResumableUpload`
              that was created
            * The ``transport`` used to initiate the upload.
        """
        client = self._require_client(client)
        if chunk_size is None:
            chunk_size = self.chunk_size
            if chunk_size is None:
                chunk_size = _DEFAULT_CHUNKSIZE

        transport = self._get_transport(client)
        if "metadata" in self._properties and "metadata" not in self._changes:
            self._changes.add("metadata")
        info = self._get_upload_arguments(client, content_type, command=command)
        headers, object_metadata, content_type = info
        if extra_headers is not None:
            headers.update(extra_headers)

        hostname = _get_host_name(client._connection)
        base_url = _RESUMABLE_URL_TEMPLATE.format(
            hostname=hostname, bucket_path=self.bucket.path, api_version=_API_VERSION
        )
        name_value_pairs = []

        if self.user_project is not None:
            name_value_pairs.append(("userProject", self.user_project))

        # When a Customer Managed Encryption Key is used to encrypt Cloud Storage object
        # at rest, object resource metadata will store the version of the Key Management
        # Service cryptographic material. If a Blob instance with KMS Key metadata set is
        # used to upload a new version of the object then the existing kmsKeyName version
        # value can't be used in the upload request and the client instead ignores it.
        if (
            self.kms_key_name is not None
            and "cryptoKeyVersions" not in self.kms_key_name
        ):
            name_value_pairs.append(("kmsKeyName", self.kms_key_name))

        if predefined_acl is not None:
            name_value_pairs.append(("predefinedAcl", predefined_acl))

        if if_generation_match is not None:
            name_value_pairs.append(("ifGenerationMatch", if_generation_match))

        if if_generation_not_match is not None:
            name_value_pairs.append(("ifGenerationNotMatch", if_generation_not_match))

        if if_metageneration_match is not None:
            name_value_pairs.append(("ifMetagenerationMatch", if_metageneration_match))

        if if_metageneration_not_match is not None:
            name_value_pairs.append(
                ("ifMetaGenerationNotMatch", if_metageneration_not_match)
            )

        upload_url = _add_query_parameters(base_url, name_value_pairs)
        upload = ResumableUpload(
            upload_url, chunk_size, headers=headers, checksum=checksum
        )

        upload._retry_strategy = _api_core_retry_to_resumable_media_retry(
            retry, num_retries
        )

        upload.initiate(
            transport,
            stream,
            object_metadata,
            content_type,
            total_bytes=size,
            stream_final=False,
            timeout=timeout,
        )

        return upload, transport

    def _do_resumable_upload(
        self,
        client,
        stream,
        content_type,
        size,
        num_retries,
        predefined_acl,
        if_generation_match,
        if_generation_not_match,
        if_metageneration_match,
        if_metageneration_not_match,
        timeout=_DEFAULT_TIMEOUT,
        checksum=None,
        retry=None,
        command=None,
    ):
        """Perform a resumable upload.

        Assumes ``chunk_size`` is not :data:`None` on the current blob.
        The default value of ``chunk_size`` is 100 MB.

        The content type of the upload will be determined in order
        of precedence:

        - The value passed in to this method (if not :data:`None`)
        - The value stored on the current blob
        - The default value ('application/octet-stream')

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use.  If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type stream: IO[bytes]
        :param stream: A bytes IO object open for reading.

        :type content_type: str
        :param content_type: Type of content being uploaded (or :data:`None`).

        :type size: int
        :param size:
            The number of bytes to be uploaded (which will be read from
            ``stream``). If not provided, the upload will be concluded once
            ``stream`` is exhausted (or :data:`None`).

        :type num_retries: int
        :param num_retries:
            Number of upload retries. By default, only uploads with
            if_generation_match set will be retried, as uploads without the
            argument are not guaranteed to be idempotent. Setting num_retries
            will override this default behavior and guarantee retries even when
            if_generation_match is not set.  (Deprecated: This argument
            will be removed in a future release.)

        :type predefined_acl: str
        :param predefined_acl: (Optional) Predefined access control list

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type checksum: str
        :param checksum:
            (Optional) The type of checksum to compute to verify
            the integrity of the object. After the upload is complete, the
            server-computed checksum of the resulting object will be checked
            and google.resumable_media.common.DataCorruption will be raised on
            a mismatch. On a validation failure, the client will attempt to
            delete the uploaded object automatically. Supported values
            are "md5", "crc32c" and None. The default is None.

        :type retry: google.api_core.retry.Retry
        :param retry: (Optional) How to retry the RPC. A None value will disable
            retries. A google.api_core.retry.Retry value will enable retries,
            and the object will configure backoff and timeout options. Custom
            predicates (customizable error codes) are not supported for media
            operations such as this one.

            This private method does not accept ConditionalRetryPolicy values
            because the information necessary to evaluate the policy is instead
            evaluated in blob._do_upload().

            See the retry.py source code and docstrings in this package
            (google.cloud.storage.retry) for information on retry types and how
            to configure them.

        :type command: str
        :param command:
            (Optional) Information about which interface for upload was used,
            to be included in the X-Goog-API-Client header. Please leave as None
            unless otherwise directed.

        :rtype: :class:`~requests.Response`
        :returns: The "200 OK" response object returned after the final chunk
                  is uploaded.
        """
        upload, transport = self._initiate_resumable_upload(
            client,
            stream,
            content_type,
            size,
            num_retries,
            predefined_acl=predefined_acl,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
            timeout=timeout,
            checksum=checksum,
            retry=retry,
            command=command,
        )
        extra_attributes = {
            "url.full": upload.resumable_url,
            "upload.chunk_size": upload.chunk_size,
            "upload.checksum": f"{checksum}",
        }
        args = {"timeout": timeout}
        with create_trace_span(
            name="Storage.ResumableUpload/transmitNextChunk",
            attributes=extra_attributes,
            client=client,
            api_request=args,
        ):
            while not upload.finished:
                try:
                    response = upload.transmit_next_chunk(transport, timeout=timeout)
                except resumable_media.DataCorruption:
                    # Attempt to delete the corrupted object.
                    self.delete()
                    raise
            return response

    def _do_upload(
        self,
        client,
        stream,
        content_type,
        size,
        num_retries,
        predefined_acl,
        if_generation_match,
        if_generation_not_match,
        if_metageneration_match,
        if_metageneration_not_match,
        timeout=_DEFAULT_TIMEOUT,
        checksum=None,
        retry=None,
        command=None,
    ):
        """Determine an upload strategy and then perform the upload.

        If the size of the data to be uploaded exceeds 8 MB a resumable media
        request will be used, otherwise the content and the metadata will be
        uploaded in a single multipart upload request.

        The content type of the upload will be determined in order
        of precedence:

        - The value passed in to this method (if not :data:`None`)
        - The value stored on the current blob
        - The default value ('application/octet-stream')

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use.  If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type stream: IO[bytes]
        :param stream: A bytes IO object open for reading.

        :type content_type: str
        :param content_type: Type of content being uploaded (or :data:`None`).

        :type size: int
        :param size:
            The number of bytes to be uploaded (which will be read from
            ``stream``). If not provided, the upload will be concluded once
            ``stream`` is exhausted (or :data:`None`).

        :type num_retries: int
        :param num_retries:
            Number of upload retries. By default, only uploads with
            if_generation_match set will be retried, as uploads without the
            argument are not guaranteed to be idempotent. Setting num_retries
            will override this default behavior and guarantee retries even when
            if_generation_match is not set.  (Deprecated: This argument
            will be removed in a future release.)

        :type predefined_acl: str
        :param predefined_acl: (Optional) Predefined access control list

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type checksum: str
        :param checksum:
            (Optional) The type of checksum to compute to verify
            the integrity of the object. If the upload is completed in a single
            request, the checksum will be entirely precomputed and the remote
            server will handle verification and error handling. If the upload
            is too large and must be transmitted in multiple requests, the
            checksum will be incrementally computed and the client will handle
            verification and error handling, raising
            google.resumable_media.common.DataCorruption on a mismatch and
            attempting to delete the corrupted file. Supported values are
            "md5", "crc32c" and None. The default is None.

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry: (Optional) How to retry the RPC. A None value will disable
            retries. A google.api_core.retry.Retry value will enable retries,
            and the object will define retriable response codes and errors and
            configure backoff and timeout options.

            A google.cloud.storage.retry.ConditionalRetryPolicy value wraps a
            Retry object and activates it only if certain conditions are met.
            This class exists to provide safe defaults for RPC calls that are
            not technically safe to retry normally (due to potential data
            duplication or other side-effects) but become safe to retry if a
            condition such as if_generation_match is set.

            See the retry.py source code and docstrings in this package
            (google.cloud.storage.retry) for information on retry types and how
            to configure them.

            Media operations (downloads and uploads) do not support non-default
            predicates in a Retry object. The default will always be used. Other
            configuration changes for Retry objects such as delays and deadlines
            are respected.

        :type command: str
        :param command:
            (Optional) Information about which interface for upload was used,
            to be included in the X-Goog-API-Client header. Please leave as None
            unless otherwise directed.

        :rtype: dict
        :returns: The parsed JSON from the "200 OK" response. This will be the
                  **only** response in the multipart case and it will be the
                  **final** response in the resumable case.
        """

        # Handle ConditionalRetryPolicy.
        if isinstance(retry, ConditionalRetryPolicy):
            # Conditional retries are designed for non-media calls, which change
            # arguments into query_params dictionaries. Media operations work
            # differently, so here we make a "fake" query_params to feed to the
            # ConditionalRetryPolicy.
            query_params = {
                "ifGenerationMatch": if_generation_match,
                "ifMetagenerationMatch": if_metageneration_match,
            }
            retry = retry.get_retry_policy_if_conditions_met(query_params=query_params)

        if size is not None and size <= _MAX_MULTIPART_SIZE:
            response = self._do_multipart_upload(
                client,
                stream,
                content_type,
                size,
                num_retries,
                predefined_acl,
                if_generation_match,
                if_generation_not_match,
                if_metageneration_match,
                if_metageneration_not_match,
                timeout=timeout,
                checksum=checksum,
                retry=retry,
                command=command,
            )
        else:
            response = self._do_resumable_upload(
                client,
                stream,
                content_type,
                size,
                num_retries,
                predefined_acl,
                if_generation_match,
                if_generation_not_match,
                if_metageneration_match,
                if_metageneration_not_match,
                timeout=timeout,
                checksum=checksum,
                retry=retry,
                command=command,
            )

        return response.json()

    def _prep_and_do_upload(
        self,
        file_obj,
        rewind=False,
        size=None,
        content_type=None,
        num_retries=None,
        client=None,
        predefined_acl=None,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        timeout=_DEFAULT_TIMEOUT,
        checksum=None,
        retry=DEFAULT_RETRY_IF_GENERATION_SPECIFIED,
        command=None,
    ):
        """Upload the contents of this blob from a file-like object.

        The content type of the upload will be determined in order
        of precedence:

        - The value passed in to this method (if not :data:`None`)
        - The value stored on the current blob
        - The default value ('application/octet-stream')

        .. note::
           The effect of uploading to an existing blob depends on the
           "versioning" and "lifecycle" policies defined on the blob's
           bucket.  In the absence of those policies, upload will
           overwrite any existing contents.

           See the [`object versioning`](https://cloud.google.com/storage/docs/object-versioning)
           and [`lifecycle`](https://cloud.google.com/storage/docs/lifecycle)
           API documents for details.

        If the size of the data to be uploaded exceeds 8 MB a resumable media
        request will be used, otherwise the content and the metadata will be
        uploaded in a single multipart upload request.

        For more fine-grained over the upload process, check out
        [`google-resumable-media`](https://googleapis.dev/python/google-resumable-media/latest/index.html).

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        :type file_obj: file
        :param file_obj: A file handle opened in binary mode for reading.

        :type rewind: bool
        :param rewind:
            If True, seek to the beginning of the file handle before writing
            the file to Cloud Storage.

        :type size: int
        :param size:
            The number of bytes to be uploaded (which will be read from
            ``file_obj``). If not provided, the upload will be concluded once
            ``file_obj`` is exhausted.

        :type content_type: str
        :param content_type: (Optional) Type of content being uploaded.

        :type num_retries: int
        :param num_retries:
            Number of upload retries. By default, only uploads with
            if_generation_match set will be retried, as uploads without the
            argument are not guaranteed to be idempotent. Setting num_retries
            will override this default behavior and guarantee retries even when
            if_generation_match is not set.  (Deprecated: This argument
            will be removed in a future release.)

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use.  If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type predefined_acl: str
        :param predefined_acl: (Optional) Predefined access control list

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type checksum: str
        :param checksum:
            (Optional) The type of checksum to compute to verify
            the integrity of the object. If the upload is completed in a single
            request, the checksum will be entirely precomputed and the remote
            server will handle verification and error handling. If the upload
            is too large and must be transmitted in multiple requests, the
            checksum will be incrementally computed and the client will handle
            verification and error handling, raising
            google.resumable_media.common.DataCorruption on a mismatch and
            attempting to delete the corrupted file. Supported values are
            "md5", "crc32c" and None. The default is None.

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry: (Optional) How to retry the RPC. A None value will disable
            retries. A google.api_core.retry.Retry value will enable retries,
            and the object will define retriable response codes and errors and
            configure backoff and timeout options.

            A google.cloud.storage.retry.ConditionalRetryPolicy value wraps a
            Retry object and activates it only if certain conditions are met.
            This class exists to provide safe defaults for RPC calls that are
            not technically safe to retry normally (due to potential data
            duplication or other side-effects) but become safe to retry if a
            condition such as if_generation_match is set.

            See the retry.py source code and docstrings in this package
            (google.cloud.storage.retry) for information on retry types and how
            to configure them.

            Media operations (downloads and uploads) do not support non-default
            predicates in a Retry object. The default will always be used. Other
            configuration changes for Retry objects such as delays and deadlines
            are respected.

        :type command: str
        :param command:
            (Optional) Information about which interface for upload was used,
            to be included in the X-Goog-API-Client header. Please leave as None
            unless otherwise directed.

        :raises: :class:`~google.cloud.exceptions.GoogleCloudError`
                 if the upload response returns an error status.
        """
        if num_retries is not None:
            warnings.warn(_NUM_RETRIES_MESSAGE, DeprecationWarning, stacklevel=2)
            # num_retries and retry are mutually exclusive. If num_retries is
            # set and retry is exactly the default, then nullify retry for
            # backwards compatibility.
            if retry is DEFAULT_RETRY_IF_GENERATION_SPECIFIED:
                retry = None

        _maybe_rewind(file_obj, rewind=rewind)
        predefined_acl = ACL.validate_predefined(predefined_acl)

        try:
            created_json = self._do_upload(
                client,
                file_obj,
                content_type,
                size,
                num_retries,
                predefined_acl,
                if_generation_match,
                if_generation_not_match,
                if_metageneration_match,
                if_metageneration_not_match,
                timeout=timeout,
                checksum=checksum,
                retry=retry,
                command=command,
            )
            self._set_properties(created_json)
        except resumable_media.InvalidResponse as exc:
            _raise_from_invalid_response(exc)

    @create_trace_span(name="Storage.Blob.uploadFromFile")
    def upload_from_file(
        self,
        file_obj,
        rewind=False,
        size=None,
        content_type=None,
        num_retries=None,
        client=None,
        predefined_acl=None,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        timeout=_DEFAULT_TIMEOUT,
        checksum=None,
        retry=DEFAULT_RETRY_IF_GENERATION_SPECIFIED,
    ):
        """Upload the contents of this blob from a file-like object.

        The content type of the upload will be determined in order
        of precedence:

        - The value passed in to this method (if not :data:`None`)
        - The value stored on the current blob
        - The default value ('application/octet-stream')

        .. note::
           The effect of uploading to an existing blob depends on the
           "versioning" and "lifecycle" policies defined on the blob's
           bucket.  In the absence of those policies, upload will
           overwrite any existing contents.

           See the [`object versioning`](https://cloud.google.com/storage/docs/object-versioning)
           and [`lifecycle`](https://cloud.google.com/storage/docs/lifecycle)
           API documents for details.

        If the size of the data to be uploaded exceeds 8 MB a resumable media
        request will be used, otherwise the content and the metadata will be
        uploaded in a single multipart upload request.

        For more fine-grained over the upload process, check out
        [`google-resumable-media`](https://googleapis.dev/python/google-resumable-media/latest/index.html).

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        :type file_obj: file
        :param file_obj: A file handle opened in binary mode for reading.

        :type rewind: bool
        :param rewind:
            If True, seek to the beginning of the file handle before writing
            the file to Cloud Storage.

        :type size: int
        :param size:
            The number of bytes to be uploaded (which will be read from
            ``file_obj``). If not provided, the upload will be concluded once
            ``file_obj`` is exhausted.

        :type content_type: str
        :param content_type: (Optional) Type of content being uploaded.

        :type num_retries: int
        :param num_retries:
            Number of upload retries. By default, only uploads with
            if_generation_match set will be retried, as uploads without the
            argument are not guaranteed to be idempotent. Setting num_retries
            will override this default behavior and guarantee retries even when
            if_generation_match is not set.  (Deprecated: This argument
            will be removed in a future release.)

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use.  If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type predefined_acl: str
        :param predefined_acl: (Optional) Predefined access control list

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type checksum: str
        :param checksum:
            (Optional) The type of checksum to compute to verify
            the integrity of the object. If the upload is completed in a single
            request, the checksum will be entirely precomputed and the remote
            server will handle verification and error handling. If the upload
            is too large and must be transmitted in multiple requests, the
            checksum will be incrementally computed and the client will handle
            verification and error handling, raising
            google.resumable_media.common.DataCorruption on a mismatch and
            attempting to delete the corrupted file. Supported values are
            "md5", "crc32c" and None. The default is None.

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry: (Optional) How to retry the RPC.
            The default value is ``DEFAULT_RETRY_IF_GENERATION_SPECIFIED``, a conditional retry
            policy which will only enable retries if ``if_generation_match`` or ``generation``
            is set, in order to ensure requests are idempotent before retrying them.
            Change the value to ``DEFAULT_RETRY`` or another `google.api_core.retry.Retry` object
            to enable retries regardless of generation precondition setting.
            See [Configuring Retries](https://cloud.google.com/python/docs/reference/storage/latest/retry_timeout).

            Media operations (downloads and uploads) do not support non-default
            predicates in a Retry object. Other configuration changes for Retry objects
            such as delays and deadlines are respected.

        :raises: :class:`~google.cloud.exceptions.GoogleCloudError`
                 if the upload response returns an error status.
        """
        self._prep_and_do_upload(
            file_obj,
            rewind=rewind,
            size=size,
            content_type=content_type,
            num_retries=num_retries,
            client=client,
            predefined_acl=predefined_acl,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
            timeout=timeout,
            checksum=checksum,
            retry=retry,
        )

    def _handle_filename_and_upload(self, filename, content_type=None, *args, **kwargs):
        """Upload this blob's contents from the content of a named file.

        :type filename: str
        :param filename: The path to the file.

        :type content_type: str
        :param content_type: (Optional) Type of content being uploaded.

        For *args and **kwargs, refer to the documentation for upload_from_filename() for more information.
        """

        content_type = self._get_content_type(content_type, filename=filename)

        with open(filename, "rb") as file_obj:
            total_bytes = os.fstat(file_obj.fileno()).st_size
            self._prep_and_do_upload(
                file_obj,
                content_type=content_type,
                size=total_bytes,
                *args,
                **kwargs,
            )

    @create_trace_span(name="Storage.Blob.uploadFromFilename")
    def upload_from_filename(
        self,
        filename,
        content_type=None,
        num_retries=None,
        client=None,
        predefined_acl=None,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        timeout=_DEFAULT_TIMEOUT,
        checksum=None,
        retry=DEFAULT_RETRY_IF_GENERATION_SPECIFIED,
    ):
        """Upload this blob's contents from the content of a named file.

        The content type of the upload will be determined in order
        of precedence:

        - The value passed in to this method (if not :data:`None`)
        - The value stored on the current blob
        - The value given by ``mimetypes.guess_type``
        - The default value ('application/octet-stream')

        .. note::
           The effect of uploading to an existing blob depends on the
           "versioning" and "lifecycle" policies defined on the blob's
           bucket.  In the absence of those policies, upload will
           overwrite any existing contents.

           See the [`object versioning`](https://cloud.google.com/storage/docs/object-versioning)
           and [`lifecycle`](https://cloud.google.com/storage/docs/lifecycle)
           API documents for details.

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        See a [code sample](https://cloud.google.com/storage/docs/samples/storage-upload-encrypted-file#storage_upload_encrypted_file-python)
        to upload a file with a
        [`customer-supplied encryption key`](https://cloud.google.com/storage/docs/encryption#customer-supplied).

        :type filename: str
        :param filename: The path to the file.

        :type content_type: str
        :param content_type: (Optional) Type of content being uploaded.

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use.  If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type num_retries: int
        :param num_retries:
            Number of upload retries. By default, only uploads with
            if_generation_match set will be retried, as uploads without the
            argument are not guaranteed to be idempotent. Setting num_retries
            will override this default behavior and guarantee retries even when
            if_generation_match is not set.  (Deprecated: This argument
            will be removed in a future release.)

        :type predefined_acl: str
        :param predefined_acl: (Optional) Predefined access control list

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type checksum: str
        :param checksum:
            (Optional) The type of checksum to compute to verify
            the integrity of the object. If the upload is completed in a single
            request, the checksum will be entirely precomputed and the remote
            server will handle verification and error handling. If the upload
            is too large and must be transmitted in multiple requests, the
            checksum will be incrementally computed and the client will handle
            verification and error handling, raising
            google.resumable_media.common.DataCorruption on a mismatch and
            attempting to delete the corrupted file. Supported values are
            "md5", "crc32c" and None. The default is None.

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry: (Optional) How to retry the RPC.
            The default value is ``DEFAULT_RETRY_IF_GENERATION_SPECIFIED``, a conditional retry
            policy which will only enable retries if ``if_generation_match`` or ``generation``
            is set, in order to ensure requests are idempotent before retrying them.
            Change the value to ``DEFAULT_RETRY`` or another `google.api_core.retry.Retry` object
            to enable retries regardless of generation precondition setting.
            See [Configuring Retries](https://cloud.google.com/python/docs/reference/storage/latest/retry_timeout).

            Media operations (downloads and uploads) do not support non-default
            predicates in a Retry object. Other configuration changes for Retry objects
            such as delays and deadlines are respected.
        """

        self._handle_filename_and_upload(
            filename,
            content_type=content_type,
            num_retries=num_retries,
            client=client,
            predefined_acl=predefined_acl,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
            timeout=timeout,
            checksum=checksum,
            retry=retry,
        )

    @create_trace_span(name="Storage.Blob.uploadFromString")
    def upload_from_string(
        self,
        data,
        content_type="text/plain",
        num_retries=None,
        client=None,
        predefined_acl=None,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        timeout=_DEFAULT_TIMEOUT,
        checksum=None,
        retry=DEFAULT_RETRY_IF_GENERATION_SPECIFIED,
    ):
        """Upload contents of this blob from the provided string.

        .. note::
           The effect of uploading to an existing blob depends on the
           "versioning" and "lifecycle" policies defined on the blob's
           bucket.  In the absence of those policies, upload will
           overwrite any existing contents.

           See the [`object versioning`](https://cloud.google.com/storage/docs/object-versioning)
           and [`lifecycle`](https://cloud.google.com/storage/docs/lifecycle)
           API documents for details.

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        :type data: bytes or str
        :param data:
            The data to store in this blob.  If the value is text, it will be
            encoded as UTF-8.

        :type content_type: str
        :param content_type:
            (Optional) Type of content being uploaded. Defaults to
            ``'text/plain'``.

        :type num_retries: int
        :param num_retries:
            Number of upload retries. By default, only uploads with
            if_generation_match set will be retried, as uploads without the
            argument are not guaranteed to be idempotent. Setting num_retries
            will override this default behavior and guarantee retries even when
            if_generation_match is not set.  (Deprecated: This argument
            will be removed in a future release.)

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use.  If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type predefined_acl: str
        :param predefined_acl: (Optional) Predefined access control list

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type checksum: str
        :param checksum:
            (Optional) The type of checksum to compute to verify
            the integrity of the object. If the upload is completed in a single
            request, the checksum will be entirely precomputed and the remote
            server will handle verification and error handling. If the upload
            is too large and must be transmitted in multiple requests, the
            checksum will be incrementally computed and the client will handle
            verification and error handling, raising
            google.resumable_media.common.DataCorruption on a mismatch and
            attempting to delete the corrupted file. Supported values are
            "md5", "crc32c" and None. The default is None.

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry: (Optional) How to retry the RPC.
            The default value is ``DEFAULT_RETRY_IF_GENERATION_SPECIFIED``, a conditional retry
            policy which will only enable retries if ``if_generation_match`` or ``generation``
            is set, in order to ensure requests are idempotent before retrying them.
            Change the value to ``DEFAULT_RETRY`` or another `google.api_core.retry.Retry` object
            to enable retries regardless of generation precondition setting.
            See [Configuring Retries](https://cloud.google.com/python/docs/reference/storage/latest/retry_timeout).

            Media operations (downloads and uploads) do not support non-default
            predicates in a Retry object. Other configuration changes for Retry objects
            such as delays and deadlines are respected.
        """
        data = _to_bytes(data, encoding="utf-8")
        string_buffer = BytesIO(data)
        self.upload_from_file(
            file_obj=string_buffer,
            size=len(data),
            content_type=content_type,
            num_retries=num_retries,
            client=client,
            predefined_acl=predefined_acl,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
            timeout=timeout,
            checksum=checksum,
            retry=retry,
        )

    @create_trace_span(name="Storage.Blob.createResumableUploadSession")
    def create_resumable_upload_session(
        self,
        content_type=None,
        size=None,
        origin=None,
        client=None,
        timeout=_DEFAULT_TIMEOUT,
        checksum=None,
        predefined_acl=None,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        retry=DEFAULT_RETRY_IF_GENERATION_SPECIFIED,
    ):
        """Create a resumable upload session.

        Resumable upload sessions allow you to start an upload session from
        one client and complete the session in another. This method is called
        by the initiator to set the metadata and limits. The initiator then
        passes the session URL to the client that will upload the binary data.
        The client performs a PUT request on the session URL to complete the
        upload. This process allows untrusted clients to upload to an
        access-controlled bucket.

        For more details, see the
        documentation on [`signed URLs`](https://cloud.google.com/storage/docs/access-control/signed-urls#signing-resumable).

        The content type of the upload will be determined in order
        of precedence:

        - The value passed in to this method (if not :data:`None`)
        - The value stored on the current blob
        - The default value ('application/octet-stream')

        .. note::
           The effect of uploading to an existing blob depends on the
           "versioning" and "lifecycle" policies defined on the blob's
           bucket.  In the absence of those policies, upload will
           overwrite any existing contents.

           See the [`object versioning`](https://cloud.google.com/storage/docs/object-versioning)
           and [`lifecycle`](https://cloud.google.com/storage/docs/lifecycle)
           API documents for details.

        If :attr:`encryption_key` is set, the blob will be encrypted with
        a [`customer-supplied`](https://cloud.google.com/storage/docs/encryption#customer-supplied)
        encryption key.

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        :type size: int
        :param size:
            (Optional) The maximum number of bytes that can be uploaded using
            this session. If the size is not known when creating the session,
            this should be left blank.

        :type content_type: str
        :param content_type: (Optional) Type of content being uploaded.

        :type origin: str
        :param origin:
            (Optional) If set, the upload can only be completed by a user-agent
            that uploads from the given origin. This can be useful when passing
            the session to a web client.

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use.  If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type checksum: str
        :param checksum:
            (Optional) The type of checksum to compute to verify
            the integrity of the object. After the upload is complete, the
            server-computed checksum of the resulting object will be checked
            and google.resumable_media.common.DataCorruption will be raised on
            a mismatch. On a validation failure, the client will attempt to
            delete the uploaded object automatically. Supported values
            are "md5", "crc32c" and None. The default is None.

        :type predefined_acl: str
        :param predefined_acl: (Optional) Predefined access control list

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry: (Optional) How to retry the RPC.
            The default value is ``DEFAULT_RETRY_IF_GENERATION_SPECIFIED``, a conditional retry
            policy which will only enable retries if ``if_generation_match`` or ``generation``
            is set, in order to ensure requests are idempotent before retrying them.
            Change the value to ``DEFAULT_RETRY`` or another `google.api_core.retry.Retry` object
            to enable retries regardless of generation precondition setting.
            See [Configuring Retries](https://cloud.google.com/python/docs/reference/storage/latest/retry_timeout).

            Media operations (downloads and uploads) do not support non-default
            predicates in a Retry object. Other configuration changes for Retry objects
            such as delays and deadlines are respected.

        :rtype: str
        :returns: The resumable upload session URL. The upload can be
                  completed by making an HTTP PUT request with the
                  file's contents.

        :raises: :class:`google.cloud.exceptions.GoogleCloudError`
                 if the session creation response returns an error status.
        """

        # Handle ConditionalRetryPolicy.
        if isinstance(retry, ConditionalRetryPolicy):
            # Conditional retries are designed for non-media calls, which change
            # arguments into query_params dictionaries. Media operations work
            # differently, so here we make a "fake" query_params to feed to the
            # ConditionalRetryPolicy.
            query_params = {
                "ifGenerationMatch": if_generation_match,
                "ifMetagenerationMatch": if_metageneration_match,
            }
            retry = retry.get_retry_policy_if_conditions_met(query_params=query_params)

        extra_headers = {}
        if origin is not None:
            # This header is specifically for client-side uploads, it
            # determines the origins allowed for CORS.
            extra_headers["Origin"] = origin

        try:
            fake_stream = BytesIO(b"")
            # Send a fake the chunk size which we **know** will be acceptable
            # to the `ResumableUpload` constructor. The chunk size only
            # matters when **sending** bytes to an upload.
            upload, _ = self._initiate_resumable_upload(
                client,
                fake_stream,
                content_type,
                size,
                None,
                predefined_acl=predefined_acl,
                if_generation_match=if_generation_match,
                if_generation_not_match=if_generation_not_match,
                if_metageneration_match=if_metageneration_match,
                if_metageneration_not_match=if_metageneration_not_match,
                extra_headers=extra_headers,
                chunk_size=self._CHUNK_SIZE_MULTIPLE,
                timeout=timeout,
                checksum=checksum,
                retry=retry,
            )

            return upload.resumable_url
        except resumable_media.InvalidResponse as exc:
            _raise_from_invalid_response(exc)

    @create_trace_span(name="Storage.Blob.getIamPolicy")
    def get_iam_policy(
        self,
        client=None,
        requested_policy_version=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=DEFAULT_RETRY,
    ):
        """Retrieve the IAM policy for the object.

        .. note::

           Blob- / object-level IAM support does not yet exist and methods
           currently call an internal ACL backend not providing any utility
           beyond the blob's :attr:`acl` at this time. The API may be enhanced
           in the future and is currently undocumented. Use :attr:`acl` for
           managing object access control.

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use.  If not passed, falls back to the
            ``client`` stored on the current object's bucket.

        :type requested_policy_version: int or ``NoneType``
        :param requested_policy_version:
            (Optional) The version of IAM policies to request.  If a policy
            with a condition is requested without setting this, the server will
            return an error.  This must be set to a value of 3 to retrieve IAM
            policies containing conditions. This is to prevent client code that
            isn't aware of IAM conditions from interpreting and modifying
            policies incorrectly.  The service might return a policy with
            version lower than the one that was requested, based on the feature
            syntax in the policy fetched.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC. See: :ref:`configuring_retries`

        :rtype: :class:`google.api_core.iam.Policy`
        :returns: the policy instance, based on the resource returned from
                  the ``getIamPolicy`` API request.
        """
        client = self._require_client(client)

        query_params = {}

        if self.user_project is not None:
            query_params["userProject"] = self.user_project

        if requested_policy_version is not None:
            query_params["optionsRequestedPolicyVersion"] = requested_policy_version

        info = client._get_resource(
            f"{self.path}/iam",
            query_params=query_params,
            timeout=timeout,
            retry=retry,
            _target_object=None,
        )
        return Policy.from_api_repr(info)

    @create_trace_span(name="Storage.Blob.setIamPolicy")
    def set_iam_policy(
        self,
        policy,
        client=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=DEFAULT_RETRY_IF_ETAG_IN_JSON,
    ):
        """Update the IAM policy for the bucket.

        .. note::

           Blob- / object-level IAM support does not yet exist and methods
           currently call an internal ACL backend not providing any utility
           beyond the blob's :attr:`acl` at this time. The API may be enhanced
           in the future and is currently undocumented. Use :attr:`acl` for
           managing object access control.

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        :type policy: :class:`google.api_core.iam.Policy`
        :param policy: policy instance used to update bucket's IAM policy.

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use.  If not passed, falls back to the
            ``client`` stored on the current bucket.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC. See: :ref:`configuring_retries`

        :rtype: :class:`google.api_core.iam.Policy`
        :returns: the policy instance, based on the resource returned from
                  the ``setIamPolicy`` API request.
        """
        client = self._require_client(client)

        query_params = {}

        if self.user_project is not None:
            query_params["userProject"] = self.user_project

        path = f"{self.path}/iam"
        resource = policy.to_api_repr()
        resource["resourceId"] = self.path
        info = client._put_resource(
            path,
            resource,
            query_params=query_params,
            timeout=timeout,
            retry=retry,
            _target_object=None,
        )
        return Policy.from_api_repr(info)

    @create_trace_span(name="Storage.Blob.testIamPermissions")
    def test_iam_permissions(
        self, permissions, client=None, timeout=_DEFAULT_TIMEOUT, retry=DEFAULT_RETRY
    ):
        """API call:  test permissions

        .. note::

           Blob- / object-level IAM support does not yet exist and methods
           currently call an internal ACL backend not providing any utility
           beyond the blob's :attr:`acl` at this time. The API may be enhanced
           in the future and is currently undocumented. Use :attr:`acl` for
           managing object access control.

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        :type permissions: list of string
        :param permissions: the permissions to check

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use.  If not passed, falls back to the
            ``client`` stored on the current bucket.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC. See: :ref:`configuring_retries`

        :rtype: list of string
        :returns: the permissions returned by the ``testIamPermissions`` API
                  request.
        """
        client = self._require_client(client)
        query_params = {"permissions": permissions}

        if self.user_project is not None:
            query_params["userProject"] = self.user_project

        path = f"{self.path}/iam/testPermissions"
        resp = client._get_resource(
            path,
            query_params=query_params,
            timeout=timeout,
            retry=retry,
            _target_object=None,
        )

        return resp.get("permissions", [])

    @create_trace_span(name="Storage.Blob.makePublic")
    def make_public(
        self,
        client=None,
        timeout=_DEFAULT_TIMEOUT,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        retry=DEFAULT_RETRY_IF_METAGENERATION_SPECIFIED,
    ):
        """Update blob's ACL, granting read access to anonymous users.

        :type client: :class:`~google.cloud.storage.client.Client` or
                      ``NoneType``
        :param client: (Optional) The client to use.  If not passed, falls back
                       to the ``client`` stored on the blob's bucket.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC. See: :ref:`configuring_retries`
        """
        self.acl.all().grant_read()
        self.acl.save(
            client=client,
            timeout=timeout,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
            retry=retry,
        )

    @create_trace_span(name="Storage.Blob.makePrivate")
    def make_private(
        self,
        client=None,
        timeout=_DEFAULT_TIMEOUT,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        retry=DEFAULT_RETRY_IF_METAGENERATION_SPECIFIED,
    ):
        """Update blob's ACL, revoking read access for anonymous users.

        :type client: :class:`~google.cloud.storage.client.Client` or
                      ``NoneType``
        :param client: (Optional) The client to use.  If not passed, falls back
                       to the ``client`` stored on the blob's bucket.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC. See: :ref:`configuring_retries`
        """
        self.acl.all().revoke_read()
        self.acl.save(
            client=client,
            timeout=timeout,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
            retry=retry,
        )

    @create_trace_span(name="Storage.Blob.compose")
    def compose(
        self,
        sources,
        client=None,
        timeout=_DEFAULT_TIMEOUT,
        if_generation_match=None,
        if_metageneration_match=None,
        if_source_generation_match=None,
        retry=DEFAULT_RETRY_IF_GENERATION_SPECIFIED,
    ):
        """Concatenate source blobs into this one.

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        See [API reference docs](https://cloud.google.com/storage/docs/json_api/v1/objects/compose)
        and a [code sample](https://cloud.google.com/storage/docs/samples/storage-compose-file#storage_compose_file-python).

        :type sources: list of :class:`Blob`
        :param sources: Blobs whose contents will be composed into this blob.

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use. If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) Makes the operation conditional on whether the
            destination object's current generation matches the given value.
            Setting to 0 makes the operation succeed only if there are no live
            versions of the object.
            Note: In a previous version, this argument worked identically to the
            ``if_source_generation_match`` argument. For
            backwards-compatibility reasons, if a list is passed in,
            this argument will behave like ``if_source_generation_match``
            and also issue a DeprecationWarning.

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) Makes the operation conditional on whether the
            destination object's current metageneration matches the given
            value.

            If a list of long is passed in, no match operation will be
            performed.  (Deprecated: type(list of long) is supported for
            backwards-compatability reasons only.)

        :type if_source_generation_match: list of long
        :param if_source_generation_match:
            (Optional) Makes the operation conditional on whether the current
            generation of each source blob matches the corresponding generation.
            The list must match ``sources`` item-to-item.

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC.
            The default value is ``DEFAULT_RETRY_IF_GENERATION_SPECIFIED``, a conditional retry
            policy which will only enable retries if ``if_generation_match`` or ``generation``
            is set, in order to ensure requests are idempotent before retrying them.
            Change the value to ``DEFAULT_RETRY`` or another `google.api_core.retry.Retry` object
            to enable retries regardless of generation precondition setting.
            See [Configuring Retries](https://cloud.google.com/python/docs/reference/storage/latest/retry_timeout).
        """
        sources_len = len(sources)
        client = self._require_client(client)
        query_params = {}

        if isinstance(if_generation_match, list):
            warnings.warn(
                _COMPOSE_IF_GENERATION_LIST_DEPRECATED,
                DeprecationWarning,
                stacklevel=2,
            )

            if if_source_generation_match is not None:
                raise ValueError(
                    _COMPOSE_IF_GENERATION_LIST_AND_IF_SOURCE_GENERATION_ERROR
                )

            if_source_generation_match = if_generation_match
            if_generation_match = None

        if isinstance(if_metageneration_match, list):
            warnings.warn(
                _COMPOSE_IF_METAGENERATION_LIST_DEPRECATED,
                DeprecationWarning,
                stacklevel=2,
            )

            if_metageneration_match = None

        if if_source_generation_match is None:
            if_source_generation_match = [None] * sources_len
        if len(if_source_generation_match) != sources_len:
            raise ValueError(_COMPOSE_IF_SOURCE_GENERATION_MISMATCH_ERROR)

        source_objects = []
        for source, source_generation in zip(sources, if_source_generation_match):
            source_object = {"name": source.name, "generation": source.generation}

            preconditions = {}
            if source_generation is not None:
                preconditions["ifGenerationMatch"] = source_generation

            if preconditions:
                source_object["objectPreconditions"] = preconditions

            source_objects.append(source_object)

        request = {
            "sourceObjects": source_objects,
            "destination": self._properties.copy(),
        }

        if self.user_project is not None:
            query_params["userProject"] = self.user_project

        _add_generation_match_parameters(
            query_params,
            if_generation_match=if_generation_match,
            if_metageneration_match=if_metageneration_match,
        )

        api_response = client._post_resource(
            f"{self.path}/compose",
            request,
            query_params=query_params,
            timeout=timeout,
            retry=retry,
            _target_object=self,
        )
        self._set_properties(api_response)

    @create_trace_span(name="Storage.Blob.rewrite")
    def rewrite(
        self,
        source,
        token=None,
        client=None,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        if_source_generation_match=None,
        if_source_generation_not_match=None,
        if_source_metageneration_match=None,
        if_source_metageneration_not_match=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=DEFAULT_RETRY_IF_GENERATION_SPECIFIED,
    ):
        """Rewrite source blob into this one.

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        .. note::

           ``rewrite`` is not supported in a ``Batch`` context.

        :type source: :class:`Blob`
        :param source: blob whose contents will be rewritten into this blob.

        :type token: str
        :param token:
            (Optional) Token returned from an earlier, not-completed call to
            rewrite the same source blob.  If passed, result will include
            updated status, total bytes written.

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use.  If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`
            Note that the generation to be matched is that of the
            ``destination`` blob.

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`
            Note that the generation to be matched is that of the
            ``destination`` blob.

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`
            Note that the metageneration to be matched is that of the
            ``destination`` blob.

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`
            Note that the metageneration to be matched is that of the
            ``destination`` blob.

        :type if_source_generation_match: long
        :param if_source_generation_match:
            (Optional) Makes the operation conditional on whether the source
            object's generation matches the given value.

        :type if_source_generation_not_match: long
        :param if_source_generation_not_match:
            (Optional) Makes the operation conditional on whether the source
            object's generation does not match the given value.

        :type if_source_metageneration_match: long
        :param if_source_metageneration_match:
            (Optional) Makes the operation conditional on whether the source
            object's current metageneration matches the given value.

        :type if_source_metageneration_not_match: long
        :param if_source_metageneration_not_match:
            (Optional) Makes the operation conditional on whether the source
            object's current metageneration does not match the given value.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC.
            The default value is ``DEFAULT_RETRY_IF_GENERATION_SPECIFIED``, a conditional retry
            policy which will only enable retries if ``if_generation_match`` or ``generation``
            is set, in order to ensure requests are idempotent before retrying them.
            Change the value to ``DEFAULT_RETRY`` or another `google.api_core.retry.Retry` object
            to enable retries regardless of generation precondition setting.
            See [Configuring Retries](https://cloud.google.com/python/docs/reference/storage/latest/retry_timeout).

        :rtype: tuple
        :returns: ``(token, bytes_rewritten, total_bytes)``, where ``token``
                  is a rewrite token (``None`` if the rewrite is complete),
                  ``bytes_rewritten`` is the number of bytes rewritten so far,
                  and ``total_bytes`` is the total number of bytes to be
                  rewritten.
        """
        client = self._require_client(client)
        headers = _get_encryption_headers(self._encryption_key)
        headers.update(_get_encryption_headers(source._encryption_key, source=True))

        query_params = self._query_params
        if "generation" in query_params:
            del query_params["generation"]

        if token:
            query_params["rewriteToken"] = token

        if source.generation:
            query_params["sourceGeneration"] = source.generation

        # When a Customer Managed Encryption Key is used to encrypt Cloud Storage object
        # at rest, object resource metadata will store the version of the Key Management
        # Service cryptographic material. If a Blob instance with KMS Key metadata set is
        # used to rewrite the object, then the existing kmsKeyName version
        # value can't be used in the rewrite request and the client instead ignores it.
        if (
            self.kms_key_name is not None
            and "cryptoKeyVersions" not in self.kms_key_name
        ):
            query_params["destinationKmsKeyName"] = self.kms_key_name

        _add_generation_match_parameters(
            query_params,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
            if_source_generation_match=if_source_generation_match,
            if_source_generation_not_match=if_source_generation_not_match,
            if_source_metageneration_match=if_source_metageneration_match,
            if_source_metageneration_not_match=if_source_metageneration_not_match,
        )

        path = f"{source.path}/rewriteTo{self.path}"
        api_response = client._post_resource(
            path,
            self._properties,
            query_params=query_params,
            headers=headers,
            timeout=timeout,
            retry=retry,
            _target_object=self,
        )
        rewritten = int(api_response["totalBytesRewritten"])
        size = int(api_response["objectSize"])

        # The resource key is set if and only if the API response is
        # completely done. Additionally, there is no rewrite token to return
        # in this case.
        if api_response["done"]:
            self._set_properties(api_response["resource"])
            return None, rewritten, size

        return api_response["rewriteToken"], rewritten, size

    @create_trace_span(name="Storage.Blob.updateStorageClass")
    def update_storage_class(
        self,
        new_class,
        client=None,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        if_source_generation_match=None,
        if_source_generation_not_match=None,
        if_source_metageneration_match=None,
        if_source_metageneration_not_match=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=DEFAULT_RETRY_IF_GENERATION_SPECIFIED,
    ):
        """Update blob's storage class via a rewrite-in-place. This helper will
        wait for the rewrite to complete before returning, so it may take some
        time for large files.

        See
        https://cloud.google.com/storage/docs/per-object-storage-class

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        :type new_class: str
        :param new_class:
            new storage class for the object.   One of:
            :attr:`~google.cloud.storage.constants.NEARLINE_STORAGE_CLASS`,
            :attr:`~google.cloud.storage.constants.COLDLINE_STORAGE_CLASS`,
            :attr:`~google.cloud.storage.constants.ARCHIVE_STORAGE_CLASS`,
            :attr:`~google.cloud.storage.constants.STANDARD_STORAGE_CLASS`,
            :attr:`~google.cloud.storage.constants.MULTI_REGIONAL_LEGACY_STORAGE_CLASS`,
            or
            :attr:`~google.cloud.storage.constants.REGIONAL_LEGACY_STORAGE_CLASS`.

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use.  If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`
            Note that the generation to be matched is that of the
            ``destination`` blob.

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`
            Note that the generation to be matched is that of the
            ``destination`` blob.

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`
            Note that the metageneration to be matched is that of the
            ``destination`` blob.

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`
            Note that the metageneration to be matched is that of the
            ``destination`` blob.

        :type if_source_generation_match: long
        :param if_source_generation_match:
            (Optional) Makes the operation conditional on whether the source
            object's generation matches the given value.

        :type if_source_generation_not_match: long
        :param if_source_generation_not_match:
            (Optional) Makes the operation conditional on whether the source
            object's generation does not match the given value.

        :type if_source_metageneration_match: long
        :param if_source_metageneration_match:
            (Optional) Makes the operation conditional on whether the source
            object's current metageneration matches the given value.

        :type if_source_metageneration_not_match: long
        :param if_source_metageneration_not_match:
            (Optional) Makes the operation conditional on whether the source
            object's current metageneration does not match the given value.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC.
            The default value is ``DEFAULT_RETRY_IF_GENERATION_SPECIFIED``, a conditional retry
            policy which will only enable retries if ``if_generation_match`` or ``generation``
            is set, in order to ensure requests are idempotent before retrying them.
            Change the value to ``DEFAULT_RETRY`` or another `google.api_core.retry.Retry` object
            to enable retries regardless of generation precondition setting.
            See [Configuring Retries](https://cloud.google.com/python/docs/reference/storage/latest/retry_timeout).
        """
        # Update current blob's storage class prior to rewrite
        self._patch_property("storageClass", new_class)

        # Execute consecutive rewrite operations until operation is done
        token, _, _ = self.rewrite(
            self,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
            if_source_generation_match=if_source_generation_match,
            if_source_generation_not_match=if_source_generation_not_match,
            if_source_metageneration_match=if_source_metageneration_match,
            if_source_metageneration_not_match=if_source_metageneration_not_match,
            timeout=timeout,
            retry=retry,
        )
        while token is not None:
            token, _, _ = self.rewrite(
                self,
                token=token,
                if_generation_match=if_generation_match,
                if_generation_not_match=if_generation_not_match,
                if_metageneration_match=if_metageneration_match,
                if_metageneration_not_match=if_metageneration_not_match,
                if_source_generation_match=if_source_generation_match,
                if_source_generation_not_match=if_source_generation_not_match,
                if_source_metageneration_match=if_source_metageneration_match,
                if_source_metageneration_not_match=if_source_metageneration_not_match,
                timeout=timeout,
                retry=retry,
            )

    @create_trace_span(name="Storage.Blob.open")
    def open(
        self,
        mode="r",
        chunk_size=None,
        ignore_flush=None,
        encoding=None,
        errors=None,
        newline=None,
        **kwargs,
    ):
        r"""Create a file handler for file-like I/O to or from this blob.

        This method can be used as a context manager, just like Python's
        built-in 'open()' function.

        While reading, as with other read methods, if blob.generation is not set
        the most recent blob generation will be used. Because the file-like IO
        reader downloads progressively in chunks, this could result in data from
        multiple versions being mixed together. If this is a concern, use
        either bucket.get_blob(), or blob.reload(), which will download the
        latest generation number and set it; or, if the generation is known, set
        it manually, for instance with bucket.blob(generation=123456).

        Checksumming (hashing) to verify data integrity is disabled for reads
        using this feature because reads are implemented using request ranges,
        which do not provide checksums to validate. See
        https://cloud.google.com/storage/docs/hashes-etags for details.

        See a [code sample](https://github.com/googleapis/python-storage/blob/main/samples/snippets/storage_fileio_write_read.py).

        Keyword arguments to pass to the underlying API calls.
        For both uploads and downloads, the following arguments are
        supported:

        - ``if_generation_match``
        - ``if_generation_not_match``
        - ``if_metageneration_match``
        - ``if_metageneration_not_match``
        - ``timeout``
        - ``retry``

        For downloads only, the following additional arguments are supported:

        - ``raw_download``

        For uploads only, the following additional arguments are supported:

        - ``content_type``
        - ``num_retries``
        - ``predefined_acl``
        - ``checksum``

        .. note::

           ``num_retries`` is supported for backwards-compatibility
           reasons only; please use ``retry`` with a Retry object or
           ConditionalRetryPolicy instead.

        :type mode: str
        :param mode:
            (Optional) A mode string, as per standard Python `open()` semantics.The first
            character must be 'r', to open the blob for reading, or 'w' to open
            it for writing. The second character, if present, must be 't' for
            (unicode) text mode, or 'b' for bytes mode. If the second character
            is omitted, text mode is the default.

        :type chunk_size: long
        :param chunk_size:
            (Optional) For reads, the minimum number of bytes to read at a time.
            If fewer bytes than the chunk_size are requested, the remainder is
            buffered. For writes, the maximum number of bytes to buffer before
            sending data to the server, and the size of each request when data
            is sent. Writes are implemented as a "resumable upload", so
            chunk_size for writes must be exactly a multiple of 256KiB as with
            other resumable uploads. The default is 40 MiB.

        :type ignore_flush: bool
        :param ignore_flush:
            (Optional) For non text-mode writes, makes flush() do nothing
            instead of raising an error. flush() without closing is not
            supported by the remote service and therefore calling it normally
            results in io.UnsupportedOperation. However, that behavior is
            incompatible with some consumers and wrappers of file objects in
            Python, such as zipfile.ZipFile or io.TextIOWrapper. Setting
            ignore_flush will cause flush() to successfully do nothing, for
            compatibility with those contexts. The correct way to actually flush
            data to the remote server is to close() (using a context manager,
            such as in the example, will cause this to happen automatically).

        :type encoding: str
        :param encoding:
            (Optional) For text mode only, the name of the encoding that the stream will
            be decoded or encoded with. If omitted, it defaults to
            locale.getpreferredencoding(False).

        :type errors: str
        :param errors:
            (Optional) For text mode only, an optional string that specifies how encoding
            and decoding errors are to be handled. Pass 'strict' to raise a
            ValueError exception if there is an encoding error (the default of
            None has the same effect), or pass 'ignore' to ignore errors. (Note
            that ignoring encoding errors can lead to data loss.) Other more
            rarely-used options are also available; see the Python 'io' module
            documentation for 'io.TextIOWrapper' for a complete list.

        :type newline: str
        :param newline:
            (Optional) For text mode only, controls how line endings are handled. It can
            be None, '', '\n', '\r', and '\r\n'. If None, reads use "universal
            newline mode" and writes use the system default. See the Python
            'io' module documentation for 'io.TextIOWrapper' for details.

        :returns: A 'BlobReader' or 'BlobWriter' from
            'google.cloud.storage.fileio', or an 'io.TextIOWrapper' around one
            of those classes, depending on the 'mode' argument.
        """
        if mode == "rb":
            if encoding or errors or newline:
                raise ValueError(
                    "encoding, errors and newline arguments are for text mode only"
                )
            if ignore_flush:
                raise ValueError(
                    "ignore_flush argument is for non-text write mode only"
                )
            return BlobReader(self, chunk_size=chunk_size, **kwargs)
        elif mode == "wb":
            if encoding or errors or newline:
                raise ValueError(
                    "encoding, errors and newline arguments are for text mode only"
                )
            return BlobWriter(
                self, chunk_size=chunk_size, ignore_flush=ignore_flush, **kwargs
            )
        elif mode in ("r", "rt"):
            if ignore_flush:
                raise ValueError(
                    "ignore_flush argument is for non-text write mode only"
                )
            return TextIOWrapper(
                BlobReader(self, chunk_size=chunk_size, **kwargs),
                encoding=encoding,
                errors=errors,
                newline=newline,
            )
        elif mode in ("w", "wt"):
            if ignore_flush is False:
                raise ValueError(
                    "ignore_flush is required for text mode writing and "
                    "cannot be set to False"
                )
            return TextIOWrapper(
                BlobWriter(self, chunk_size=chunk_size, ignore_flush=True, **kwargs),
                encoding=encoding,
                errors=errors,
                newline=newline,
            )
        else:
            raise NotImplementedError(
                "Supported modes strings are 'r', 'rb', 'rt', 'w', 'wb', and 'wt' only."
            )

    cache_control = _scalar_property("cacheControl")
    """HTTP 'Cache-Control' header for this object.

    See [`RFC 7234`](https://tools.ietf.org/html/rfc7234#section-5.2)
    and [`API reference docs`](https://cloud.google.com/storage/docs/json_api/v1/objects).

    :rtype: str or ``NoneType``

    """

    content_disposition = _scalar_property("contentDisposition")
    """HTTP 'Content-Disposition' header for this object.

    See [`RFC 6266`](https://tools.ietf.org/html/rfc7234#section-5.2) and
    [`API reference docs`](https://cloud.google.com/storage/docs/json_api/v1/objects).

    :rtype: str or ``NoneType``
    """

    content_encoding = _scalar_property("contentEncoding")
    """HTTP 'Content-Encoding' header for this object.

    See [`RFC 7231`](https://tools.ietf.org/html/rfc7231#section-3.1.2.2) and
    [`API reference docs`](https://cloud.google.com/storage/docs/json_api/v1/objects).

    :rtype: str or ``NoneType``
    """

    content_language = _scalar_property("contentLanguage")
    """HTTP 'Content-Language' header for this object.

    See [`BCP47`](https://tools.ietf.org/html/bcp47) and
    [`API reference docs`](https://cloud.google.com/storage/docs/json_api/v1/objects).

    :rtype: str or ``NoneType``
    """

    content_type = _scalar_property(_CONTENT_TYPE_FIELD)
    """HTTP 'Content-Type' header for this object.

    See [`RFC 2616`](https://tools.ietf.org/html/rfc2616#section-14.17) and
    [`API reference docs`](https://cloud.google.com/storage/docs/json_api/v1/objects).

    :rtype: str or ``NoneType``
    """

    crc32c = _scalar_property("crc32c")
    """CRC32C checksum for this object.

    This returns the blob's CRC32C checksum. To retrieve the value, first use a
    reload method of the Blob class which loads the blob's properties from the server.

    See [`RFC 4960`](https://tools.ietf.org/html/rfc4960#appendix-B) and
    [`API reference docs`](https://cloud.google.com/storage/docs/json_api/v1/objects).

    If not set before upload, the server will compute the hash.

    :rtype: str or ``NoneType``
    """

    def _prep_and_do_download(
        self,
        file_obj,
        client=None,
        start=None,
        end=None,
        raw_download=False,
        if_etag_match=None,
        if_etag_not_match=None,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        timeout=_DEFAULT_TIMEOUT,
        checksum="md5",
        retry=DEFAULT_RETRY,
        command=None,
    ):
        """Download the contents of a blob object into a file-like object.

        See https://cloud.google.com/storage/docs/downloading-objects

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        :type file_obj: file
        :param file_obj: A file handle to which to write the blob's data.

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client:
            (Optional) The client to use. If not passed, falls back to the
            ``client`` stored on the blob's bucket.

        :type start: int
        :param start: (Optional) The first byte in a range to be downloaded.

        :type end: int
        :param end: (Optional) The last byte in a range to be downloaded.

        :type raw_download: bool
        :param raw_download:
            (Optional) If true, download the object without any expansion.

        :type if_etag_match: Union[str, Set[str]]
        :param if_etag_match:
            (Optional) See :ref:`using-if-etag-match`

        :type if_etag_not_match: Union[str, Set[str]]
        :param if_etag_not_match:
            (Optional) See :ref:`using-if-etag-not-match`

        :type if_generation_match: long
        :param if_generation_match:
            (Optional) See :ref:`using-if-generation-match`

        :type if_generation_not_match: long
        :param if_generation_not_match:
            (Optional) See :ref:`using-if-generation-not-match`

        :type if_metageneration_match: long
        :param if_metageneration_match:
            (Optional) See :ref:`using-if-metageneration-match`

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match:
            (Optional) See :ref:`using-if-metageneration-not-match`

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type checksum: str
        :param checksum:
            (Optional) The type of checksum to compute to verify the integrity
            of the object. The response headers must contain a checksum of the
            requested type. If the headers lack an appropriate checksum (for
            instance in the case of transcoded or ranged downloads where the
            remote service does not know the correct checksum, including
            downloads where chunk_size is set) an INFO-level log will be
            emitted. Supported values are "md5", "crc32c" and None. The default
            is "md5".

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry: (Optional) How to retry the RPC. A None value will disable
            retries. A google.api_core.retry.Retry value will enable retries,
            and the object will define retriable response codes and errors and
            configure backoff and timeout options.

            A google.cloud.storage.retry.ConditionalRetryPolicy value wraps a
            Retry object and activates it only if certain conditions are met.
            This class exists to provide safe defaults for RPC calls that are
            not technically safe to retry normally (due to potential data
            duplication or other side-effects) but become safe to retry if a
            condition such as if_metageneration_match is set.

            See the retry.py source code and docstrings in this package
            (google.cloud.storage.retry) for information on retry types and how
            to configure them.

            Media operations (downloads and uploads) do not support non-default
            predicates in a Retry object. The default will always be used. Other
            configuration changes for Retry objects such as delays and deadlines
            are respected.

        :type command: str
        :param command:
            (Optional) Information about which interface for download was used,
            to be included in the X-Goog-API-Client header. Please leave as None
            unless otherwise directed.
        """
        # Handle ConditionalRetryPolicy.
        if isinstance(retry, ConditionalRetryPolicy):
            # Conditional retries are designed for non-media calls, which change
            # arguments into query_params dictionaries. Media operations work
            # differently, so here we make a "fake" query_params to feed to the
            # ConditionalRetryPolicy.
            query_params = {
                "ifGenerationMatch": if_generation_match,
                "ifMetagenerationMatch": if_metageneration_match,
            }
            retry = retry.get_retry_policy_if_conditions_met(query_params=query_params)

        client = self._require_client(client)

        download_url = self._get_download_url(
            client,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
        )
        headers = _get_encryption_headers(self._encryption_key)
        headers["accept-encoding"] = "gzip"
        _add_etag_match_headers(
            headers,
            if_etag_match=if_etag_match,
            if_etag_not_match=if_etag_not_match,
        )
        # Add any client attached custom headers to be sent with the request.
        headers = {
            **_get_default_headers(client._connection.user_agent, command=command),
            **headers,
            **client._extra_headers,
        }

        transport = client._http

        try:
            self._do_download(
                transport,
                file_obj,
                download_url,
                headers,
                start,
                end,
                raw_download,
                timeout=timeout,
                checksum=checksum,
                retry=retry,
            )
        except resumable_media.InvalidResponse as exc:
            _raise_from_invalid_response(exc)

    @property
    def component_count(self):
        """Number of underlying components that make up this object.

        See https://cloud.google.com/storage/docs/json_api/v1/objects

        :rtype: int or ``NoneType``
        :returns: The component count (in case of a composed object) or
                  ``None`` if the blob's resource has not been loaded from
                  the server.  This property will not be set on objects
                  not created via ``compose``.
        """
        component_count = self._properties.get("componentCount")
        if component_count is not None:
            return int(component_count)

    @property
    def etag(self):
        """Retrieve the ETag for the object.

        See [`RFC 2616 (etags)`](https://tools.ietf.org/html/rfc2616#section-3.11) and
        [`API reference docs`](https://cloud.google.com/storage/docs/json_api/v1/objects).

        :rtype: str or ``NoneType``
        :returns: The blob etag or ``None`` if the blob's resource has not
                  been loaded from the server.
        """
        return self._properties.get("etag")

    event_based_hold = _scalar_property("eventBasedHold")
    """Is an event-based hold active on the object?

    See [`API reference docs`](https://cloud.google.com/storage/docs/json_api/v1/objects).

    If the property is not set locally, returns :data:`None`.

    :rtype: bool or ``NoneType``
    """

    @property
    def generation(self):
        """Retrieve the generation for the object.

        See https://cloud.google.com/storage/docs/json_api/v1/objects

        :rtype: int or ``NoneType``
        :returns: The generation of the blob or ``None`` if the blob's
                  resource has not been loaded from the server.
        """
        generation = self._properties.get("generation")
        if generation is not None:
            return int(generation)

    @property
    def id(self):
        """Retrieve the ID for the object.

        See https://cloud.google.com/storage/docs/json_api/v1/objects

        The ID consists of the bucket name, object name, and generation number.

        :rtype: str or ``NoneType``
        :returns: The ID of the blob or ``None`` if the blob's
                  resource has not been loaded from the server.
        """
        return self._properties.get("id")

    md5_hash = _scalar_property("md5Hash")
    """MD5 hash for this object.

    This returns the blob's MD5 hash. To retrieve the value, first use a
    reload method of the Blob class which loads the blob's properties from the server.

    See [`RFC 1321`](https://tools.ietf.org/html/rfc1321) and
    [`API reference docs`](https://cloud.google.com/storage/docs/json_api/v1/objects).

    If not set before upload, the server will compute the hash.

    :rtype: str or ``NoneType``
    """

    @property
    def media_link(self):
        """Retrieve the media download URI for the object.

        See https://cloud.google.com/storage/docs/json_api/v1/objects

        :rtype: str or ``NoneType``
        :returns: The media link for the blob or ``None`` if the blob's
                  resource has not been loaded from the server.
        """
        return self._properties.get("mediaLink")

    @property
    def metadata(self):
        """Retrieve arbitrary/application specific metadata for the object.

        See https://cloud.google.com/storage/docs/json_api/v1/objects

        :setter: Update arbitrary/application specific metadata for the
                 object.
        :getter: Retrieve arbitrary/application specific metadata for
                 the object.

        :rtype: dict or ``NoneType``
        :returns: The metadata associated with the blob or ``None`` if the
                  property is not set.
        """
        return copy.deepcopy(self._properties.get("metadata"))

    @metadata.setter
    def metadata(self, value):
        """Update arbitrary/application specific metadata for the object.

        Values are stored to GCS as strings. To delete a key, set its value to
        None and call blob.patch().

        See https://cloud.google.com/storage/docs/json_api/v1/objects

        :type value: dict
        :param value: The blob metadata to set.
        """
        if value is not None:
            value = {k: str(v) if v is not None else None for k, v in value.items()}
        self._patch_property("metadata", value)

    @property
    def metageneration(self):
        """Retrieve the metageneration for the object.

        See https://cloud.google.com/storage/docs/json_api/v1/objects

        :rtype: int or ``NoneType``
        :returns: The metageneration of the blob or ``None`` if the blob's
                  resource has not been loaded from the server.
        """
        metageneration = self._properties.get("metageneration")
        if metageneration is not None:
            return int(metageneration)

    @property
    def owner(self):
        """Retrieve info about the owner of the object.

        See https://cloud.google.com/storage/docs/json_api/v1/objects

        :rtype: dict or ``NoneType``
        :returns: Mapping of owner's role/ID, or ``None`` if the blob's
                  resource has not been loaded from the server.
        """
        return copy.deepcopy(self._properties.get("owner"))

    @property
    def retention_expiration_time(self):
        """Retrieve timestamp at which the object's retention period expires.

        See https://cloud.google.com/storage/docs/json_api/v1/objects

        :rtype: :class:`datetime.datetime` or ``NoneType``
        :returns: Datetime object parsed from RFC3339 valid timestamp, or
                  ``None`` if the property is not set locally.
        """
        value = self._properties.get("retentionExpirationTime")
        if value is not None:
            return _rfc3339_nanos_to_datetime(value)

    @property
    def self_link(self):
        """Retrieve the URI for the object.

        See https://cloud.google.com/storage/docs/json_api/v1/objects

        :rtype: str or ``NoneType``
        :returns: The self link for the blob or ``None`` if the blob's
                  resource has not been loaded from the server.
        """
        return self._properties.get("selfLink")

    @property
    def size(self):
        """Size of the object, in bytes.

        See https://cloud.google.com/storage/docs/json_api/v1/objects

        :rtype: int or ``NoneType``
        :returns: The size of the blob or ``None`` if the blob's
                  resource has not been loaded from the server.
        """
        size = self._properties.get("size")
        if size is not None:
            return int(size)

    @property
    def kms_key_name(self):
        """Resource name of Cloud KMS key used to encrypt the blob's contents.

        :rtype: str or ``NoneType``
        :returns:
            The resource name or ``None`` if no Cloud KMS key was used,
            or the blob's resource has not been loaded from the server.
        """
        return self._properties.get("kmsKeyName")

    @kms_key_name.setter
    def kms_key_name(self, value):
        """Set KMS encryption key for object.

        :type value: str or ``NoneType``
        :param value: new KMS key name (None to clear any existing key).
        """
        self._patch_property("kmsKeyName", value)

    storage_class = _scalar_property("storageClass")
    """Retrieve the storage class for the object.

    This can only be set at blob / object **creation** time. If you'd
    like to change the storage class **after** the blob / object already
    exists in a bucket, call :meth:`update_storage_class` (which uses
    :meth:`rewrite`).

    See https://cloud.google.com/storage/docs/storage-classes

    :rtype: str or ``NoneType``
    :returns:
        If set, one of
        :attr:`~google.cloud.storage.constants.STANDARD_STORAGE_CLASS`,
        :attr:`~google.cloud.storage.constants.NEARLINE_STORAGE_CLASS`,
        :attr:`~google.cloud.storage.constants.COLDLINE_STORAGE_CLASS`,
        :attr:`~google.cloud.storage.constants.ARCHIVE_STORAGE_CLASS`,
        :attr:`~google.cloud.storage.constants.MULTI_REGIONAL_LEGACY_STORAGE_CLASS`,
        :attr:`~google.cloud.storage.constants.REGIONAL_LEGACY_STORAGE_CLASS`,
        :attr:`~google.cloud.storage.constants.DURABLE_REDUCED_AVAILABILITY_STORAGE_CLASS`,
        else ``None``.
    """

    temporary_hold = _scalar_property("temporaryHold")
    """Is a temporary hold active on the object?

    See [`API reference docs`](https://cloud.google.com/storage/docs/json_api/v1/objects).

    If the property is not set locally, returns :data:`None`.

    :rtype: bool or ``NoneType``
    """

    @property
    def time_deleted(self):
        """Retrieve the timestamp at which the object was deleted.

        See https://cloud.google.com/storage/docs/json_api/v1/objects

        :rtype: :class:`datetime.datetime` or ``NoneType``
        :returns: Datetime object parsed from RFC3339 valid timestamp, or
                  ``None`` if the blob's resource has not been loaded from
                  the server (see :meth:`reload`). If the blob has
                  not been deleted, this will never be set.
        """
        value = self._properties.get("timeDeleted")
        if value is not None:
            return _rfc3339_nanos_to_datetime(value)

    @property
    def time_created(self):
        """Retrieve the timestamp at which the object was created.

        See https://cloud.google.com/storage/docs/json_api/v1/objects

        :rtype: :class:`datetime.datetime` or ``NoneType``
        :returns: Datetime object parsed from RFC3339 valid timestamp, or
                  ``None`` if the blob's resource has not been loaded from
                  the server (see :meth:`reload`).
        """
        value = self._properties.get("timeCreated")
        if value is not None:
            return _rfc3339_nanos_to_datetime(value)

    @property
    def updated(self):
        """Retrieve the timestamp at which the object was updated.

        See https://cloud.google.com/storage/docs/json_api/v1/objects

        :rtype: :class:`datetime.datetime` or ``NoneType``
        :returns: Datetime object parsed from RFC3339 valid timestamp, or
                  ``None`` if the blob's resource has not been loaded from
                  the server (see :meth:`reload`).
        """
        value = self._properties.get("updated")
        if value is not None:
            return _rfc3339_nanos_to_datetime(value)

    @property
    def custom_time(self):
        """Retrieve the custom time for the object.

        See https://cloud.google.com/storage/docs/json_api/v1/objects

        :rtype: :class:`datetime.datetime` or ``NoneType``
        :returns: Datetime object parsed from RFC3339 valid timestamp, or
                  ``None`` if the blob's resource has not been loaded from
                  the server (see :meth:`reload`).
        """
        value = self._properties.get("customTime")
        if value is not None:
            return _rfc3339_nanos_to_datetime(value)

    @custom_time.setter
    def custom_time(self, value):
        """Set the custom time for the object.

        Once set on the server side object, this value can't be unset, but may
        only changed to a custom datetime in the future.

        If :attr:`custom_time` must be unset, either perform a rewrite
        operation or upload the data again.

        See https://cloud.google.com/storage/docs/json_api/v1/objects

        :type value: :class:`datetime.datetime`
        :param value: new value
        """
        if value is not None:
            value = _datetime_to_rfc3339(value)

        self._patch_property("customTime", value)

    @property
    def retention(self):
        """Retrieve the retention configuration for this object.

        :rtype: :class:`Retention`
        :returns: an instance for managing the object's retention configuration.
        """
        info = self._properties.get("retention", {})
        return Retention.from_api_repr(info, self)

    @property
    def soft_delete_time(self):
        """If this object has been soft-deleted, returns the time at which it became soft-deleted.

        :rtype: :class:`datetime.datetime` or ``NoneType``
        :returns:
            (readonly) The time that the object became soft-deleted.
             Note this property is only set for soft-deleted objects.
        """
        soft_delete_time = self._properties.get("softDeleteTime")
        if soft_delete_time is not None:
            return _rfc3339_nanos_to_datetime(soft_delete_time)

    @property
    def hard_delete_time(self):
        """If this object has been soft-deleted, returns the time at which it will be permanently deleted.

        :rtype: :class:`datetime.datetime` or ``NoneType``
        :returns:
            (readonly) The time that the object will be permanently deleted.
            Note this property is only set for soft-deleted objects.
        """
        hard_delete_time = self._properties.get("hardDeleteTime")
        if hard_delete_time is not None:
            return _rfc3339_nanos_to_datetime(hard_delete_time)


def _get_host_name(connection):
    """Returns the host name from the given connection.

    :type connection: :class:`~google.cloud.storage._http.Connection`
    :param connection: The connection object.

    :rtype: str
    :returns: The host name.
    """
    # TODO: After google-cloud-core 1.6.0 is stable and we upgrade it
    # to 1.6.0 in setup.py, we no longer need to check the attribute
    # existence. We can simply return connection.get_api_base_url_for_mtls().
    return (
        connection.API_BASE_URL
        if not hasattr(connection, "get_api_base_url_for_mtls")
        else connection.get_api_base_url_for_mtls()
    )


def _get_encryption_headers(key, source=False):
    """Builds customer encryption key headers

    :type key: bytes
    :param key: 32 byte key to build request key and hash.

    :type source: bool
    :param source: If true, return headers for the "source" blob; otherwise,
                   return headers for the "destination" blob.

    :rtype: dict
    :returns: dict of HTTP headers being sent in request.
    """
    if key is None:
        return {}

    key = _to_bytes(key)
    key_hash = hashlib.sha256(key).digest()
    key_hash = base64.b64encode(key_hash)
    key = base64.b64encode(key)

    if source:
        prefix = "X-Goog-Copy-Source-Encryption-"
    else:
        prefix = "X-Goog-Encryption-"

    return {
        prefix + "Algorithm": "AES256",
        prefix + "Key": _bytes_to_unicode(key),
        prefix + "Key-Sha256": _bytes_to_unicode(key_hash),
    }


def _quote(value, safe=b"~"):
    """URL-quote a string.

    If the value is unicode, this method first UTF-8 encodes it as bytes and
    then quotes the bytes. (In Python 3, ``urllib.parse.quote`` does this
    encoding automatically, but in Python 2, non-ASCII characters cannot be
    quoted.)

    :type value: str or bytes
    :param value: The value to be URL-quoted.

    :type safe: bytes
    :param safe: Bytes *not* to be quoted.  By default, includes only ``b'~'``.

    :rtype: str
    :returns: The encoded value (bytes in Python 2, unicode in Python 3).
    """
    value = _to_bytes(value, encoding="utf-8")
    return quote(value, safe=safe)


def _maybe_rewind(stream, rewind=False):
    """Rewind the stream if desired.

    :type stream: IO[bytes]
    :param stream: A bytes IO object open for reading.

    :type rewind: bool
    :param rewind: Indicates if we should seek to the beginning of the stream.
    """
    if rewind:
        stream.seek(0, os.SEEK_SET)


def _raise_from_invalid_response(error):
    """Re-wrap and raise an ``InvalidResponse`` exception.

    :type error: :exc:`google.resumable_media.InvalidResponse`
    :param error: A caught exception from the ``google-resumable-media``
                  library.

    :raises: :class:`~google.cloud.exceptions.GoogleCloudError` corresponding
             to the failed status code
    """
    response = error.response

    # The 'response.text' gives the actual reason of error, where 'error' gives
    # the message of expected status code.
    if response.text:
        error_message = response.text + ": " + str(error)
    else:
        error_message = str(error)

    message = f"{response.request.method} {response.request.url}: {error_message}"

    raise exceptions.from_http_status(response.status_code, message, response=response)


def _add_query_parameters(base_url, name_value_pairs):
    """Add one query parameter to a base URL.

    :type base_url: string
    :param base_url: Base URL (may already contain query parameters)

    :type name_value_pairs: list of (string, string) tuples.
    :param name_value_pairs: Names and values of the query parameters to add

    :rtype: string
    :returns: URL with additional query strings appended.
    """
    if len(name_value_pairs) == 0:
        return base_url

    scheme, netloc, path, query, frag = urlsplit(base_url)
    query = parse_qsl(query)
    query.extend(name_value_pairs)
    return urlunsplit((scheme, netloc, path, urlencode(query), frag))


class Retention(dict):
    """Map an object's retention configuration.

    :type blob: :class:`Blob`
    :params blob: blob for which this retention configuration applies to.

    :type mode: str or ``NoneType``
    :params mode:
        (Optional) The mode of the retention configuration, which can be either Unlocked or Locked.
        See: https://cloud.google.com/storage/docs/object-lock

    :type retain_until_time: :class:`datetime.datetime` or ``NoneType``
    :params retain_until_time:
        (Optional) The earliest time that the object can be deleted or replaced, which is the
        retention configuration set for this object.

    :type retention_expiration_time: :class:`datetime.datetime` or ``NoneType``
    :params retention_expiration_time:
        (Optional) The earliest time that the object can be deleted, which depends on any
        retention configuration set for the object and any retention policy set for the bucket
        that contains the object. This value should normally only be set by the back-end API.
    """

    def __init__(
        self,
        blob,
        mode=None,
        retain_until_time=None,
        retention_expiration_time=None,
    ):
        data = {"mode": mode}
        if retain_until_time is not None:
            retain_until_time = _datetime_to_rfc3339(retain_until_time)
        data["retainUntilTime"] = retain_until_time

        if retention_expiration_time is not None:
            retention_expiration_time = _datetime_to_rfc3339(retention_expiration_time)
        data["retentionExpirationTime"] = retention_expiration_time

        super(Retention, self).__init__(data)
        self._blob = blob

    @classmethod
    def from_api_repr(cls, resource, blob):
        """Factory:  construct instance from resource.

        :type blob: :class:`Blob`
        :params blob: Blob for which this retention configuration applies to.

        :type resource: dict
        :param resource: mapping as returned from API call.

        :rtype: :class:`Retention`
        :returns: Retention configuration created from resource.
        """
        instance = cls(blob)
        instance.update(resource)
        return instance

    @property
    def blob(self):
        """Blob for which this retention configuration applies to.

        :rtype: :class:`Blob`
        :returns: the instance's blob.
        """
        return self._blob

    @property
    def mode(self):
        """The mode of the retention configuration. Options are 'Unlocked' or 'Locked'.

        :rtype: string
        :returns: The mode of the retention configuration, which can be either set to 'Unlocked' or 'Locked'.
        """
        return self.get("mode")

    @mode.setter
    def mode(self, value):
        self["mode"] = value
        self.blob._patch_property("retention", self)

    @property
    def retain_until_time(self):
        """The earliest time that the object can be deleted or replaced, which is the
        retention configuration set for this object.

        :rtype: :class:`datetime.datetime` or ``NoneType``
        :returns: Datetime object parsed from RFC3339 valid timestamp, or
                  ``None`` if the blob's resource has not been loaded from
                  the server (see :meth:`reload`).
        """
        value = self.get("retainUntilTime")
        if value is not None:
            return _rfc3339_nanos_to_datetime(value)

    @retain_until_time.setter
    def retain_until_time(self, value):
        """Set the retain_until_time for the object retention configuration.

        :type value: :class:`datetime.datetime`
        :param value: The earliest time that the object can be deleted or replaced.
        """
        if value is not None:
            value = _datetime_to_rfc3339(value)
        self["retainUntilTime"] = value
        self.blob._patch_property("retention", self)

    @property
    def retention_expiration_time(self):
        """The earliest time that the object can be deleted, which depends on any
        retention configuration set for the object and any retention policy set for
        the bucket that contains the object.

        :rtype: :class:`datetime.datetime` or ``NoneType``
        :returns:
            (readonly) The earliest time that the object can be deleted.
        """
        retention_expiration_time = self.get("retentionExpirationTime")
        if retention_expiration_time is not None:
            return _rfc3339_nanos_to_datetime(retention_expiration_time)
