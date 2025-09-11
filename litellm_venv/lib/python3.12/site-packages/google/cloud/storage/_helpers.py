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

"""Helper functions for Cloud Storage utility classes.

These are *not* part of the API.
"""

import base64
import datetime
from hashlib import md5
import os
from urllib.parse import urlsplit
from urllib.parse import urlunsplit
from uuid import uuid4

from google import resumable_media
from google.auth import environment_vars
from google.cloud.storage.constants import _DEFAULT_TIMEOUT
from google.cloud.storage.retry import DEFAULT_RETRY
from google.cloud.storage.retry import DEFAULT_RETRY_IF_METAGENERATION_SPECIFIED


STORAGE_EMULATOR_ENV_VAR = "STORAGE_EMULATOR_HOST"  # Despite name, includes scheme.
"""Environment variable defining host for Storage emulator."""

_API_ENDPOINT_OVERRIDE_ENV_VAR = "API_ENDPOINT_OVERRIDE"  # Includes scheme.
"""This is an experimental configuration variable. Use api_endpoint instead."""

_API_VERSION_OVERRIDE_ENV_VAR = "API_VERSION_OVERRIDE"
"""This is an experimental configuration variable used for internal testing."""

_DEFAULT_UNIVERSE_DOMAIN = "googleapis.com"

_STORAGE_HOST_TEMPLATE = "storage.{universe_domain}"

_TRUE_DEFAULT_STORAGE_HOST = _STORAGE_HOST_TEMPLATE.format(
    universe_domain=_DEFAULT_UNIVERSE_DOMAIN
)

_DEFAULT_SCHEME = "https://"

_API_VERSION = os.getenv(_API_VERSION_OVERRIDE_ENV_VAR, "v1")
"""API version of the default storage host"""

# etag match parameters in snake case and equivalent header
_ETAG_MATCH_PARAMETERS = (
    ("if_etag_match", "If-Match"),
    ("if_etag_not_match", "If-None-Match"),
)

# generation match parameters in camel and snake cases
_GENERATION_MATCH_PARAMETERS = (
    ("if_generation_match", "ifGenerationMatch"),
    ("if_generation_not_match", "ifGenerationNotMatch"),
    ("if_metageneration_match", "ifMetagenerationMatch"),
    ("if_metageneration_not_match", "ifMetagenerationNotMatch"),
    ("if_source_generation_match", "ifSourceGenerationMatch"),
    ("if_source_generation_not_match", "ifSourceGenerationNotMatch"),
    ("if_source_metageneration_match", "ifSourceMetagenerationMatch"),
    ("if_source_metageneration_not_match", "ifSourceMetagenerationNotMatch"),
)

_NUM_RETRIES_MESSAGE = (
    "`num_retries` has been deprecated and will be removed in a future "
    "release. Use the `retry` argument with a Retry or ConditionalRetryPolicy "
    "object, or None, instead."
)

# _NOW() returns the current local date and time.
# It is preferred to use timezone-aware datetimes _NOW(_UTC),
# which returns the current UTC date and time.
_NOW = datetime.datetime.now
_UTC = datetime.timezone.utc


def _get_storage_emulator_override():
    return os.environ.get(STORAGE_EMULATOR_ENV_VAR, None)


def _get_default_storage_base_url():
    return os.getenv(
        _API_ENDPOINT_OVERRIDE_ENV_VAR, _DEFAULT_SCHEME + _TRUE_DEFAULT_STORAGE_HOST
    )


def _get_api_endpoint_override():
    """This is an experimental configuration variable. Use api_endpoint instead."""
    if _get_default_storage_base_url() != _DEFAULT_SCHEME + _TRUE_DEFAULT_STORAGE_HOST:
        return _get_default_storage_base_url()
    return None


def _virtual_hosted_style_base_url(url, bucket, trailing_slash=False):
    """Returns the scheme and netloc sections of the url, with the bucket
    prepended to the netloc.

    Not intended for use with netlocs which include a username and password.
    """
    parsed_url = urlsplit(url)
    new_netloc = f"{bucket}.{parsed_url.netloc}"
    base_url = urlunsplit(
        (parsed_url.scheme, new_netloc, "/" if trailing_slash else "", "", "")
    )
    return base_url


def _use_client_cert():
    return os.getenv("GOOGLE_API_USE_CLIENT_CERTIFICATE") == "true"


def _get_environ_project():
    return os.getenv(
        environment_vars.PROJECT,
        os.getenv(environment_vars.LEGACY_PROJECT),
    )


def _validate_name(name):
    """Pre-flight ``Bucket`` name validation.

    :type name: str or :data:`NoneType`
    :param name: Proposed bucket name.

    :rtype: str or :data:`NoneType`
    :returns: ``name`` if valid.
    """
    if name is None:
        return

    # The first and last characters must be alphanumeric.
    if not all([name[0].isalnum(), name[-1].isalnum()]):
        raise ValueError("Bucket names must start and end with a number or letter.")
    return name


class _PropertyMixin(object):
    """Abstract mixin for cloud storage classes with associated properties.

    Non-abstract subclasses should implement:
      - path
      - client
      - user_project

    :type name: str
    :param name: The name of the object. Bucket names must start and end with a
                 number or letter.
    """

    def __init__(self, name=None):
        self.name = name
        self._properties = {}
        self._changes = set()

    @property
    def path(self):
        """Abstract getter for the object path."""
        raise NotImplementedError

    @property
    def client(self):
        """Abstract getter for the object client."""
        raise NotImplementedError

    @property
    def user_project(self):
        """Abstract getter for the object user_project."""
        raise NotImplementedError

    def _require_client(self, client):
        """Check client or verify over-ride.

        :type client: :class:`~google.cloud.storage.client.Client` or
                      ``NoneType``
        :param client: the client to use.  If not passed, falls back to the
                       ``client`` stored on the current object.

        :rtype: :class:`google.cloud.storage.client.Client`
        :returns: The client passed in or the currently bound client.
        """
        if client is None:
            client = self.client
        return client

    def _encryption_headers(self):
        """Return any encryption headers needed to fetch the object.

        .. note::
           Defined here because :meth:`reload` calls it, but this method is
           really only relevant for :class:`~google.cloud.storage.blob.Blob`.

        :rtype: dict
        :returns: a mapping of encryption-related headers.
        """
        return {}

    @property
    def _query_params(self):
        """Default query parameters."""
        params = {}
        if self.user_project is not None:
            params["userProject"] = self.user_project
        return params

    def reload(
        self,
        client=None,
        projection="noAcl",
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
        """Reload properties from Cloud Storage.

        If :attr:`user_project` is set, bills the API request to that project.

        :type client: :class:`~google.cloud.storage.client.Client` or
                      ``NoneType``
        :param client: the client to use. If not passed, falls back to the
                       ``client`` stored on the current object.

        :type projection: str
        :param projection: (Optional) If used, must be 'full' or 'noAcl'.
                           Defaults to ``'noAcl'``. Specifies the set of
                           properties to return.

        :type if_etag_match: Union[str, Set[str]]
        :param if_etag_match: (Optional) See :ref:`using-if-etag-match`

        :type if_etag_not_match: Union[str, Set[str]])
        :param if_etag_not_match: (Optional) See :ref:`using-if-etag-not-match`

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
            (Optional) If True, looks for a soft-deleted object. Will only return
            the object metadata if the object exists and is in a soft-deleted state.
            :attr:`generation` is required to be set on the blob if ``soft_deleted`` is set to True.
            See: https://cloud.google.com/storage/docs/soft-delete
        """
        client = self._require_client(client)
        query_params = self._query_params
        # Pass only '?projection=noAcl' here because 'acl' and related
        # are handled via custom endpoints.
        query_params["projection"] = projection
        _add_generation_match_parameters(
            query_params,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
        )
        if soft_deleted is not None:
            query_params["softDeleted"] = soft_deleted
            # Soft delete reload requires a generation, even for targets
            # that don't include them in default query params (buckets).
            query_params["generation"] = self.generation
        headers = self._encryption_headers()
        _add_etag_match_headers(
            headers, if_etag_match=if_etag_match, if_etag_not_match=if_etag_not_match
        )
        api_response = client._get_resource(
            self.path,
            query_params=query_params,
            headers=headers,
            timeout=timeout,
            retry=retry,
            _target_object=self,
        )
        self._set_properties(api_response)

    def _patch_property(self, name, value):
        """Update field of this object's properties.

        This method will only update the field provided and will not
        touch the other fields.

        It **will not** reload the properties from the server. The behavior is
        local only and syncing occurs via :meth:`patch`.

        :type name: str
        :param name: The field name to update.

        :type value: object
        :param value: The value being updated.
        """
        self._changes.add(name)
        self._properties[name] = value

    def _set_properties(self, value):
        """Set the properties for the current object.

        :type value: dict or :class:`google.cloud.storage.batch._FutureDict`
        :param value: The properties to be set.
        """
        self._properties = value
        # If the values are reset, the changes must as well.
        self._changes = set()

    def patch(
        self,
        client=None,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=DEFAULT_RETRY_IF_METAGENERATION_SPECIFIED,
        override_unlocked_retention=False,
    ):
        """Sends all changed properties in a PATCH request.

        Updates the ``_properties`` with the response from the backend.

        If :attr:`user_project` is set, bills the API request to that project.

        :type client: :class:`~google.cloud.storage.client.Client` or
                      ``NoneType``
        :param client: the client to use. If not passed, falls back to the
                       ``client`` stored on the current object.

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

        :type override_unlocked_retention: bool
        :param override_unlocked_retention:
            (Optional) override_unlocked_retention must be set to True if the operation includes
            a retention property that changes the mode from Unlocked to Locked, reduces the
            retainUntilTime, or removes the retention configuration from the object. See:
            https://cloud.google.com/storage/docs/json_api/v1/objects/patch
        """
        client = self._require_client(client)
        query_params = self._query_params
        # Pass '?projection=full' here because 'PATCH' documented not
        # to work properly w/ 'noAcl'.
        query_params["projection"] = "full"
        if override_unlocked_retention:
            query_params["overrideUnlockedRetention"] = override_unlocked_retention
        _add_generation_match_parameters(
            query_params,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
        )
        update_properties = {key: self._properties[key] for key in self._changes}

        # Make the API call.
        api_response = client._patch_resource(
            self.path,
            update_properties,
            query_params=query_params,
            _target_object=self,
            timeout=timeout,
            retry=retry,
        )
        self._set_properties(api_response)

    def update(
        self,
        client=None,
        if_generation_match=None,
        if_generation_not_match=None,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=DEFAULT_RETRY_IF_METAGENERATION_SPECIFIED,
        override_unlocked_retention=False,
    ):
        """Sends all properties in a PUT request.

        Updates the ``_properties`` with the response from the backend.

        If :attr:`user_project` is set, bills the API request to that project.

        :type client: :class:`~google.cloud.storage.client.Client` or
                      ``NoneType``
        :param client: the client to use. If not passed, falls back to the
                       ``client`` stored on the current object.

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

        :type override_unlocked_retention: bool
        :param override_unlocked_retention:
            (Optional) override_unlocked_retention must be set to True if the operation includes
            a retention property that changes the mode from Unlocked to Locked, reduces the
            retainUntilTime, or removes the retention configuration from the object. See:
            https://cloud.google.com/storage/docs/json_api/v1/objects/patch
        """
        client = self._require_client(client)

        query_params = self._query_params
        query_params["projection"] = "full"
        if override_unlocked_retention:
            query_params["overrideUnlockedRetention"] = override_unlocked_retention
        _add_generation_match_parameters(
            query_params,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
        )

        api_response = client._put_resource(
            self.path,
            self._properties,
            query_params=query_params,
            timeout=timeout,
            retry=retry,
            _target_object=self,
        )
        self._set_properties(api_response)


def _scalar_property(fieldname):
    """Create a property descriptor around the :class:`_PropertyMixin` helpers."""

    def _getter(self):
        """Scalar property getter."""
        return self._properties.get(fieldname)

    def _setter(self, value):
        """Scalar property setter."""
        self._patch_property(fieldname, value)

    return property(_getter, _setter)


def _write_buffer_to_hash(buffer_object, hash_obj, digest_block_size=8192):
    """Read blocks from a buffer and update a hash with them.

    :type buffer_object: bytes buffer
    :param buffer_object: Buffer containing bytes used to update a hash object.

    :type hash_obj: object that implements update
    :param hash_obj: A hash object (MD5 or CRC32-C).

    :type digest_block_size: int
    :param digest_block_size: The block size to write to the hash.
                              Defaults to 8192.
    """
    block = buffer_object.read(digest_block_size)

    while len(block) > 0:
        hash_obj.update(block)
        # Update the block for the next iteration.
        block = buffer_object.read(digest_block_size)


def _base64_md5hash(buffer_object):
    """Get MD5 hash of bytes (as base64).

    :type buffer_object: bytes buffer
    :param buffer_object: Buffer containing bytes used to compute an MD5
                          hash (as base64).

    :rtype: str
    :returns: A base64 encoded digest of the MD5 hash.
    """
    hash_obj = md5()
    _write_buffer_to_hash(buffer_object, hash_obj)
    digest_bytes = hash_obj.digest()
    return base64.b64encode(digest_bytes)


def _add_etag_match_headers(headers, **match_parameters):
    """Add generation match parameters into the given parameters list.

    :type headers: dict
    :param headers: Headers dict.

    :type match_parameters: dict
    :param match_parameters: if*etag*match parameters to add.
    """
    for snakecase_name, header_name in _ETAG_MATCH_PARAMETERS:
        value = match_parameters.get(snakecase_name)

        if value is not None:
            if isinstance(value, str):
                value = [value]
            headers[header_name] = ", ".join(value)


def _add_generation_match_parameters(parameters, **match_parameters):
    """Add generation match parameters into the given parameters list.

    :type parameters: list or dict
    :param parameters: Parameters list or dict.

    :type match_parameters: dict
    :param match_parameters: if*generation*match parameters to add.

    :raises: :exc:`ValueError` if ``parameters`` is not a ``list()``
             or a ``dict()``.
    """
    for snakecase_name, camelcase_name in _GENERATION_MATCH_PARAMETERS:
        value = match_parameters.get(snakecase_name)

        if value is not None:
            if isinstance(parameters, list):
                parameters.append((camelcase_name, value))

            elif isinstance(parameters, dict):
                parameters[camelcase_name] = value

            else:
                raise ValueError(
                    "`parameters` argument should be a dict() or a list()."
                )


def _raise_if_more_than_one_set(**kwargs):
    """Raise ``ValueError`` exception if more than one parameter was set.

    :type error: :exc:`ValueError`
    :param error: Description of which fields were set

    :raises: :class:`~ValueError` containing the fields that were set
    """
    if sum(arg is not None for arg in kwargs.values()) > 1:
        escaped_keys = [f"'{name}'" for name in kwargs.keys()]

        keys_but_last = ", ".join(escaped_keys[:-1])
        last_key = escaped_keys[-1]

        msg = f"Pass at most one of {keys_but_last} and {last_key}"

        raise ValueError(msg)


def _bucket_bound_hostname_url(host, scheme=None):
    """Helper to build bucket bound hostname URL.

    :type host: str
    :param host: Host name.

    :type scheme: str
    :param scheme: (Optional) Web scheme. If passed, use it
                   as a scheme in the result URL.

    :rtype: str
    :returns: A bucket bound hostname URL.
    """
    url_parts = urlsplit(host)
    if url_parts.scheme and url_parts.netloc:
        return host

    return f"{scheme}://{host}"


def _api_core_retry_to_resumable_media_retry(retry, num_retries=None):
    """Convert google.api.core.Retry to google.resumable_media.RetryStrategy.

    Custom predicates are not translated.

    :type retry: google.api_core.Retry
    :param retry: (Optional) The google.api_core.Retry object to translate.

    :type num_retries: int
    :param num_retries: (Optional) The number of retries desired. This is
        supported for backwards compatibility and is mutually exclusive with
        `retry`.

    :rtype: google.resumable_media.RetryStrategy
    :returns: A RetryStrategy with all applicable attributes copied from input,
              or a RetryStrategy with max_retries set to 0 if None was input.
    """

    if retry is not None and num_retries is not None:
        raise ValueError("num_retries and retry arguments are mutually exclusive")

    elif retry is not None:
        return resumable_media.RetryStrategy(
            max_sleep=retry._maximum,
            max_cumulative_retry=retry._deadline,
            initial_delay=retry._initial,
            multiplier=retry._multiplier,
        )
    elif num_retries is not None:
        return resumable_media.RetryStrategy(max_retries=num_retries)
    else:
        return resumable_media.RetryStrategy(max_retries=0)


def _get_invocation_id():
    return "gccl-invocation-id/" + str(uuid4())


def _get_default_headers(
    user_agent,
    content_type="application/json; charset=UTF-8",
    x_upload_content_type=None,
    command=None,
):
    """Get the headers for a request.

    :type user_agent: str
    :param user_agent: The user-agent for requests.

    :type command: str
    :param command:
        (Optional) Information about which interface for the operation was
        used, to be included in the X-Goog-API-Client header. Please leave
        as None unless otherwise directed.

    :rtype: dict
    :returns: The headers to be used for the request.
    """
    x_goog_api_client = f"{user_agent} {_get_invocation_id()}"

    if command:
        x_goog_api_client += f" gccl-gcs-cmd/{command}"

    return {
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": user_agent,
        "X-Goog-API-Client": x_goog_api_client,
        "content-type": content_type,
        "x-upload-content-type": x_upload_content_type or content_type,
    }
