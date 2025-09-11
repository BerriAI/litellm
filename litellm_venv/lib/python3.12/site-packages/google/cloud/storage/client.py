# Copyright 2015 Google LLC
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

"""Client for interacting with the Google Cloud Storage API."""

import base64
import binascii
import collections
import datetime
import functools
import json
import warnings
import google.api_core.client_options

from google.auth.credentials import AnonymousCredentials

from google.api_core import page_iterator
from google.cloud._helpers import _LocalStack
from google.cloud.client import ClientWithProject
from google.cloud.exceptions import NotFound

from google.cloud.storage._helpers import _add_generation_match_parameters
from google.cloud.storage._helpers import _bucket_bound_hostname_url
from google.cloud.storage._helpers import _get_api_endpoint_override
from google.cloud.storage._helpers import _get_environ_project
from google.cloud.storage._helpers import _get_storage_emulator_override
from google.cloud.storage._helpers import _use_client_cert
from google.cloud.storage._helpers import _virtual_hosted_style_base_url
from google.cloud.storage._helpers import _DEFAULT_UNIVERSE_DOMAIN
from google.cloud.storage._helpers import _DEFAULT_SCHEME
from google.cloud.storage._helpers import _STORAGE_HOST_TEMPLATE
from google.cloud.storage._helpers import _NOW
from google.cloud.storage._helpers import _UTC
from google.cloud.storage._opentelemetry_tracing import create_trace_span

from google.cloud.storage._http import Connection
from google.cloud.storage._signing import (
    get_expiration_seconds_v4,
    get_v4_now_dtstamps,
    ensure_signed_credentials,
    _sign_message,
)
from google.cloud.storage.batch import Batch
from google.cloud.storage.bucket import Bucket, _item_to_blob, _blobs_page_start
from google.cloud.storage.blob import Blob
from google.cloud.storage.hmac_key import HMACKeyMetadata
from google.cloud.storage.acl import BucketACL
from google.cloud.storage.acl import DefaultObjectACL
from google.cloud.storage.constants import _DEFAULT_TIMEOUT
from google.cloud.storage.retry import DEFAULT_RETRY


_marker = object()


class Client(ClientWithProject):
    """Client to bundle configuration needed for API requests.

    :type project: str or None
    :param project: the project which the client acts on behalf of. Will be
                    passed when creating a topic.  If not passed,
                    falls back to the default inferred from the environment.

    :type credentials: :class:`~google.auth.credentials.Credentials`
    :param credentials: (Optional) The OAuth2 Credentials to use for this
                        client. If not passed (and if no ``_http`` object is
                        passed), falls back to the default inferred from the
                        environment.

    :type _http: :class:`~requests.Session`
    :param _http: (Optional) HTTP object to make requests. Can be any object
                  that defines ``request()`` with the same interface as
                  :meth:`requests.Session.request`. If not passed, an
                  ``_http`` object is created that is bound to the
                  ``credentials`` for the current object.
                  This parameter should be considered private, and could
                  change in the future.

    :type client_info: :class:`~google.api_core.client_info.ClientInfo`
    :param client_info:
        The client info used to send a user-agent string along with API
        requests. If ``None``, then default info will be used. Generally,
        you only need to set this if you're developing your own library
        or partner tool.

    :type client_options: :class:`~google.api_core.client_options.ClientOptions` or :class:`dict`
    :param client_options: (Optional) Client options used to set user options on the client.
        A non-default universe domain or api endpoint should be set through client_options.

    :type use_auth_w_custom_endpoint: bool
    :param use_auth_w_custom_endpoint:
        (Optional) Whether authentication is required under custom endpoints.
        If false, uses AnonymousCredentials and bypasses authentication.
        Defaults to True. Note this is only used when a custom endpoint is set in conjunction.

    :type extra_headers: dict
    :param extra_headers:
        (Optional) Custom headers to be sent with the requests attached to the client.
        For example, you can add custom audit logging headers.
    """

    SCOPE = (
        "https://www.googleapis.com/auth/devstorage.full_control",
        "https://www.googleapis.com/auth/devstorage.read_only",
        "https://www.googleapis.com/auth/devstorage.read_write",
    )
    """The scopes required for authenticating as a Cloud Storage consumer."""

    def __init__(
        self,
        project=_marker,
        credentials=None,
        _http=None,
        client_info=None,
        client_options=None,
        use_auth_w_custom_endpoint=True,
        extra_headers={},
    ):
        self._base_connection = None

        if project is None:
            no_project = True
            project = "<none>"
        else:
            no_project = False

        if project is _marker:
            project = None

        # Save the initial value of constructor arguments before they
        # are passed along, for use in __reduce__ defined elsewhere.
        self._initial_client_info = client_info
        self._initial_client_options = client_options
        self._extra_headers = extra_headers

        connection_kw_args = {"client_info": client_info}

        if client_options:
            if isinstance(client_options, dict):
                client_options = google.api_core.client_options.from_dict(
                    client_options
                )

        if client_options and client_options.universe_domain:
            self._universe_domain = client_options.universe_domain
        else:
            self._universe_domain = None

        storage_emulator_override = _get_storage_emulator_override()
        api_endpoint_override = _get_api_endpoint_override()

        # Determine the api endpoint. The rules are as follows:

        # 1. If the `api_endpoint` is set in `client_options`, use that as the
        #    endpoint.
        if client_options and client_options.api_endpoint:
            api_endpoint = client_options.api_endpoint

        # 2. Elif the "STORAGE_EMULATOR_HOST" env var is set, then use that as the
        #    endpoint.
        elif storage_emulator_override:
            api_endpoint = storage_emulator_override

        # 3. Elif the "API_ENDPOINT_OVERRIDE" env var is set, then use that as the
        #    endpoint.
        elif api_endpoint_override:
            api_endpoint = api_endpoint_override

        # 4. Elif the `universe_domain` is set in `client_options`,
        #    create the endpoint using that as the default.
        #
        #    Mutual TLS is not compatible with a non-default universe domain
        #    at this time. If such settings are enabled along with the
        #    "GOOGLE_API_USE_CLIENT_CERTIFICATE" env variable, a ValueError will
        #    be raised.

        elif self._universe_domain:
            # The final decision of whether to use mTLS takes place in
            # google-auth-library-python. We peek at the environment variable
            # here only to issue an exception in case of a conflict.
            if _use_client_cert():
                raise ValueError(
                    'The "GOOGLE_API_USE_CLIENT_CERTIFICATE" env variable is '
                    'set to "true" and a non-default universe domain is '
                    "configured. mTLS is not supported in any universe other than"
                    "googleapis.com."
                )
            api_endpoint = _DEFAULT_SCHEME + _STORAGE_HOST_TEMPLATE.format(
                universe_domain=self._universe_domain
            )

        # 5. Else, use the default, which is to use the default
        #    universe domain of "googleapis.com" and create the endpoint
        #    "storage.googleapis.com" from that.
        else:
            api_endpoint = None

        connection_kw_args["api_endpoint"] = api_endpoint

        self._is_emulator_set = True if storage_emulator_override else False

        # If a custom endpoint is set, the client checks for credentials
        # or finds the default credentials based on the current environment.
        # Authentication may be bypassed under certain conditions:
        # (1) STORAGE_EMULATOR_HOST is set (for backwards compatibility), OR
        # (2) use_auth_w_custom_endpoint is set to False.
        if connection_kw_args["api_endpoint"] is not None:
            if self._is_emulator_set or not use_auth_w_custom_endpoint:
                if credentials is None:
                    credentials = AnonymousCredentials()
                if project is None:
                    project = _get_environ_project()
                if project is None:
                    no_project = True
                    project = "<none>"

        super(Client, self).__init__(
            project=project,
            credentials=credentials,
            client_options=client_options,
            _http=_http,
        )

        # Validate that the universe domain of the credentials matches the
        # universe domain of the client.
        if self._credentials.universe_domain != self.universe_domain:
            raise ValueError(
                "The configured universe domain ({client_ud}) does not match "
                "the universe domain found in the credentials ({cred_ud}). If "
                "you haven't configured the universe domain explicitly, "
                "`googleapis.com` is the default.".format(
                    client_ud=self.universe_domain,
                    cred_ud=self._credentials.universe_domain,
                )
            )

        if no_project:
            self.project = None

        # Pass extra_headers to Connection
        connection = Connection(self, **connection_kw_args)
        connection.extra_headers = extra_headers
        self._connection = connection
        self._batch_stack = _LocalStack()

    @classmethod
    def create_anonymous_client(cls):
        """Factory: return client with anonymous credentials.

        .. note::

           Such a client has only limited access to "public" buckets:
           listing their contents and downloading their blobs.

        :rtype: :class:`google.cloud.storage.client.Client`
        :returns: Instance w/ anonymous credentials and no project.
        """
        client = cls(project="<none>", credentials=AnonymousCredentials())
        client.project = None
        return client

    @property
    def universe_domain(self):
        return self._universe_domain or _DEFAULT_UNIVERSE_DOMAIN

    @property
    def api_endpoint(self):
        return self._connection.API_BASE_URL

    @property
    def _connection(self):
        """Get connection or batch on the client.

        :rtype: :class:`google.cloud.storage._http.Connection`
        :returns: The connection set on the client, or the batch
                  if one is set.
        """
        if self.current_batch is not None:
            return self.current_batch
        else:
            return self._base_connection

    @_connection.setter
    def _connection(self, value):
        """Set connection on the client.

        Intended to be used by constructor (since the base class calls)
            self._connection = connection
        Will raise if the connection is set more than once.

        :type value: :class:`google.cloud.storage._http.Connection`
        :param value: The connection set on the client.

        :raises: :class:`ValueError` if connection has already been set.
        """
        if self._base_connection is not None:
            raise ValueError("Connection already set on client")
        self._base_connection = value

    def _push_batch(self, batch):
        """Push a batch onto our stack.

        "Protected", intended for use by batch context mgrs.

        :type batch: :class:`google.cloud.storage.batch.Batch`
        :param batch: newly-active batch
        """
        self._batch_stack.push(batch)

    def _pop_batch(self):
        """Pop a batch from our stack.

        "Protected", intended for use by batch context mgrs.

        :raises: IndexError if the stack is empty.
        :rtype: :class:`google.cloud.storage.batch.Batch`
        :returns: the top-most batch/transaction, after removing it.
        """
        return self._batch_stack.pop()

    @property
    def current_batch(self):
        """Currently-active batch.

        :rtype: :class:`google.cloud.storage.batch.Batch` or ``NoneType`` (if
                no batch is active).
        :returns: The batch at the top of the batch stack.
        """
        return self._batch_stack.top

    @create_trace_span(name="Storage.Client.getServiceAccountEmail")
    def get_service_account_email(
        self, project=None, timeout=_DEFAULT_TIMEOUT, retry=DEFAULT_RETRY
    ):
        """Get the email address of the project's GCS service account

        :type project: str
        :param project:
            (Optional) Project ID to use for retreiving GCS service account
            email address.  Defaults to the client's project.
        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC. See: :ref:`configuring_retries`

        :rtype: str
        :returns: service account email address
        """
        if project is None:
            project = self.project

        path = f"/projects/{project}/serviceAccount"
        api_response = self._get_resource(path, timeout=timeout, retry=retry)
        return api_response["email_address"]

    def bucket(self, bucket_name, user_project=None, generation=None):
        """Factory constructor for bucket object.

        .. note::
          This will not make an HTTP request; it simply instantiates
          a bucket object owned by this client.

        :type bucket_name: str
        :param bucket_name: The name of the bucket to be instantiated.

        :type user_project: str
        :param user_project: (Optional) The project ID to be billed for API
                             requests made via the bucket.

        :type generation: int
        :param generation: (Optional) If present, selects a specific revision of
                           this bucket.

        :rtype: :class:`google.cloud.storage.bucket.Bucket`
        :returns: The bucket object created.
        """
        return Bucket(
            client=self,
            name=bucket_name,
            user_project=user_project,
            generation=generation,
        )

    def batch(self, raise_exception=True):
        """Factory constructor for batch object.

        .. note::
          This will not make an HTTP request; it simply instantiates
          a batch object owned by this client.

        :type raise_exception: bool
        :param raise_exception:
            (Optional) Defaults to True. If True, instead of adding exceptions
            to the list of return responses, the final exception will be raised.
            Note that exceptions are unwrapped after all operations are complete
            in success or failure, and only the last exception is raised.

        :rtype: :class:`google.cloud.storage.batch.Batch`
        :returns: The batch object created.
        """
        return Batch(client=self, raise_exception=raise_exception)

    def _get_resource(
        self,
        path,
        query_params=None,
        headers=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=DEFAULT_RETRY,
        _target_object=None,
    ):
        """Helper for bucket / blob methods making API 'GET' calls.

        Args:
            path str:
                The path of the resource to fetch.

            query_params Optional[dict]:
                HTTP query parameters to be passed

            headers Optional[dict]:
                HTTP headers to be passed

            timeout (Optional[Union[float, Tuple[float, float]]]):
                The amount of time, in seconds, to wait for the server response.

                Can also be passed as a tuple (connect_timeout, read_timeout).
                See :meth:`requests.Session.request` documentation for details.

            retry (Optional[Union[google.api_core.retry.Retry, google.cloud.storage.retry.ConditionalRetryPolicy]]):
                How to retry the RPC. A None value will disable retries.
                A google.api_core.retry.Retry value will enable retries, and the object will
                define retriable response codes and errors and configure backoff and timeout options.

                A google.cloud.storage.retry.ConditionalRetryPolicy value wraps a Retry object and
                activates it only if certain conditions are met. This class exists to provide safe defaults
                for RPC calls that are not technically safe to retry normally (due to potential data
                duplication or other side-effects) but become safe to retry if a condition such as
                if_metageneration_match is set.

                See the retry.py source code and docstrings in this package (google.cloud.storage.retry) for
                information on retry types and how to configure them.

            _target_object (Union[ \
                :class:`~google.cloud.storage.bucket.Bucket`, \
                :class:`~google.cloud.storage.bucket.blob`, \
            ]):
                Object to which future data is to be applied -- only relevant
                in the context of a batch.

        Returns:
            dict
                The JSON resource fetched

        Raises:
            google.cloud.exceptions.NotFound
                If the bucket is not found.
        """
        return self._connection.api_request(
            method="GET",
            path=path,
            query_params=query_params,
            headers=headers,
            timeout=timeout,
            retry=retry,
            _target_object=_target_object,
        )

    def _list_resource(
        self,
        path,
        item_to_value,
        page_token=None,
        max_results=None,
        extra_params=None,
        page_start=page_iterator._do_nothing_page_start,
        page_size=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=DEFAULT_RETRY,
    ):
        kwargs = {
            "method": "GET",
            "path": path,
            "timeout": timeout,
        }
        with create_trace_span(
            name="Storage.Client._list_resource_returns_iterator",
            client=self,
            api_request=kwargs,
            retry=retry,
        ):
            api_request = functools.partial(
                self._connection.api_request, timeout=timeout, retry=retry
            )
        return page_iterator.HTTPIterator(
            client=self,
            api_request=api_request,
            path=path,
            item_to_value=item_to_value,
            page_token=page_token,
            max_results=max_results,
            extra_params=extra_params,
            page_start=page_start,
            page_size=page_size,
        )

    def _patch_resource(
        self,
        path,
        data,
        query_params=None,
        headers=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=None,
        _target_object=None,
    ):
        """Helper for bucket / blob methods making API 'PATCH' calls.

        Args:
            path str:
                The path of the resource to fetch.

            data dict:
                The data to be patched.

            query_params Optional[dict]:
                HTTP query parameters to be passed

            headers Optional[dict]:
                HTTP headers to be passed

            timeout (Optional[Union[float, Tuple[float, float]]]):
                The amount of time, in seconds, to wait for the server response.

                Can also be passed as a tuple (connect_timeout, read_timeout).
                See :meth:`requests.Session.request` documentation for details.

            retry (Optional[Union[google.api_core.retry.Retry, google.cloud.storage.retry.ConditionalRetryPolicy]]):
                How to retry the RPC. A None value will disable retries.
                A google.api_core.retry.Retry value will enable retries, and the object will
                define retriable response codes and errors and configure backoff and timeout options.

                A google.cloud.storage.retry.ConditionalRetryPolicy value wraps a Retry object and
                activates it only if certain conditions are met. This class exists to provide safe defaults
                for RPC calls that are not technically safe to retry normally (due to potential data
                duplication or other side-effects) but become safe to retry if a condition such as
                if_metageneration_match is set.

                See the retry.py source code and docstrings in this package (google.cloud.storage.retry) for
                information on retry types and how to configure them.

            _target_object (Union[ \
                :class:`~google.cloud.storage.bucket.Bucket`, \
                :class:`~google.cloud.storage.bucket.blob`, \
            ]):
                Object to which future data is to be applied -- only relevant
                in the context of a batch.

        Returns:
            dict
                The JSON resource fetched

        Raises:
            google.cloud.exceptions.NotFound
                If the bucket is not found.
        """
        return self._connection.api_request(
            method="PATCH",
            path=path,
            data=data,
            query_params=query_params,
            headers=headers,
            timeout=timeout,
            retry=retry,
            _target_object=_target_object,
        )

    def _put_resource(
        self,
        path,
        data,
        query_params=None,
        headers=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=None,
        _target_object=None,
    ):
        """Helper for bucket / blob methods making API 'PUT' calls.

        Args:
            path str:
                The path of the resource to fetch.

            data dict:
                The data to be patched.

            query_params Optional[dict]:
                HTTP query parameters to be passed

            headers Optional[dict]:
                HTTP headers to be passed

            timeout (Optional[Union[float, Tuple[float, float]]]):
                The amount of time, in seconds, to wait for the server response.

                Can also be passed as a tuple (connect_timeout, read_timeout).
                See :meth:`requests.Session.request` documentation for details.

            retry (Optional[Union[google.api_core.retry.Retry, google.cloud.storage.retry.ConditionalRetryPolicy]]):
                How to retry the RPC. A None value will disable retries.
                A google.api_core.retry.Retry value will enable retries, and the object will
                define retriable response codes and errors and configure backoff and timeout options.

                A google.cloud.storage.retry.ConditionalRetryPolicy value wraps a Retry object and
                activates it only if certain conditions are met. This class exists to provide safe defaults
                for RPC calls that are not technically safe to retry normally (due to potential data
                duplication or other side-effects) but become safe to retry if a condition such as
                if_metageneration_match is set.

                See the retry.py source code and docstrings in this package (google.cloud.storage.retry) for
                information on retry types and how to configure them.

            _target_object (Union[ \
                :class:`~google.cloud.storage.bucket.Bucket`, \
                :class:`~google.cloud.storage.bucket.blob`, \
            ]):
                Object to which future data is to be applied -- only relevant
                in the context of a batch.

        Returns:
            dict
                The JSON resource fetched

        Raises:
            google.cloud.exceptions.NotFound
                If the bucket is not found.
        """
        return self._connection.api_request(
            method="PUT",
            path=path,
            data=data,
            query_params=query_params,
            headers=headers,
            timeout=timeout,
            retry=retry,
            _target_object=_target_object,
        )

    def _post_resource(
        self,
        path,
        data,
        query_params=None,
        headers=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=None,
        _target_object=None,
    ):
        """Helper for bucket / blob methods making API 'POST' calls.

        Args:
            path str:
                The path of the resource to which to post.

            data dict:
                The data to be posted.

            query_params Optional[dict]:
                HTTP query parameters to be passed

            headers Optional[dict]:
                HTTP headers to be passed

            timeout (Optional[Union[float, Tuple[float, float]]]):
                The amount of time, in seconds, to wait for the server response.

                Can also be passed as a tuple (connect_timeout, read_timeout).
                See :meth:`requests.Session.request` documentation for details.

            retry (Optional[Union[google.api_core.retry.Retry, google.cloud.storage.retry.ConditionalRetryPolicy]]):
                How to retry the RPC. A None value will disable retries.
                A google.api_core.retry.Retry value will enable retries, and the object will
                define retriable response codes and errors and configure backoff and timeout options.

                A google.cloud.storage.retry.ConditionalRetryPolicy value wraps a Retry object and
                activates it only if certain conditions are met. This class exists to provide safe defaults
                for RPC calls that are not technically safe to retry normally (due to potential data
                duplication or other side-effects) but become safe to retry if a condition such as
                if_metageneration_match is set.

                See the retry.py source code and docstrings in this package (google.cloud.storage.retry) for
                information on retry types and how to configure them.

            _target_object (Union[ \
                :class:`~google.cloud.storage.bucket.Bucket`, \
                :class:`~google.cloud.storage.bucket.blob`, \
            ]):
                Object to which future data is to be applied -- only relevant
                in the context of a batch.

        Returns:
            dict
                The JSON resource returned from the post.

        Raises:
            google.cloud.exceptions.NotFound
                If the bucket is not found.
        """

        return self._connection.api_request(
            method="POST",
            path=path,
            data=data,
            query_params=query_params,
            headers=headers,
            timeout=timeout,
            retry=retry,
            _target_object=_target_object,
        )

    def _delete_resource(
        self,
        path,
        query_params=None,
        headers=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=DEFAULT_RETRY,
        _target_object=None,
    ):
        """Helper for bucket / blob methods making API 'DELETE' calls.

        Args:
            path str:
                The path of the resource to delete.

            query_params Optional[dict]:
                HTTP query parameters to be passed

            headers Optional[dict]:
                HTTP headers to be passed

            timeout (Optional[Union[float, Tuple[float, float]]]):
                The amount of time, in seconds, to wait for the server response.

                Can also be passed as a tuple (connect_timeout, read_timeout).
                See :meth:`requests.Session.request` documentation for details.

            retry (Optional[Union[google.api_core.retry.Retry, google.cloud.storage.retry.ConditionalRetryPolicy]]):
                How to retry the RPC. A None value will disable retries.
                A google.api_core.retry.Retry value will enable retries, and the object will
                define retriable response codes and errors and configure backoff and timeout options.

                A google.cloud.storage.retry.ConditionalRetryPolicy value wraps a Retry object and
                activates it only if certain conditions are met. This class exists to provide safe defaults
                for RPC calls that are not technically safe to retry normally (due to potential data
                duplication or other side-effects) but become safe to retry if a condition such as
                if_metageneration_match is set.

                See the retry.py source code and docstrings in this package (google.cloud.storage.retry) for
                information on retry types and how to configure them.

            _target_object (Union[ \
                :class:`~google.cloud.storage.bucket.Bucket`, \
                :class:`~google.cloud.storage.bucket.blob`, \
            ]):
                Object to which future data is to be applied -- only relevant
                in the context of a batch.

        Returns:
            dict
                The JSON resource fetched

        Raises:
            google.cloud.exceptions.NotFound
                If the bucket is not found.
        """
        return self._connection.api_request(
            method="DELETE",
            path=path,
            query_params=query_params,
            headers=headers,
            timeout=timeout,
            retry=retry,
            _target_object=_target_object,
        )

    def _bucket_arg_to_bucket(self, bucket_or_name, generation=None):
        """Helper to return given bucket or create new by name.

        Args:
            bucket_or_name (Union[ \
                :class:`~google.cloud.storage.bucket.Bucket`, \
                 str, \
            ]):
                The bucket resource to pass or name to create.
            generation (Optional[int]):
                The bucket generation. If generation is specified,
                bucket_or_name must be a name (str).

        Returns:
            google.cloud.storage.bucket.Bucket
                The newly created bucket or the given one.
        """
        if isinstance(bucket_or_name, Bucket):
            if generation:
                raise ValueError(
                    "The generation can only be specified if a "
                    "name is used to specify a bucket, not a Bucket object. "
                    "Create a new Bucket object with the correct generation "
                    "instead."
                )
            bucket = bucket_or_name
            if bucket.client is None:
                bucket._client = self
        else:
            bucket = Bucket(self, name=bucket_or_name, generation=generation)
        return bucket

    @create_trace_span(name="Storage.Client.getBucket")
    def get_bucket(
        self,
        bucket_or_name,
        timeout=_DEFAULT_TIMEOUT,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        retry=DEFAULT_RETRY,
        *,
        generation=None,
        soft_deleted=None,
    ):
        """Retrieve a bucket via a GET request.

        See [API reference docs](https://cloud.google.com/storage/docs/json_api/v1/buckets/get) and a [code sample](https://cloud.google.com/storage/docs/samples/storage-get-bucket-metadata#storage_get_bucket_metadata-python).

        Args:
            bucket_or_name (Union[ \
                :class:`~google.cloud.storage.bucket.Bucket`, \
                 str, \
            ]):
                The bucket resource to pass or name to create.

            timeout (Optional[Union[float, Tuple[float, float]]]):
                The amount of time, in seconds, to wait for the server response.

                Can also be passed as a tuple (connect_timeout, read_timeout).
                See :meth:`requests.Session.request` documentation for details.

            if_metageneration_match (Optional[int]):
                Make the operation conditional on whether the
                bucket's current metageneration matches the given value.

            if_metageneration_not_match (Optional[int]):
                Make the operation conditional on whether the bucket's
                current metageneration does not match the given value.

            retry (Optional[Union[google.api_core.retry.Retry, google.cloud.storage.retry.ConditionalRetryPolicy]]):
                How to retry the RPC. A None value will disable retries.
                A google.api_core.retry.Retry value will enable retries, and the object will
                define retriable response codes and errors and configure backoff and timeout options.

                A google.cloud.storage.retry.ConditionalRetryPolicy value wraps a Retry object and
                activates it only if certain conditions are met. This class exists to provide safe defaults
                for RPC calls that are not technically safe to retry normally (due to potential data
                duplication or other side-effects) but become safe to retry if a condition such as
                if_metageneration_match is set.

                See the retry.py source code and docstrings in this package (google.cloud.storage.retry) for
                information on retry types and how to configure them.

            generation (Optional[int]):
                The generation of the bucket. The generation can be used to
                specify a specific soft-deleted version of the bucket, in
                conjunction with the ``soft_deleted`` argument below. If
                ``soft_deleted`` is not True, the generation is unused.

            soft_deleted (Optional[bool]):
                If True, looks for a soft-deleted bucket. Will only return
                the bucket metadata if the bucket exists and is in a
                soft-deleted state. The bucket ``generation`` is required if
                ``soft_deleted`` is set to True.
                See: https://cloud.google.com/storage/docs/soft-delete

        Returns:
            google.cloud.storage.bucket.Bucket
                The bucket matching the name provided.

        Raises:
            google.cloud.exceptions.NotFound
                If the bucket is not found.
        """
        bucket = self._bucket_arg_to_bucket(bucket_or_name, generation=generation)
        bucket.reload(
            client=self,
            timeout=timeout,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
            retry=retry,
            soft_deleted=soft_deleted,
        )
        return bucket

    @create_trace_span(name="Storage.Client.lookupBucket")
    def lookup_bucket(
        self,
        bucket_name,
        timeout=_DEFAULT_TIMEOUT,
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        retry=DEFAULT_RETRY,
    ):
        """Get a bucket by name, returning None if not found.

        You can use this if you would rather check for a None value
        than catching a NotFound exception.

        :type bucket_name: str
        :param bucket_name: The name of the bucket to get.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type if_metageneration_match: long
        :param if_metageneration_match: (Optional) Make the operation conditional on whether the
                                        blob's current metageneration matches the given value.

        :type if_metageneration_not_match: long
        :param if_metageneration_not_match: (Optional) Make the operation conditional on whether the
                                            blob's current metageneration does not match the given value.

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC. See: :ref:`configuring_retries`

        :rtype: :class:`google.cloud.storage.bucket.Bucket` or ``NoneType``
        :returns: The bucket matching the name provided or None if not found.
        """
        try:
            return self.get_bucket(
                bucket_name,
                timeout=timeout,
                if_metageneration_match=if_metageneration_match,
                if_metageneration_not_match=if_metageneration_not_match,
                retry=retry,
            )
        except NotFound:
            return None

    @create_trace_span(name="Storage.Client.createBucket")
    def create_bucket(
        self,
        bucket_or_name,
        requester_pays=None,
        project=None,
        user_project=None,
        location=None,
        data_locations=None,
        predefined_acl=None,
        predefined_default_object_acl=None,
        enable_object_retention=False,
        timeout=_DEFAULT_TIMEOUT,
        retry=DEFAULT_RETRY,
    ):
        """Create a new bucket via a POST request.

        See [API reference docs](https://cloud.google.com/storage/docs/json_api/v1/buckets/insert) and a [code sample](https://cloud.google.com/storage/docs/samples/storage-create-bucket#storage_create_bucket-python).

        Args:
            bucket_or_name (Union[ \
                :class:`~google.cloud.storage.bucket.Bucket`, \
                 str, \
            ]):
                The bucket resource to pass or name to create.
            requester_pays (bool):
                DEPRECATED. Use Bucket().requester_pays instead.
                (Optional) Whether requester pays for API requests for
                this bucket and its blobs.
            project (str):
                (Optional) The project under which the bucket is to be created.
                If not passed, uses the project set on the client.
            user_project (str):
                (Optional) The project ID to be billed for API requests
                made via created bucket.
            location (str):
                (Optional) The location of the bucket. If not passed,
                the default location, US, will be used. If specifying a dual-region,
                `data_locations` should be set in conjunction. See:
                https://cloud.google.com/storage/docs/locations
            data_locations (list of str):
                (Optional) The list of regional locations of a custom dual-region bucket.
                Dual-regions require exactly 2 regional locations. See:
                https://cloud.google.com/storage/docs/locations
            predefined_acl (str):
                (Optional) Name of predefined ACL to apply to bucket. See:
                https://cloud.google.com/storage/docs/access-control/lists#predefined-acl
            predefined_default_object_acl (str):
                (Optional) Name of predefined ACL to apply to bucket's objects. See:
                https://cloud.google.com/storage/docs/access-control/lists#predefined-acl
            enable_object_retention (bool):
                (Optional) Whether object retention should be enabled on this bucket. See:
                https://cloud.google.com/storage/docs/object-lock
            timeout (Optional[Union[float, Tuple[float, float]]]):
                The amount of time, in seconds, to wait for the server response.

                Can also be passed as a tuple (connect_timeout, read_timeout).
                See :meth:`requests.Session.request` documentation for details.

            retry (Optional[Union[google.api_core.retry.Retry, google.cloud.storage.retry.ConditionalRetryPolicy]]):
                How to retry the RPC. A None value will disable retries.
                A google.api_core.retry.Retry value will enable retries, and the object will
                define retriable response codes and errors and configure backoff and timeout options.

                A google.cloud.storage.retry.ConditionalRetryPolicy value wraps a Retry object and
                activates it only if certain conditions are met. This class exists to provide safe defaults
                for RPC calls that are not technically safe to retry normally (due to potential data
                duplication or other side-effects) but become safe to retry if a condition such as
                if_metageneration_match is set.

                See the retry.py source code and docstrings in this package (google.cloud.storage.retry) for
                information on retry types and how to configure them.

        Returns:
            google.cloud.storage.bucket.Bucket
                The newly created bucket.

        Raises:
            google.cloud.exceptions.Conflict
                If the bucket already exists.
        """
        bucket = self._bucket_arg_to_bucket(bucket_or_name)
        query_params = {}

        if project is None:
            project = self.project

        # Use no project if STORAGE_EMULATOR_HOST is set
        if self._is_emulator_set:
            if project is None:
                project = _get_environ_project()
            if project is None:
                project = "<none>"

        # Only include the project parameter if a project is set.
        # If a project is not set, falls back to API validation (BadRequest).
        if project is not None:
            query_params = {"project": project}

        if requester_pays is not None:
            warnings.warn(
                "requester_pays arg is deprecated. Use Bucket().requester_pays instead.",
                PendingDeprecationWarning,
                stacklevel=1,
            )
            bucket.requester_pays = requester_pays

        if predefined_acl is not None:
            predefined_acl = BucketACL.validate_predefined(predefined_acl)
            query_params["predefinedAcl"] = predefined_acl

        if predefined_default_object_acl is not None:
            predefined_default_object_acl = DefaultObjectACL.validate_predefined(
                predefined_default_object_acl
            )
            query_params["predefinedDefaultObjectAcl"] = predefined_default_object_acl

        if user_project is not None:
            query_params["userProject"] = user_project

        if enable_object_retention:
            query_params["enableObjectRetention"] = enable_object_retention

        properties = {key: bucket._properties[key] for key in bucket._changes}
        properties["name"] = bucket.name

        if location is not None:
            properties["location"] = location

        if data_locations is not None:
            properties["customPlacementConfig"] = {"dataLocations": data_locations}

        api_response = self._post_resource(
            "/b",
            properties,
            query_params=query_params,
            timeout=timeout,
            retry=retry,
            _target_object=bucket,
        )

        bucket._set_properties(api_response)
        return bucket

    @create_trace_span(name="Storage.Client.downloadBlobToFile")
    def download_blob_to_file(
        self,
        blob_or_uri,
        file_obj,
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
        """Download the contents of a blob object or blob URI into a file-like object.

        See https://cloud.google.com/storage/docs/downloading-objects

        Args:
            blob_or_uri (Union[ \
            :class:`~google.cloud.storage.blob.Blob`, \
             str, \
            ]):
                The blob resource to pass or URI to download.

            file_obj (file):
                A file handle to which to write the blob's data.

            start (int):
                (Optional) The first byte in a range to be downloaded.

            end (int):
                (Optional) The last byte in a range to be downloaded.

            raw_download (bool):
                (Optional) If true, download the object without any expansion.

            if_etag_match (Union[str, Set[str]]):
                (Optional) See :ref:`using-if-etag-match`

            if_etag_not_match (Union[str, Set[str]]):
                (Optional) See :ref:`using-if-etag-not-match`

            if_generation_match (long):
                (Optional) See :ref:`using-if-generation-match`

            if_generation_not_match (long):
                (Optional) See :ref:`using-if-generation-not-match`

            if_metageneration_match (long):
                (Optional) See :ref:`using-if-metageneration-match`

            if_metageneration_not_match (long):
                (Optional) See :ref:`using-if-metageneration-not-match`

            timeout ([Union[float, Tuple[float, float]]]):
                (Optional) The amount of time, in seconds, to wait
                for the server response.  See: :ref:`configuring_timeouts`

            checksum (str):
                (Optional) The type of checksum to compute to verify the integrity
                of the object. The response headers must contain a checksum of the
                requested type. If the headers lack an appropriate checksum (for
                instance in the case of transcoded or ranged downloads where the
                remote service does not know the correct checksum, including
                downloads where chunk_size is set) an INFO-level log will be
                emitted. Supported values are "md5", "crc32c" and None. The default
                is "md5".
            retry (google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy)
                (Optional) How to retry the RPC. A None value will disable
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
        """

        if not isinstance(blob_or_uri, Blob):
            blob_or_uri = Blob.from_string(blob_or_uri)

        blob_or_uri._prep_and_do_download(
            file_obj,
            client=self,
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

    @create_trace_span(name="Storage.Client.listBlobs")
    def list_blobs(
        self,
        bucket_or_name,
        max_results=None,
        page_token=None,
        prefix=None,
        delimiter=None,
        start_offset=None,
        end_offset=None,
        include_trailing_delimiter=None,
        versions=None,
        projection="noAcl",
        fields=None,
        page_size=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=DEFAULT_RETRY,
        match_glob=None,
        include_folders_as_prefixes=None,
        soft_deleted=None,
    ):
        """Return an iterator used to find blobs in the bucket.

        If :attr:`user_project` is set, bills the API request to that project.

        .. note::
          List prefixes (directories) in a bucket using a prefix and delimiter.
          See a [code sample](https://cloud.google.com/storage/docs/samples/storage-list-files-with-prefix#storage_list_files_with_prefix-python)
          listing objects using a prefix filter.

        Args:
            bucket_or_name (Union[ \
                :class:`~google.cloud.storage.bucket.Bucket`, \
                 str, \
            ]):
                The bucket resource to pass or name to create.

            max_results (int):
                (Optional) The maximum number of blobs to return.

            page_token (str):
                (Optional) If present, return the next batch of blobs, using the
                value, which must correspond to the ``nextPageToken`` value
                returned in the previous response.  Deprecated: use the ``pages``
                property of the returned iterator instead of manually passing the
                token.

            prefix (str):
                (Optional) Prefix used to filter blobs.

            delimiter (str):
                (Optional) Delimiter, used with ``prefix`` to
                emulate hierarchy.

            start_offset (str):
                (Optional) Filter results to objects whose names are
                lexicographically equal to or after ``startOffset``. If
                ``endOffset`` is also set, the objects listed will have names
                between ``startOffset`` (inclusive) and ``endOffset``
                (exclusive).

            end_offset (str):
                (Optional) Filter results to objects whose names are
                lexicographically before ``endOffset``. If ``startOffset`` is
                also set, the objects listed will have names between
                ``startOffset`` (inclusive) and ``endOffset`` (exclusive).

            include_trailing_delimiter (boolean):
                (Optional) If true, objects that end in exactly one instance of
                ``delimiter`` will have their metadata included in ``items`` in
                addition to ``prefixes``.

            versions (bool):
                (Optional) Whether object versions should be returned
                as separate blobs.

            projection (str):
                (Optional) If used, must be 'full' or 'noAcl'.
                Defaults to ``'noAcl'``. Specifies the set of
                properties to return.

            fields (str):
                (Optional) Selector specifying which fields to include
                in a partial response. Must be a list of fields. For
                example to get a partial response with just the next
                page token and the name and language of each blob returned:
                ``'items(name,contentLanguage),nextPageToken'``.
                See: https://cloud.google.com/storage/docs/json_api/v1/parameters#fields

            page_size (int):
                (Optional) Maximum number of blobs to return in each page.
                Defaults to a value set by the API.

            timeout (Optional[Union[float, Tuple[float, float]]]):
                The amount of time, in seconds, to wait for the server response.

                Can also be passed as a tuple (connect_timeout, read_timeout).
                See :meth:`requests.Session.request` documentation for details.

            retry (Optional[Union[google.api_core.retry.Retry, google.cloud.storage.retry.ConditionalRetryPolicy]]):
                How to retry the RPC. A None value will disable retries.
                A google.api_core.retry.Retry value will enable retries, and the object will
                define retriable response codes and errors and configure backoff and timeout options.

                A google.cloud.storage.retry.ConditionalRetryPolicy value wraps a Retry object and
                activates it only if certain conditions are met. This class exists to provide safe defaults
                for RPC calls that are not technically safe to retry normally (due to potential data
                duplication or other side-effects) but become safe to retry if a condition such as
                if_metageneration_match is set.

                See the retry.py source code and docstrings in this package (google.cloud.storage.retry) for
                information on retry types and how to configure them.

            match_glob (str):
                (Optional) A glob pattern used to filter results (for example, foo*bar).
                The string value must be UTF-8 encoded. See:
                https://cloud.google.com/storage/docs/json_api/v1/objects/list#list-object-glob

            include_folders_as_prefixes (bool):
                (Optional) If true, includes Folders and Managed Folders in the set of
                ``prefixes`` returned by the query. Only applicable if ``delimiter`` is set to /.
                See: https://cloud.google.com/storage/docs/managed-folders

            soft_deleted (bool):
                (Optional) If true, only soft-deleted objects will be listed as distinct results in order of increasing
                generation number. This parameter can only be used successfully if the bucket has a soft delete policy.
                Note ``soft_deleted`` and ``versions`` cannot be set to True simultaneously. See:
                https://cloud.google.com/storage/docs/soft-delete

        Returns:
            Iterator of all :class:`~google.cloud.storage.blob.Blob`
            in this bucket matching the arguments. The RPC call
            returns a response when the iterator is consumed.

            As part of the response, you'll also get back an iterator.prefixes entity that lists object names
            up to and including the requested delimiter. Duplicate entries are omitted from this list.
        """
        bucket = self._bucket_arg_to_bucket(bucket_or_name)

        extra_params = {"projection": projection}

        if prefix is not None:
            extra_params["prefix"] = prefix

        if delimiter is not None:
            extra_params["delimiter"] = delimiter

        if match_glob is not None:
            extra_params["matchGlob"] = match_glob

        if start_offset is not None:
            extra_params["startOffset"] = start_offset

        if end_offset is not None:
            extra_params["endOffset"] = end_offset

        if include_trailing_delimiter is not None:
            extra_params["includeTrailingDelimiter"] = include_trailing_delimiter

        if versions is not None:
            extra_params["versions"] = versions

        if fields is not None:
            extra_params["fields"] = fields

        if include_folders_as_prefixes is not None:
            extra_params["includeFoldersAsPrefixes"] = include_folders_as_prefixes

        if soft_deleted is not None:
            extra_params["softDeleted"] = soft_deleted

        if bucket.user_project is not None:
            extra_params["userProject"] = bucket.user_project

        path = bucket.path + "/o"
        iterator = self._list_resource(
            path,
            _item_to_blob,
            page_token=page_token,
            max_results=max_results,
            extra_params=extra_params,
            page_start=_blobs_page_start,
            page_size=page_size,
            timeout=timeout,
            retry=retry,
        )
        iterator.bucket = bucket
        iterator.prefixes = set()
        return iterator

    @create_trace_span(name="Storage.Client.listBuckets")
    def list_buckets(
        self,
        max_results=None,
        page_token=None,
        prefix=None,
        projection="noAcl",
        fields=None,
        project=None,
        page_size=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=DEFAULT_RETRY,
        *,
        soft_deleted=None,
    ):
        """Get all buckets in the project associated to the client.

        This will not populate the list of blobs available in each
        bucket.

        See [API reference docs](https://cloud.google.com/storage/docs/json_api/v1/buckets/list) and a [code sample](https://cloud.google.com/storage/docs/samples/storage-list-buckets#storage_list_buckets-python).

        :type max_results: int
        :param max_results: (Optional) The maximum number of buckets to return.

        :type page_token: str
        :param page_token:
            (Optional) If present, return the next batch of buckets, using the
            value, which must correspond to the ``nextPageToken`` value
            returned in the previous response.  Deprecated: use the ``pages``
            property of the returned iterator instead of manually passing the
            token.

        :type prefix: str
        :param prefix: (Optional) Filter results to buckets whose names begin
                       with this prefix.

        :type projection: str
        :param projection:
            (Optional) Specifies the set of properties to return. If used, must
            be 'full' or 'noAcl'. Defaults to 'noAcl'.

        :type fields: str
        :param fields:
            (Optional) Selector specifying which fields to include in a partial
            response. Must be a list of fields. For example to get a partial
            response with just the next page token and the language of each
            bucket returned: 'items/id,nextPageToken'

        :type project: str
        :param project: (Optional) The project whose buckets are to be listed.
                        If not passed, uses the project set on the client.

        :type page_size: int
        :param page_size: (Optional) Maximum number of buckets to return in each page.
            Defaults to a value set by the API.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC. See: :ref:`configuring_retries`

        :type soft_deleted: bool
        :param soft_deleted:
            (Optional) If true, only soft-deleted buckets will be listed as distinct results in order of increasing
            generation number. This parameter can only be used successfully if the bucket has a soft delete policy.
            See: https://cloud.google.com/storage/docs/soft-delete

        :rtype: :class:`~google.api_core.page_iterator.Iterator`
        :raises ValueError: if both ``project`` is ``None`` and the client's
                            project is also ``None``.
        :returns: Iterator of all :class:`~google.cloud.storage.bucket.Bucket`
                  belonging to this project.
        """
        extra_params = {}

        if project is None:
            project = self.project

        # Use no project if STORAGE_EMULATOR_HOST is set
        if self._is_emulator_set:
            if project is None:
                project = _get_environ_project()
            if project is None:
                project = "<none>"

        # Only include the project parameter if a project is set.
        # If a project is not set, falls back to API validation (BadRequest).
        if project is not None:
            extra_params = {"project": project}

        if prefix is not None:
            extra_params["prefix"] = prefix

        extra_params["projection"] = projection

        if fields is not None:
            extra_params["fields"] = fields

        if soft_deleted is not None:
            extra_params["softDeleted"] = soft_deleted

        return self._list_resource(
            "/b",
            _item_to_bucket,
            page_token=page_token,
            max_results=max_results,
            extra_params=extra_params,
            page_size=page_size,
            timeout=timeout,
            retry=retry,
        )

    def restore_bucket(
        self,
        bucket_name,
        generation,
        projection="noAcl",
        if_metageneration_match=None,
        if_metageneration_not_match=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=DEFAULT_RETRY,
    ):
        """Restores a soft-deleted bucket.

        :type bucket_name: str
        :param bucket_name: The name of the bucket to be restored.

        :type generation: int
        :param generation: Selects the specific revision of the bucket.

        :type projection: str
        :param projection:
            (Optional) Specifies the set of properties to return. If used, must
            be 'full' or 'noAcl'. Defaults to 'noAcl'.

        if_metageneration_match (Optional[int]):
            Make the operation conditional on whether the
            blob's current metageneration matches the given value.

        if_metageneration_not_match (Optional[int]):
            Make the operation conditional on whether the blob's
            current metageneration does not match the given value.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC.

            Users can configure non-default retry behavior. A ``None`` value will
            disable retries. See [Configuring Retries](https://cloud.google.com/python/docs/reference/storage/latest/retry_timeout).

        :rtype: :class:`google.cloud.storage.bucket.Bucket`
        :returns: The restored Bucket.
        """
        query_params = {"generation": generation, "projection": projection}

        _add_generation_match_parameters(
            query_params,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
        )

        bucket = self.bucket(bucket_name)
        api_response = self._post_resource(
            f"{bucket.path}/restore",
            None,
            query_params=query_params,
            timeout=timeout,
            retry=retry,
        )
        bucket._set_properties(api_response)
        return bucket

    @create_trace_span(name="Storage.Client.createHmacKey")
    def create_hmac_key(
        self,
        service_account_email,
        project_id=None,
        user_project=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=None,
    ):
        """Create an HMAC key for a service account.

        :type service_account_email: str
        :param service_account_email: e-mail address of the service account

        :type project_id: str
        :param project_id: (Optional) Explicit project ID for the key.
            Defaults to the client's project.

        :type user_project: str
        :param user_project: (Optional) This parameter is currently ignored.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry: (Optional) How to retry the RPC. A None value will disable retries.
            A google.api_core.retry.Retry value will enable retries, and the object will
            define retriable response codes and errors and configure backoff and timeout options.

            A google.cloud.storage.retry.ConditionalRetryPolicy value wraps a Retry object and
            activates it only if certain conditions are met. This class exists to provide safe defaults
            for RPC calls that are not technically safe to retry normally (due to potential data
            duplication or other side-effects) but become safe to retry if a condition such as
            if_metageneration_match is set.

            See the retry.py source code and docstrings in this package (google.cloud.storage.retry) for
            information on retry types and how to configure them.

        :rtype:
            Tuple[:class:`~google.cloud.storage.hmac_key.HMACKeyMetadata`, str]
        :returns: metadata for the created key, plus the bytes of the key's secret, which is an 40-character base64-encoded string.
        """
        if project_id is None:
            project_id = self.project

        path = f"/projects/{project_id}/hmacKeys"
        qs_params = {"serviceAccountEmail": service_account_email}

        if user_project is not None:
            qs_params["userProject"] = user_project

        api_response = self._post_resource(
            path,
            None,
            query_params=qs_params,
            timeout=timeout,
            retry=retry,
        )
        metadata = HMACKeyMetadata(self)
        metadata._properties = api_response["metadata"]
        secret = api_response["secret"]
        return metadata, secret

    @create_trace_span(name="Storage.Client.listHmacKeys")
    def list_hmac_keys(
        self,
        max_results=None,
        service_account_email=None,
        show_deleted_keys=None,
        project_id=None,
        user_project=None,
        timeout=_DEFAULT_TIMEOUT,
        retry=DEFAULT_RETRY,
    ):
        """List HMAC keys for a project.

        :type max_results: int
        :param max_results:
            (Optional) Max number of keys to return in a given page.

        :type service_account_email: str
        :param service_account_email:
            (Optional) Limit keys to those created by the given service account.

        :type show_deleted_keys: bool
        :param show_deleted_keys:
            (Optional) Included deleted keys in the list. Default is to
            exclude them.

        :type project_id: str
        :param project_id: (Optional) Explicit project ID for the key.
            Defaults to the client's project.

        :type user_project: str
        :param user_project: (Optional) This parameter is currently ignored.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC. See: :ref:`configuring_retries`

        :rtype:
            Tuple[:class:`~google.cloud.storage.hmac_key.HMACKeyMetadata`, str]
        :returns: metadata for the created key, plus the bytes of the key's secret, which is an 40-character base64-encoded string.
        """
        if project_id is None:
            project_id = self.project

        path = f"/projects/{project_id}/hmacKeys"
        extra_params = {}

        if service_account_email is not None:
            extra_params["serviceAccountEmail"] = service_account_email

        if show_deleted_keys is not None:
            extra_params["showDeletedKeys"] = show_deleted_keys

        if user_project is not None:
            extra_params["userProject"] = user_project

        return self._list_resource(
            path,
            _item_to_hmac_key_metadata,
            max_results=max_results,
            extra_params=extra_params,
            timeout=timeout,
            retry=retry,
        )

    @create_trace_span(name="Storage.Client.getHmacKeyMetadata")
    def get_hmac_key_metadata(
        self, access_id, project_id=None, user_project=None, timeout=_DEFAULT_TIMEOUT
    ):
        """Return a metadata instance for the given HMAC key.

        :type access_id: str
        :param access_id: Unique ID of an existing key.

        :type project_id: str
        :param project_id: (Optional) Project ID of an existing key.
            Defaults to client's project.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type user_project: str
        :param user_project: (Optional) This parameter is currently ignored.
        """
        metadata = HMACKeyMetadata(self, access_id, project_id, user_project)
        metadata.reload(timeout=timeout)  # raises NotFound for missing key
        return metadata

    def generate_signed_post_policy_v4(
        self,
        bucket_name,
        blob_name,
        expiration,
        conditions=None,
        fields=None,
        credentials=None,
        virtual_hosted_style=False,
        bucket_bound_hostname=None,
        scheme="http",
        service_account_email=None,
        access_token=None,
    ):
        """Generate a V4 signed policy object. Generated policy object allows user to upload objects with a POST request.

        .. note::

            Assumes ``credentials`` implements the
            :class:`google.auth.credentials.Signing` interface. Also assumes
            ``credentials`` has a ``service_account_email`` property which
            identifies the credentials.

        See a [code sample](https://github.com/googleapis/python-storage/blob/main/samples/snippets/storage_generate_signed_post_policy_v4.py).

        :type bucket_name: str
        :param bucket_name: Bucket name.

        :type blob_name: str
        :param blob_name: Object name.

        :type expiration: Union[Integer, datetime.datetime, datetime.timedelta]
        :param expiration: Policy expiration time. If a ``datetime`` instance is
                           passed without an explicit ``tzinfo`` set,  it will be
                           assumed to be ``UTC``.

        :type conditions: list
        :param conditions: (Optional) List of POST policy conditions, which are
                           used to restrict what is allowed in the request.

        :type fields: dict
        :param fields: (Optional) Additional elements to include into request.

        :type credentials: :class:`google.auth.credentials.Signing`
        :param credentials: (Optional) Credentials object with an associated private
                            key to sign text.

        :type virtual_hosted_style: bool
        :param virtual_hosted_style:
            (Optional) If True, construct the URL relative to the bucket
            virtual hostname, e.g., '<bucket-name>.storage.googleapis.com'.
            Incompatible with bucket_bound_hostname.

        :type bucket_bound_hostname: str
        :param bucket_bound_hostname:
            (Optional) If passed, construct the URL relative to the bucket-bound hostname.
            Value can be bare or with a scheme, e.g., 'example.com' or 'http://example.com'.
            Incompatible with virtual_hosted_style.
            See: https://cloud.google.com/storage/docs/request-endpoints#cname

        :type scheme: str
        :param scheme:
            (Optional) If ``bucket_bound_hostname`` is passed as a bare hostname, use
            this value as a scheme. ``https`` will work only when using a CDN.
            Defaults to ``"http"``.

        :type service_account_email: str
        :param service_account_email: (Optional) E-mail address of the service account.

        :type access_token: str
        :param access_token: (Optional) Access token for a service account.

        :raises: :exc:`ValueError` when mutually exclusive arguments are used.

        :rtype: dict
        :returns: Signed POST policy.
        """
        if virtual_hosted_style and bucket_bound_hostname:
            raise ValueError(
                "Only one of virtual_hosted_style and bucket_bound_hostname "
                "can be specified."
            )

        credentials = self._credentials if credentials is None else credentials
        client_email = service_account_email
        if not access_token or not service_account_email:
            ensure_signed_credentials(credentials)
            client_email = credentials.signer_email

        # prepare policy conditions and fields
        timestamp, datestamp = get_v4_now_dtstamps()

        x_goog_credential = "{email}/{datestamp}/auto/storage/goog4_request".format(
            email=client_email, datestamp=datestamp
        )
        required_conditions = [
            {"bucket": bucket_name},
            {"key": blob_name},
            {"x-goog-date": timestamp},
            {"x-goog-credential": x_goog_credential},
            {"x-goog-algorithm": "GOOG4-RSA-SHA256"},
        ]

        conditions = conditions or []
        policy_fields = {}
        for key, value in sorted((fields or {}).items()):
            if not key.startswith("x-ignore-"):
                policy_fields[key] = value
                conditions.append({key: value})

        conditions += required_conditions

        # calculate policy expiration time
        now = _NOW(_UTC).replace(tzinfo=None)
        if expiration is None:
            expiration = now + datetime.timedelta(hours=1)

        policy_expires = now + datetime.timedelta(
            seconds=get_expiration_seconds_v4(expiration)
        )

        # encode policy for signing
        policy = json.dumps(
            collections.OrderedDict(
                sorted(
                    {
                        "conditions": conditions,
                        "expiration": policy_expires.isoformat() + "Z",
                    }.items()
                )
            ),
            separators=(",", ":"),
        )
        str_to_sign = base64.b64encode(policy.encode("utf-8"))

        # sign the policy and get its cryptographic signature
        if access_token and service_account_email:
            signature = _sign_message(str_to_sign, access_token, service_account_email)
            signature_bytes = base64.b64decode(signature)
        else:
            signature_bytes = credentials.sign_bytes(str_to_sign)

        # get hexadecimal representation of the signature
        signature = binascii.hexlify(signature_bytes).decode("utf-8")

        policy_fields.update(
            {
                "key": blob_name,
                "x-goog-algorithm": "GOOG4-RSA-SHA256",
                "x-goog-credential": x_goog_credential,
                "x-goog-date": timestamp,
                "x-goog-signature": signature,
                "policy": str_to_sign.decode("utf-8"),
            }
        )
        # designate URL
        if virtual_hosted_style:
            url = _virtual_hosted_style_base_url(
                self.api_endpoint, bucket_name, trailing_slash=True
            )
        elif bucket_bound_hostname:
            url = f"{_bucket_bound_hostname_url(bucket_bound_hostname, scheme)}/"
        else:
            url = f"{self.api_endpoint}/{bucket_name}/"

        return {"url": url, "fields": policy_fields}


def _item_to_bucket(iterator, item):
    """Convert a JSON bucket to the native object.

    :type iterator: :class:`~google.api_core.page_iterator.Iterator`
    :param iterator: The iterator that has retrieved the item.

    :type item: dict
    :param item: An item to be converted to a bucket.

    :rtype: :class:`.Bucket`
    :returns: The next bucket in the page.
    """
    name = item.get("name")
    bucket = Bucket(iterator.client, name)
    bucket._set_properties(item)
    return bucket


def _item_to_hmac_key_metadata(iterator, item):
    """Convert a JSON key metadata resource to the native object.

    :type iterator: :class:`~google.api_core.page_iterator.Iterator`
    :param iterator: The iterator that has retrieved the item.

    :type item: dict
    :param item: An item to be converted to a key metadata instance.

    :rtype: :class:`~google.cloud.storage.hmac_key.HMACKeyMetadata`
    :returns: The next key metadata instance in the page.
    """
    metadata = HMACKeyMetadata(iterator.client)
    metadata._properties = item
    return metadata
