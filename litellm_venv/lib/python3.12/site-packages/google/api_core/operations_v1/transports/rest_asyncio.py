# -*- coding: utf-8 -*-
# Copyright 2024 Google LLC
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
#

import json
from typing import Any, Callable, Coroutine, Dict, Optional, Sequence, Tuple

from google.auth import __version__ as auth_version

try:
    from google.auth.aio.transport.sessions import AsyncAuthorizedSession  # type: ignore
except ImportError as e:  # pragma: NO COVER
    raise ImportError(
        "The `async_rest` extra of `google-api-core` is required to use long-running operations.  Install it by running "
        "`pip install google-api-core[async_rest]`."
    ) from e

from google.api_core import exceptions as core_exceptions  # type: ignore
from google.api_core import gapic_v1  # type: ignore
from google.api_core import path_template  # type: ignore
from google.api_core import rest_helpers  # type: ignore
from google.api_core import retry_async as retries_async  # type: ignore
from google.auth.aio import credentials as ga_credentials_async  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import empty_pb2  # type: ignore
from google.protobuf import json_format  # type: ignore

from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO, OperationsTransport

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=BASE_DEFAULT_CLIENT_INFO.gapic_version,
    grpc_version=None,
    rest_version=f"google-auth@{auth_version}",
)


class AsyncOperationsRestTransport(OperationsTransport):
    """Asynchronous REST backend transport for Operations.

    Manages async long-running operations with an API service.

    When an API method normally takes long time to complete, it can be
    designed to return [Operation][google.api_core.operations_v1.Operation] to the
    client, and the client can use this interface to receive the real
    response asynchronously by polling the operation resource, or pass
    the operation resource to another API (such as Google Cloud Pub/Sub
    API) to receive the response. Any API service that returns
    long-running operations should implement the ``Operations``
    interface so developers can have a consistent client experience.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1
    """

    def __init__(
        self,
        *,
        host: str = "longrunning.googleapis.com",
        credentials: Optional[ga_credentials_async.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        url_scheme: str = "https",
        http_options: Optional[Dict] = None,
        path_prefix: str = "v1",
        # TODO(https://github.com/googleapis/python-api-core/issues/715): Add docstring for `credentials_file` to async REST transport.
        # TODO(https://github.com/googleapis/python-api-core/issues/716): Add docstring for `scopes` to async REST transport.
        # TODO(https://github.com/googleapis/python-api-core/issues/717): Add docstring for `quota_project_id` to async REST transport.
        # TODO(https://github.com/googleapis/python-api-core/issues/718): Add docstring for `client_cert_source` to async REST transport.
    ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to.
            credentials (Optional[google.auth.aio.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.
            url_scheme: the protocol scheme for the API endpoint.  Normally
                "https", but for testing or local servers,
                "http" can be specified.
            http_options: a dictionary of http_options for transcoding, to override
                the defaults from operations.proto.  Each method has an entry
                with the corresponding http rules as value.
            path_prefix: path prefix (usually represents API version). Set to
                "v1" by default.

        """
        unsupported_params = {
            # TODO(https://github.com/googleapis/python-api-core/issues/715): Add support for `credentials_file` to async REST transport.
            "google.api_core.client_options.ClientOptions.credentials_file": credentials_file,
            # TODO(https://github.com/googleapis/python-api-core/issues/716): Add support for `scopes` to async REST transport.
            "google.api_core.client_options.ClientOptions.scopes": scopes,
            # TODO(https://github.com/googleapis/python-api-core/issues/717): Add support for `quota_project_id` to async REST transport.
            "google.api_core.client_options.ClientOptions.quota_project_id": quota_project_id,
            # TODO(https://github.com/googleapis/python-api-core/issues/718): Add support for `client_cert_source` to async REST transport.
            "google.api_core.client_options.ClientOptions.client_cert_source": client_cert_source_for_mtls,
            # TODO(https://github.com/googleapis/python-api-core/issues/718): Add support for `client_cert_source` to async REST transport.
            "google.api_core.client_options.ClientOptions.client_cert_source": client_cert_source_for_mtls,
        }
        provided_unsupported_params = [
            name for name, value in unsupported_params.items() if value is not None
        ]
        if provided_unsupported_params:
            raise core_exceptions.AsyncRestUnsupportedParameterError(
                f"The following provided parameters are not supported for `transport=rest_asyncio`: {', '.join(provided_unsupported_params)}"
            )

        super().__init__(
            host=host,
            # TODO(https://github.com/googleapis/python-api-core/issues/709): Remove `type: ignore` when the linked issue is resolved.
            credentials=credentials,  # type: ignore
            client_info=client_info,
            # TODO(https://github.com/googleapis/python-api-core/issues/725): Set always_use_jwt_access token when supported.
            always_use_jwt_access=False,
        )
        # TODO(https://github.com/googleapis/python-api-core/issues/708): add support for
        # `default_host` in AsyncAuthorizedSession for feature parity with the synchronous
        # code.
        # TODO(https://github.com/googleapis/python-api-core/issues/709): Remove `type: ignore` when the linked issue is resolved.
        self._session = AsyncAuthorizedSession(self._credentials)  # type: ignore
        # TODO(https://github.com/googleapis/python-api-core/issues/720): Add wrap logic directly to the property methods for callables.
        self._prep_wrapped_messages(client_info)
        self._http_options = http_options or {}
        self._path_prefix = path_prefix

    def _prep_wrapped_messages(self, client_info):
        # Precompute the wrapped methods.
        self._wrapped_methods = {
            self.list_operations: gapic_v1.method_async.wrap_method(
                self.list_operations,
                default_retry=retries_async.AsyncRetry(
                    initial=0.5,
                    maximum=10.0,
                    multiplier=2.0,
                    predicate=retries_async.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=10.0,
                ),
                default_timeout=10.0,
                client_info=client_info,
                kind="rest_asyncio",
            ),
            self.get_operation: gapic_v1.method_async.wrap_method(
                self.get_operation,
                default_retry=retries_async.AsyncRetry(
                    initial=0.5,
                    maximum=10.0,
                    multiplier=2.0,
                    predicate=retries_async.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=10.0,
                ),
                default_timeout=10.0,
                client_info=client_info,
                kind="rest_asyncio",
            ),
            self.delete_operation: gapic_v1.method_async.wrap_method(
                self.delete_operation,
                default_retry=retries_async.AsyncRetry(
                    initial=0.5,
                    maximum=10.0,
                    multiplier=2.0,
                    predicate=retries_async.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=10.0,
                ),
                default_timeout=10.0,
                client_info=client_info,
                kind="rest_asyncio",
            ),
            self.cancel_operation: gapic_v1.method_async.wrap_method(
                self.cancel_operation,
                default_retry=retries_async.AsyncRetry(
                    initial=0.5,
                    maximum=10.0,
                    multiplier=2.0,
                    predicate=retries_async.if_exception_type(
                        core_exceptions.ServiceUnavailable,
                    ),
                    deadline=10.0,
                ),
                default_timeout=10.0,
                client_info=client_info,
                kind="rest_asyncio",
            ),
        }

    async def _list_operations(
        self,
        request: operations_pb2.ListOperationsRequest,
        *,
        # TODO(https://github.com/googleapis/python-api-core/issues/722): Leverage `retry`
        # to allow configuring retryable error codes.
        retry=gapic_v1.method_async.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> operations_pb2.ListOperationsResponse:
        r"""Asynchronously call the list operations method over HTTP.

        Args:
            request (~.operations_pb2.ListOperationsRequest):
                The request object. The request message for
                [Operations.ListOperations][google.api_core.operations_v1.Operations.ListOperations].
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            ~.operations_pb2.ListOperationsResponse:
                The response message for
                [Operations.ListOperations][google.api_core.operations_v1.Operations.ListOperations].

        """

        http_options = [
            {
                "method": "get",
                "uri": "/{}/{{name=**}}/operations".format(self._path_prefix),
            },
        ]
        if "google.longrunning.Operations.ListOperations" in self._http_options:
            http_options = self._http_options[
                "google.longrunning.Operations.ListOperations"
            ]

        request_kwargs = self._convert_protobuf_message_to_dict(request)
        transcoded_request = path_template.transcode(http_options, **request_kwargs)

        uri = transcoded_request["uri"]
        method = transcoded_request["method"]

        # Jsonify the query params
        query_params_request = operations_pb2.ListOperationsRequest()
        json_format.ParseDict(transcoded_request["query_params"], query_params_request)
        query_params = json_format.MessageToDict(
            query_params_request,
            preserving_proto_field_name=False,
            use_integers_for_enums=False,
        )

        # Send the request
        headers = dict(metadata)
        headers["Content-Type"] = "application/json"
        # TODO(https://github.com/googleapis/python-api-core/issues/721): Update incorrect use of `uri`` variable name.
        response = await getattr(self._session, method)(
            "{host}{uri}".format(host=self._host, uri=uri),
            timeout=timeout,
            headers=headers,
            params=rest_helpers.flatten_query_params(query_params),
        )
        content = await response.read()

        # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
        # subclass.
        if response.status_code >= 400:
            payload = json.loads(content.decode("utf-8"))
            request_url = "{host}{uri}".format(host=self._host, uri=uri)
            raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

        # Return the response
        api_response = operations_pb2.ListOperationsResponse()
        json_format.Parse(content, api_response, ignore_unknown_fields=False)
        return api_response

    async def _get_operation(
        self,
        request: operations_pb2.GetOperationRequest,
        *,
        # TODO(https://github.com/googleapis/python-api-core/issues/722): Leverage `retry`
        # to allow configuring retryable error codes.
        retry=gapic_v1.method_async.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> operations_pb2.Operation:
        r"""Asynchronously call the get operation method over HTTP.

        Args:
            request (~.operations_pb2.GetOperationRequest):
                The request object. The request message for
                [Operations.GetOperation][google.api_core.operations_v1.Operations.GetOperation].
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            ~.operations_pb2.Operation:
                This resource represents a long-
                running operation that is the result of a
                network API call.

        """

        http_options = [
            {
                "method": "get",
                "uri": "/{}/{{name=**/operations/*}}".format(self._path_prefix),
            },
        ]
        if "google.longrunning.Operations.GetOperation" in self._http_options:
            http_options = self._http_options[
                "google.longrunning.Operations.GetOperation"
            ]

        request_kwargs = self._convert_protobuf_message_to_dict(request)
        transcoded_request = path_template.transcode(http_options, **request_kwargs)

        uri = transcoded_request["uri"]
        method = transcoded_request["method"]

        # Jsonify the query params
        query_params_request = operations_pb2.GetOperationRequest()
        json_format.ParseDict(transcoded_request["query_params"], query_params_request)
        query_params = json_format.MessageToDict(
            query_params_request,
            preserving_proto_field_name=False,
            use_integers_for_enums=False,
        )

        # Send the request
        headers = dict(metadata)
        headers["Content-Type"] = "application/json"
        # TODO(https://github.com/googleapis/python-api-core/issues/721): Update incorrect use of `uri`` variable name.
        response = await getattr(self._session, method)(
            "{host}{uri}".format(host=self._host, uri=uri),
            timeout=timeout,
            headers=headers,
            params=rest_helpers.flatten_query_params(query_params),
        )
        content = await response.read()

        # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
        # subclass.
        if response.status_code >= 400:
            payload = json.loads(content.decode("utf-8"))
            request_url = "{host}{uri}".format(host=self._host, uri=uri)
            raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

        # Return the response
        api_response = operations_pb2.Operation()
        json_format.Parse(content, api_response, ignore_unknown_fields=False)
        return api_response

    async def _delete_operation(
        self,
        request: operations_pb2.DeleteOperationRequest,
        *,
        # TODO(https://github.com/googleapis/python-api-core/issues/722): Leverage `retry`
        # to allow configuring retryable error codes.
        retry=gapic_v1.method_async.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> empty_pb2.Empty:
        r"""Asynchronously call the delete operation method over HTTP.

        Args:
            request (~.operations_pb2.DeleteOperationRequest):
                The request object. The request message for
                [Operations.DeleteOperation][google.api_core.operations_v1.Operations.DeleteOperation].

            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """

        http_options = [
            {
                "method": "delete",
                "uri": "/{}/{{name=**/operations/*}}".format(self._path_prefix),
            },
        ]
        if "google.longrunning.Operations.DeleteOperation" in self._http_options:
            http_options = self._http_options[
                "google.longrunning.Operations.DeleteOperation"
            ]

        request_kwargs = self._convert_protobuf_message_to_dict(request)
        transcoded_request = path_template.transcode(http_options, **request_kwargs)

        uri = transcoded_request["uri"]
        method = transcoded_request["method"]

        # Jsonify the query params
        query_params_request = operations_pb2.DeleteOperationRequest()
        json_format.ParseDict(transcoded_request["query_params"], query_params_request)
        query_params = json_format.MessageToDict(
            query_params_request,
            preserving_proto_field_name=False,
            use_integers_for_enums=False,
        )

        # Send the request
        headers = dict(metadata)
        headers["Content-Type"] = "application/json"
        # TODO(https://github.com/googleapis/python-api-core/issues/721): Update incorrect use of `uri`` variable name.
        response = await getattr(self._session, method)(
            "{host}{uri}".format(host=self._host, uri=uri),
            timeout=timeout,
            headers=headers,
            params=rest_helpers.flatten_query_params(query_params),
        )

        # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
        # subclass.
        if response.status_code >= 400:
            content = await response.read()
            payload = json.loads(content.decode("utf-8"))
            request_url = "{host}{uri}".format(host=self._host, uri=uri)
            raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

        return empty_pb2.Empty()

    async def _cancel_operation(
        self,
        request: operations_pb2.CancelOperationRequest,
        *,
        # TODO(https://github.com/googleapis/python-api-core/issues/722): Leverage `retry`
        # to allow configuring retryable error codes.
        retry=gapic_v1.method_async.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
        # TODO(https://github.com/googleapis/python-api-core/issues/722): Add `retry` parameter
        # to allow configuring retryable error codes.
    ) -> empty_pb2.Empty:
        r"""Asynchronously call the cancel operation method over HTTP.

        Args:
            request (~.operations_pb2.CancelOperationRequest):
                The request object. The request message for
                [Operations.CancelOperation][google.api_core.operations_v1.Operations.CancelOperation].
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """

        http_options = [
            {
                "method": "post",
                "uri": "/{}/{{name=**/operations/*}}:cancel".format(self._path_prefix),
                "body": "*",
            },
        ]
        if "google.longrunning.Operations.CancelOperation" in self._http_options:
            http_options = self._http_options[
                "google.longrunning.Operations.CancelOperation"
            ]

        request_kwargs = self._convert_protobuf_message_to_dict(request)
        transcoded_request = path_template.transcode(http_options, **request_kwargs)

        # Jsonify the request body
        body_request = operations_pb2.CancelOperationRequest()
        json_format.ParseDict(transcoded_request["body"], body_request)
        body = json_format.MessageToDict(
            body_request,
            preserving_proto_field_name=False,
            use_integers_for_enums=False,
        )
        uri = transcoded_request["uri"]
        method = transcoded_request["method"]

        # Jsonify the query params
        query_params_request = operations_pb2.CancelOperationRequest()
        json_format.ParseDict(transcoded_request["query_params"], query_params_request)
        query_params = json_format.MessageToDict(
            query_params_request,
            preserving_proto_field_name=False,
            use_integers_for_enums=False,
        )

        # Send the request
        headers = dict(metadata)
        headers["Content-Type"] = "application/json"
        # TODO(https://github.com/googleapis/python-api-core/issues/721): Update incorrect use of `uri`` variable name.
        response = await getattr(self._session, method)(
            "{host}{uri}".format(host=self._host, uri=uri),
            timeout=timeout,
            headers=headers,
            params=rest_helpers.flatten_query_params(query_params),
            data=body,
        )

        # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
        # subclass.
        if response.status_code >= 400:
            content = await response.read()
            payload = json.loads(content.decode("utf-8"))
            request_url = "{host}{uri}".format(host=self._host, uri=uri)
            raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

        return empty_pb2.Empty()

    @property
    def list_operations(
        self,
    ) -> Callable[
        [operations_pb2.ListOperationsRequest],
        Coroutine[Any, Any, operations_pb2.ListOperationsResponse],
    ]:
        return self._list_operations

    @property
    def get_operation(
        self,
    ) -> Callable[
        [operations_pb2.GetOperationRequest],
        Coroutine[Any, Any, operations_pb2.Operation],
    ]:
        return self._get_operation

    @property
    def delete_operation(
        self,
    ) -> Callable[
        [operations_pb2.DeleteOperationRequest], Coroutine[Any, Any, empty_pb2.Empty]
    ]:
        return self._delete_operation

    @property
    def cancel_operation(
        self,
    ) -> Callable[
        [operations_pb2.CancelOperationRequest], Coroutine[Any, Any, empty_pb2.Empty]
    ]:
        return self._cancel_operation


__all__ = ("AsyncOperationsRestTransport",)
