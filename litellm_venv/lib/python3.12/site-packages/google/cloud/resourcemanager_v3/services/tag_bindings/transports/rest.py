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
import dataclasses
import json  # type: ignore
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings

from google.api_core import gapic_v1, operations_v1, rest_helpers, rest_streaming
from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.requests import AuthorizedSession  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import json_format
from requests import __version__ as requests_version

from google.cloud.resourcemanager_v3.types import tag_bindings

from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .rest_base import _BaseTagBindingsRestTransport

try:
    OptionalRetry = Union[retries.Retry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.Retry, object, None]  # type: ignore

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = logging.getLogger(__name__)

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=BASE_DEFAULT_CLIENT_INFO.gapic_version,
    grpc_version=None,
    rest_version=f"requests@{requests_version}",
)


class TagBindingsRestInterceptor:
    """Interceptor for TagBindings.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the TagBindingsRestTransport.

    .. code-block:: python
        class MyCustomTagBindingsInterceptor(TagBindingsRestInterceptor):
            def pre_create_tag_binding(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_tag_binding(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_tag_binding(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_delete_tag_binding(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_effective_tags(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_effective_tags(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_tag_bindings(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_tag_bindings(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = TagBindingsRestTransport(interceptor=MyCustomTagBindingsInterceptor())
        client = TagBindingsClient(transport=transport)


    """

    def pre_create_tag_binding(
        self,
        request: tag_bindings.CreateTagBindingRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        tag_bindings.CreateTagBindingRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for create_tag_binding

        Override in a subclass to manipulate the request or metadata
        before they are sent to the TagBindings server.
        """
        return request, metadata

    def post_create_tag_binding(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for create_tag_binding

        DEPRECATED. Please use the `post_create_tag_binding_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the TagBindings server but before
        it is returned to user code. This `post_create_tag_binding` interceptor runs
        before the `post_create_tag_binding_with_metadata` interceptor.
        """
        return response

    def post_create_tag_binding_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_tag_binding

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the TagBindings server but before it is returned to user code.

        We recommend only using this `post_create_tag_binding_with_metadata`
        interceptor in new development instead of the `post_create_tag_binding` interceptor.
        When both interceptors are used, this `post_create_tag_binding_with_metadata` interceptor runs after the
        `post_create_tag_binding` interceptor. The (possibly modified) response returned by
        `post_create_tag_binding` will be passed to
        `post_create_tag_binding_with_metadata`.
        """
        return response, metadata

    def pre_delete_tag_binding(
        self,
        request: tag_bindings.DeleteTagBindingRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        tag_bindings.DeleteTagBindingRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for delete_tag_binding

        Override in a subclass to manipulate the request or metadata
        before they are sent to the TagBindings server.
        """
        return request, metadata

    def post_delete_tag_binding(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for delete_tag_binding

        DEPRECATED. Please use the `post_delete_tag_binding_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the TagBindings server but before
        it is returned to user code. This `post_delete_tag_binding` interceptor runs
        before the `post_delete_tag_binding_with_metadata` interceptor.
        """
        return response

    def post_delete_tag_binding_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for delete_tag_binding

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the TagBindings server but before it is returned to user code.

        We recommend only using this `post_delete_tag_binding_with_metadata`
        interceptor in new development instead of the `post_delete_tag_binding` interceptor.
        When both interceptors are used, this `post_delete_tag_binding_with_metadata` interceptor runs after the
        `post_delete_tag_binding` interceptor. The (possibly modified) response returned by
        `post_delete_tag_binding` will be passed to
        `post_delete_tag_binding_with_metadata`.
        """
        return response, metadata

    def pre_list_effective_tags(
        self,
        request: tag_bindings.ListEffectiveTagsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        tag_bindings.ListEffectiveTagsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for list_effective_tags

        Override in a subclass to manipulate the request or metadata
        before they are sent to the TagBindings server.
        """
        return request, metadata

    def post_list_effective_tags(
        self, response: tag_bindings.ListEffectiveTagsResponse
    ) -> tag_bindings.ListEffectiveTagsResponse:
        """Post-rpc interceptor for list_effective_tags

        DEPRECATED. Please use the `post_list_effective_tags_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the TagBindings server but before
        it is returned to user code. This `post_list_effective_tags` interceptor runs
        before the `post_list_effective_tags_with_metadata` interceptor.
        """
        return response

    def post_list_effective_tags_with_metadata(
        self,
        response: tag_bindings.ListEffectiveTagsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        tag_bindings.ListEffectiveTagsResponse, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for list_effective_tags

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the TagBindings server but before it is returned to user code.

        We recommend only using this `post_list_effective_tags_with_metadata`
        interceptor in new development instead of the `post_list_effective_tags` interceptor.
        When both interceptors are used, this `post_list_effective_tags_with_metadata` interceptor runs after the
        `post_list_effective_tags` interceptor. The (possibly modified) response returned by
        `post_list_effective_tags` will be passed to
        `post_list_effective_tags_with_metadata`.
        """
        return response, metadata

    def pre_list_tag_bindings(
        self,
        request: tag_bindings.ListTagBindingsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        tag_bindings.ListTagBindingsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for list_tag_bindings

        Override in a subclass to manipulate the request or metadata
        before they are sent to the TagBindings server.
        """
        return request, metadata

    def post_list_tag_bindings(
        self, response: tag_bindings.ListTagBindingsResponse
    ) -> tag_bindings.ListTagBindingsResponse:
        """Post-rpc interceptor for list_tag_bindings

        DEPRECATED. Please use the `post_list_tag_bindings_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the TagBindings server but before
        it is returned to user code. This `post_list_tag_bindings` interceptor runs
        before the `post_list_tag_bindings_with_metadata` interceptor.
        """
        return response

    def post_list_tag_bindings_with_metadata(
        self,
        response: tag_bindings.ListTagBindingsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        tag_bindings.ListTagBindingsResponse, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for list_tag_bindings

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the TagBindings server but before it is returned to user code.

        We recommend only using this `post_list_tag_bindings_with_metadata`
        interceptor in new development instead of the `post_list_tag_bindings` interceptor.
        When both interceptors are used, this `post_list_tag_bindings_with_metadata` interceptor runs after the
        `post_list_tag_bindings` interceptor. The (possibly modified) response returned by
        `post_list_tag_bindings` will be passed to
        `post_list_tag_bindings_with_metadata`.
        """
        return response, metadata

    def pre_get_operation(
        self,
        request: operations_pb2.GetOperationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.GetOperationRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_operation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the TagBindings server.
        """
        return request, metadata

    def post_get_operation(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for get_operation

        Override in a subclass to manipulate the response
        after it is returned by the TagBindings server but before
        it is returned to user code.
        """
        return response


@dataclasses.dataclass
class TagBindingsRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: TagBindingsRestInterceptor


class TagBindingsRestTransport(_BaseTagBindingsRestTransport):
    """REST backend synchronous transport for TagBindings.

    Allow users to create and manage TagBindings between
    TagValues and different Google Cloud resources throughout the
    GCP resource hierarchy.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1
    """

    def __init__(
        self,
        *,
        host: str = "cloudresourcemanager.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        url_scheme: str = "https",
        interceptor: Optional[TagBindingsRestInterceptor] = None,
        api_audience: Optional[str] = None,
    ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to (default: 'cloudresourcemanager.googleapis.com').
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.

            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if ``channel`` is provided.
            scopes (Optional(Sequence[str])): A list of scopes. This argument is
                ignored if ``channel`` is provided.
            client_cert_source_for_mtls (Callable[[], Tuple[bytes, bytes]]): Client
                certificate to configure mutual TLS HTTP channel. It is ignored
                if ``channel`` is provided.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you are developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.
            url_scheme: the protocol scheme for the API endpoint.  Normally
                "https", but for testing or local servers,
                "http" can be specified.
        """
        # Run the base constructor
        # TODO(yon-mg): resolve other ctor params i.e. scopes, quota, etc.
        # TODO: When custom host (api_endpoint) is set, `scopes` must *also* be set on the
        # credentials object
        super().__init__(
            host=host,
            credentials=credentials,
            client_info=client_info,
            always_use_jwt_access=always_use_jwt_access,
            url_scheme=url_scheme,
            api_audience=api_audience,
        )
        self._session = AuthorizedSession(
            self._credentials, default_host=self.DEFAULT_HOST
        )
        self._operations_client: Optional[operations_v1.AbstractOperationsClient] = None
        if client_cert_source_for_mtls:
            self._session.configure_mtls_channel(client_cert_source_for_mtls)
        self._interceptor = interceptor or TagBindingsRestInterceptor()
        self._prep_wrapped_messages(client_info)

    @property
    def operations_client(self) -> operations_v1.AbstractOperationsClient:
        """Create the client designed to process long-running operations.

        This property caches on the instance; repeated calls return the same
        client.
        """
        # Only create a new client if we do not already have one.
        if self._operations_client is None:
            http_options: Dict[str, List[Dict[str, str]]] = {
                "google.longrunning.Operations.GetOperation": [
                    {
                        "method": "get",
                        "uri": "/v3/{name=operations/**}",
                    },
                ],
            }

            rest_transport = operations_v1.OperationsRestTransport(
                host=self._host,
                # use the credentials which are saved
                credentials=self._credentials,
                scopes=self._scopes,
                http_options=http_options,
                path_prefix="v3",
            )

            self._operations_client = operations_v1.AbstractOperationsClient(
                transport=rest_transport
            )

        # Return the client from cache.
        return self._operations_client

    class _CreateTagBinding(
        _BaseTagBindingsRestTransport._BaseCreateTagBinding, TagBindingsRestStub
    ):
        def __hash__(self):
            return hash("TagBindingsRestTransport.CreateTagBinding")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: tag_bindings.CreateTagBindingRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the create tag binding method over HTTP.

            Args:
                request (~.tag_bindings.CreateTagBindingRequest):
                    The request object. The request message to create a
                TagBinding.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.operations_pb2.Operation:
                    This resource represents a
                long-running operation that is the
                result of a network API call.

            """

            http_options = (
                _BaseTagBindingsRestTransport._BaseCreateTagBinding._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_tag_binding(
                request, metadata
            )
            transcoded_request = _BaseTagBindingsRestTransport._BaseCreateTagBinding._get_transcoded_request(
                http_options, request
            )

            body = _BaseTagBindingsRestTransport._BaseCreateTagBinding._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseTagBindingsRestTransport._BaseCreateTagBinding._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.resourcemanager_v3.TagBindingsClient.CreateTagBinding",
                    extra={
                        "serviceName": "google.cloud.resourcemanager.v3.TagBindings",
                        "rpcName": "CreateTagBinding",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = TagBindingsRestTransport._CreateTagBinding._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = operations_pb2.Operation()
            json_format.Parse(response.content, resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_create_tag_binding(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_tag_binding_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.resourcemanager_v3.TagBindingsClient.create_tag_binding",
                    extra={
                        "serviceName": "google.cloud.resourcemanager.v3.TagBindings",
                        "rpcName": "CreateTagBinding",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _DeleteTagBinding(
        _BaseTagBindingsRestTransport._BaseDeleteTagBinding, TagBindingsRestStub
    ):
        def __hash__(self):
            return hash("TagBindingsRestTransport.DeleteTagBinding")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: tag_bindings.DeleteTagBindingRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the delete tag binding method over HTTP.

            Args:
                request (~.tag_bindings.DeleteTagBindingRequest):
                    The request object. The request message to delete a
                TagBinding.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.operations_pb2.Operation:
                    This resource represents a
                long-running operation that is the
                result of a network API call.

            """

            http_options = (
                _BaseTagBindingsRestTransport._BaseDeleteTagBinding._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_tag_binding(
                request, metadata
            )
            transcoded_request = _BaseTagBindingsRestTransport._BaseDeleteTagBinding._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseTagBindingsRestTransport._BaseDeleteTagBinding._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.resourcemanager_v3.TagBindingsClient.DeleteTagBinding",
                    extra={
                        "serviceName": "google.cloud.resourcemanager.v3.TagBindings",
                        "rpcName": "DeleteTagBinding",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = TagBindingsRestTransport._DeleteTagBinding._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = operations_pb2.Operation()
            json_format.Parse(response.content, resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_delete_tag_binding(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_delete_tag_binding_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.resourcemanager_v3.TagBindingsClient.delete_tag_binding",
                    extra={
                        "serviceName": "google.cloud.resourcemanager.v3.TagBindings",
                        "rpcName": "DeleteTagBinding",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListEffectiveTags(
        _BaseTagBindingsRestTransport._BaseListEffectiveTags, TagBindingsRestStub
    ):
        def __hash__(self):
            return hash("TagBindingsRestTransport.ListEffectiveTags")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: tag_bindings.ListEffectiveTagsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> tag_bindings.ListEffectiveTagsResponse:
            r"""Call the list effective tags method over HTTP.

            Args:
                request (~.tag_bindings.ListEffectiveTagsRequest):
                    The request object. The request message to
                ListEffectiveTags
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.tag_bindings.ListEffectiveTagsResponse:
                    The response of ListEffectiveTags.
            """

            http_options = (
                _BaseTagBindingsRestTransport._BaseListEffectiveTags._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_effective_tags(
                request, metadata
            )
            transcoded_request = _BaseTagBindingsRestTransport._BaseListEffectiveTags._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseTagBindingsRestTransport._BaseListEffectiveTags._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.resourcemanager_v3.TagBindingsClient.ListEffectiveTags",
                    extra={
                        "serviceName": "google.cloud.resourcemanager.v3.TagBindings",
                        "rpcName": "ListEffectiveTags",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = TagBindingsRestTransport._ListEffectiveTags._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = tag_bindings.ListEffectiveTagsResponse()
            pb_resp = tag_bindings.ListEffectiveTagsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_effective_tags(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_effective_tags_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = tag_bindings.ListEffectiveTagsResponse.to_json(
                        response
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.resourcemanager_v3.TagBindingsClient.list_effective_tags",
                    extra={
                        "serviceName": "google.cloud.resourcemanager.v3.TagBindings",
                        "rpcName": "ListEffectiveTags",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListTagBindings(
        _BaseTagBindingsRestTransport._BaseListTagBindings, TagBindingsRestStub
    ):
        def __hash__(self):
            return hash("TagBindingsRestTransport.ListTagBindings")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: tag_bindings.ListTagBindingsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> tag_bindings.ListTagBindingsResponse:
            r"""Call the list tag bindings method over HTTP.

            Args:
                request (~.tag_bindings.ListTagBindingsRequest):
                    The request object. The request message to list all
                TagBindings for a parent.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.tag_bindings.ListTagBindingsResponse:
                    The ListTagBindings response.
            """

            http_options = (
                _BaseTagBindingsRestTransport._BaseListTagBindings._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_tag_bindings(
                request, metadata
            )
            transcoded_request = _BaseTagBindingsRestTransport._BaseListTagBindings._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseTagBindingsRestTransport._BaseListTagBindings._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.resourcemanager_v3.TagBindingsClient.ListTagBindings",
                    extra={
                        "serviceName": "google.cloud.resourcemanager.v3.TagBindings",
                        "rpcName": "ListTagBindings",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = TagBindingsRestTransport._ListTagBindings._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = tag_bindings.ListTagBindingsResponse()
            pb_resp = tag_bindings.ListTagBindingsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_tag_bindings(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_tag_bindings_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = tag_bindings.ListTagBindingsResponse.to_json(
                        response
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.resourcemanager_v3.TagBindingsClient.list_tag_bindings",
                    extra={
                        "serviceName": "google.cloud.resourcemanager.v3.TagBindings",
                        "rpcName": "ListTagBindings",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    @property
    def create_tag_binding(
        self,
    ) -> Callable[[tag_bindings.CreateTagBindingRequest], operations_pb2.Operation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateTagBinding(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_tag_binding(
        self,
    ) -> Callable[[tag_bindings.DeleteTagBindingRequest], operations_pb2.Operation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteTagBinding(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_effective_tags(
        self,
    ) -> Callable[
        [tag_bindings.ListEffectiveTagsRequest], tag_bindings.ListEffectiveTagsResponse
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListEffectiveTags(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_tag_bindings(
        self,
    ) -> Callable[
        [tag_bindings.ListTagBindingsRequest], tag_bindings.ListTagBindingsResponse
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListTagBindings(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_operation(self):
        return self._GetOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _GetOperation(
        _BaseTagBindingsRestTransport._BaseGetOperation, TagBindingsRestStub
    ):
        def __hash__(self):
            return hash("TagBindingsRestTransport.GetOperation")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: operations_pb2.GetOperationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the get operation method over HTTP.

            Args:
                request (operations_pb2.GetOperationRequest):
                    The request object for GetOperation method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                operations_pb2.Operation: Response from GetOperation method.
            """

            http_options = (
                _BaseTagBindingsRestTransport._BaseGetOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_operation(request, metadata)
            transcoded_request = (
                _BaseTagBindingsRestTransport._BaseGetOperation._get_transcoded_request(
                    http_options, request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseTagBindingsRestTransport._BaseGetOperation._get_query_params_json(
                    transcoded_request
                )
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.resourcemanager_v3.TagBindingsClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.resourcemanager.v3.TagBindings",
                        "rpcName": "GetOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = TagBindingsRestTransport._GetOperation._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            content = response.content.decode("utf-8")
            resp = operations_pb2.Operation()
            resp = json_format.Parse(content, resp)
            resp = self._interceptor.post_get_operation(resp)
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.resourcemanager_v3.TagBindingsAsyncClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.resourcemanager.v3.TagBindings",
                        "rpcName": "GetOperation",
                        "httpResponse": http_response,
                        "metadata": http_response["headers"],
                    },
                )
            return resp

    @property
    def kind(self) -> str:
        return "rest"

    def close(self):
        self._session.close()


__all__ = ("TagBindingsRestTransport",)
