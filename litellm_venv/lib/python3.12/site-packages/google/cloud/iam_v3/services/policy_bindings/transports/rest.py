# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
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
from google.cloud.location import locations_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
import google.protobuf
from google.protobuf import json_format
from requests import __version__ as requests_version

from google.cloud.iam_v3.types import policy_binding_resources, policy_bindings_service

from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .rest_base import _BasePolicyBindingsRestTransport

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

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


class PolicyBindingsRestInterceptor:
    """Interceptor for PolicyBindings.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the PolicyBindingsRestTransport.

    .. code-block:: python
        class MyCustomPolicyBindingsInterceptor(PolicyBindingsRestInterceptor):
            def pre_create_policy_binding(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_policy_binding(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_policy_binding(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_delete_policy_binding(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_policy_binding(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_policy_binding(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_policy_bindings(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_policy_bindings(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_search_target_policy_bindings(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_search_target_policy_bindings(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_policy_binding(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_policy_binding(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = PolicyBindingsRestTransport(interceptor=MyCustomPolicyBindingsInterceptor())
        client = PolicyBindingsClient(transport=transport)


    """

    def pre_create_policy_binding(
        self,
        request: policy_bindings_service.CreatePolicyBindingRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        policy_bindings_service.CreatePolicyBindingRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for create_policy_binding

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PolicyBindings server.
        """
        return request, metadata

    def post_create_policy_binding(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for create_policy_binding

        DEPRECATED. Please use the `post_create_policy_binding_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PolicyBindings server but before
        it is returned to user code. This `post_create_policy_binding` interceptor runs
        before the `post_create_policy_binding_with_metadata` interceptor.
        """
        return response

    def post_create_policy_binding_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_policy_binding

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PolicyBindings server but before it is returned to user code.

        We recommend only using this `post_create_policy_binding_with_metadata`
        interceptor in new development instead of the `post_create_policy_binding` interceptor.
        When both interceptors are used, this `post_create_policy_binding_with_metadata` interceptor runs after the
        `post_create_policy_binding` interceptor. The (possibly modified) response returned by
        `post_create_policy_binding` will be passed to
        `post_create_policy_binding_with_metadata`.
        """
        return response, metadata

    def pre_delete_policy_binding(
        self,
        request: policy_bindings_service.DeletePolicyBindingRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        policy_bindings_service.DeletePolicyBindingRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for delete_policy_binding

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PolicyBindings server.
        """
        return request, metadata

    def post_delete_policy_binding(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for delete_policy_binding

        DEPRECATED. Please use the `post_delete_policy_binding_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PolicyBindings server but before
        it is returned to user code. This `post_delete_policy_binding` interceptor runs
        before the `post_delete_policy_binding_with_metadata` interceptor.
        """
        return response

    def post_delete_policy_binding_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for delete_policy_binding

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PolicyBindings server but before it is returned to user code.

        We recommend only using this `post_delete_policy_binding_with_metadata`
        interceptor in new development instead of the `post_delete_policy_binding` interceptor.
        When both interceptors are used, this `post_delete_policy_binding_with_metadata` interceptor runs after the
        `post_delete_policy_binding` interceptor. The (possibly modified) response returned by
        `post_delete_policy_binding` will be passed to
        `post_delete_policy_binding_with_metadata`.
        """
        return response, metadata

    def pre_get_policy_binding(
        self,
        request: policy_bindings_service.GetPolicyBindingRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        policy_bindings_service.GetPolicyBindingRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_policy_binding

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PolicyBindings server.
        """
        return request, metadata

    def post_get_policy_binding(
        self, response: policy_binding_resources.PolicyBinding
    ) -> policy_binding_resources.PolicyBinding:
        """Post-rpc interceptor for get_policy_binding

        DEPRECATED. Please use the `post_get_policy_binding_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PolicyBindings server but before
        it is returned to user code. This `post_get_policy_binding` interceptor runs
        before the `post_get_policy_binding_with_metadata` interceptor.
        """
        return response

    def post_get_policy_binding_with_metadata(
        self,
        response: policy_binding_resources.PolicyBinding,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        policy_binding_resources.PolicyBinding, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for get_policy_binding

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PolicyBindings server but before it is returned to user code.

        We recommend only using this `post_get_policy_binding_with_metadata`
        interceptor in new development instead of the `post_get_policy_binding` interceptor.
        When both interceptors are used, this `post_get_policy_binding_with_metadata` interceptor runs after the
        `post_get_policy_binding` interceptor. The (possibly modified) response returned by
        `post_get_policy_binding` will be passed to
        `post_get_policy_binding_with_metadata`.
        """
        return response, metadata

    def pre_list_policy_bindings(
        self,
        request: policy_bindings_service.ListPolicyBindingsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        policy_bindings_service.ListPolicyBindingsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_policy_bindings

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PolicyBindings server.
        """
        return request, metadata

    def post_list_policy_bindings(
        self, response: policy_bindings_service.ListPolicyBindingsResponse
    ) -> policy_bindings_service.ListPolicyBindingsResponse:
        """Post-rpc interceptor for list_policy_bindings

        DEPRECATED. Please use the `post_list_policy_bindings_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PolicyBindings server but before
        it is returned to user code. This `post_list_policy_bindings` interceptor runs
        before the `post_list_policy_bindings_with_metadata` interceptor.
        """
        return response

    def post_list_policy_bindings_with_metadata(
        self,
        response: policy_bindings_service.ListPolicyBindingsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        policy_bindings_service.ListPolicyBindingsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_policy_bindings

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PolicyBindings server but before it is returned to user code.

        We recommend only using this `post_list_policy_bindings_with_metadata`
        interceptor in new development instead of the `post_list_policy_bindings` interceptor.
        When both interceptors are used, this `post_list_policy_bindings_with_metadata` interceptor runs after the
        `post_list_policy_bindings` interceptor. The (possibly modified) response returned by
        `post_list_policy_bindings` will be passed to
        `post_list_policy_bindings_with_metadata`.
        """
        return response, metadata

    def pre_search_target_policy_bindings(
        self,
        request: policy_bindings_service.SearchTargetPolicyBindingsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        policy_bindings_service.SearchTargetPolicyBindingsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for search_target_policy_bindings

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PolicyBindings server.
        """
        return request, metadata

    def post_search_target_policy_bindings(
        self, response: policy_bindings_service.SearchTargetPolicyBindingsResponse
    ) -> policy_bindings_service.SearchTargetPolicyBindingsResponse:
        """Post-rpc interceptor for search_target_policy_bindings

        DEPRECATED. Please use the `post_search_target_policy_bindings_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PolicyBindings server but before
        it is returned to user code. This `post_search_target_policy_bindings` interceptor runs
        before the `post_search_target_policy_bindings_with_metadata` interceptor.
        """
        return response

    def post_search_target_policy_bindings_with_metadata(
        self,
        response: policy_bindings_service.SearchTargetPolicyBindingsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        policy_bindings_service.SearchTargetPolicyBindingsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for search_target_policy_bindings

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PolicyBindings server but before it is returned to user code.

        We recommend only using this `post_search_target_policy_bindings_with_metadata`
        interceptor in new development instead of the `post_search_target_policy_bindings` interceptor.
        When both interceptors are used, this `post_search_target_policy_bindings_with_metadata` interceptor runs after the
        `post_search_target_policy_bindings` interceptor. The (possibly modified) response returned by
        `post_search_target_policy_bindings` will be passed to
        `post_search_target_policy_bindings_with_metadata`.
        """
        return response, metadata

    def pre_update_policy_binding(
        self,
        request: policy_bindings_service.UpdatePolicyBindingRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        policy_bindings_service.UpdatePolicyBindingRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for update_policy_binding

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PolicyBindings server.
        """
        return request, metadata

    def post_update_policy_binding(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for update_policy_binding

        DEPRECATED. Please use the `post_update_policy_binding_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PolicyBindings server but before
        it is returned to user code. This `post_update_policy_binding` interceptor runs
        before the `post_update_policy_binding_with_metadata` interceptor.
        """
        return response

    def post_update_policy_binding_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_policy_binding

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PolicyBindings server but before it is returned to user code.

        We recommend only using this `post_update_policy_binding_with_metadata`
        interceptor in new development instead of the `post_update_policy_binding` interceptor.
        When both interceptors are used, this `post_update_policy_binding_with_metadata` interceptor runs after the
        `post_update_policy_binding` interceptor. The (possibly modified) response returned by
        `post_update_policy_binding` will be passed to
        `post_update_policy_binding_with_metadata`.
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
        before they are sent to the PolicyBindings server.
        """
        return request, metadata

    def post_get_operation(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for get_operation

        Override in a subclass to manipulate the response
        after it is returned by the PolicyBindings server but before
        it is returned to user code.
        """
        return response


@dataclasses.dataclass
class PolicyBindingsRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: PolicyBindingsRestInterceptor


class PolicyBindingsRestTransport(_BasePolicyBindingsRestTransport):
    """REST backend synchronous transport for PolicyBindings.

    An interface for managing Identity and Access Management
    (IAM) policy bindings.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1
    """

    def __init__(
        self,
        *,
        host: str = "iam.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        url_scheme: str = "https",
        interceptor: Optional[PolicyBindingsRestInterceptor] = None,
        api_audience: Optional[str] = None,
    ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to (default: 'iam.googleapis.com').
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
        self._interceptor = interceptor or PolicyBindingsRestInterceptor()
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
                        "uri": "/v3/{name=projects/*/locations/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v3/{name=folders/*/locations/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v3/{name=organizations/*/locations/*/operations/*}",
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

    class _CreatePolicyBinding(
        _BasePolicyBindingsRestTransport._BaseCreatePolicyBinding,
        PolicyBindingsRestStub,
    ):
        def __hash__(self):
            return hash("PolicyBindingsRestTransport.CreatePolicyBinding")

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
            request: policy_bindings_service.CreatePolicyBindingRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the create policy binding method over HTTP.

            Args:
                request (~.policy_bindings_service.CreatePolicyBindingRequest):
                    The request object. Request message for
                CreatePolicyBinding method.
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
                _BasePolicyBindingsRestTransport._BaseCreatePolicyBinding._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_policy_binding(
                request, metadata
            )
            transcoded_request = _BasePolicyBindingsRestTransport._BaseCreatePolicyBinding._get_transcoded_request(
                http_options, request
            )

            body = _BasePolicyBindingsRestTransport._BaseCreatePolicyBinding._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BasePolicyBindingsRestTransport._BaseCreatePolicyBinding._get_query_params_json(
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
                    f"Sending request for google.iam_v3.PolicyBindingsClient.CreatePolicyBinding",
                    extra={
                        "serviceName": "google.iam.v3.PolicyBindings",
                        "rpcName": "CreatePolicyBinding",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = PolicyBindingsRestTransport._CreatePolicyBinding._get_response(
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

            resp = self._interceptor.post_create_policy_binding(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_policy_binding_with_metadata(
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
                    "Received response for google.iam_v3.PolicyBindingsClient.create_policy_binding",
                    extra={
                        "serviceName": "google.iam.v3.PolicyBindings",
                        "rpcName": "CreatePolicyBinding",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _DeletePolicyBinding(
        _BasePolicyBindingsRestTransport._BaseDeletePolicyBinding,
        PolicyBindingsRestStub,
    ):
        def __hash__(self):
            return hash("PolicyBindingsRestTransport.DeletePolicyBinding")

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
            request: policy_bindings_service.DeletePolicyBindingRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the delete policy binding method over HTTP.

            Args:
                request (~.policy_bindings_service.DeletePolicyBindingRequest):
                    The request object. Request message for
                DeletePolicyBinding method.
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
                _BasePolicyBindingsRestTransport._BaseDeletePolicyBinding._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_policy_binding(
                request, metadata
            )
            transcoded_request = _BasePolicyBindingsRestTransport._BaseDeletePolicyBinding._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BasePolicyBindingsRestTransport._BaseDeletePolicyBinding._get_query_params_json(
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
                    f"Sending request for google.iam_v3.PolicyBindingsClient.DeletePolicyBinding",
                    extra={
                        "serviceName": "google.iam.v3.PolicyBindings",
                        "rpcName": "DeletePolicyBinding",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = PolicyBindingsRestTransport._DeletePolicyBinding._get_response(
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

            resp = self._interceptor.post_delete_policy_binding(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_delete_policy_binding_with_metadata(
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
                    "Received response for google.iam_v3.PolicyBindingsClient.delete_policy_binding",
                    extra={
                        "serviceName": "google.iam.v3.PolicyBindings",
                        "rpcName": "DeletePolicyBinding",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetPolicyBinding(
        _BasePolicyBindingsRestTransport._BaseGetPolicyBinding, PolicyBindingsRestStub
    ):
        def __hash__(self):
            return hash("PolicyBindingsRestTransport.GetPolicyBinding")

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
            request: policy_bindings_service.GetPolicyBindingRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> policy_binding_resources.PolicyBinding:
            r"""Call the get policy binding method over HTTP.

            Args:
                request (~.policy_bindings_service.GetPolicyBindingRequest):
                    The request object. Request message for GetPolicyBinding
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.policy_binding_resources.PolicyBinding:
                    IAM policy binding resource.
            """

            http_options = (
                _BasePolicyBindingsRestTransport._BaseGetPolicyBinding._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_policy_binding(
                request, metadata
            )
            transcoded_request = _BasePolicyBindingsRestTransport._BaseGetPolicyBinding._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BasePolicyBindingsRestTransport._BaseGetPolicyBinding._get_query_params_json(
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
                    f"Sending request for google.iam_v3.PolicyBindingsClient.GetPolicyBinding",
                    extra={
                        "serviceName": "google.iam.v3.PolicyBindings",
                        "rpcName": "GetPolicyBinding",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = PolicyBindingsRestTransport._GetPolicyBinding._get_response(
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
            resp = policy_binding_resources.PolicyBinding()
            pb_resp = policy_binding_resources.PolicyBinding.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_policy_binding(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_policy_binding_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = policy_binding_resources.PolicyBinding.to_json(
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
                    "Received response for google.iam_v3.PolicyBindingsClient.get_policy_binding",
                    extra={
                        "serviceName": "google.iam.v3.PolicyBindings",
                        "rpcName": "GetPolicyBinding",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListPolicyBindings(
        _BasePolicyBindingsRestTransport._BaseListPolicyBindings, PolicyBindingsRestStub
    ):
        def __hash__(self):
            return hash("PolicyBindingsRestTransport.ListPolicyBindings")

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
            request: policy_bindings_service.ListPolicyBindingsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> policy_bindings_service.ListPolicyBindingsResponse:
            r"""Call the list policy bindings method over HTTP.

            Args:
                request (~.policy_bindings_service.ListPolicyBindingsRequest):
                    The request object. Request message for
                ListPolicyBindings method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.policy_bindings_service.ListPolicyBindingsResponse:
                    Response message for
                ListPolicyBindings method.

            """

            http_options = (
                _BasePolicyBindingsRestTransport._BaseListPolicyBindings._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_policy_bindings(
                request, metadata
            )
            transcoded_request = _BasePolicyBindingsRestTransport._BaseListPolicyBindings._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BasePolicyBindingsRestTransport._BaseListPolicyBindings._get_query_params_json(
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
                    f"Sending request for google.iam_v3.PolicyBindingsClient.ListPolicyBindings",
                    extra={
                        "serviceName": "google.iam.v3.PolicyBindings",
                        "rpcName": "ListPolicyBindings",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = PolicyBindingsRestTransport._ListPolicyBindings._get_response(
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
            resp = policy_bindings_service.ListPolicyBindingsResponse()
            pb_resp = policy_bindings_service.ListPolicyBindingsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_policy_bindings(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_policy_bindings_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        policy_bindings_service.ListPolicyBindingsResponse.to_json(
                            response
                        )
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.iam_v3.PolicyBindingsClient.list_policy_bindings",
                    extra={
                        "serviceName": "google.iam.v3.PolicyBindings",
                        "rpcName": "ListPolicyBindings",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _SearchTargetPolicyBindings(
        _BasePolicyBindingsRestTransport._BaseSearchTargetPolicyBindings,
        PolicyBindingsRestStub,
    ):
        def __hash__(self):
            return hash("PolicyBindingsRestTransport.SearchTargetPolicyBindings")

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
            request: policy_bindings_service.SearchTargetPolicyBindingsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> policy_bindings_service.SearchTargetPolicyBindingsResponse:
            r"""Call the search target policy
            bindings method over HTTP.

                Args:
                    request (~.policy_bindings_service.SearchTargetPolicyBindingsRequest):
                        The request object. Request message for
                    SearchTargetPolicyBindings method.
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.policy_bindings_service.SearchTargetPolicyBindingsResponse:
                        Response message for
                    SearchTargetPolicyBindings method.

            """

            http_options = (
                _BasePolicyBindingsRestTransport._BaseSearchTargetPolicyBindings._get_http_options()
            )

            request, metadata = self._interceptor.pre_search_target_policy_bindings(
                request, metadata
            )
            transcoded_request = _BasePolicyBindingsRestTransport._BaseSearchTargetPolicyBindings._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BasePolicyBindingsRestTransport._BaseSearchTargetPolicyBindings._get_query_params_json(
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
                    f"Sending request for google.iam_v3.PolicyBindingsClient.SearchTargetPolicyBindings",
                    extra={
                        "serviceName": "google.iam.v3.PolicyBindings",
                        "rpcName": "SearchTargetPolicyBindings",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                PolicyBindingsRestTransport._SearchTargetPolicyBindings._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = policy_bindings_service.SearchTargetPolicyBindingsResponse()
            pb_resp = policy_bindings_service.SearchTargetPolicyBindingsResponse.pb(
                resp
            )

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_search_target_policy_bindings(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            (
                resp,
                _,
            ) = self._interceptor.post_search_target_policy_bindings_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = policy_bindings_service.SearchTargetPolicyBindingsResponse.to_json(
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
                    "Received response for google.iam_v3.PolicyBindingsClient.search_target_policy_bindings",
                    extra={
                        "serviceName": "google.iam.v3.PolicyBindings",
                        "rpcName": "SearchTargetPolicyBindings",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdatePolicyBinding(
        _BasePolicyBindingsRestTransport._BaseUpdatePolicyBinding,
        PolicyBindingsRestStub,
    ):
        def __hash__(self):
            return hash("PolicyBindingsRestTransport.UpdatePolicyBinding")

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
            request: policy_bindings_service.UpdatePolicyBindingRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the update policy binding method over HTTP.

            Args:
                request (~.policy_bindings_service.UpdatePolicyBindingRequest):
                    The request object. Request message for
                UpdatePolicyBinding method.
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
                _BasePolicyBindingsRestTransport._BaseUpdatePolicyBinding._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_policy_binding(
                request, metadata
            )
            transcoded_request = _BasePolicyBindingsRestTransport._BaseUpdatePolicyBinding._get_transcoded_request(
                http_options, request
            )

            body = _BasePolicyBindingsRestTransport._BaseUpdatePolicyBinding._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BasePolicyBindingsRestTransport._BaseUpdatePolicyBinding._get_query_params_json(
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
                    f"Sending request for google.iam_v3.PolicyBindingsClient.UpdatePolicyBinding",
                    extra={
                        "serviceName": "google.iam.v3.PolicyBindings",
                        "rpcName": "UpdatePolicyBinding",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = PolicyBindingsRestTransport._UpdatePolicyBinding._get_response(
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

            resp = self._interceptor.post_update_policy_binding(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_policy_binding_with_metadata(
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
                    "Received response for google.iam_v3.PolicyBindingsClient.update_policy_binding",
                    extra={
                        "serviceName": "google.iam.v3.PolicyBindings",
                        "rpcName": "UpdatePolicyBinding",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    @property
    def create_policy_binding(
        self,
    ) -> Callable[
        [policy_bindings_service.CreatePolicyBindingRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreatePolicyBinding(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_policy_binding(
        self,
    ) -> Callable[
        [policy_bindings_service.DeletePolicyBindingRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeletePolicyBinding(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_policy_binding(
        self,
    ) -> Callable[
        [policy_bindings_service.GetPolicyBindingRequest],
        policy_binding_resources.PolicyBinding,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetPolicyBinding(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_policy_bindings(
        self,
    ) -> Callable[
        [policy_bindings_service.ListPolicyBindingsRequest],
        policy_bindings_service.ListPolicyBindingsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListPolicyBindings(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def search_target_policy_bindings(
        self,
    ) -> Callable[
        [policy_bindings_service.SearchTargetPolicyBindingsRequest],
        policy_bindings_service.SearchTargetPolicyBindingsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._SearchTargetPolicyBindings(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_policy_binding(
        self,
    ) -> Callable[
        [policy_bindings_service.UpdatePolicyBindingRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdatePolicyBinding(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_operation(self):
        return self._GetOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _GetOperation(
        _BasePolicyBindingsRestTransport._BaseGetOperation, PolicyBindingsRestStub
    ):
        def __hash__(self):
            return hash("PolicyBindingsRestTransport.GetOperation")

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
                _BasePolicyBindingsRestTransport._BaseGetOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_operation(request, metadata)
            transcoded_request = _BasePolicyBindingsRestTransport._BaseGetOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BasePolicyBindingsRestTransport._BaseGetOperation._get_query_params_json(
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
                    f"Sending request for google.iam_v3.PolicyBindingsClient.GetOperation",
                    extra={
                        "serviceName": "google.iam.v3.PolicyBindings",
                        "rpcName": "GetOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = PolicyBindingsRestTransport._GetOperation._get_response(
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
                    "Received response for google.iam_v3.PolicyBindingsAsyncClient.GetOperation",
                    extra={
                        "serviceName": "google.iam.v3.PolicyBindings",
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


__all__ = ("PolicyBindingsRestTransport",)
