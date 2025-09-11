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

from google.cloud.iam_v3.types import (
    principal_access_boundary_policies_service,
    principal_access_boundary_policy_resources,
)

from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .rest_base import _BasePrincipalAccessBoundaryPoliciesRestTransport

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


class PrincipalAccessBoundaryPoliciesRestInterceptor:
    """Interceptor for PrincipalAccessBoundaryPolicies.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the PrincipalAccessBoundaryPoliciesRestTransport.

    .. code-block:: python
        class MyCustomPrincipalAccessBoundaryPoliciesInterceptor(PrincipalAccessBoundaryPoliciesRestInterceptor):
            def pre_create_principal_access_boundary_policy(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_principal_access_boundary_policy(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_principal_access_boundary_policy(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_delete_principal_access_boundary_policy(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_principal_access_boundary_policy(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_principal_access_boundary_policy(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_principal_access_boundary_policies(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_principal_access_boundary_policies(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_search_principal_access_boundary_policy_bindings(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_search_principal_access_boundary_policy_bindings(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_principal_access_boundary_policy(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_principal_access_boundary_policy(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = PrincipalAccessBoundaryPoliciesRestTransport(interceptor=MyCustomPrincipalAccessBoundaryPoliciesInterceptor())
        client = PrincipalAccessBoundaryPoliciesClient(transport=transport)


    """

    def pre_create_principal_access_boundary_policy(
        self,
        request: principal_access_boundary_policies_service.CreatePrincipalAccessBoundaryPolicyRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        principal_access_boundary_policies_service.CreatePrincipalAccessBoundaryPolicyRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for create_principal_access_boundary_policy

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PrincipalAccessBoundaryPolicies server.
        """
        return request, metadata

    def post_create_principal_access_boundary_policy(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for create_principal_access_boundary_policy

        DEPRECATED. Please use the `post_create_principal_access_boundary_policy_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PrincipalAccessBoundaryPolicies server but before
        it is returned to user code. This `post_create_principal_access_boundary_policy` interceptor runs
        before the `post_create_principal_access_boundary_policy_with_metadata` interceptor.
        """
        return response

    def post_create_principal_access_boundary_policy_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_principal_access_boundary_policy

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PrincipalAccessBoundaryPolicies server but before it is returned to user code.

        We recommend only using this `post_create_principal_access_boundary_policy_with_metadata`
        interceptor in new development instead of the `post_create_principal_access_boundary_policy` interceptor.
        When both interceptors are used, this `post_create_principal_access_boundary_policy_with_metadata` interceptor runs after the
        `post_create_principal_access_boundary_policy` interceptor. The (possibly modified) response returned by
        `post_create_principal_access_boundary_policy` will be passed to
        `post_create_principal_access_boundary_policy_with_metadata`.
        """
        return response, metadata

    def pre_delete_principal_access_boundary_policy(
        self,
        request: principal_access_boundary_policies_service.DeletePrincipalAccessBoundaryPolicyRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        principal_access_boundary_policies_service.DeletePrincipalAccessBoundaryPolicyRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for delete_principal_access_boundary_policy

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PrincipalAccessBoundaryPolicies server.
        """
        return request, metadata

    def post_delete_principal_access_boundary_policy(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for delete_principal_access_boundary_policy

        DEPRECATED. Please use the `post_delete_principal_access_boundary_policy_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PrincipalAccessBoundaryPolicies server but before
        it is returned to user code. This `post_delete_principal_access_boundary_policy` interceptor runs
        before the `post_delete_principal_access_boundary_policy_with_metadata` interceptor.
        """
        return response

    def post_delete_principal_access_boundary_policy_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for delete_principal_access_boundary_policy

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PrincipalAccessBoundaryPolicies server but before it is returned to user code.

        We recommend only using this `post_delete_principal_access_boundary_policy_with_metadata`
        interceptor in new development instead of the `post_delete_principal_access_boundary_policy` interceptor.
        When both interceptors are used, this `post_delete_principal_access_boundary_policy_with_metadata` interceptor runs after the
        `post_delete_principal_access_boundary_policy` interceptor. The (possibly modified) response returned by
        `post_delete_principal_access_boundary_policy` will be passed to
        `post_delete_principal_access_boundary_policy_with_metadata`.
        """
        return response, metadata

    def pre_get_principal_access_boundary_policy(
        self,
        request: principal_access_boundary_policies_service.GetPrincipalAccessBoundaryPolicyRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        principal_access_boundary_policies_service.GetPrincipalAccessBoundaryPolicyRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_principal_access_boundary_policy

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PrincipalAccessBoundaryPolicies server.
        """
        return request, metadata

    def post_get_principal_access_boundary_policy(
        self,
        response: principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy,
    ) -> principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy:
        """Post-rpc interceptor for get_principal_access_boundary_policy

        DEPRECATED. Please use the `post_get_principal_access_boundary_policy_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PrincipalAccessBoundaryPolicies server but before
        it is returned to user code. This `post_get_principal_access_boundary_policy` interceptor runs
        before the `post_get_principal_access_boundary_policy_with_metadata` interceptor.
        """
        return response

    def post_get_principal_access_boundary_policy_with_metadata(
        self,
        response: principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for get_principal_access_boundary_policy

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PrincipalAccessBoundaryPolicies server but before it is returned to user code.

        We recommend only using this `post_get_principal_access_boundary_policy_with_metadata`
        interceptor in new development instead of the `post_get_principal_access_boundary_policy` interceptor.
        When both interceptors are used, this `post_get_principal_access_boundary_policy_with_metadata` interceptor runs after the
        `post_get_principal_access_boundary_policy` interceptor. The (possibly modified) response returned by
        `post_get_principal_access_boundary_policy` will be passed to
        `post_get_principal_access_boundary_policy_with_metadata`.
        """
        return response, metadata

    def pre_list_principal_access_boundary_policies(
        self,
        request: principal_access_boundary_policies_service.ListPrincipalAccessBoundaryPoliciesRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        principal_access_boundary_policies_service.ListPrincipalAccessBoundaryPoliciesRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_principal_access_boundary_policies

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PrincipalAccessBoundaryPolicies server.
        """
        return request, metadata

    def post_list_principal_access_boundary_policies(
        self,
        response: principal_access_boundary_policies_service.ListPrincipalAccessBoundaryPoliciesResponse,
    ) -> (
        principal_access_boundary_policies_service.ListPrincipalAccessBoundaryPoliciesResponse
    ):
        """Post-rpc interceptor for list_principal_access_boundary_policies

        DEPRECATED. Please use the `post_list_principal_access_boundary_policies_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PrincipalAccessBoundaryPolicies server but before
        it is returned to user code. This `post_list_principal_access_boundary_policies` interceptor runs
        before the `post_list_principal_access_boundary_policies_with_metadata` interceptor.
        """
        return response

    def post_list_principal_access_boundary_policies_with_metadata(
        self,
        response: principal_access_boundary_policies_service.ListPrincipalAccessBoundaryPoliciesResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        principal_access_boundary_policies_service.ListPrincipalAccessBoundaryPoliciesResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_principal_access_boundary_policies

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PrincipalAccessBoundaryPolicies server but before it is returned to user code.

        We recommend only using this `post_list_principal_access_boundary_policies_with_metadata`
        interceptor in new development instead of the `post_list_principal_access_boundary_policies` interceptor.
        When both interceptors are used, this `post_list_principal_access_boundary_policies_with_metadata` interceptor runs after the
        `post_list_principal_access_boundary_policies` interceptor. The (possibly modified) response returned by
        `post_list_principal_access_boundary_policies` will be passed to
        `post_list_principal_access_boundary_policies_with_metadata`.
        """
        return response, metadata

    def pre_search_principal_access_boundary_policy_bindings(
        self,
        request: principal_access_boundary_policies_service.SearchPrincipalAccessBoundaryPolicyBindingsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        principal_access_boundary_policies_service.SearchPrincipalAccessBoundaryPolicyBindingsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for search_principal_access_boundary_policy_bindings

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PrincipalAccessBoundaryPolicies server.
        """
        return request, metadata

    def post_search_principal_access_boundary_policy_bindings(
        self,
        response: principal_access_boundary_policies_service.SearchPrincipalAccessBoundaryPolicyBindingsResponse,
    ) -> (
        principal_access_boundary_policies_service.SearchPrincipalAccessBoundaryPolicyBindingsResponse
    ):
        """Post-rpc interceptor for search_principal_access_boundary_policy_bindings

        DEPRECATED. Please use the `post_search_principal_access_boundary_policy_bindings_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PrincipalAccessBoundaryPolicies server but before
        it is returned to user code. This `post_search_principal_access_boundary_policy_bindings` interceptor runs
        before the `post_search_principal_access_boundary_policy_bindings_with_metadata` interceptor.
        """
        return response

    def post_search_principal_access_boundary_policy_bindings_with_metadata(
        self,
        response: principal_access_boundary_policies_service.SearchPrincipalAccessBoundaryPolicyBindingsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        principal_access_boundary_policies_service.SearchPrincipalAccessBoundaryPolicyBindingsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for search_principal_access_boundary_policy_bindings

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PrincipalAccessBoundaryPolicies server but before it is returned to user code.

        We recommend only using this `post_search_principal_access_boundary_policy_bindings_with_metadata`
        interceptor in new development instead of the `post_search_principal_access_boundary_policy_bindings` interceptor.
        When both interceptors are used, this `post_search_principal_access_boundary_policy_bindings_with_metadata` interceptor runs after the
        `post_search_principal_access_boundary_policy_bindings` interceptor. The (possibly modified) response returned by
        `post_search_principal_access_boundary_policy_bindings` will be passed to
        `post_search_principal_access_boundary_policy_bindings_with_metadata`.
        """
        return response, metadata

    def pre_update_principal_access_boundary_policy(
        self,
        request: principal_access_boundary_policies_service.UpdatePrincipalAccessBoundaryPolicyRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        principal_access_boundary_policies_service.UpdatePrincipalAccessBoundaryPolicyRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for update_principal_access_boundary_policy

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PrincipalAccessBoundaryPolicies server.
        """
        return request, metadata

    def post_update_principal_access_boundary_policy(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for update_principal_access_boundary_policy

        DEPRECATED. Please use the `post_update_principal_access_boundary_policy_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PrincipalAccessBoundaryPolicies server but before
        it is returned to user code. This `post_update_principal_access_boundary_policy` interceptor runs
        before the `post_update_principal_access_boundary_policy_with_metadata` interceptor.
        """
        return response

    def post_update_principal_access_boundary_policy_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_principal_access_boundary_policy

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PrincipalAccessBoundaryPolicies server but before it is returned to user code.

        We recommend only using this `post_update_principal_access_boundary_policy_with_metadata`
        interceptor in new development instead of the `post_update_principal_access_boundary_policy` interceptor.
        When both interceptors are used, this `post_update_principal_access_boundary_policy_with_metadata` interceptor runs after the
        `post_update_principal_access_boundary_policy` interceptor. The (possibly modified) response returned by
        `post_update_principal_access_boundary_policy` will be passed to
        `post_update_principal_access_boundary_policy_with_metadata`.
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
        before they are sent to the PrincipalAccessBoundaryPolicies server.
        """
        return request, metadata

    def post_get_operation(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for get_operation

        Override in a subclass to manipulate the response
        after it is returned by the PrincipalAccessBoundaryPolicies server but before
        it is returned to user code.
        """
        return response


@dataclasses.dataclass
class PrincipalAccessBoundaryPoliciesRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: PrincipalAccessBoundaryPoliciesRestInterceptor


class PrincipalAccessBoundaryPoliciesRestTransport(
    _BasePrincipalAccessBoundaryPoliciesRestTransport
):
    """REST backend synchronous transport for PrincipalAccessBoundaryPolicies.

    Manages Identity and Access Management (IAM) principal access
    boundary policies.

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
        interceptor: Optional[PrincipalAccessBoundaryPoliciesRestInterceptor] = None,
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
        self._interceptor = (
            interceptor or PrincipalAccessBoundaryPoliciesRestInterceptor()
        )
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

    class _CreatePrincipalAccessBoundaryPolicy(
        _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseCreatePrincipalAccessBoundaryPolicy,
        PrincipalAccessBoundaryPoliciesRestStub,
    ):
        def __hash__(self):
            return hash(
                "PrincipalAccessBoundaryPoliciesRestTransport.CreatePrincipalAccessBoundaryPolicy"
            )

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
            request: principal_access_boundary_policies_service.CreatePrincipalAccessBoundaryPolicyRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the create principal access
            boundary policy method over HTTP.

                Args:
                    request (~.principal_access_boundary_policies_service.CreatePrincipalAccessBoundaryPolicyRequest):
                        The request object. Request message for
                    CreatePrincipalAccessBoundaryPolicyRequest
                    method.
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
                _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseCreatePrincipalAccessBoundaryPolicy._get_http_options()
            )

            (
                request,
                metadata,
            ) = self._interceptor.pre_create_principal_access_boundary_policy(
                request, metadata
            )
            transcoded_request = _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseCreatePrincipalAccessBoundaryPolicy._get_transcoded_request(
                http_options, request
            )

            body = _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseCreatePrincipalAccessBoundaryPolicy._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseCreatePrincipalAccessBoundaryPolicy._get_query_params_json(
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
                    f"Sending request for google.iam_v3.PrincipalAccessBoundaryPoliciesClient.CreatePrincipalAccessBoundaryPolicy",
                    extra={
                        "serviceName": "google.iam.v3.PrincipalAccessBoundaryPolicies",
                        "rpcName": "CreatePrincipalAccessBoundaryPolicy",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = PrincipalAccessBoundaryPoliciesRestTransport._CreatePrincipalAccessBoundaryPolicy._get_response(
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

            resp = self._interceptor.post_create_principal_access_boundary_policy(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            (
                resp,
                _,
            ) = self._interceptor.post_create_principal_access_boundary_policy_with_metadata(
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
                    "Received response for google.iam_v3.PrincipalAccessBoundaryPoliciesClient.create_principal_access_boundary_policy",
                    extra={
                        "serviceName": "google.iam.v3.PrincipalAccessBoundaryPolicies",
                        "rpcName": "CreatePrincipalAccessBoundaryPolicy",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _DeletePrincipalAccessBoundaryPolicy(
        _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseDeletePrincipalAccessBoundaryPolicy,
        PrincipalAccessBoundaryPoliciesRestStub,
    ):
        def __hash__(self):
            return hash(
                "PrincipalAccessBoundaryPoliciesRestTransport.DeletePrincipalAccessBoundaryPolicy"
            )

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
            request: principal_access_boundary_policies_service.DeletePrincipalAccessBoundaryPolicyRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the delete principal access
            boundary policy method over HTTP.

                Args:
                    request (~.principal_access_boundary_policies_service.DeletePrincipalAccessBoundaryPolicyRequest):
                        The request object. Request message for
                    DeletePrincipalAccessBoundaryPolicy
                    method.
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
                _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseDeletePrincipalAccessBoundaryPolicy._get_http_options()
            )

            (
                request,
                metadata,
            ) = self._interceptor.pre_delete_principal_access_boundary_policy(
                request, metadata
            )
            transcoded_request = _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseDeletePrincipalAccessBoundaryPolicy._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseDeletePrincipalAccessBoundaryPolicy._get_query_params_json(
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
                    f"Sending request for google.iam_v3.PrincipalAccessBoundaryPoliciesClient.DeletePrincipalAccessBoundaryPolicy",
                    extra={
                        "serviceName": "google.iam.v3.PrincipalAccessBoundaryPolicies",
                        "rpcName": "DeletePrincipalAccessBoundaryPolicy",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = PrincipalAccessBoundaryPoliciesRestTransport._DeletePrincipalAccessBoundaryPolicy._get_response(
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

            resp = self._interceptor.post_delete_principal_access_boundary_policy(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            (
                resp,
                _,
            ) = self._interceptor.post_delete_principal_access_boundary_policy_with_metadata(
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
                    "Received response for google.iam_v3.PrincipalAccessBoundaryPoliciesClient.delete_principal_access_boundary_policy",
                    extra={
                        "serviceName": "google.iam.v3.PrincipalAccessBoundaryPolicies",
                        "rpcName": "DeletePrincipalAccessBoundaryPolicy",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetPrincipalAccessBoundaryPolicy(
        _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseGetPrincipalAccessBoundaryPolicy,
        PrincipalAccessBoundaryPoliciesRestStub,
    ):
        def __hash__(self):
            return hash(
                "PrincipalAccessBoundaryPoliciesRestTransport.GetPrincipalAccessBoundaryPolicy"
            )

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
            request: principal_access_boundary_policies_service.GetPrincipalAccessBoundaryPolicyRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy:
            r"""Call the get principal access
            boundary policy method over HTTP.

                Args:
                    request (~.principal_access_boundary_policies_service.GetPrincipalAccessBoundaryPolicyRequest):
                        The request object. Request message for
                    GetPrincipalAccessBoundaryPolicy method.
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy:
                        An IAM principal access boundary
                    policy resource.

            """

            http_options = (
                _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseGetPrincipalAccessBoundaryPolicy._get_http_options()
            )

            (
                request,
                metadata,
            ) = self._interceptor.pre_get_principal_access_boundary_policy(
                request, metadata
            )
            transcoded_request = _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseGetPrincipalAccessBoundaryPolicy._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseGetPrincipalAccessBoundaryPolicy._get_query_params_json(
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
                    f"Sending request for google.iam_v3.PrincipalAccessBoundaryPoliciesClient.GetPrincipalAccessBoundaryPolicy",
                    extra={
                        "serviceName": "google.iam.v3.PrincipalAccessBoundaryPolicies",
                        "rpcName": "GetPrincipalAccessBoundaryPolicy",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = PrincipalAccessBoundaryPoliciesRestTransport._GetPrincipalAccessBoundaryPolicy._get_response(
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
            resp = (
                principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy()
            )
            pb_resp = principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy.pb(
                resp
            )

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_principal_access_boundary_policy(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            (
                resp,
                _,
            ) = self._interceptor.post_get_principal_access_boundary_policy_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy.to_json(
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
                    "Received response for google.iam_v3.PrincipalAccessBoundaryPoliciesClient.get_principal_access_boundary_policy",
                    extra={
                        "serviceName": "google.iam.v3.PrincipalAccessBoundaryPolicies",
                        "rpcName": "GetPrincipalAccessBoundaryPolicy",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListPrincipalAccessBoundaryPolicies(
        _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseListPrincipalAccessBoundaryPolicies,
        PrincipalAccessBoundaryPoliciesRestStub,
    ):
        def __hash__(self):
            return hash(
                "PrincipalAccessBoundaryPoliciesRestTransport.ListPrincipalAccessBoundaryPolicies"
            )

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
            request: principal_access_boundary_policies_service.ListPrincipalAccessBoundaryPoliciesRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> (
            principal_access_boundary_policies_service.ListPrincipalAccessBoundaryPoliciesResponse
        ):
            r"""Call the list principal access
            boundary policies method over HTTP.

                Args:
                    request (~.principal_access_boundary_policies_service.ListPrincipalAccessBoundaryPoliciesRequest):
                        The request object. Request message for
                    ListPrincipalAccessBoundaryPolicies
                    method.
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.principal_access_boundary_policies_service.ListPrincipalAccessBoundaryPoliciesResponse:
                        Response message for
                    ListPrincipalAccessBoundaryPolicies
                    method.

            """

            http_options = (
                _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseListPrincipalAccessBoundaryPolicies._get_http_options()
            )

            (
                request,
                metadata,
            ) = self._interceptor.pre_list_principal_access_boundary_policies(
                request, metadata
            )
            transcoded_request = _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseListPrincipalAccessBoundaryPolicies._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseListPrincipalAccessBoundaryPolicies._get_query_params_json(
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
                    f"Sending request for google.iam_v3.PrincipalAccessBoundaryPoliciesClient.ListPrincipalAccessBoundaryPolicies",
                    extra={
                        "serviceName": "google.iam.v3.PrincipalAccessBoundaryPolicies",
                        "rpcName": "ListPrincipalAccessBoundaryPolicies",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = PrincipalAccessBoundaryPoliciesRestTransport._ListPrincipalAccessBoundaryPolicies._get_response(
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
            resp = (
                principal_access_boundary_policies_service.ListPrincipalAccessBoundaryPoliciesResponse()
            )
            pb_resp = principal_access_boundary_policies_service.ListPrincipalAccessBoundaryPoliciesResponse.pb(
                resp
            )

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_principal_access_boundary_policies(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            (
                resp,
                _,
            ) = self._interceptor.post_list_principal_access_boundary_policies_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = principal_access_boundary_policies_service.ListPrincipalAccessBoundaryPoliciesResponse.to_json(
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
                    "Received response for google.iam_v3.PrincipalAccessBoundaryPoliciesClient.list_principal_access_boundary_policies",
                    extra={
                        "serviceName": "google.iam.v3.PrincipalAccessBoundaryPolicies",
                        "rpcName": "ListPrincipalAccessBoundaryPolicies",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _SearchPrincipalAccessBoundaryPolicyBindings(
        _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseSearchPrincipalAccessBoundaryPolicyBindings,
        PrincipalAccessBoundaryPoliciesRestStub,
    ):
        def __hash__(self):
            return hash(
                "PrincipalAccessBoundaryPoliciesRestTransport.SearchPrincipalAccessBoundaryPolicyBindings"
            )

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
            request: principal_access_boundary_policies_service.SearchPrincipalAccessBoundaryPolicyBindingsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> (
            principal_access_boundary_policies_service.SearchPrincipalAccessBoundaryPolicyBindingsResponse
        ):
            r"""Call the search principal access
            boundary policy bindings method over HTTP.

                Args:
                    request (~.principal_access_boundary_policies_service.SearchPrincipalAccessBoundaryPolicyBindingsRequest):
                        The request object. Request message for
                    SearchPrincipalAccessBoundaryPolicyBindings
                    rpc.
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.principal_access_boundary_policies_service.SearchPrincipalAccessBoundaryPolicyBindingsResponse:
                        Response message for
                    SearchPrincipalAccessBoundaryPolicyBindings
                    rpc.

            """

            http_options = (
                _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseSearchPrincipalAccessBoundaryPolicyBindings._get_http_options()
            )

            (
                request,
                metadata,
            ) = self._interceptor.pre_search_principal_access_boundary_policy_bindings(
                request, metadata
            )
            transcoded_request = _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseSearchPrincipalAccessBoundaryPolicyBindings._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseSearchPrincipalAccessBoundaryPolicyBindings._get_query_params_json(
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
                    f"Sending request for google.iam_v3.PrincipalAccessBoundaryPoliciesClient.SearchPrincipalAccessBoundaryPolicyBindings",
                    extra={
                        "serviceName": "google.iam.v3.PrincipalAccessBoundaryPolicies",
                        "rpcName": "SearchPrincipalAccessBoundaryPolicyBindings",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = PrincipalAccessBoundaryPoliciesRestTransport._SearchPrincipalAccessBoundaryPolicyBindings._get_response(
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
            resp = (
                principal_access_boundary_policies_service.SearchPrincipalAccessBoundaryPolicyBindingsResponse()
            )
            pb_resp = principal_access_boundary_policies_service.SearchPrincipalAccessBoundaryPolicyBindingsResponse.pb(
                resp
            )

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = (
                self._interceptor.post_search_principal_access_boundary_policy_bindings(
                    resp
                )
            )
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            (
                resp,
                _,
            ) = self._interceptor.post_search_principal_access_boundary_policy_bindings_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = principal_access_boundary_policies_service.SearchPrincipalAccessBoundaryPolicyBindingsResponse.to_json(
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
                    "Received response for google.iam_v3.PrincipalAccessBoundaryPoliciesClient.search_principal_access_boundary_policy_bindings",
                    extra={
                        "serviceName": "google.iam.v3.PrincipalAccessBoundaryPolicies",
                        "rpcName": "SearchPrincipalAccessBoundaryPolicyBindings",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdatePrincipalAccessBoundaryPolicy(
        _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseUpdatePrincipalAccessBoundaryPolicy,
        PrincipalAccessBoundaryPoliciesRestStub,
    ):
        def __hash__(self):
            return hash(
                "PrincipalAccessBoundaryPoliciesRestTransport.UpdatePrincipalAccessBoundaryPolicy"
            )

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
            request: principal_access_boundary_policies_service.UpdatePrincipalAccessBoundaryPolicyRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the update principal access
            boundary policy method over HTTP.

                Args:
                    request (~.principal_access_boundary_policies_service.UpdatePrincipalAccessBoundaryPolicyRequest):
                        The request object. Request message for
                    UpdatePrincipalAccessBoundaryPolicy
                    method.
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
                _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseUpdatePrincipalAccessBoundaryPolicy._get_http_options()
            )

            (
                request,
                metadata,
            ) = self._interceptor.pre_update_principal_access_boundary_policy(
                request, metadata
            )
            transcoded_request = _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseUpdatePrincipalAccessBoundaryPolicy._get_transcoded_request(
                http_options, request
            )

            body = _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseUpdatePrincipalAccessBoundaryPolicy._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseUpdatePrincipalAccessBoundaryPolicy._get_query_params_json(
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
                    f"Sending request for google.iam_v3.PrincipalAccessBoundaryPoliciesClient.UpdatePrincipalAccessBoundaryPolicy",
                    extra={
                        "serviceName": "google.iam.v3.PrincipalAccessBoundaryPolicies",
                        "rpcName": "UpdatePrincipalAccessBoundaryPolicy",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = PrincipalAccessBoundaryPoliciesRestTransport._UpdatePrincipalAccessBoundaryPolicy._get_response(
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

            resp = self._interceptor.post_update_principal_access_boundary_policy(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            (
                resp,
                _,
            ) = self._interceptor.post_update_principal_access_boundary_policy_with_metadata(
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
                    "Received response for google.iam_v3.PrincipalAccessBoundaryPoliciesClient.update_principal_access_boundary_policy",
                    extra={
                        "serviceName": "google.iam.v3.PrincipalAccessBoundaryPolicies",
                        "rpcName": "UpdatePrincipalAccessBoundaryPolicy",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    @property
    def create_principal_access_boundary_policy(
        self,
    ) -> Callable[
        [
            principal_access_boundary_policies_service.CreatePrincipalAccessBoundaryPolicyRequest
        ],
        operations_pb2.Operation,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreatePrincipalAccessBoundaryPolicy(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_principal_access_boundary_policy(
        self,
    ) -> Callable[
        [
            principal_access_boundary_policies_service.DeletePrincipalAccessBoundaryPolicyRequest
        ],
        operations_pb2.Operation,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeletePrincipalAccessBoundaryPolicy(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_principal_access_boundary_policy(
        self,
    ) -> Callable[
        [
            principal_access_boundary_policies_service.GetPrincipalAccessBoundaryPolicyRequest
        ],
        principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetPrincipalAccessBoundaryPolicy(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_principal_access_boundary_policies(
        self,
    ) -> Callable[
        [
            principal_access_boundary_policies_service.ListPrincipalAccessBoundaryPoliciesRequest
        ],
        principal_access_boundary_policies_service.ListPrincipalAccessBoundaryPoliciesResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListPrincipalAccessBoundaryPolicies(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def search_principal_access_boundary_policy_bindings(
        self,
    ) -> Callable[
        [
            principal_access_boundary_policies_service.SearchPrincipalAccessBoundaryPolicyBindingsRequest
        ],
        principal_access_boundary_policies_service.SearchPrincipalAccessBoundaryPolicyBindingsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._SearchPrincipalAccessBoundaryPolicyBindings(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_principal_access_boundary_policy(
        self,
    ) -> Callable[
        [
            principal_access_boundary_policies_service.UpdatePrincipalAccessBoundaryPolicyRequest
        ],
        operations_pb2.Operation,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdatePrincipalAccessBoundaryPolicy(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_operation(self):
        return self._GetOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _GetOperation(
        _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseGetOperation,
        PrincipalAccessBoundaryPoliciesRestStub,
    ):
        def __hash__(self):
            return hash("PrincipalAccessBoundaryPoliciesRestTransport.GetOperation")

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
                _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseGetOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_operation(request, metadata)
            transcoded_request = _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseGetOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BasePrincipalAccessBoundaryPoliciesRestTransport._BaseGetOperation._get_query_params_json(
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
                    f"Sending request for google.iam_v3.PrincipalAccessBoundaryPoliciesClient.GetOperation",
                    extra={
                        "serviceName": "google.iam.v3.PrincipalAccessBoundaryPolicies",
                        "rpcName": "GetOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = PrincipalAccessBoundaryPoliciesRestTransport._GetOperation._get_response(
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
                    "Received response for google.iam_v3.PrincipalAccessBoundaryPoliciesAsyncClient.GetOperation",
                    extra={
                        "serviceName": "google.iam.v3.PrincipalAccessBoundaryPolicies",
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


__all__ = ("PrincipalAccessBoundaryPoliciesRestTransport",)
