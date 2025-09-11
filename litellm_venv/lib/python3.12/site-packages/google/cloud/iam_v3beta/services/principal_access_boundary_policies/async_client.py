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
from collections import OrderedDict
import logging as std_logging
import re
from typing import (
    Callable,
    Dict,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry_async as retries
from google.api_core.client_options import ClientOptions
from google.auth import credentials as ga_credentials  # type: ignore
from google.oauth2 import service_account  # type: ignore
import google.protobuf

from google.cloud.iam_v3beta import gapic_version as package_version

try:
    OptionalRetry = Union[retries.AsyncRetry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.AsyncRetry, object, None]  # type: ignore

from google.api_core import operation  # type: ignore
from google.api_core import operation_async  # type: ignore
from google.cloud.location import locations_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import empty_pb2  # type: ignore
from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore

from google.cloud.iam_v3beta.services.principal_access_boundary_policies import pagers
from google.cloud.iam_v3beta.types import (
    operation_metadata,
    policy_binding_resources,
    principal_access_boundary_policies_service,
    principal_access_boundary_policy_resources,
)

from .client import PrincipalAccessBoundaryPoliciesClient
from .transports.base import (
    DEFAULT_CLIENT_INFO,
    PrincipalAccessBoundaryPoliciesTransport,
)
from .transports.grpc_asyncio import PrincipalAccessBoundaryPoliciesGrpcAsyncIOTransport

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = std_logging.getLogger(__name__)


class PrincipalAccessBoundaryPoliciesAsyncClient:
    """Manages Identity and Access Management (IAM) principal access
    boundary policies.
    """

    _client: PrincipalAccessBoundaryPoliciesClient

    # Copy defaults from the synchronous client for use here.
    # Note: DEFAULT_ENDPOINT is deprecated. Use _DEFAULT_ENDPOINT_TEMPLATE instead.
    DEFAULT_ENDPOINT = PrincipalAccessBoundaryPoliciesClient.DEFAULT_ENDPOINT
    DEFAULT_MTLS_ENDPOINT = PrincipalAccessBoundaryPoliciesClient.DEFAULT_MTLS_ENDPOINT
    _DEFAULT_ENDPOINT_TEMPLATE = (
        PrincipalAccessBoundaryPoliciesClient._DEFAULT_ENDPOINT_TEMPLATE
    )
    _DEFAULT_UNIVERSE = PrincipalAccessBoundaryPoliciesClient._DEFAULT_UNIVERSE

    policy_binding_path = staticmethod(
        PrincipalAccessBoundaryPoliciesClient.policy_binding_path
    )
    parse_policy_binding_path = staticmethod(
        PrincipalAccessBoundaryPoliciesClient.parse_policy_binding_path
    )
    principal_access_boundary_policy_path = staticmethod(
        PrincipalAccessBoundaryPoliciesClient.principal_access_boundary_policy_path
    )
    parse_principal_access_boundary_policy_path = staticmethod(
        PrincipalAccessBoundaryPoliciesClient.parse_principal_access_boundary_policy_path
    )
    common_billing_account_path = staticmethod(
        PrincipalAccessBoundaryPoliciesClient.common_billing_account_path
    )
    parse_common_billing_account_path = staticmethod(
        PrincipalAccessBoundaryPoliciesClient.parse_common_billing_account_path
    )
    common_folder_path = staticmethod(
        PrincipalAccessBoundaryPoliciesClient.common_folder_path
    )
    parse_common_folder_path = staticmethod(
        PrincipalAccessBoundaryPoliciesClient.parse_common_folder_path
    )
    common_organization_path = staticmethod(
        PrincipalAccessBoundaryPoliciesClient.common_organization_path
    )
    parse_common_organization_path = staticmethod(
        PrincipalAccessBoundaryPoliciesClient.parse_common_organization_path
    )
    common_project_path = staticmethod(
        PrincipalAccessBoundaryPoliciesClient.common_project_path
    )
    parse_common_project_path = staticmethod(
        PrincipalAccessBoundaryPoliciesClient.parse_common_project_path
    )
    common_location_path = staticmethod(
        PrincipalAccessBoundaryPoliciesClient.common_location_path
    )
    parse_common_location_path = staticmethod(
        PrincipalAccessBoundaryPoliciesClient.parse_common_location_path
    )

    @classmethod
    def from_service_account_info(cls, info: dict, *args, **kwargs):
        """Creates an instance of this client using the provided credentials
            info.

        Args:
            info (dict): The service account private key info.
            args: Additional arguments to pass to the constructor.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            PrincipalAccessBoundaryPoliciesAsyncClient: The constructed client.
        """
        return PrincipalAccessBoundaryPoliciesClient.from_service_account_info.__func__(PrincipalAccessBoundaryPoliciesAsyncClient, info, *args, **kwargs)  # type: ignore

    @classmethod
    def from_service_account_file(cls, filename: str, *args, **kwargs):
        """Creates an instance of this client using the provided credentials
            file.

        Args:
            filename (str): The path to the service account private key json
                file.
            args: Additional arguments to pass to the constructor.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            PrincipalAccessBoundaryPoliciesAsyncClient: The constructed client.
        """
        return PrincipalAccessBoundaryPoliciesClient.from_service_account_file.__func__(PrincipalAccessBoundaryPoliciesAsyncClient, filename, *args, **kwargs)  # type: ignore

    from_service_account_json = from_service_account_file

    @classmethod
    def get_mtls_endpoint_and_cert_source(
        cls, client_options: Optional[ClientOptions] = None
    ):
        """Return the API endpoint and client cert source for mutual TLS.

        The client cert source is determined in the following order:
        (1) if `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable is not "true", the
        client cert source is None.
        (2) if `client_options.client_cert_source` is provided, use the provided one; if the
        default client cert source exists, use the default one; otherwise the client cert
        source is None.

        The API endpoint is determined in the following order:
        (1) if `client_options.api_endpoint` if provided, use the provided one.
        (2) if `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable is "always", use the
        default mTLS endpoint; if the environment variable is "never", use the default API
        endpoint; otherwise if client cert source exists, use the default mTLS endpoint, otherwise
        use the default API endpoint.

        More details can be found at https://google.aip.dev/auth/4114.

        Args:
            client_options (google.api_core.client_options.ClientOptions): Custom options for the
                client. Only the `api_endpoint` and `client_cert_source` properties may be used
                in this method.

        Returns:
            Tuple[str, Callable[[], Tuple[bytes, bytes]]]: returns the API endpoint and the
                client cert source to use.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If any errors happen.
        """
        return PrincipalAccessBoundaryPoliciesClient.get_mtls_endpoint_and_cert_source(client_options)  # type: ignore

    @property
    def transport(self) -> PrincipalAccessBoundaryPoliciesTransport:
        """Returns the transport used by the client instance.

        Returns:
            PrincipalAccessBoundaryPoliciesTransport: The transport used by the client instance.
        """
        return self._client.transport

    @property
    def api_endpoint(self):
        """Return the API endpoint used by the client instance.

        Returns:
            str: The API endpoint used by the client instance.
        """
        return self._client._api_endpoint

    @property
    def universe_domain(self) -> str:
        """Return the universe domain used by the client instance.

        Returns:
            str: The universe domain used
                by the client instance.
        """
        return self._client._universe_domain

    get_transport_class = PrincipalAccessBoundaryPoliciesClient.get_transport_class

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Optional[
            Union[
                str,
                PrincipalAccessBoundaryPoliciesTransport,
                Callable[..., PrincipalAccessBoundaryPoliciesTransport],
            ]
        ] = "grpc_asyncio",
        client_options: Optional[ClientOptions] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the principal access boundary policies async client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Optional[Union[str,PrincipalAccessBoundaryPoliciesTransport,Callable[..., PrincipalAccessBoundaryPoliciesTransport]]]):
                The transport to use, or a Callable that constructs and returns a new transport to use.
                If a Callable is given, it will be called with the same set of initialization
                arguments as used in the PrincipalAccessBoundaryPoliciesTransport constructor.
                If set to None, a transport is chosen automatically.
            client_options (Optional[Union[google.api_core.client_options.ClientOptions, dict]]):
                Custom options for the client.

                1. The ``api_endpoint`` property can be used to override the
                default endpoint provided by the client when ``transport`` is
                not explicitly provided. Only if this property is not set and
                ``transport`` was not explicitly provided, the endpoint is
                determined by the GOOGLE_API_USE_MTLS_ENDPOINT environment
                variable, which have one of the following values:
                "always" (always use the default mTLS endpoint), "never" (always
                use the default regular endpoint) and "auto" (auto-switch to the
                default mTLS endpoint if client certificate is present; this is
                the default value).

                2. If the GOOGLE_API_USE_CLIENT_CERTIFICATE environment variable
                is "true", then the ``client_cert_source`` property can be used
                to provide a client certificate for mTLS transport. If
                not provided, the default SSL client certificate will be used if
                present. If GOOGLE_API_USE_CLIENT_CERTIFICATE is "false" or not
                set, no client certificate will be used.

                3. The ``universe_domain`` property can be used to override the
                default "googleapis.com" universe. Note that ``api_endpoint``
                property still takes precedence; and ``universe_domain`` is
                currently not supported for mTLS.

            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.

        Raises:
            google.auth.exceptions.MutualTlsChannelError: If mutual TLS transport
                creation failed for any reason.
        """
        self._client = PrincipalAccessBoundaryPoliciesClient(
            credentials=credentials,
            transport=transport,
            client_options=client_options,
            client_info=client_info,
        )

        if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
            std_logging.DEBUG
        ):  # pragma: NO COVER
            _LOGGER.debug(
                "Created client `google.iam_v3beta.PrincipalAccessBoundaryPoliciesAsyncClient`.",
                extra={
                    "serviceName": "google.iam.v3beta.PrincipalAccessBoundaryPolicies",
                    "universeDomain": getattr(
                        self._client._transport._credentials, "universe_domain", ""
                    ),
                    "credentialsType": f"{type(self._client._transport._credentials).__module__}.{type(self._client._transport._credentials).__qualname__}",
                    "credentialsInfo": getattr(
                        self.transport._credentials, "get_cred_info", lambda: None
                    )(),
                }
                if hasattr(self._client._transport, "_credentials")
                else {
                    "serviceName": "google.iam.v3beta.PrincipalAccessBoundaryPolicies",
                    "credentialsType": None,
                },
            )

    async def create_principal_access_boundary_policy(
        self,
        request: Optional[
            Union[
                principal_access_boundary_policies_service.CreatePrincipalAccessBoundaryPolicyRequest,
                dict,
            ]
        ] = None,
        *,
        parent: Optional[str] = None,
        principal_access_boundary_policy: Optional[
            principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy
        ] = None,
        principal_access_boundary_policy_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Creates a principal access boundary policy, and
        returns a long running operation.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_v3beta

            async def sample_create_principal_access_boundary_policy():
                # Create a client
                client = iam_v3beta.PrincipalAccessBoundaryPoliciesAsyncClient()

                # Initialize request argument(s)
                request = iam_v3beta.CreatePrincipalAccessBoundaryPolicyRequest(
                    parent="parent_value",
                    principal_access_boundary_policy_id="principal_access_boundary_policy_id_value",
                )

                # Make the request
                operation = client.create_principal_access_boundary_policy(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.iam_v3beta.types.CreatePrincipalAccessBoundaryPolicyRequest, dict]]):
                The request object. Request message for
                CreatePrincipalAccessBoundaryPolicyRequest
                method.
            parent (:class:`str`):
                Required. The parent resource where this principal
                access boundary policy will be created. Only
                organizations are supported.

                Format:
                ``organizations/{organization_id}/locations/{location}``

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            principal_access_boundary_policy (:class:`google.cloud.iam_v3beta.types.PrincipalAccessBoundaryPolicy`):
                Required. The principal access
                boundary policy to create.

                This corresponds to the ``principal_access_boundary_policy`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            principal_access_boundary_policy_id (:class:`str`):
                Required. The ID to use for the principal access
                boundary policy, which will become the final component
                of the principal access boundary policy's resource name.

                This value must start with a lowercase letter followed
                by up to 62 lowercase letters, numbers, hyphens, or
                dots. Pattern, /[a-z][a-z0-9-.]{2,62}/.

                This corresponds to the ``principal_access_boundary_policy_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation_async.AsyncOperation:
                An object representing a long-running operation.

                The result type for the operation will be
                :class:`google.cloud.iam_v3beta.types.PrincipalAccessBoundaryPolicy`
                An IAM principal access boundary policy resource.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [
            parent,
            principal_access_boundary_policy,
            principal_access_boundary_policy_id,
        ]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(
            request,
            principal_access_boundary_policies_service.CreatePrincipalAccessBoundaryPolicyRequest,
        ):
            request = principal_access_boundary_policies_service.CreatePrincipalAccessBoundaryPolicyRequest(
                request
            )

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if principal_access_boundary_policy is not None:
            request.principal_access_boundary_policy = principal_access_boundary_policy
        if principal_access_boundary_policy_id is not None:
            request.principal_access_boundary_policy_id = (
                principal_access_boundary_policy_id
            )

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_principal_access_boundary_policy
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation_async.from_gapic(
            response,
            self._client._transport.operations_client,
            principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy,
            metadata_type=operation_metadata.OperationMetadata,
        )

        # Done; return the response.
        return response

    async def get_principal_access_boundary_policy(
        self,
        request: Optional[
            Union[
                principal_access_boundary_policies_service.GetPrincipalAccessBoundaryPolicyRequest,
                dict,
            ]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy:
        r"""Gets a principal access boundary policy.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_v3beta

            async def sample_get_principal_access_boundary_policy():
                # Create a client
                client = iam_v3beta.PrincipalAccessBoundaryPoliciesAsyncClient()

                # Initialize request argument(s)
                request = iam_v3beta.GetPrincipalAccessBoundaryPolicyRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_principal_access_boundary_policy(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.iam_v3beta.types.GetPrincipalAccessBoundaryPolicyRequest, dict]]):
                The request object. Request message for
                GetPrincipalAccessBoundaryPolicy method.
            name (:class:`str`):
                Required. The name of the principal access boundary
                policy to retrieve.

                Format:
                ``organizations/{organization_id}/locations/{location}/principalAccessBoundaryPolicies/{principal_access_boundary_policy_id}``

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_v3beta.types.PrincipalAccessBoundaryPolicy:
                An IAM principal access boundary
                policy resource.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(
            request,
            principal_access_boundary_policies_service.GetPrincipalAccessBoundaryPolicyRequest,
        ):
            request = principal_access_boundary_policies_service.GetPrincipalAccessBoundaryPolicyRequest(
                request
            )

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_principal_access_boundary_policy
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def update_principal_access_boundary_policy(
        self,
        request: Optional[
            Union[
                principal_access_boundary_policies_service.UpdatePrincipalAccessBoundaryPolicyRequest,
                dict,
            ]
        ] = None,
        *,
        principal_access_boundary_policy: Optional[
            principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy
        ] = None,
        update_mask: Optional[field_mask_pb2.FieldMask] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Updates a principal access boundary policy.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_v3beta

            async def sample_update_principal_access_boundary_policy():
                # Create a client
                client = iam_v3beta.PrincipalAccessBoundaryPoliciesAsyncClient()

                # Initialize request argument(s)
                request = iam_v3beta.UpdatePrincipalAccessBoundaryPolicyRequest(
                )

                # Make the request
                operation = client.update_principal_access_boundary_policy(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.iam_v3beta.types.UpdatePrincipalAccessBoundaryPolicyRequest, dict]]):
                The request object. Request message for
                UpdatePrincipalAccessBoundaryPolicy
                method.
            principal_access_boundary_policy (:class:`google.cloud.iam_v3beta.types.PrincipalAccessBoundaryPolicy`):
                Required. The principal access boundary policy to
                update.

                The principal access boundary policy's ``name`` field is
                used to identify the policy to update.

                This corresponds to the ``principal_access_boundary_policy`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            update_mask (:class:`google.protobuf.field_mask_pb2.FieldMask`):
                Optional. The list of fields to
                update

                This corresponds to the ``update_mask`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation_async.AsyncOperation:
                An object representing a long-running operation.

                The result type for the operation will be
                :class:`google.cloud.iam_v3beta.types.PrincipalAccessBoundaryPolicy`
                An IAM principal access boundary policy resource.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [principal_access_boundary_policy, update_mask]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(
            request,
            principal_access_boundary_policies_service.UpdatePrincipalAccessBoundaryPolicyRequest,
        ):
            request = principal_access_boundary_policies_service.UpdatePrincipalAccessBoundaryPolicyRequest(
                request
            )

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if principal_access_boundary_policy is not None:
            request.principal_access_boundary_policy = principal_access_boundary_policy
        if update_mask is not None:
            request.update_mask = update_mask

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_principal_access_boundary_policy
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (
                    (
                        "principal_access_boundary_policy.name",
                        request.principal_access_boundary_policy.name,
                    ),
                )
            ),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation_async.from_gapic(
            response,
            self._client._transport.operations_client,
            principal_access_boundary_policy_resources.PrincipalAccessBoundaryPolicy,
            metadata_type=operation_metadata.OperationMetadata,
        )

        # Done; return the response.
        return response

    async def delete_principal_access_boundary_policy(
        self,
        request: Optional[
            Union[
                principal_access_boundary_policies_service.DeletePrincipalAccessBoundaryPolicyRequest,
                dict,
            ]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Deletes a principal access boundary policy.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_v3beta

            async def sample_delete_principal_access_boundary_policy():
                # Create a client
                client = iam_v3beta.PrincipalAccessBoundaryPoliciesAsyncClient()

                # Initialize request argument(s)
                request = iam_v3beta.DeletePrincipalAccessBoundaryPolicyRequest(
                    name="name_value",
                )

                # Make the request
                operation = client.delete_principal_access_boundary_policy(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.iam_v3beta.types.DeletePrincipalAccessBoundaryPolicyRequest, dict]]):
                The request object. Request message for
                DeletePrincipalAccessBoundaryPolicy
                method.
            name (:class:`str`):
                Required. The name of the principal access boundary
                policy to delete.

                Format:
                ``organizations/{organization_id}/locations/{location}/principalAccessBoundaryPolicies/{principal_access_boundary_policy_id}``

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.api_core.operation_async.AsyncOperation:
                An object representing a long-running operation.

                The result type for the operation will be :class:`google.protobuf.empty_pb2.Empty` A generic empty message that you can re-use to avoid defining duplicated
                   empty messages in your APIs. A typical example is to
                   use it as the request or the response type of an API
                   method. For instance:

                      service Foo {
                         rpc Bar(google.protobuf.Empty) returns
                         (google.protobuf.Empty);

                      }

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(
            request,
            principal_access_boundary_policies_service.DeletePrincipalAccessBoundaryPolicyRequest,
        ):
            request = principal_access_boundary_policies_service.DeletePrincipalAccessBoundaryPolicyRequest(
                request
            )

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_principal_access_boundary_policy
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Wrap the response in an operation future.
        response = operation_async.from_gapic(
            response,
            self._client._transport.operations_client,
            empty_pb2.Empty,
            metadata_type=operation_metadata.OperationMetadata,
        )

        # Done; return the response.
        return response

    async def list_principal_access_boundary_policies(
        self,
        request: Optional[
            Union[
                principal_access_boundary_policies_service.ListPrincipalAccessBoundaryPoliciesRequest,
                dict,
            ]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListPrincipalAccessBoundaryPoliciesAsyncPager:
        r"""Lists principal access boundary policies.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_v3beta

            async def sample_list_principal_access_boundary_policies():
                # Create a client
                client = iam_v3beta.PrincipalAccessBoundaryPoliciesAsyncClient()

                # Initialize request argument(s)
                request = iam_v3beta.ListPrincipalAccessBoundaryPoliciesRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_principal_access_boundary_policies(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.iam_v3beta.types.ListPrincipalAccessBoundaryPoliciesRequest, dict]]):
                The request object. Request message for
                ListPrincipalAccessBoundaryPolicies
                method.
            parent (:class:`str`):
                Required. The parent resource, which owns the collection
                of principal access boundary policies.

                Format:
                ``organizations/{organization_id}/locations/{location}``

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_v3beta.services.principal_access_boundary_policies.pagers.ListPrincipalAccessBoundaryPoliciesAsyncPager:
                Response message for
                ListPrincipalAccessBoundaryPolicies
                method.  Iterating over this object will
                yield results and resolve additional
                pages automatically.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(
            request,
            principal_access_boundary_policies_service.ListPrincipalAccessBoundaryPoliciesRequest,
        ):
            request = principal_access_boundary_policies_service.ListPrincipalAccessBoundaryPoliciesRequest(
                request
            )

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_principal_access_boundary_policies
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListPrincipalAccessBoundaryPoliciesAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def search_principal_access_boundary_policy_bindings(
        self,
        request: Optional[
            Union[
                principal_access_boundary_policies_service.SearchPrincipalAccessBoundaryPolicyBindingsRequest,
                dict,
            ]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.SearchPrincipalAccessBoundaryPolicyBindingsAsyncPager:
        r"""Returns all policy bindings that bind a specific
        policy if a user has searchPolicyBindings permission on
        that policy.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_v3beta

            async def sample_search_principal_access_boundary_policy_bindings():
                # Create a client
                client = iam_v3beta.PrincipalAccessBoundaryPoliciesAsyncClient()

                # Initialize request argument(s)
                request = iam_v3beta.SearchPrincipalAccessBoundaryPolicyBindingsRequest(
                    name="name_value",
                )

                # Make the request
                page_result = client.search_principal_access_boundary_policy_bindings(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.iam_v3beta.types.SearchPrincipalAccessBoundaryPolicyBindingsRequest, dict]]):
                The request object. Request message for
                SearchPrincipalAccessBoundaryPolicyBindings
                rpc.
            name (:class:`str`):
                Required. The name of the principal access boundary
                policy. Format:
                ``organizations/{organization_id}/locations/{location}/principalAccessBoundaryPolicies/{principal_access_boundary_policy_id}``

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            google.cloud.iam_v3beta.services.principal_access_boundary_policies.pagers.SearchPrincipalAccessBoundaryPolicyBindingsAsyncPager:
                Response message for
                SearchPrincipalAccessBoundaryPolicyBindings
                rpc.  Iterating over this object will
                yield results and resolve additional
                pages automatically.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = (
            len([param for param in flattened_params if param is not None]) > 0
        )
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(
            request,
            principal_access_boundary_policies_service.SearchPrincipalAccessBoundaryPolicyBindingsRequest,
        ):
            request = principal_access_boundary_policies_service.SearchPrincipalAccessBoundaryPolicyBindingsRequest(
                request
            )

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.search_principal_access_boundary_policy_bindings
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.SearchPrincipalAccessBoundaryPolicyBindingsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def get_operation(
        self,
        request: Optional[operations_pb2.GetOperationRequest] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operations_pb2.Operation:
        r"""Gets the latest state of a long-running operation.

        Args:
            request (:class:`~.operations_pb2.GetOperationRequest`):
                The request object. Request message for
                `GetOperation` method.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors,
                    if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        Returns:
            ~.operations_pb2.Operation:
                An ``Operation`` object.
        """
        # Create or coerce a protobuf request object.
        # The request isn't a proto-plus wrapped type,
        # so it must be constructed via keyword expansion.
        if isinstance(request, dict):
            request = operations_pb2.GetOperationRequest(**request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self.transport._wrapped_methods[self._client._transport.get_operation]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def __aenter__(self) -> "PrincipalAccessBoundaryPoliciesAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.transport.close()


DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


__all__ = ("PrincipalAccessBoundaryPoliciesAsyncClient",)
