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

from google.cloud.iam_v3 import gapic_version as package_version

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
from google.type import expr_pb2  # type: ignore

from google.cloud.iam_v3.services.policy_bindings import pagers
from google.cloud.iam_v3.types import (
    operation_metadata,
    policy_binding_resources,
    policy_bindings_service,
)

from .client import PolicyBindingsClient
from .transports.base import DEFAULT_CLIENT_INFO, PolicyBindingsTransport
from .transports.grpc_asyncio import PolicyBindingsGrpcAsyncIOTransport

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = std_logging.getLogger(__name__)


class PolicyBindingsAsyncClient:
    """An interface for managing Identity and Access Management
    (IAM) policy bindings.
    """

    _client: PolicyBindingsClient

    # Copy defaults from the synchronous client for use here.
    # Note: DEFAULT_ENDPOINT is deprecated. Use _DEFAULT_ENDPOINT_TEMPLATE instead.
    DEFAULT_ENDPOINT = PolicyBindingsClient.DEFAULT_ENDPOINT
    DEFAULT_MTLS_ENDPOINT = PolicyBindingsClient.DEFAULT_MTLS_ENDPOINT
    _DEFAULT_ENDPOINT_TEMPLATE = PolicyBindingsClient._DEFAULT_ENDPOINT_TEMPLATE
    _DEFAULT_UNIVERSE = PolicyBindingsClient._DEFAULT_UNIVERSE

    policy_binding_path = staticmethod(PolicyBindingsClient.policy_binding_path)
    parse_policy_binding_path = staticmethod(
        PolicyBindingsClient.parse_policy_binding_path
    )
    common_billing_account_path = staticmethod(
        PolicyBindingsClient.common_billing_account_path
    )
    parse_common_billing_account_path = staticmethod(
        PolicyBindingsClient.parse_common_billing_account_path
    )
    common_folder_path = staticmethod(PolicyBindingsClient.common_folder_path)
    parse_common_folder_path = staticmethod(
        PolicyBindingsClient.parse_common_folder_path
    )
    common_organization_path = staticmethod(
        PolicyBindingsClient.common_organization_path
    )
    parse_common_organization_path = staticmethod(
        PolicyBindingsClient.parse_common_organization_path
    )
    common_project_path = staticmethod(PolicyBindingsClient.common_project_path)
    parse_common_project_path = staticmethod(
        PolicyBindingsClient.parse_common_project_path
    )
    common_location_path = staticmethod(PolicyBindingsClient.common_location_path)
    parse_common_location_path = staticmethod(
        PolicyBindingsClient.parse_common_location_path
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
            PolicyBindingsAsyncClient: The constructed client.
        """
        return PolicyBindingsClient.from_service_account_info.__func__(PolicyBindingsAsyncClient, info, *args, **kwargs)  # type: ignore

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
            PolicyBindingsAsyncClient: The constructed client.
        """
        return PolicyBindingsClient.from_service_account_file.__func__(PolicyBindingsAsyncClient, filename, *args, **kwargs)  # type: ignore

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
        return PolicyBindingsClient.get_mtls_endpoint_and_cert_source(client_options)  # type: ignore

    @property
    def transport(self) -> PolicyBindingsTransport:
        """Returns the transport used by the client instance.

        Returns:
            PolicyBindingsTransport: The transport used by the client instance.
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

    get_transport_class = PolicyBindingsClient.get_transport_class

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Optional[
            Union[str, PolicyBindingsTransport, Callable[..., PolicyBindingsTransport]]
        ] = "grpc_asyncio",
        client_options: Optional[ClientOptions] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the policy bindings async client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Optional[Union[str,PolicyBindingsTransport,Callable[..., PolicyBindingsTransport]]]):
                The transport to use, or a Callable that constructs and returns a new transport to use.
                If a Callable is given, it will be called with the same set of initialization
                arguments as used in the PolicyBindingsTransport constructor.
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
        self._client = PolicyBindingsClient(
            credentials=credentials,
            transport=transport,
            client_options=client_options,
            client_info=client_info,
        )

        if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
            std_logging.DEBUG
        ):  # pragma: NO COVER
            _LOGGER.debug(
                "Created client `google.iam_v3.PolicyBindingsAsyncClient`.",
                extra={
                    "serviceName": "google.iam.v3.PolicyBindings",
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
                    "serviceName": "google.iam.v3.PolicyBindings",
                    "credentialsType": None,
                },
            )

    async def create_policy_binding(
        self,
        request: Optional[
            Union[policy_bindings_service.CreatePolicyBindingRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        policy_binding: Optional[policy_binding_resources.PolicyBinding] = None,
        policy_binding_id: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Creates a policy binding and returns a long-running
        operation. Callers will need the IAM permissions on both
        the policy and target. Once the binding is created, the
        policy is applied to the target.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_v3

            async def sample_create_policy_binding():
                # Create a client
                client = iam_v3.PolicyBindingsAsyncClient()

                # Initialize request argument(s)
                policy_binding = iam_v3.PolicyBinding()
                policy_binding.target.principal_set = "principal_set_value"
                policy_binding.policy = "policy_value"

                request = iam_v3.CreatePolicyBindingRequest(
                    parent="parent_value",
                    policy_binding_id="policy_binding_id_value",
                    policy_binding=policy_binding,
                )

                # Make the request
                operation = client.create_policy_binding(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.iam_v3.types.CreatePolicyBindingRequest, dict]]):
                The request object. Request message for
                CreatePolicyBinding method.
            parent (:class:`str`):
                Required. The parent resource where this policy binding
                will be created. The binding parent is the closest
                Resource Manager resource (project, folder or
                organization) to the binding target.

                Format:

                -  ``projects/{project_id}/locations/{location}``
                -  ``projects/{project_number}/locations/{location}``
                -  ``folders/{folder_id}/locations/{location}``
                -  ``organizations/{organization_id}/locations/{location}``

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            policy_binding (:class:`google.cloud.iam_v3.types.PolicyBinding`):
                Required. The policy binding to
                create.

                This corresponds to the ``policy_binding`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            policy_binding_id (:class:`str`):
                Required. The ID to use for the policy binding, which
                will become the final component of the policy binding's
                resource name.

                This value must start with a lowercase letter followed
                by up to 62 lowercase letters, numbers, hyphens, or
                dots. Pattern, /[a-z][a-z0-9-.]{2,62}/.

                This corresponds to the ``policy_binding_id`` field
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
                :class:`google.cloud.iam_v3.types.PolicyBinding` IAM
                policy binding resource.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, policy_binding, policy_binding_id]
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
        if not isinstance(request, policy_bindings_service.CreatePolicyBindingRequest):
            request = policy_bindings_service.CreatePolicyBindingRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if policy_binding is not None:
            request.policy_binding = policy_binding
        if policy_binding_id is not None:
            request.policy_binding_id = policy_binding_id

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_policy_binding
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
            policy_binding_resources.PolicyBinding,
            metadata_type=operation_metadata.OperationMetadata,
        )

        # Done; return the response.
        return response

    async def get_policy_binding(
        self,
        request: Optional[
            Union[policy_bindings_service.GetPolicyBindingRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> policy_binding_resources.PolicyBinding:
        r"""Gets a policy binding.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_v3

            async def sample_get_policy_binding():
                # Create a client
                client = iam_v3.PolicyBindingsAsyncClient()

                # Initialize request argument(s)
                request = iam_v3.GetPolicyBindingRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_policy_binding(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.iam_v3.types.GetPolicyBindingRequest, dict]]):
                The request object. Request message for GetPolicyBinding
                method.
            name (:class:`str`):
                Required. The name of the policy binding to retrieve.

                Format:

                -  ``projects/{project_id}/locations/{location}/policyBindings/{policy_binding_id}``
                -  ``projects/{project_number}/locations/{location}/policyBindings/{policy_binding_id}``
                -  ``folders/{folder_id}/locations/{location}/policyBindings/{policy_binding_id}``
                -  ``organizations/{organization_id}/locations/{location}/policyBindings/{policy_binding_id}``

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
            google.cloud.iam_v3.types.PolicyBinding:
                IAM policy binding resource.
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
        if not isinstance(request, policy_bindings_service.GetPolicyBindingRequest):
            request = policy_bindings_service.GetPolicyBindingRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_policy_binding
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

    async def update_policy_binding(
        self,
        request: Optional[
            Union[policy_bindings_service.UpdatePolicyBindingRequest, dict]
        ] = None,
        *,
        policy_binding: Optional[policy_binding_resources.PolicyBinding] = None,
        update_mask: Optional[field_mask_pb2.FieldMask] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Updates a policy binding and returns a long-running
        operation. Callers will need the IAM permissions on the
        policy and target in the binding to update, and the IAM
        permission to remove the existing policy from the
        binding. Target is immutable and cannot be updated. Once
        the binding is updated, the new policy is applied to the
        target.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_v3

            async def sample_update_policy_binding():
                # Create a client
                client = iam_v3.PolicyBindingsAsyncClient()

                # Initialize request argument(s)
                policy_binding = iam_v3.PolicyBinding()
                policy_binding.target.principal_set = "principal_set_value"
                policy_binding.policy = "policy_value"

                request = iam_v3.UpdatePolicyBindingRequest(
                    policy_binding=policy_binding,
                )

                # Make the request
                operation = client.update_policy_binding(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.iam_v3.types.UpdatePolicyBindingRequest, dict]]):
                The request object. Request message for
                UpdatePolicyBinding method.
            policy_binding (:class:`google.cloud.iam_v3.types.PolicyBinding`):
                Required. The policy binding to update.

                The policy binding's ``name`` field is used to identify
                the policy binding to update.

                This corresponds to the ``policy_binding`` field
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
                :class:`google.cloud.iam_v3.types.PolicyBinding` IAM
                policy binding resource.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [policy_binding, update_mask]
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
        if not isinstance(request, policy_bindings_service.UpdatePolicyBindingRequest):
            request = policy_bindings_service.UpdatePolicyBindingRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if policy_binding is not None:
            request.policy_binding = policy_binding
        if update_mask is not None:
            request.update_mask = update_mask

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_policy_binding
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata(
                (("policy_binding.name", request.policy_binding.name),)
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
            policy_binding_resources.PolicyBinding,
            metadata_type=operation_metadata.OperationMetadata,
        )

        # Done; return the response.
        return response

    async def delete_policy_binding(
        self,
        request: Optional[
            Union[policy_bindings_service.DeletePolicyBindingRequest, dict]
        ] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> operation_async.AsyncOperation:
        r"""Deletes a policy binding and returns a long-running
        operation. Callers will need the IAM permissions on both
        the policy and target. Once the binding is deleted, the
        policy no longer applies to the target.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_v3

            async def sample_delete_policy_binding():
                # Create a client
                client = iam_v3.PolicyBindingsAsyncClient()

                # Initialize request argument(s)
                request = iam_v3.DeletePolicyBindingRequest(
                    name="name_value",
                )

                # Make the request
                operation = client.delete_policy_binding(request=request)

                print("Waiting for operation to complete...")

                response = (await operation).result()

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[google.cloud.iam_v3.types.DeletePolicyBindingRequest, dict]]):
                The request object. Request message for
                DeletePolicyBinding method.
            name (:class:`str`):
                Required. The name of the policy binding to delete.

                Format:

                -  ``projects/{project_id}/locations/{location}/policyBindings/{policy_binding_id}``
                -  ``projects/{project_number}/locations/{location}/policyBindings/{policy_binding_id}``
                -  ``folders/{folder_id}/locations/{location}/policyBindings/{policy_binding_id}``
                -  ``organizations/{organization_id}/locations/{location}/policyBindings/{policy_binding_id}``

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
        if not isinstance(request, policy_bindings_service.DeletePolicyBindingRequest):
            request = policy_bindings_service.DeletePolicyBindingRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_policy_binding
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

    async def list_policy_bindings(
        self,
        request: Optional[
            Union[policy_bindings_service.ListPolicyBindingsRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.ListPolicyBindingsAsyncPager:
        r"""Lists policy bindings.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_v3

            async def sample_list_policy_bindings():
                # Create a client
                client = iam_v3.PolicyBindingsAsyncClient()

                # Initialize request argument(s)
                request = iam_v3.ListPolicyBindingsRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_policy_bindings(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.iam_v3.types.ListPolicyBindingsRequest, dict]]):
                The request object. Request message for
                ListPolicyBindings method.
            parent (:class:`str`):
                Required. The parent resource, which owns the collection
                of policy bindings.

                Format:

                -  ``projects/{project_id}/locations/{location}``
                -  ``projects/{project_number}/locations/{location}``
                -  ``folders/{folder_id}/locations/{location}``
                -  ``organizations/{organization_id}/locations/{location}``

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
            google.cloud.iam_v3.services.policy_bindings.pagers.ListPolicyBindingsAsyncPager:
                Response message for
                ListPolicyBindings method.
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

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
        if not isinstance(request, policy_bindings_service.ListPolicyBindingsRequest):
            request = policy_bindings_service.ListPolicyBindingsRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_policy_bindings
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
        response = pagers.ListPolicyBindingsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def search_target_policy_bindings(
        self,
        request: Optional[
            Union[policy_bindings_service.SearchTargetPolicyBindingsRequest, dict]
        ] = None,
        *,
        parent: Optional[str] = None,
        target: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
    ) -> pagers.SearchTargetPolicyBindingsAsyncPager:
        r"""Search policy bindings by target. Returns all policy
        binding objects bound directly to target.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import iam_v3

            async def sample_search_target_policy_bindings():
                # Create a client
                client = iam_v3.PolicyBindingsAsyncClient()

                # Initialize request argument(s)
                request = iam_v3.SearchTargetPolicyBindingsRequest(
                    target="target_value",
                    parent="parent_value",
                )

                # Make the request
                page_result = client.search_target_policy_bindings(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[google.cloud.iam_v3.types.SearchTargetPolicyBindingsRequest, dict]]):
                The request object. Request message for
                SearchTargetPolicyBindings method.
            parent (:class:`str`):
                Required. The parent resource where this search will be
                performed. This should be the nearest Resource Manager
                resource (project, folder, or organization) to the
                target.

                Format:

                -  ``projects/{project_id}/locations/{location}``
                -  ``projects/{project_number}/locations/{location}``
                -  ``folders/{folder_id}/locations/{location}``
                -  ``organizations/{organization_id}/locations/{location}``

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            target (:class:`str`):
                Required. The target resource, which is bound to the
                policy in the binding.

                Format:

                -  ``//iam.googleapis.com/locations/global/workforcePools/POOL_ID``
                -  ``//iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/POOL_ID``
                -  ``//iam.googleapis.com/locations/global/workspace/WORKSPACE_ID``
                -  ``//cloudresourcemanager.googleapis.com/projects/{project_number}``
                -  ``//cloudresourcemanager.googleapis.com/folders/{folder_id}``
                -  ``//cloudresourcemanager.googleapis.com/organizations/{organization_id}``

                This corresponds to the ``target`` field
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
            google.cloud.iam_v3.services.policy_bindings.pagers.SearchTargetPolicyBindingsAsyncPager:
                Response message for
                SearchTargetPolicyBindings method.
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [parent, target]
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
            request, policy_bindings_service.SearchTargetPolicyBindingsRequest
        ):
            request = policy_bindings_service.SearchTargetPolicyBindingsRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if parent is not None:
            request.parent = parent
        if target is not None:
            request.target = target

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.search_target_policy_bindings
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
        response = pagers.SearchTargetPolicyBindingsAsyncPager(
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

    async def __aenter__(self) -> "PolicyBindingsAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.transport.close()


DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


__all__ = ("PolicyBindingsAsyncClient",)
