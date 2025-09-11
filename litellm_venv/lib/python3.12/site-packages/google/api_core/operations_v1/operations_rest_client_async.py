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
from typing import Optional, Sequence, Tuple, Union

from google.api_core import client_options as client_options_lib  # type: ignore
from google.api_core import gapic_v1  # type: ignore
from google.api_core.operations_v1 import pagers_async as pagers
from google.api_core.operations_v1.transports.base import (
    DEFAULT_CLIENT_INFO,
    OperationsTransport,
)
from google.api_core.operations_v1.abstract_operations_base_client import (
    AbstractOperationsBaseClient,
)
from google.longrunning import operations_pb2

try:
    from google.auth.aio import credentials as ga_credentials  # type: ignore
except ImportError as e:  # pragma: NO COVER
    raise ImportError(
        "The `async_rest` extra of `google-api-core` is required to use long-running operations.  Install it by running "
        "`pip install google-api-core[async_rest]`."
    ) from e


class AsyncOperationsRestClient(AbstractOperationsBaseClient):
    """Manages long-running operations with a REST API service for the asynchronous client.

    When an API method normally takes long time to complete, it can be
    designed to return [Operation][google.api_core.operations_v1.Operation] to the
    client, and the client can use this interface to receive the real
    response asynchronously by polling the operation resource, or pass
    the operation resource to another API (such as Google Cloud Pub/Sub
    API) to receive the response. Any API service that returns
    long-running operations should implement the ``Operations``
    interface so developers can have a consistent client experience.
    """

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Union[str, OperationsTransport, None] = None,
        client_options: Optional[client_options_lib.ClientOptions] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the operations client.

        Args:
            credentials (Optional[google.auth.aio.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Union[str, OperationsTransport]): The
                transport to use. If set to None, this defaults to 'rest_asyncio'.
            client_options (google.api_core.client_options.ClientOptions): Custom options for the
                client. It won't take effect if a ``transport`` instance is provided.
                (1) The ``api_endpoint`` property can be used to override the
                default endpoint provided by the client. GOOGLE_API_USE_MTLS_ENDPOINT
                environment variable can also be used to override the endpoint:
                "always" (always use the default mTLS endpoint), "never" (always
                use the default regular endpoint) and "auto" (auto switch to the
                default mTLS endpoint if client certificate is present, this is
                the default value). However, the ``api_endpoint`` property takes
                precedence if provided.
                (2) If GOOGLE_API_USE_CLIENT_CERTIFICATE environment variable
                is "true", then the ``client_cert_source`` property can be used
                to provide client certificate for mutual TLS transport. If
                not provided, the default SSL client certificate will be used if
                present. If GOOGLE_API_USE_CLIENT_CERTIFICATE is "false" or not
                set, no client certificate will be used.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If mutual TLS transport
                creation failed for any reason.
        """
        super().__init__(
            credentials=credentials,  # type: ignore
            # NOTE: If a transport is not provided, we force the client to use the async
            # REST transport.
            transport=transport or "rest_asyncio",
            client_options=client_options,
            client_info=client_info,
        )

    async def get_operation(
        self,
        name: str,
        *,
        # TODO(https://github.com/googleapis/python-api-core/issues/722): Leverage `retry`
        # to allow configuring retryable error codes.
        retry=gapic_v1.method_async.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> operations_pb2.Operation:
        r"""Gets the latest state of a long-running operation.
        Clients can use this method to poll the operation result
        at intervals as recommended by the API service.

        Args:
            name (str):
                The name of the operation resource.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.longrunning.operations_pb2.Operation:
                This resource represents a long-
                running operation that is the result of a
                network API call.

        """

        request = operations_pb2.GetOperationRequest(name=name)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.get_operation]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata or ()) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def list_operations(
        self,
        name: str,
        filter_: Optional[str] = None,
        *,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        # TODO(https://github.com/googleapis/python-api-core/issues/722): Leverage `retry`
        # to allow configuring retryable error codes.
        retry=gapic_v1.method_async.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> pagers.ListOperationsAsyncPager:
        r"""Lists operations that match the specified filter in the request.
        If the server doesn't support this method, it returns
        ``UNIMPLEMENTED``.

        NOTE: the ``name`` binding allows API services to override the
        binding to use different resource name schemes, such as
        ``users/*/operations``. To override the binding, API services
        can add a binding such as ``"/v1/{name=users/*}/operations"`` to
        their service configuration. For backwards compatibility, the
        default name includes the operations collection id, however
        overriding users must ensure the name binding is the parent
        resource, without the operations collection id.

        Args:
            name (str):
                The name of the operation's parent
                resource.
            filter_ (str):
                The standard list filter.
                This corresponds to the ``filter`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.api_core.operations_v1.pagers.ListOperationsPager:
                The response message for
                [Operations.ListOperations][google.api_core.operations_v1.Operations.ListOperations].

                Iterating over this object will yield results and
                resolve additional pages automatically.

        """
        # Create a protobuf request object.
        request = operations_pb2.ListOperationsRequest(name=name, filter=filter_)
        if page_size is not None:
            request.page_size = page_size
        if page_token is not None:
            request.page_token = page_token

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.list_operations]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata or ()) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListOperationsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def delete_operation(
        self,
        name: str,
        *,
        # TODO(https://github.com/googleapis/python-api-core/issues/722): Leverage `retry`
        # to allow configuring retryable error codes.
        retry=gapic_v1.method_async.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> None:
        r"""Deletes a long-running operation. This method indicates that the
        client is no longer interested in the operation result. It does
        not cancel the operation. If the server doesn't support this
        method, it returns ``google.rpc.Code.UNIMPLEMENTED``.

        Args:
            name (str):
                The name of the operation resource to
                be deleted.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        # Create the request object.
        request = operations_pb2.DeleteOperationRequest(name=name)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.delete_operation]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata or ()) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

    async def cancel_operation(
        self,
        name: Optional[str] = None,
        *,
        # TODO(https://github.com/googleapis/python-api-core/issues/722): Leverage `retry`
        # to allow configuring retryable error codes.
        retry=gapic_v1.method_async.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> None:
        r"""Starts asynchronous cancellation on a long-running operation.
        The server makes a best effort to cancel the operation, but
        success is not guaranteed. If the server doesn't support this
        method, it returns ``google.rpc.Code.UNIMPLEMENTED``. Clients
        can use
        [Operations.GetOperation][google.api_core.operations_v1.Operations.GetOperation]
        or other methods to check whether the cancellation succeeded or
        whether the operation completed despite cancellation. On
        successful cancellation, the operation is not deleted; instead,
        it becomes an operation with an
        [Operation.error][google.api_core.operations_v1.Operation.error] value with
        a [google.rpc.Status.code][google.rpc.Status.code] of 1,
        corresponding to ``Code.CANCELLED``.

        Args:
            name (str):
                The name of the operation resource to
                be cancelled.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        # Create the request object.
        request = operations_pb2.CancelOperationRequest(name=name)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.cancel_operation]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata or ()) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        await rpc(
            request,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )
