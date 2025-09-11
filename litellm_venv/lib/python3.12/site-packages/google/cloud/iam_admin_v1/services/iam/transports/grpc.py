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
import json
import logging as std_logging
import pickle
from typing import Callable, Dict, Optional, Sequence, Tuple, Union
import warnings

from google.api_core import gapic_v1, grpc_helpers
import google.auth  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.protobuf import empty_pb2  # type: ignore
from google.protobuf.json_format import MessageToJson
import google.protobuf.message
import grpc  # type: ignore
import proto  # type: ignore

from google.cloud.iam_admin_v1.types import iam

from .base import DEFAULT_CLIENT_INFO, IAMTransport

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = std_logging.getLogger(__name__)


class _LoggingClientInterceptor(grpc.UnaryUnaryClientInterceptor):  # pragma: NO COVER
    def intercept_unary_unary(self, continuation, client_call_details, request):
        logging_enabled = CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
            std_logging.DEBUG
        )
        if logging_enabled:  # pragma: NO COVER
            request_metadata = client_call_details.metadata
            if isinstance(request, proto.Message):
                request_payload = type(request).to_json(request)
            elif isinstance(request, google.protobuf.message.Message):
                request_payload = MessageToJson(request)
            else:
                request_payload = f"{type(request).__name__}: {pickle.dumps(request)}"

            request_metadata = {
                key: value.decode("utf-8") if isinstance(value, bytes) else value
                for key, value in request_metadata
            }
            grpc_request = {
                "payload": request_payload,
                "requestMethod": "grpc",
                "metadata": dict(request_metadata),
            }
            _LOGGER.debug(
                f"Sending request for {client_call_details.method}",
                extra={
                    "serviceName": "google.iam.admin.v1.IAM",
                    "rpcName": str(client_call_details.method),
                    "request": grpc_request,
                    "metadata": grpc_request["metadata"],
                },
            )
        response = continuation(client_call_details, request)
        if logging_enabled:  # pragma: NO COVER
            response_metadata = response.trailing_metadata()
            # Convert gRPC metadata `<class 'grpc.aio._metadata.Metadata'>` to list of tuples
            metadata = (
                dict([(k, str(v)) for k, v in response_metadata])
                if response_metadata
                else None
            )
            result = response.result()
            if isinstance(result, proto.Message):
                response_payload = type(result).to_json(result)
            elif isinstance(result, google.protobuf.message.Message):
                response_payload = MessageToJson(result)
            else:
                response_payload = f"{type(result).__name__}: {pickle.dumps(result)}"
            grpc_response = {
                "payload": response_payload,
                "metadata": metadata,
                "status": "OK",
            }
            _LOGGER.debug(
                f"Received response for {client_call_details.method}.",
                extra={
                    "serviceName": "google.iam.admin.v1.IAM",
                    "rpcName": client_call_details.method,
                    "response": grpc_response,
                    "metadata": grpc_response["metadata"],
                },
            )
        return response


class IAMGrpcTransport(IAMTransport):
    """gRPC backend transport for IAM.

    Creates and manages Identity and Access Management (IAM) resources.

    You can use this service to work with all of the following
    resources:

    -  **Service accounts**, which identify an application or a virtual
       machine (VM) instance rather than a person
    -  **Service account keys**, which service accounts use to
       authenticate with Google APIs
    -  **IAM policies for service accounts**, which specify the roles
       that a principal has for the service account
    -  **IAM custom roles**, which help you limit the number of
       permissions that you grant to principals

    In addition, you can use this service to complete the following
    tasks, among others:

    -  Test whether a service account can use specific permissions
    -  Check which roles you can grant for a specific resource
    -  Lint, or validate, condition expressions in an IAM policy

    When you read data from the IAM API, each read is eventually
    consistent. In other words, if you write data with the IAM API, then
    immediately read that data, the read operation might return an older
    version of the data. To deal with this behavior, your application
    can retry the request with truncated exponential backoff.

    In contrast, writing data to the IAM API is sequentially consistent.
    In other words, write operations are always processed in the order
    in which they were received.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends protocol buffers over the wire using gRPC (which is built on
    top of HTTP/2); the ``grpcio`` package must be installed.
    """

    _stubs: Dict[str, Callable]

    def __init__(
        self,
        *,
        host: str = "iam.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        channel: Optional[Union[grpc.Channel, Callable[..., grpc.Channel]]] = None,
        api_mtls_endpoint: Optional[str] = None,
        client_cert_source: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        ssl_channel_credentials: Optional[grpc.ChannelCredentials] = None,
        client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
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
                This argument is ignored if a ``channel`` instance is provided.
            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if a ``channel`` instance is provided.
            scopes (Optional(Sequence[str])): A list of scopes. This argument is
                ignored if a ``channel`` instance is provided.
            channel (Optional[Union[grpc.Channel, Callable[..., grpc.Channel]]]):
                A ``Channel`` instance through which to make calls, or a Callable
                that constructs and returns one. If set to None, ``self.create_channel``
                is used to create the channel. If a Callable is given, it will be called
                with the same arguments as used in ``self.create_channel``.
            api_mtls_endpoint (Optional[str]): Deprecated. The mutual TLS endpoint.
                If provided, it overrides the ``host`` argument and tries to create
                a mutual TLS channel with client SSL credentials from
                ``client_cert_source`` or application default SSL credentials.
            client_cert_source (Optional[Callable[[], Tuple[bytes, bytes]]]):
                Deprecated. A callback to provide client SSL certificate bytes and
                private key bytes, both in PEM format. It is ignored if
                ``api_mtls_endpoint`` is None.
            ssl_channel_credentials (grpc.ChannelCredentials): SSL credentials
                for the grpc channel. It is ignored if a ``channel`` instance is provided.
            client_cert_source_for_mtls (Optional[Callable[[], Tuple[bytes, bytes]]]):
                A callback to provide client certificate bytes and private key bytes,
                both in PEM format. It is used to configure a mutual TLS channel. It is
                ignored if a ``channel`` instance or ``ssl_channel_credentials`` is provided.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.

        Raises:
          google.auth.exceptions.MutualTLSChannelError: If mutual TLS transport
              creation failed for any reason.
          google.api_core.exceptions.DuplicateCredentialArgs: If both ``credentials``
              and ``credentials_file`` are passed.
        """
        self._grpc_channel = None
        self._ssl_channel_credentials = ssl_channel_credentials
        self._stubs: Dict[str, Callable] = {}

        if api_mtls_endpoint:
            warnings.warn("api_mtls_endpoint is deprecated", DeprecationWarning)
        if client_cert_source:
            warnings.warn("client_cert_source is deprecated", DeprecationWarning)

        if isinstance(channel, grpc.Channel):
            # Ignore credentials if a channel was passed.
            credentials = None
            self._ignore_credentials = True
            # If a channel was explicitly provided, set it.
            self._grpc_channel = channel
            self._ssl_channel_credentials = None

        else:
            if api_mtls_endpoint:
                host = api_mtls_endpoint

                # Create SSL credentials with client_cert_source or application
                # default SSL credentials.
                if client_cert_source:
                    cert, key = client_cert_source()
                    self._ssl_channel_credentials = grpc.ssl_channel_credentials(
                        certificate_chain=cert, private_key=key
                    )
                else:
                    self._ssl_channel_credentials = SslCredentials().ssl_credentials

            else:
                if client_cert_source_for_mtls and not ssl_channel_credentials:
                    cert, key = client_cert_source_for_mtls()
                    self._ssl_channel_credentials = grpc.ssl_channel_credentials(
                        certificate_chain=cert, private_key=key
                    )

        # The base transport sets the host, credentials and scopes
        super().__init__(
            host=host,
            credentials=credentials,
            credentials_file=credentials_file,
            scopes=scopes,
            quota_project_id=quota_project_id,
            client_info=client_info,
            always_use_jwt_access=always_use_jwt_access,
            api_audience=api_audience,
        )

        if not self._grpc_channel:
            # initialize with the provided callable or the default channel
            channel_init = channel or type(self).create_channel
            self._grpc_channel = channel_init(
                self._host,
                # use the credentials which are saved
                credentials=self._credentials,
                # Set ``credentials_file`` to ``None`` here as
                # the credentials that we saved earlier should be used.
                credentials_file=None,
                scopes=self._scopes,
                ssl_credentials=self._ssl_channel_credentials,
                quota_project_id=quota_project_id,
                options=[
                    ("grpc.max_send_message_length", -1),
                    ("grpc.max_receive_message_length", -1),
                ],
            )

        self._interceptor = _LoggingClientInterceptor()
        self._logged_channel = grpc.intercept_channel(
            self._grpc_channel, self._interceptor
        )

        # Wrap messages. This must be done after self._logged_channel exists
        self._prep_wrapped_messages(client_info)

    @classmethod
    def create_channel(
        cls,
        host: str = "iam.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        quota_project_id: Optional[str] = None,
        **kwargs,
    ) -> grpc.Channel:
        """Create and return a gRPC channel object.
        Args:
            host (Optional[str]): The host for the channel to use.
            credentials (Optional[~.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify this application to the service. If
                none are specified, the client will attempt to ascertain
                the credentials from the environment.
            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is mutually exclusive with credentials.
            scopes (Optional[Sequence[str]]): A optional list of scopes needed for this
                service. These are only used when credentials are not specified and
                are passed to :func:`google.auth.default`.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            kwargs (Optional[dict]): Keyword arguments, which are passed to the
                channel creation.
        Returns:
            grpc.Channel: A gRPC channel object.

        Raises:
            google.api_core.exceptions.DuplicateCredentialArgs: If both ``credentials``
              and ``credentials_file`` are passed.
        """

        return grpc_helpers.create_channel(
            host,
            credentials=credentials,
            credentials_file=credentials_file,
            quota_project_id=quota_project_id,
            default_scopes=cls.AUTH_SCOPES,
            scopes=scopes,
            default_host=cls.DEFAULT_HOST,
            **kwargs,
        )

    @property
    def grpc_channel(self) -> grpc.Channel:
        """Return the channel designed to connect to this service."""
        return self._grpc_channel

    @property
    def list_service_accounts(
        self,
    ) -> Callable[[iam.ListServiceAccountsRequest], iam.ListServiceAccountsResponse]:
        r"""Return a callable for the list service accounts method over gRPC.

        Lists every [ServiceAccount][google.iam.admin.v1.ServiceAccount]
        that belongs to a specific project.

        Returns:
            Callable[[~.ListServiceAccountsRequest],
                    ~.ListServiceAccountsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_service_accounts" not in self._stubs:
            self._stubs["list_service_accounts"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/ListServiceAccounts",
                request_serializer=iam.ListServiceAccountsRequest.serialize,
                response_deserializer=iam.ListServiceAccountsResponse.deserialize,
            )
        return self._stubs["list_service_accounts"]

    @property
    def get_service_account(
        self,
    ) -> Callable[[iam.GetServiceAccountRequest], iam.ServiceAccount]:
        r"""Return a callable for the get service account method over gRPC.

        Gets a [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        Returns:
            Callable[[~.GetServiceAccountRequest],
                    ~.ServiceAccount]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_service_account" not in self._stubs:
            self._stubs["get_service_account"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/GetServiceAccount",
                request_serializer=iam.GetServiceAccountRequest.serialize,
                response_deserializer=iam.ServiceAccount.deserialize,
            )
        return self._stubs["get_service_account"]

    @property
    def create_service_account(
        self,
    ) -> Callable[[iam.CreateServiceAccountRequest], iam.ServiceAccount]:
        r"""Return a callable for the create service account method over gRPC.

        Creates a [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        Returns:
            Callable[[~.CreateServiceAccountRequest],
                    ~.ServiceAccount]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_service_account" not in self._stubs:
            self._stubs["create_service_account"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/CreateServiceAccount",
                request_serializer=iam.CreateServiceAccountRequest.serialize,
                response_deserializer=iam.ServiceAccount.deserialize,
            )
        return self._stubs["create_service_account"]

    @property
    def update_service_account(
        self,
    ) -> Callable[[iam.ServiceAccount], iam.ServiceAccount]:
        r"""Return a callable for the update service account method over gRPC.

        **Note:** We are in the process of deprecating this method. Use
        [PatchServiceAccount][google.iam.admin.v1.IAM.PatchServiceAccount]
        instead.

        Updates a [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        You can update only the ``display_name`` field.

        Returns:
            Callable[[~.ServiceAccount],
                    ~.ServiceAccount]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_service_account" not in self._stubs:
            self._stubs["update_service_account"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/UpdateServiceAccount",
                request_serializer=iam.ServiceAccount.serialize,
                response_deserializer=iam.ServiceAccount.deserialize,
            )
        return self._stubs["update_service_account"]

    @property
    def patch_service_account(
        self,
    ) -> Callable[[iam.PatchServiceAccountRequest], iam.ServiceAccount]:
        r"""Return a callable for the patch service account method over gRPC.

        Patches a [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        Returns:
            Callable[[~.PatchServiceAccountRequest],
                    ~.ServiceAccount]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "patch_service_account" not in self._stubs:
            self._stubs["patch_service_account"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/PatchServiceAccount",
                request_serializer=iam.PatchServiceAccountRequest.serialize,
                response_deserializer=iam.ServiceAccount.deserialize,
            )
        return self._stubs["patch_service_account"]

    @property
    def delete_service_account(
        self,
    ) -> Callable[[iam.DeleteServiceAccountRequest], empty_pb2.Empty]:
        r"""Return a callable for the delete service account method over gRPC.

        Deletes a [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        **Warning:** After you delete a service account, you might not
        be able to undelete it. If you know that you need to re-enable
        the service account in the future, use
        [DisableServiceAccount][google.iam.admin.v1.IAM.DisableServiceAccount]
        instead.

        If you delete a service account, IAM permanently removes the
        service account 30 days later. Google Cloud cannot recover the
        service account after it is permanently removed, even if you
        file a support request.

        To help avoid unplanned outages, we recommend that you disable
        the service account before you delete it. Use
        [DisableServiceAccount][google.iam.admin.v1.IAM.DisableServiceAccount]
        to disable the service account, then wait at least 24 hours and
        watch for unintended consequences. If there are no unintended
        consequences, you can delete the service account.

        Returns:
            Callable[[~.DeleteServiceAccountRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_service_account" not in self._stubs:
            self._stubs["delete_service_account"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/DeleteServiceAccount",
                request_serializer=iam.DeleteServiceAccountRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_service_account"]

    @property
    def undelete_service_account(
        self,
    ) -> Callable[
        [iam.UndeleteServiceAccountRequest], iam.UndeleteServiceAccountResponse
    ]:
        r"""Return a callable for the undelete service account method over gRPC.

        Restores a deleted
        [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        **Important:** It is not always possible to restore a deleted
        service account. Use this method only as a last resort.

        After you delete a service account, IAM permanently removes the
        service account 30 days later. There is no way to restore a
        deleted service account that has been permanently removed.

        Returns:
            Callable[[~.UndeleteServiceAccountRequest],
                    ~.UndeleteServiceAccountResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "undelete_service_account" not in self._stubs:
            self._stubs["undelete_service_account"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/UndeleteServiceAccount",
                request_serializer=iam.UndeleteServiceAccountRequest.serialize,
                response_deserializer=iam.UndeleteServiceAccountResponse.deserialize,
            )
        return self._stubs["undelete_service_account"]

    @property
    def enable_service_account(
        self,
    ) -> Callable[[iam.EnableServiceAccountRequest], empty_pb2.Empty]:
        r"""Return a callable for the enable service account method over gRPC.

        Enables a [ServiceAccount][google.iam.admin.v1.ServiceAccount]
        that was disabled by
        [DisableServiceAccount][google.iam.admin.v1.IAM.DisableServiceAccount].

        If the service account is already enabled, then this method has
        no effect.

        If the service account was disabled by other means—for example,
        if Google disabled the service account because it was
        compromised—you cannot use this method to enable the service
        account.

        Returns:
            Callable[[~.EnableServiceAccountRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "enable_service_account" not in self._stubs:
            self._stubs["enable_service_account"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/EnableServiceAccount",
                request_serializer=iam.EnableServiceAccountRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["enable_service_account"]

    @property
    def disable_service_account(
        self,
    ) -> Callable[[iam.DisableServiceAccountRequest], empty_pb2.Empty]:
        r"""Return a callable for the disable service account method over gRPC.

        Disables a [ServiceAccount][google.iam.admin.v1.ServiceAccount]
        immediately.

        If an application uses the service account to authenticate, that
        application can no longer call Google APIs or access Google
        Cloud resources. Existing access tokens for the service account
        are rejected, and requests for new access tokens will fail.

        To re-enable the service account, use
        [EnableServiceAccount][google.iam.admin.v1.IAM.EnableServiceAccount].
        After you re-enable the service account, its existing access
        tokens will be accepted, and you can request new access tokens.

        To help avoid unplanned outages, we recommend that you disable
        the service account before you delete it. Use this method to
        disable the service account, then wait at least 24 hours and
        watch for unintended consequences. If there are no unintended
        consequences, you can delete the service account with
        [DeleteServiceAccount][google.iam.admin.v1.IAM.DeleteServiceAccount].

        Returns:
            Callable[[~.DisableServiceAccountRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "disable_service_account" not in self._stubs:
            self._stubs["disable_service_account"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/DisableServiceAccount",
                request_serializer=iam.DisableServiceAccountRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["disable_service_account"]

    @property
    def list_service_account_keys(
        self,
    ) -> Callable[
        [iam.ListServiceAccountKeysRequest], iam.ListServiceAccountKeysResponse
    ]:
        r"""Return a callable for the list service account keys method over gRPC.

        Lists every
        [ServiceAccountKey][google.iam.admin.v1.ServiceAccountKey] for a
        service account.

        Returns:
            Callable[[~.ListServiceAccountKeysRequest],
                    ~.ListServiceAccountKeysResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_service_account_keys" not in self._stubs:
            self._stubs["list_service_account_keys"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/ListServiceAccountKeys",
                request_serializer=iam.ListServiceAccountKeysRequest.serialize,
                response_deserializer=iam.ListServiceAccountKeysResponse.deserialize,
            )
        return self._stubs["list_service_account_keys"]

    @property
    def get_service_account_key(
        self,
    ) -> Callable[[iam.GetServiceAccountKeyRequest], iam.ServiceAccountKey]:
        r"""Return a callable for the get service account key method over gRPC.

        Gets a
        [ServiceAccountKey][google.iam.admin.v1.ServiceAccountKey].

        Returns:
            Callable[[~.GetServiceAccountKeyRequest],
                    ~.ServiceAccountKey]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_service_account_key" not in self._stubs:
            self._stubs["get_service_account_key"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/GetServiceAccountKey",
                request_serializer=iam.GetServiceAccountKeyRequest.serialize,
                response_deserializer=iam.ServiceAccountKey.deserialize,
            )
        return self._stubs["get_service_account_key"]

    @property
    def create_service_account_key(
        self,
    ) -> Callable[[iam.CreateServiceAccountKeyRequest], iam.ServiceAccountKey]:
        r"""Return a callable for the create service account key method over gRPC.

        Creates a
        [ServiceAccountKey][google.iam.admin.v1.ServiceAccountKey].

        Returns:
            Callable[[~.CreateServiceAccountKeyRequest],
                    ~.ServiceAccountKey]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_service_account_key" not in self._stubs:
            self._stubs[
                "create_service_account_key"
            ] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/CreateServiceAccountKey",
                request_serializer=iam.CreateServiceAccountKeyRequest.serialize,
                response_deserializer=iam.ServiceAccountKey.deserialize,
            )
        return self._stubs["create_service_account_key"]

    @property
    def upload_service_account_key(
        self,
    ) -> Callable[[iam.UploadServiceAccountKeyRequest], iam.ServiceAccountKey]:
        r"""Return a callable for the upload service account key method over gRPC.

        Uploads the public key portion of a key pair that you manage,
        and associates the public key with a
        [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        After you upload the public key, you can use the private key
        from the key pair as a service account key.

        Returns:
            Callable[[~.UploadServiceAccountKeyRequest],
                    ~.ServiceAccountKey]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "upload_service_account_key" not in self._stubs:
            self._stubs[
                "upload_service_account_key"
            ] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/UploadServiceAccountKey",
                request_serializer=iam.UploadServiceAccountKeyRequest.serialize,
                response_deserializer=iam.ServiceAccountKey.deserialize,
            )
        return self._stubs["upload_service_account_key"]

    @property
    def delete_service_account_key(
        self,
    ) -> Callable[[iam.DeleteServiceAccountKeyRequest], empty_pb2.Empty]:
        r"""Return a callable for the delete service account key method over gRPC.

        Deletes a
        [ServiceAccountKey][google.iam.admin.v1.ServiceAccountKey].
        Deleting a service account key does not revoke short-lived
        credentials that have been issued based on the service account
        key.

        Returns:
            Callable[[~.DeleteServiceAccountKeyRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_service_account_key" not in self._stubs:
            self._stubs[
                "delete_service_account_key"
            ] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/DeleteServiceAccountKey",
                request_serializer=iam.DeleteServiceAccountKeyRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_service_account_key"]

    @property
    def disable_service_account_key(
        self,
    ) -> Callable[[iam.DisableServiceAccountKeyRequest], empty_pb2.Empty]:
        r"""Return a callable for the disable service account key method over gRPC.

        Disable a
        [ServiceAccountKey][google.iam.admin.v1.ServiceAccountKey]. A
        disabled service account key can be re-enabled with
        [EnableServiceAccountKey][google.iam.admin.v1.IAM.EnableServiceAccountKey].

        Returns:
            Callable[[~.DisableServiceAccountKeyRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "disable_service_account_key" not in self._stubs:
            self._stubs[
                "disable_service_account_key"
            ] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/DisableServiceAccountKey",
                request_serializer=iam.DisableServiceAccountKeyRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["disable_service_account_key"]

    @property
    def enable_service_account_key(
        self,
    ) -> Callable[[iam.EnableServiceAccountKeyRequest], empty_pb2.Empty]:
        r"""Return a callable for the enable service account key method over gRPC.

        Enable a
        [ServiceAccountKey][google.iam.admin.v1.ServiceAccountKey].

        Returns:
            Callable[[~.EnableServiceAccountKeyRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "enable_service_account_key" not in self._stubs:
            self._stubs[
                "enable_service_account_key"
            ] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/EnableServiceAccountKey",
                request_serializer=iam.EnableServiceAccountKeyRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["enable_service_account_key"]

    @property
    def sign_blob(self) -> Callable[[iam.SignBlobRequest], iam.SignBlobResponse]:
        r"""Return a callable for the sign blob method over gRPC.

        **Note:** This method is deprecated. Use the
        ```signBlob`` <https://cloud.google.com/iam/help/rest-credentials/v1/projects.serviceAccounts/signBlob>`__
        method in the IAM Service Account Credentials API instead. If
        you currently use this method, see the `migration
        guide <https://cloud.google.com/iam/help/credentials/migrate-api>`__
        for instructions.

        Signs a blob using the system-managed private key for a
        [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        Returns:
            Callable[[~.SignBlobRequest],
                    ~.SignBlobResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "sign_blob" not in self._stubs:
            self._stubs["sign_blob"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/SignBlob",
                request_serializer=iam.SignBlobRequest.serialize,
                response_deserializer=iam.SignBlobResponse.deserialize,
            )
        return self._stubs["sign_blob"]

    @property
    def sign_jwt(self) -> Callable[[iam.SignJwtRequest], iam.SignJwtResponse]:
        r"""Return a callable for the sign jwt method over gRPC.

        **Note:** This method is deprecated. Use the
        ```signJwt`` <https://cloud.google.com/iam/help/rest-credentials/v1/projects.serviceAccounts/signJwt>`__
        method in the IAM Service Account Credentials API instead. If
        you currently use this method, see the `migration
        guide <https://cloud.google.com/iam/help/credentials/migrate-api>`__
        for instructions.

        Signs a JSON Web Token (JWT) using the system-managed private
        key for a [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        Returns:
            Callable[[~.SignJwtRequest],
                    ~.SignJwtResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "sign_jwt" not in self._stubs:
            self._stubs["sign_jwt"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/SignJwt",
                request_serializer=iam.SignJwtRequest.serialize,
                response_deserializer=iam.SignJwtResponse.deserialize,
            )
        return self._stubs["sign_jwt"]

    @property
    def get_iam_policy(
        self,
    ) -> Callable[[iam_policy_pb2.GetIamPolicyRequest], policy_pb2.Policy]:
        r"""Return a callable for the get iam policy method over gRPC.

        Gets the IAM policy that is attached to a
        [ServiceAccount][google.iam.admin.v1.ServiceAccount]. This IAM
        policy specifies which principals have access to the service
        account.

        This method does not tell you whether the service account has
        been granted any roles on other resources. To check whether a
        service account has role grants on a resource, use the
        ``getIamPolicy`` method for that resource. For example, to view
        the role grants for a project, call the Resource Manager API's
        ```projects.getIamPolicy`` <https://cloud.google.com/resource-manager/reference/rest/v1/projects/getIamPolicy>`__
        method.

        Returns:
            Callable[[~.GetIamPolicyRequest],
                    ~.Policy]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_iam_policy" not in self._stubs:
            self._stubs["get_iam_policy"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/GetIamPolicy",
                request_serializer=iam_policy_pb2.GetIamPolicyRequest.SerializeToString,
                response_deserializer=policy_pb2.Policy.FromString,
            )
        return self._stubs["get_iam_policy"]

    @property
    def set_iam_policy(
        self,
    ) -> Callable[[iam_policy_pb2.SetIamPolicyRequest], policy_pb2.Policy]:
        r"""Return a callable for the set iam policy method over gRPC.

        Sets the IAM policy that is attached to a
        [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        Use this method to grant or revoke access to the service
        account. For example, you could grant a principal the ability to
        impersonate the service account.

        This method does not enable the service account to access other
        resources. To grant roles to a service account on a resource,
        follow these steps:

        1. Call the resource's ``getIamPolicy`` method to get its
           current IAM policy.
        2. Edit the policy so that it binds the service account to an
           IAM role for the resource.
        3. Call the resource's ``setIamPolicy`` method to update its IAM
           policy.

        For detailed instructions, see `Manage access to project,
        folders, and
        organizations <https://cloud.google.com/iam/help/service-accounts/granting-access-to-service-accounts>`__
        or `Manage access to other
        resources <https://cloud.google.com/iam/help/access/manage-other-resources>`__.

        Returns:
            Callable[[~.SetIamPolicyRequest],
                    ~.Policy]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "set_iam_policy" not in self._stubs:
            self._stubs["set_iam_policy"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/SetIamPolicy",
                request_serializer=iam_policy_pb2.SetIamPolicyRequest.SerializeToString,
                response_deserializer=policy_pb2.Policy.FromString,
            )
        return self._stubs["set_iam_policy"]

    @property
    def test_iam_permissions(
        self,
    ) -> Callable[
        [iam_policy_pb2.TestIamPermissionsRequest],
        iam_policy_pb2.TestIamPermissionsResponse,
    ]:
        r"""Return a callable for the test iam permissions method over gRPC.

        Tests whether the caller has the specified permissions on a
        [ServiceAccount][google.iam.admin.v1.ServiceAccount].

        Returns:
            Callable[[~.TestIamPermissionsRequest],
                    ~.TestIamPermissionsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "test_iam_permissions" not in self._stubs:
            self._stubs["test_iam_permissions"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/TestIamPermissions",
                request_serializer=iam_policy_pb2.TestIamPermissionsRequest.SerializeToString,
                response_deserializer=iam_policy_pb2.TestIamPermissionsResponse.FromString,
            )
        return self._stubs["test_iam_permissions"]

    @property
    def query_grantable_roles(
        self,
    ) -> Callable[[iam.QueryGrantableRolesRequest], iam.QueryGrantableRolesResponse]:
        r"""Return a callable for the query grantable roles method over gRPC.

        Lists roles that can be granted on a Google Cloud
        resource. A role is grantable if the IAM policy for the
        resource can contain bindings to the role.

        Returns:
            Callable[[~.QueryGrantableRolesRequest],
                    ~.QueryGrantableRolesResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "query_grantable_roles" not in self._stubs:
            self._stubs["query_grantable_roles"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/QueryGrantableRoles",
                request_serializer=iam.QueryGrantableRolesRequest.serialize,
                response_deserializer=iam.QueryGrantableRolesResponse.deserialize,
            )
        return self._stubs["query_grantable_roles"]

    @property
    def list_roles(self) -> Callable[[iam.ListRolesRequest], iam.ListRolesResponse]:
        r"""Return a callable for the list roles method over gRPC.

        Lists every predefined [Role][google.iam.admin.v1.Role] that IAM
        supports, or every custom role that is defined for an
        organization or project.

        Returns:
            Callable[[~.ListRolesRequest],
                    ~.ListRolesResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_roles" not in self._stubs:
            self._stubs["list_roles"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/ListRoles",
                request_serializer=iam.ListRolesRequest.serialize,
                response_deserializer=iam.ListRolesResponse.deserialize,
            )
        return self._stubs["list_roles"]

    @property
    def get_role(self) -> Callable[[iam.GetRoleRequest], iam.Role]:
        r"""Return a callable for the get role method over gRPC.

        Gets the definition of a [Role][google.iam.admin.v1.Role].

        Returns:
            Callable[[~.GetRoleRequest],
                    ~.Role]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_role" not in self._stubs:
            self._stubs["get_role"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/GetRole",
                request_serializer=iam.GetRoleRequest.serialize,
                response_deserializer=iam.Role.deserialize,
            )
        return self._stubs["get_role"]

    @property
    def create_role(self) -> Callable[[iam.CreateRoleRequest], iam.Role]:
        r"""Return a callable for the create role method over gRPC.

        Creates a new custom [Role][google.iam.admin.v1.Role].

        Returns:
            Callable[[~.CreateRoleRequest],
                    ~.Role]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_role" not in self._stubs:
            self._stubs["create_role"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/CreateRole",
                request_serializer=iam.CreateRoleRequest.serialize,
                response_deserializer=iam.Role.deserialize,
            )
        return self._stubs["create_role"]

    @property
    def update_role(self) -> Callable[[iam.UpdateRoleRequest], iam.Role]:
        r"""Return a callable for the update role method over gRPC.

        Updates the definition of a custom
        [Role][google.iam.admin.v1.Role].

        Returns:
            Callable[[~.UpdateRoleRequest],
                    ~.Role]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_role" not in self._stubs:
            self._stubs["update_role"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/UpdateRole",
                request_serializer=iam.UpdateRoleRequest.serialize,
                response_deserializer=iam.Role.deserialize,
            )
        return self._stubs["update_role"]

    @property
    def delete_role(self) -> Callable[[iam.DeleteRoleRequest], iam.Role]:
        r"""Return a callable for the delete role method over gRPC.

        Deletes a custom [Role][google.iam.admin.v1.Role].

        When you delete a custom role, the following changes occur
        immediately:

        -  You cannot bind a principal to the custom role in an IAM
           [Policy][google.iam.v1.Policy].
        -  Existing bindings to the custom role are not changed, but
           they have no effect.
        -  By default, the response from
           [ListRoles][google.iam.admin.v1.IAM.ListRoles] does not
           include the custom role.

        You have 7 days to undelete the custom role. After 7 days, the
        following changes occur:

        -  The custom role is permanently deleted and cannot be
           recovered.
        -  If an IAM policy contains a binding to the custom role, the
           binding is permanently removed.

        Returns:
            Callable[[~.DeleteRoleRequest],
                    ~.Role]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_role" not in self._stubs:
            self._stubs["delete_role"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/DeleteRole",
                request_serializer=iam.DeleteRoleRequest.serialize,
                response_deserializer=iam.Role.deserialize,
            )
        return self._stubs["delete_role"]

    @property
    def undelete_role(self) -> Callable[[iam.UndeleteRoleRequest], iam.Role]:
        r"""Return a callable for the undelete role method over gRPC.

        Undeletes a custom [Role][google.iam.admin.v1.Role].

        Returns:
            Callable[[~.UndeleteRoleRequest],
                    ~.Role]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "undelete_role" not in self._stubs:
            self._stubs["undelete_role"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/UndeleteRole",
                request_serializer=iam.UndeleteRoleRequest.serialize,
                response_deserializer=iam.Role.deserialize,
            )
        return self._stubs["undelete_role"]

    @property
    def query_testable_permissions(
        self,
    ) -> Callable[
        [iam.QueryTestablePermissionsRequest], iam.QueryTestablePermissionsResponse
    ]:
        r"""Return a callable for the query testable permissions method over gRPC.

        Lists every permission that you can test on a
        resource. A permission is testable if you can check
        whether a principal has that permission on the resource.

        Returns:
            Callable[[~.QueryTestablePermissionsRequest],
                    ~.QueryTestablePermissionsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "query_testable_permissions" not in self._stubs:
            self._stubs[
                "query_testable_permissions"
            ] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/QueryTestablePermissions",
                request_serializer=iam.QueryTestablePermissionsRequest.serialize,
                response_deserializer=iam.QueryTestablePermissionsResponse.deserialize,
            )
        return self._stubs["query_testable_permissions"]

    @property
    def query_auditable_services(
        self,
    ) -> Callable[
        [iam.QueryAuditableServicesRequest], iam.QueryAuditableServicesResponse
    ]:
        r"""Return a callable for the query auditable services method over gRPC.

        Returns a list of services that allow you to opt into audit logs
        that are not generated by default.

        To learn more about audit logs, see the `Logging
        documentation <https://cloud.google.com/logging/docs/audit>`__.

        Returns:
            Callable[[~.QueryAuditableServicesRequest],
                    ~.QueryAuditableServicesResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "query_auditable_services" not in self._stubs:
            self._stubs["query_auditable_services"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/QueryAuditableServices",
                request_serializer=iam.QueryAuditableServicesRequest.serialize,
                response_deserializer=iam.QueryAuditableServicesResponse.deserialize,
            )
        return self._stubs["query_auditable_services"]

    @property
    def lint_policy(self) -> Callable[[iam.LintPolicyRequest], iam.LintPolicyResponse]:
        r"""Return a callable for the lint policy method over gRPC.

        Lints, or validates, an IAM policy. Currently checks the
        [google.iam.v1.Binding.condition][google.iam.v1.Binding.condition]
        field, which contains a condition expression for a role binding.

        Successful calls to this method always return an HTTP ``200 OK``
        status code, even if the linter detects an issue in the IAM
        policy.

        Returns:
            Callable[[~.LintPolicyRequest],
                    ~.LintPolicyResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "lint_policy" not in self._stubs:
            self._stubs["lint_policy"] = self._logged_channel.unary_unary(
                "/google.iam.admin.v1.IAM/LintPolicy",
                request_serializer=iam.LintPolicyRequest.serialize,
                response_deserializer=iam.LintPolicyResponse.deserialize,
            )
        return self._stubs["lint_policy"]

    def close(self):
        self._logged_channel.close()

    @property
    def kind(self) -> str:
        return "grpc"


__all__ = ("IAMGrpcTransport",)
