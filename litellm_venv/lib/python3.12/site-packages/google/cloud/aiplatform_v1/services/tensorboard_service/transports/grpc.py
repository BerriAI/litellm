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
import warnings
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

from google.api_core import grpc_helpers
from google.api_core import operations_v1
from google.api_core import gapic_v1
import google.auth  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore

import grpc  # type: ignore

from google.cloud.aiplatform_v1.types import tensorboard
from google.cloud.aiplatform_v1.types import tensorboard_experiment
from google.cloud.aiplatform_v1.types import (
    tensorboard_experiment as gca_tensorboard_experiment,
)
from google.cloud.aiplatform_v1.types import tensorboard_run
from google.cloud.aiplatform_v1.types import tensorboard_run as gca_tensorboard_run
from google.cloud.aiplatform_v1.types import tensorboard_service
from google.cloud.aiplatform_v1.types import tensorboard_time_series
from google.cloud.aiplatform_v1.types import (
    tensorboard_time_series as gca_tensorboard_time_series,
)
from google.cloud.location import locations_pb2  # type: ignore
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from .base import TensorboardServiceTransport, DEFAULT_CLIENT_INFO


class TensorboardServiceGrpcTransport(TensorboardServiceTransport):
    """gRPC backend transport for TensorboardService.

    TensorboardService

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
        host: str = "aiplatform.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        channel: Optional[grpc.Channel] = None,
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
                 The hostname to connect to (default: 'aiplatform.googleapis.com').
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
                This argument is ignored if ``channel`` is provided.
            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if ``channel`` is provided.
            scopes (Optional(Sequence[str])): A list of scopes. This argument is
                ignored if ``channel`` is provided.
            channel (Optional[grpc.Channel]): A ``Channel`` instance through
                which to make calls.
            api_mtls_endpoint (Optional[str]): Deprecated. The mutual TLS endpoint.
                If provided, it overrides the ``host`` argument and tries to create
                a mutual TLS channel with client SSL credentials from
                ``client_cert_source`` or application default SSL credentials.
            client_cert_source (Optional[Callable[[], Tuple[bytes, bytes]]]):
                Deprecated. A callback to provide client SSL certificate bytes and
                private key bytes, both in PEM format. It is ignored if
                ``api_mtls_endpoint`` is None.
            ssl_channel_credentials (grpc.ChannelCredentials): SSL credentials
                for the grpc channel. It is ignored if ``channel`` is provided.
            client_cert_source_for_mtls (Optional[Callable[[], Tuple[bytes, bytes]]]):
                A callback to provide client certificate bytes and private key bytes,
                both in PEM format. It is used to configure a mutual TLS channel. It is
                ignored if ``channel`` or ``ssl_channel_credentials`` is provided.
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
        self._operations_client: Optional[operations_v1.OperationsClient] = None

        if api_mtls_endpoint:
            warnings.warn("api_mtls_endpoint is deprecated", DeprecationWarning)
        if client_cert_source:
            warnings.warn("client_cert_source is deprecated", DeprecationWarning)

        if channel:
            # Ignore credentials if a channel was passed.
            credentials = False
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
            self._grpc_channel = type(self).create_channel(
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

        # Wrap messages. This must be done after self._grpc_channel exists
        self._prep_wrapped_messages(client_info)

    @classmethod
    def create_channel(
        cls,
        host: str = "aiplatform.googleapis.com",
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
    def operations_client(self) -> operations_v1.OperationsClient:
        """Create the client designed to process long-running operations.

        This property caches on the instance; repeated calls return the same
        client.
        """
        # Quick check: Only create a new client if we do not already have one.
        if self._operations_client is None:
            self._operations_client = operations_v1.OperationsClient(self.grpc_channel)

        # Return the client from cache.
        return self._operations_client

    @property
    def create_tensorboard(
        self,
    ) -> Callable[
        [tensorboard_service.CreateTensorboardRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the create tensorboard method over gRPC.

        Creates a Tensorboard.

        Returns:
            Callable[[~.CreateTensorboardRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_tensorboard" not in self._stubs:
            self._stubs["create_tensorboard"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/CreateTensorboard",
                request_serializer=tensorboard_service.CreateTensorboardRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["create_tensorboard"]

    @property
    def get_tensorboard(
        self,
    ) -> Callable[[tensorboard_service.GetTensorboardRequest], tensorboard.Tensorboard]:
        r"""Return a callable for the get tensorboard method over gRPC.

        Gets a Tensorboard.

        Returns:
            Callable[[~.GetTensorboardRequest],
                    ~.Tensorboard]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_tensorboard" not in self._stubs:
            self._stubs["get_tensorboard"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/GetTensorboard",
                request_serializer=tensorboard_service.GetTensorboardRequest.serialize,
                response_deserializer=tensorboard.Tensorboard.deserialize,
            )
        return self._stubs["get_tensorboard"]

    @property
    def update_tensorboard(
        self,
    ) -> Callable[
        [tensorboard_service.UpdateTensorboardRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the update tensorboard method over gRPC.

        Updates a Tensorboard.

        Returns:
            Callable[[~.UpdateTensorboardRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_tensorboard" not in self._stubs:
            self._stubs["update_tensorboard"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/UpdateTensorboard",
                request_serializer=tensorboard_service.UpdateTensorboardRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["update_tensorboard"]

    @property
    def list_tensorboards(
        self,
    ) -> Callable[
        [tensorboard_service.ListTensorboardsRequest],
        tensorboard_service.ListTensorboardsResponse,
    ]:
        r"""Return a callable for the list tensorboards method over gRPC.

        Lists Tensorboards in a Location.

        Returns:
            Callable[[~.ListTensorboardsRequest],
                    ~.ListTensorboardsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_tensorboards" not in self._stubs:
            self._stubs["list_tensorboards"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/ListTensorboards",
                request_serializer=tensorboard_service.ListTensorboardsRequest.serialize,
                response_deserializer=tensorboard_service.ListTensorboardsResponse.deserialize,
            )
        return self._stubs["list_tensorboards"]

    @property
    def delete_tensorboard(
        self,
    ) -> Callable[
        [tensorboard_service.DeleteTensorboardRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the delete tensorboard method over gRPC.

        Deletes a Tensorboard.

        Returns:
            Callable[[~.DeleteTensorboardRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_tensorboard" not in self._stubs:
            self._stubs["delete_tensorboard"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/DeleteTensorboard",
                request_serializer=tensorboard_service.DeleteTensorboardRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["delete_tensorboard"]

    @property
    def read_tensorboard_usage(
        self,
    ) -> Callable[
        [tensorboard_service.ReadTensorboardUsageRequest],
        tensorboard_service.ReadTensorboardUsageResponse,
    ]:
        r"""Return a callable for the read tensorboard usage method over gRPC.

        Returns a list of monthly active users for a given
        TensorBoard instance.

        Returns:
            Callable[[~.ReadTensorboardUsageRequest],
                    ~.ReadTensorboardUsageResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "read_tensorboard_usage" not in self._stubs:
            self._stubs["read_tensorboard_usage"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/ReadTensorboardUsage",
                request_serializer=tensorboard_service.ReadTensorboardUsageRequest.serialize,
                response_deserializer=tensorboard_service.ReadTensorboardUsageResponse.deserialize,
            )
        return self._stubs["read_tensorboard_usage"]

    @property
    def read_tensorboard_size(
        self,
    ) -> Callable[
        [tensorboard_service.ReadTensorboardSizeRequest],
        tensorboard_service.ReadTensorboardSizeResponse,
    ]:
        r"""Return a callable for the read tensorboard size method over gRPC.

        Returns the storage size for a given TensorBoard
        instance.

        Returns:
            Callable[[~.ReadTensorboardSizeRequest],
                    ~.ReadTensorboardSizeResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "read_tensorboard_size" not in self._stubs:
            self._stubs["read_tensorboard_size"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/ReadTensorboardSize",
                request_serializer=tensorboard_service.ReadTensorboardSizeRequest.serialize,
                response_deserializer=tensorboard_service.ReadTensorboardSizeResponse.deserialize,
            )
        return self._stubs["read_tensorboard_size"]

    @property
    def create_tensorboard_experiment(
        self,
    ) -> Callable[
        [tensorboard_service.CreateTensorboardExperimentRequest],
        gca_tensorboard_experiment.TensorboardExperiment,
    ]:
        r"""Return a callable for the create tensorboard experiment method over gRPC.

        Creates a TensorboardExperiment.

        Returns:
            Callable[[~.CreateTensorboardExperimentRequest],
                    ~.TensorboardExperiment]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_tensorboard_experiment" not in self._stubs:
            self._stubs[
                "create_tensorboard_experiment"
            ] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/CreateTensorboardExperiment",
                request_serializer=tensorboard_service.CreateTensorboardExperimentRequest.serialize,
                response_deserializer=gca_tensorboard_experiment.TensorboardExperiment.deserialize,
            )
        return self._stubs["create_tensorboard_experiment"]

    @property
    def get_tensorboard_experiment(
        self,
    ) -> Callable[
        [tensorboard_service.GetTensorboardExperimentRequest],
        tensorboard_experiment.TensorboardExperiment,
    ]:
        r"""Return a callable for the get tensorboard experiment method over gRPC.

        Gets a TensorboardExperiment.

        Returns:
            Callable[[~.GetTensorboardExperimentRequest],
                    ~.TensorboardExperiment]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_tensorboard_experiment" not in self._stubs:
            self._stubs["get_tensorboard_experiment"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/GetTensorboardExperiment",
                request_serializer=tensorboard_service.GetTensorboardExperimentRequest.serialize,
                response_deserializer=tensorboard_experiment.TensorboardExperiment.deserialize,
            )
        return self._stubs["get_tensorboard_experiment"]

    @property
    def update_tensorboard_experiment(
        self,
    ) -> Callable[
        [tensorboard_service.UpdateTensorboardExperimentRequest],
        gca_tensorboard_experiment.TensorboardExperiment,
    ]:
        r"""Return a callable for the update tensorboard experiment method over gRPC.

        Updates a TensorboardExperiment.

        Returns:
            Callable[[~.UpdateTensorboardExperimentRequest],
                    ~.TensorboardExperiment]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_tensorboard_experiment" not in self._stubs:
            self._stubs[
                "update_tensorboard_experiment"
            ] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/UpdateTensorboardExperiment",
                request_serializer=tensorboard_service.UpdateTensorboardExperimentRequest.serialize,
                response_deserializer=gca_tensorboard_experiment.TensorboardExperiment.deserialize,
            )
        return self._stubs["update_tensorboard_experiment"]

    @property
    def list_tensorboard_experiments(
        self,
    ) -> Callable[
        [tensorboard_service.ListTensorboardExperimentsRequest],
        tensorboard_service.ListTensorboardExperimentsResponse,
    ]:
        r"""Return a callable for the list tensorboard experiments method over gRPC.

        Lists TensorboardExperiments in a Location.

        Returns:
            Callable[[~.ListTensorboardExperimentsRequest],
                    ~.ListTensorboardExperimentsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_tensorboard_experiments" not in self._stubs:
            self._stubs["list_tensorboard_experiments"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/ListTensorboardExperiments",
                request_serializer=tensorboard_service.ListTensorboardExperimentsRequest.serialize,
                response_deserializer=tensorboard_service.ListTensorboardExperimentsResponse.deserialize,
            )
        return self._stubs["list_tensorboard_experiments"]

    @property
    def delete_tensorboard_experiment(
        self,
    ) -> Callable[
        [tensorboard_service.DeleteTensorboardExperimentRequest],
        operations_pb2.Operation,
    ]:
        r"""Return a callable for the delete tensorboard experiment method over gRPC.

        Deletes a TensorboardExperiment.

        Returns:
            Callable[[~.DeleteTensorboardExperimentRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_tensorboard_experiment" not in self._stubs:
            self._stubs[
                "delete_tensorboard_experiment"
            ] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/DeleteTensorboardExperiment",
                request_serializer=tensorboard_service.DeleteTensorboardExperimentRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["delete_tensorboard_experiment"]

    @property
    def create_tensorboard_run(
        self,
    ) -> Callable[
        [tensorboard_service.CreateTensorboardRunRequest],
        gca_tensorboard_run.TensorboardRun,
    ]:
        r"""Return a callable for the create tensorboard run method over gRPC.

        Creates a TensorboardRun.

        Returns:
            Callable[[~.CreateTensorboardRunRequest],
                    ~.TensorboardRun]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_tensorboard_run" not in self._stubs:
            self._stubs["create_tensorboard_run"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/CreateTensorboardRun",
                request_serializer=tensorboard_service.CreateTensorboardRunRequest.serialize,
                response_deserializer=gca_tensorboard_run.TensorboardRun.deserialize,
            )
        return self._stubs["create_tensorboard_run"]

    @property
    def batch_create_tensorboard_runs(
        self,
    ) -> Callable[
        [tensorboard_service.BatchCreateTensorboardRunsRequest],
        tensorboard_service.BatchCreateTensorboardRunsResponse,
    ]:
        r"""Return a callable for the batch create tensorboard runs method over gRPC.

        Batch create TensorboardRuns.

        Returns:
            Callable[[~.BatchCreateTensorboardRunsRequest],
                    ~.BatchCreateTensorboardRunsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "batch_create_tensorboard_runs" not in self._stubs:
            self._stubs[
                "batch_create_tensorboard_runs"
            ] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/BatchCreateTensorboardRuns",
                request_serializer=tensorboard_service.BatchCreateTensorboardRunsRequest.serialize,
                response_deserializer=tensorboard_service.BatchCreateTensorboardRunsResponse.deserialize,
            )
        return self._stubs["batch_create_tensorboard_runs"]

    @property
    def get_tensorboard_run(
        self,
    ) -> Callable[
        [tensorboard_service.GetTensorboardRunRequest], tensorboard_run.TensorboardRun
    ]:
        r"""Return a callable for the get tensorboard run method over gRPC.

        Gets a TensorboardRun.

        Returns:
            Callable[[~.GetTensorboardRunRequest],
                    ~.TensorboardRun]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_tensorboard_run" not in self._stubs:
            self._stubs["get_tensorboard_run"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/GetTensorboardRun",
                request_serializer=tensorboard_service.GetTensorboardRunRequest.serialize,
                response_deserializer=tensorboard_run.TensorboardRun.deserialize,
            )
        return self._stubs["get_tensorboard_run"]

    @property
    def update_tensorboard_run(
        self,
    ) -> Callable[
        [tensorboard_service.UpdateTensorboardRunRequest],
        gca_tensorboard_run.TensorboardRun,
    ]:
        r"""Return a callable for the update tensorboard run method over gRPC.

        Updates a TensorboardRun.

        Returns:
            Callable[[~.UpdateTensorboardRunRequest],
                    ~.TensorboardRun]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_tensorboard_run" not in self._stubs:
            self._stubs["update_tensorboard_run"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/UpdateTensorboardRun",
                request_serializer=tensorboard_service.UpdateTensorboardRunRequest.serialize,
                response_deserializer=gca_tensorboard_run.TensorboardRun.deserialize,
            )
        return self._stubs["update_tensorboard_run"]

    @property
    def list_tensorboard_runs(
        self,
    ) -> Callable[
        [tensorboard_service.ListTensorboardRunsRequest],
        tensorboard_service.ListTensorboardRunsResponse,
    ]:
        r"""Return a callable for the list tensorboard runs method over gRPC.

        Lists TensorboardRuns in a Location.

        Returns:
            Callable[[~.ListTensorboardRunsRequest],
                    ~.ListTensorboardRunsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_tensorboard_runs" not in self._stubs:
            self._stubs["list_tensorboard_runs"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/ListTensorboardRuns",
                request_serializer=tensorboard_service.ListTensorboardRunsRequest.serialize,
                response_deserializer=tensorboard_service.ListTensorboardRunsResponse.deserialize,
            )
        return self._stubs["list_tensorboard_runs"]

    @property
    def delete_tensorboard_run(
        self,
    ) -> Callable[
        [tensorboard_service.DeleteTensorboardRunRequest], operations_pb2.Operation
    ]:
        r"""Return a callable for the delete tensorboard run method over gRPC.

        Deletes a TensorboardRun.

        Returns:
            Callable[[~.DeleteTensorboardRunRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_tensorboard_run" not in self._stubs:
            self._stubs["delete_tensorboard_run"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/DeleteTensorboardRun",
                request_serializer=tensorboard_service.DeleteTensorboardRunRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["delete_tensorboard_run"]

    @property
    def batch_create_tensorboard_time_series(
        self,
    ) -> Callable[
        [tensorboard_service.BatchCreateTensorboardTimeSeriesRequest],
        tensorboard_service.BatchCreateTensorboardTimeSeriesResponse,
    ]:
        r"""Return a callable for the batch create tensorboard time
        series method over gRPC.

        Batch create TensorboardTimeSeries that belong to a
        TensorboardExperiment.

        Returns:
            Callable[[~.BatchCreateTensorboardTimeSeriesRequest],
                    ~.BatchCreateTensorboardTimeSeriesResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "batch_create_tensorboard_time_series" not in self._stubs:
            self._stubs[
                "batch_create_tensorboard_time_series"
            ] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/BatchCreateTensorboardTimeSeries",
                request_serializer=tensorboard_service.BatchCreateTensorboardTimeSeriesRequest.serialize,
                response_deserializer=tensorboard_service.BatchCreateTensorboardTimeSeriesResponse.deserialize,
            )
        return self._stubs["batch_create_tensorboard_time_series"]

    @property
    def create_tensorboard_time_series(
        self,
    ) -> Callable[
        [tensorboard_service.CreateTensorboardTimeSeriesRequest],
        gca_tensorboard_time_series.TensorboardTimeSeries,
    ]:
        r"""Return a callable for the create tensorboard time series method over gRPC.

        Creates a TensorboardTimeSeries.

        Returns:
            Callable[[~.CreateTensorboardTimeSeriesRequest],
                    ~.TensorboardTimeSeries]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_tensorboard_time_series" not in self._stubs:
            self._stubs[
                "create_tensorboard_time_series"
            ] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/CreateTensorboardTimeSeries",
                request_serializer=tensorboard_service.CreateTensorboardTimeSeriesRequest.serialize,
                response_deserializer=gca_tensorboard_time_series.TensorboardTimeSeries.deserialize,
            )
        return self._stubs["create_tensorboard_time_series"]

    @property
    def get_tensorboard_time_series(
        self,
    ) -> Callable[
        [tensorboard_service.GetTensorboardTimeSeriesRequest],
        tensorboard_time_series.TensorboardTimeSeries,
    ]:
        r"""Return a callable for the get tensorboard time series method over gRPC.

        Gets a TensorboardTimeSeries.

        Returns:
            Callable[[~.GetTensorboardTimeSeriesRequest],
                    ~.TensorboardTimeSeries]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_tensorboard_time_series" not in self._stubs:
            self._stubs["get_tensorboard_time_series"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/GetTensorboardTimeSeries",
                request_serializer=tensorboard_service.GetTensorboardTimeSeriesRequest.serialize,
                response_deserializer=tensorboard_time_series.TensorboardTimeSeries.deserialize,
            )
        return self._stubs["get_tensorboard_time_series"]

    @property
    def update_tensorboard_time_series(
        self,
    ) -> Callable[
        [tensorboard_service.UpdateTensorboardTimeSeriesRequest],
        gca_tensorboard_time_series.TensorboardTimeSeries,
    ]:
        r"""Return a callable for the update tensorboard time series method over gRPC.

        Updates a TensorboardTimeSeries.

        Returns:
            Callable[[~.UpdateTensorboardTimeSeriesRequest],
                    ~.TensorboardTimeSeries]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_tensorboard_time_series" not in self._stubs:
            self._stubs[
                "update_tensorboard_time_series"
            ] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/UpdateTensorboardTimeSeries",
                request_serializer=tensorboard_service.UpdateTensorboardTimeSeriesRequest.serialize,
                response_deserializer=gca_tensorboard_time_series.TensorboardTimeSeries.deserialize,
            )
        return self._stubs["update_tensorboard_time_series"]

    @property
    def list_tensorboard_time_series(
        self,
    ) -> Callable[
        [tensorboard_service.ListTensorboardTimeSeriesRequest],
        tensorboard_service.ListTensorboardTimeSeriesResponse,
    ]:
        r"""Return a callable for the list tensorboard time series method over gRPC.

        Lists TensorboardTimeSeries in a Location.

        Returns:
            Callable[[~.ListTensorboardTimeSeriesRequest],
                    ~.ListTensorboardTimeSeriesResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_tensorboard_time_series" not in self._stubs:
            self._stubs["list_tensorboard_time_series"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/ListTensorboardTimeSeries",
                request_serializer=tensorboard_service.ListTensorboardTimeSeriesRequest.serialize,
                response_deserializer=tensorboard_service.ListTensorboardTimeSeriesResponse.deserialize,
            )
        return self._stubs["list_tensorboard_time_series"]

    @property
    def delete_tensorboard_time_series(
        self,
    ) -> Callable[
        [tensorboard_service.DeleteTensorboardTimeSeriesRequest],
        operations_pb2.Operation,
    ]:
        r"""Return a callable for the delete tensorboard time series method over gRPC.

        Deletes a TensorboardTimeSeries.

        Returns:
            Callable[[~.DeleteTensorboardTimeSeriesRequest],
                    ~.Operation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_tensorboard_time_series" not in self._stubs:
            self._stubs[
                "delete_tensorboard_time_series"
            ] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/DeleteTensorboardTimeSeries",
                request_serializer=tensorboard_service.DeleteTensorboardTimeSeriesRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["delete_tensorboard_time_series"]

    @property
    def batch_read_tensorboard_time_series_data(
        self,
    ) -> Callable[
        [tensorboard_service.BatchReadTensorboardTimeSeriesDataRequest],
        tensorboard_service.BatchReadTensorboardTimeSeriesDataResponse,
    ]:
        r"""Return a callable for the batch read tensorboard time
        series data method over gRPC.

        Reads multiple TensorboardTimeSeries' data. The data
        point number limit is 1000 for scalars, 100 for tensors
        and blob references. If the number of data points stored
        is less than the limit, all data is returned. Otherwise,
        the number limit of data points is randomly selected
        from this time series and returned.

        Returns:
            Callable[[~.BatchReadTensorboardTimeSeriesDataRequest],
                    ~.BatchReadTensorboardTimeSeriesDataResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "batch_read_tensorboard_time_series_data" not in self._stubs:
            self._stubs[
                "batch_read_tensorboard_time_series_data"
            ] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/BatchReadTensorboardTimeSeriesData",
                request_serializer=tensorboard_service.BatchReadTensorboardTimeSeriesDataRequest.serialize,
                response_deserializer=tensorboard_service.BatchReadTensorboardTimeSeriesDataResponse.deserialize,
            )
        return self._stubs["batch_read_tensorboard_time_series_data"]

    @property
    def read_tensorboard_time_series_data(
        self,
    ) -> Callable[
        [tensorboard_service.ReadTensorboardTimeSeriesDataRequest],
        tensorboard_service.ReadTensorboardTimeSeriesDataResponse,
    ]:
        r"""Return a callable for the read tensorboard time series
        data method over gRPC.

        Reads a TensorboardTimeSeries' data. By default, if the number
        of data points stored is less than 1000, all data is returned.
        Otherwise, 1000 data points is randomly selected from this time
        series and returned. This value can be changed by changing
        max_data_points, which can't be greater than 10k.

        Returns:
            Callable[[~.ReadTensorboardTimeSeriesDataRequest],
                    ~.ReadTensorboardTimeSeriesDataResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "read_tensorboard_time_series_data" not in self._stubs:
            self._stubs[
                "read_tensorboard_time_series_data"
            ] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/ReadTensorboardTimeSeriesData",
                request_serializer=tensorboard_service.ReadTensorboardTimeSeriesDataRequest.serialize,
                response_deserializer=tensorboard_service.ReadTensorboardTimeSeriesDataResponse.deserialize,
            )
        return self._stubs["read_tensorboard_time_series_data"]

    @property
    def read_tensorboard_blob_data(
        self,
    ) -> Callable[
        [tensorboard_service.ReadTensorboardBlobDataRequest],
        tensorboard_service.ReadTensorboardBlobDataResponse,
    ]:
        r"""Return a callable for the read tensorboard blob data method over gRPC.

        Gets bytes of TensorboardBlobs.
        This is to allow reading blob data stored in consumer
        project's Cloud Storage bucket without users having to
        obtain Cloud Storage access permission.

        Returns:
            Callable[[~.ReadTensorboardBlobDataRequest],
                    ~.ReadTensorboardBlobDataResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "read_tensorboard_blob_data" not in self._stubs:
            self._stubs["read_tensorboard_blob_data"] = self.grpc_channel.unary_stream(
                "/google.cloud.aiplatform.v1.TensorboardService/ReadTensorboardBlobData",
                request_serializer=tensorboard_service.ReadTensorboardBlobDataRequest.serialize,
                response_deserializer=tensorboard_service.ReadTensorboardBlobDataResponse.deserialize,
            )
        return self._stubs["read_tensorboard_blob_data"]

    @property
    def write_tensorboard_experiment_data(
        self,
    ) -> Callable[
        [tensorboard_service.WriteTensorboardExperimentDataRequest],
        tensorboard_service.WriteTensorboardExperimentDataResponse,
    ]:
        r"""Return a callable for the write tensorboard experiment
        data method over gRPC.

        Write time series data points of multiple
        TensorboardTimeSeries in multiple TensorboardRun's. If
        any data fail to be ingested, an error is returned.

        Returns:
            Callable[[~.WriteTensorboardExperimentDataRequest],
                    ~.WriteTensorboardExperimentDataResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "write_tensorboard_experiment_data" not in self._stubs:
            self._stubs[
                "write_tensorboard_experiment_data"
            ] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/WriteTensorboardExperimentData",
                request_serializer=tensorboard_service.WriteTensorboardExperimentDataRequest.serialize,
                response_deserializer=tensorboard_service.WriteTensorboardExperimentDataResponse.deserialize,
            )
        return self._stubs["write_tensorboard_experiment_data"]

    @property
    def write_tensorboard_run_data(
        self,
    ) -> Callable[
        [tensorboard_service.WriteTensorboardRunDataRequest],
        tensorboard_service.WriteTensorboardRunDataResponse,
    ]:
        r"""Return a callable for the write tensorboard run data method over gRPC.

        Write time series data points into multiple
        TensorboardTimeSeries under a TensorboardRun. If any
        data fail to be ingested, an error is returned.

        Returns:
            Callable[[~.WriteTensorboardRunDataRequest],
                    ~.WriteTensorboardRunDataResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "write_tensorboard_run_data" not in self._stubs:
            self._stubs["write_tensorboard_run_data"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/WriteTensorboardRunData",
                request_serializer=tensorboard_service.WriteTensorboardRunDataRequest.serialize,
                response_deserializer=tensorboard_service.WriteTensorboardRunDataResponse.deserialize,
            )
        return self._stubs["write_tensorboard_run_data"]

    @property
    def export_tensorboard_time_series_data(
        self,
    ) -> Callable[
        [tensorboard_service.ExportTensorboardTimeSeriesDataRequest],
        tensorboard_service.ExportTensorboardTimeSeriesDataResponse,
    ]:
        r"""Return a callable for the export tensorboard time series
        data method over gRPC.

        Exports a TensorboardTimeSeries' data. Data is
        returned in paginated responses.

        Returns:
            Callable[[~.ExportTensorboardTimeSeriesDataRequest],
                    ~.ExportTensorboardTimeSeriesDataResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "export_tensorboard_time_series_data" not in self._stubs:
            self._stubs[
                "export_tensorboard_time_series_data"
            ] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.TensorboardService/ExportTensorboardTimeSeriesData",
                request_serializer=tensorboard_service.ExportTensorboardTimeSeriesDataRequest.serialize,
                response_deserializer=tensorboard_service.ExportTensorboardTimeSeriesDataResponse.deserialize,
            )
        return self._stubs["export_tensorboard_time_series_data"]

    def close(self):
        self.grpc_channel.close()

    @property
    def delete_operation(
        self,
    ) -> Callable[[operations_pb2.DeleteOperationRequest], None]:
        r"""Return a callable for the delete_operation method over gRPC."""
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_operation" not in self._stubs:
            self._stubs["delete_operation"] = self.grpc_channel.unary_unary(
                "/google.longrunning.Operations/DeleteOperation",
                request_serializer=operations_pb2.DeleteOperationRequest.SerializeToString,
                response_deserializer=None,
            )
        return self._stubs["delete_operation"]

    @property
    def cancel_operation(
        self,
    ) -> Callable[[operations_pb2.CancelOperationRequest], None]:
        r"""Return a callable for the cancel_operation method over gRPC."""
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "cancel_operation" not in self._stubs:
            self._stubs["cancel_operation"] = self.grpc_channel.unary_unary(
                "/google.longrunning.Operations/CancelOperation",
                request_serializer=operations_pb2.CancelOperationRequest.SerializeToString,
                response_deserializer=None,
            )
        return self._stubs["cancel_operation"]

    @property
    def wait_operation(
        self,
    ) -> Callable[[operations_pb2.WaitOperationRequest], None]:
        r"""Return a callable for the wait_operation method over gRPC."""
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_operation" not in self._stubs:
            self._stubs["wait_operation"] = self.grpc_channel.unary_unary(
                "/google.longrunning.Operations/WaitOperation",
                request_serializer=operations_pb2.WaitOperationRequest.SerializeToString,
                response_deserializer=None,
            )
        return self._stubs["wait_operation"]

    @property
    def get_operation(
        self,
    ) -> Callable[[operations_pb2.GetOperationRequest], operations_pb2.Operation]:
        r"""Return a callable for the get_operation method over gRPC."""
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_operation" not in self._stubs:
            self._stubs["get_operation"] = self.grpc_channel.unary_unary(
                "/google.longrunning.Operations/GetOperation",
                request_serializer=operations_pb2.GetOperationRequest.SerializeToString,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["get_operation"]

    @property
    def list_operations(
        self,
    ) -> Callable[
        [operations_pb2.ListOperationsRequest], operations_pb2.ListOperationsResponse
    ]:
        r"""Return a callable for the list_operations method over gRPC."""
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_operations" not in self._stubs:
            self._stubs["list_operations"] = self.grpc_channel.unary_unary(
                "/google.longrunning.Operations/ListOperations",
                request_serializer=operations_pb2.ListOperationsRequest.SerializeToString,
                response_deserializer=operations_pb2.ListOperationsResponse.FromString,
            )
        return self._stubs["list_operations"]

    @property
    def list_locations(
        self,
    ) -> Callable[
        [locations_pb2.ListLocationsRequest], locations_pb2.ListLocationsResponse
    ]:
        r"""Return a callable for the list locations method over gRPC."""
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_locations" not in self._stubs:
            self._stubs["list_locations"] = self.grpc_channel.unary_unary(
                "/google.cloud.location.Locations/ListLocations",
                request_serializer=locations_pb2.ListLocationsRequest.SerializeToString,
                response_deserializer=locations_pb2.ListLocationsResponse.FromString,
            )
        return self._stubs["list_locations"]

    @property
    def get_location(
        self,
    ) -> Callable[[locations_pb2.GetLocationRequest], locations_pb2.Location]:
        r"""Return a callable for the list locations method over gRPC."""
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_location" not in self._stubs:
            self._stubs["get_location"] = self.grpc_channel.unary_unary(
                "/google.cloud.location.Locations/GetLocation",
                request_serializer=locations_pb2.GetLocationRequest.SerializeToString,
                response_deserializer=locations_pb2.Location.FromString,
            )
        return self._stubs["get_location"]

    @property
    def set_iam_policy(
        self,
    ) -> Callable[[iam_policy_pb2.SetIamPolicyRequest], policy_pb2.Policy]:
        r"""Return a callable for the set iam policy method over gRPC.
        Sets the IAM access control policy on the specified
        function. Replaces any existing policy.
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
            self._stubs["set_iam_policy"] = self.grpc_channel.unary_unary(
                "/google.iam.v1.IAMPolicy/SetIamPolicy",
                request_serializer=iam_policy_pb2.SetIamPolicyRequest.SerializeToString,
                response_deserializer=policy_pb2.Policy.FromString,
            )
        return self._stubs["set_iam_policy"]

    @property
    def get_iam_policy(
        self,
    ) -> Callable[[iam_policy_pb2.GetIamPolicyRequest], policy_pb2.Policy]:
        r"""Return a callable for the get iam policy method over gRPC.
        Gets the IAM access control policy for a function.
        Returns an empty policy if the function exists and does
        not have a policy set.
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
            self._stubs["get_iam_policy"] = self.grpc_channel.unary_unary(
                "/google.iam.v1.IAMPolicy/GetIamPolicy",
                request_serializer=iam_policy_pb2.GetIamPolicyRequest.SerializeToString,
                response_deserializer=policy_pb2.Policy.FromString,
            )
        return self._stubs["get_iam_policy"]

    @property
    def test_iam_permissions(
        self,
    ) -> Callable[
        [iam_policy_pb2.TestIamPermissionsRequest],
        iam_policy_pb2.TestIamPermissionsResponse,
    ]:
        r"""Return a callable for the test iam permissions method over gRPC.
        Tests the specified permissions against the IAM access control
        policy for a function. If the function does not exist, this will
        return an empty set of permissions, not a NOT_FOUND error.
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
            self._stubs["test_iam_permissions"] = self.grpc_channel.unary_unary(
                "/google.iam.v1.IAMPolicy/TestIamPermissions",
                request_serializer=iam_policy_pb2.TestIamPermissionsRequest.SerializeToString,
                response_deserializer=iam_policy_pb2.TestIamPermissionsResponse.FromString,
            )
        return self._stubs["test_iam_permissions"]

    @property
    def kind(self) -> str:
        return "grpc"


__all__ = ("TensorboardServiceGrpcTransport",)
