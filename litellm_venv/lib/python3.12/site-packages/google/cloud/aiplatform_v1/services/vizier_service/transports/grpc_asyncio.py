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
from typing import Awaitable, Callable, Dict, Optional, Sequence, Tuple, Union

from google.api_core import gapic_v1
from google.api_core import grpc_helpers_async
from google.api_core import operations_v1
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore

import grpc  # type: ignore
from grpc.experimental import aio  # type: ignore

from google.cloud.aiplatform_v1.types import study
from google.cloud.aiplatform_v1.types import study as gca_study
from google.cloud.aiplatform_v1.types import vizier_service
from google.cloud.location import locations_pb2  # type: ignore
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import empty_pb2  # type: ignore
from .base import VizierServiceTransport, DEFAULT_CLIENT_INFO
from .grpc import VizierServiceGrpcTransport


class VizierServiceGrpcAsyncIOTransport(VizierServiceTransport):
    """gRPC AsyncIO backend transport for VizierService.

    Vertex AI Vizier API.

    Vertex AI Vizier is a service to solve blackbox optimization
    problems, such as tuning machine learning hyperparameters and
    searching over deep learning architectures.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends protocol buffers over the wire using gRPC (which is built on
    top of HTTP/2); the ``grpcio`` package must be installed.
    """

    _grpc_channel: aio.Channel
    _stubs: Dict[str, Callable] = {}

    @classmethod
    def create_channel(
        cls,
        host: str = "aiplatform.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        quota_project_id: Optional[str] = None,
        **kwargs,
    ) -> aio.Channel:
        """Create and return a gRPC AsyncIO channel object.
        Args:
            host (Optional[str]): The host for the channel to use.
            credentials (Optional[~.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify this application to the service. If
                none are specified, the client will attempt to ascertain
                the credentials from the environment.
            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if ``channel`` is provided.
            scopes (Optional[Sequence[str]]): A optional list of scopes needed for this
                service. These are only used when credentials are not specified and
                are passed to :func:`google.auth.default`.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            kwargs (Optional[dict]): Keyword arguments, which are passed to the
                channel creation.
        Returns:
            aio.Channel: A gRPC AsyncIO channel object.
        """

        return grpc_helpers_async.create_channel(
            host,
            credentials=credentials,
            credentials_file=credentials_file,
            quota_project_id=quota_project_id,
            default_scopes=cls.AUTH_SCOPES,
            scopes=scopes,
            default_host=cls.DEFAULT_HOST,
            **kwargs,
        )

    def __init__(
        self,
        *,
        host: str = "aiplatform.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        channel: Optional[aio.Channel] = None,
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
            scopes (Optional[Sequence[str]]): A optional list of scopes needed for this
                service. These are only used when credentials are not specified and
                are passed to :func:`google.auth.default`.
            channel (Optional[aio.Channel]): A ``Channel`` instance through
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
            google.auth.exceptions.MutualTlsChannelError: If mutual TLS transport
              creation failed for any reason.
          google.api_core.exceptions.DuplicateCredentialArgs: If both ``credentials``
              and ``credentials_file`` are passed.
        """
        self._grpc_channel = None
        self._ssl_channel_credentials = ssl_channel_credentials
        self._stubs: Dict[str, Callable] = {}
        self._operations_client: Optional[operations_v1.OperationsAsyncClient] = None

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

    @property
    def grpc_channel(self) -> aio.Channel:
        """Create the channel designed to connect to this service.

        This property caches on the instance; repeated calls return
        the same channel.
        """
        # Return the channel from cache.
        return self._grpc_channel

    @property
    def operations_client(self) -> operations_v1.OperationsAsyncClient:
        """Create the client designed to process long-running operations.

        This property caches on the instance; repeated calls return the same
        client.
        """
        # Quick check: Only create a new client if we do not already have one.
        if self._operations_client is None:
            self._operations_client = operations_v1.OperationsAsyncClient(
                self.grpc_channel
            )

        # Return the client from cache.
        return self._operations_client

    @property
    def create_study(
        self,
    ) -> Callable[[vizier_service.CreateStudyRequest], Awaitable[gca_study.Study]]:
        r"""Return a callable for the create study method over gRPC.

        Creates a Study. A resource name will be generated
        after creation of the Study.

        Returns:
            Callable[[~.CreateStudyRequest],
                    Awaitable[~.Study]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_study" not in self._stubs:
            self._stubs["create_study"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.VizierService/CreateStudy",
                request_serializer=vizier_service.CreateStudyRequest.serialize,
                response_deserializer=gca_study.Study.deserialize,
            )
        return self._stubs["create_study"]

    @property
    def get_study(
        self,
    ) -> Callable[[vizier_service.GetStudyRequest], Awaitable[study.Study]]:
        r"""Return a callable for the get study method over gRPC.

        Gets a Study by name.

        Returns:
            Callable[[~.GetStudyRequest],
                    Awaitable[~.Study]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_study" not in self._stubs:
            self._stubs["get_study"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.VizierService/GetStudy",
                request_serializer=vizier_service.GetStudyRequest.serialize,
                response_deserializer=study.Study.deserialize,
            )
        return self._stubs["get_study"]

    @property
    def list_studies(
        self,
    ) -> Callable[
        [vizier_service.ListStudiesRequest],
        Awaitable[vizier_service.ListStudiesResponse],
    ]:
        r"""Return a callable for the list studies method over gRPC.

        Lists all the studies in a region for an associated
        project.

        Returns:
            Callable[[~.ListStudiesRequest],
                    Awaitable[~.ListStudiesResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_studies" not in self._stubs:
            self._stubs["list_studies"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.VizierService/ListStudies",
                request_serializer=vizier_service.ListStudiesRequest.serialize,
                response_deserializer=vizier_service.ListStudiesResponse.deserialize,
            )
        return self._stubs["list_studies"]

    @property
    def delete_study(
        self,
    ) -> Callable[[vizier_service.DeleteStudyRequest], Awaitable[empty_pb2.Empty]]:
        r"""Return a callable for the delete study method over gRPC.

        Deletes a Study.

        Returns:
            Callable[[~.DeleteStudyRequest],
                    Awaitable[~.Empty]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_study" not in self._stubs:
            self._stubs["delete_study"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.VizierService/DeleteStudy",
                request_serializer=vizier_service.DeleteStudyRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_study"]

    @property
    def lookup_study(
        self,
    ) -> Callable[[vizier_service.LookupStudyRequest], Awaitable[study.Study]]:
        r"""Return a callable for the lookup study method over gRPC.

        Looks a study up using the user-defined display_name field
        instead of the fully qualified resource name.

        Returns:
            Callable[[~.LookupStudyRequest],
                    Awaitable[~.Study]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "lookup_study" not in self._stubs:
            self._stubs["lookup_study"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.VizierService/LookupStudy",
                request_serializer=vizier_service.LookupStudyRequest.serialize,
                response_deserializer=study.Study.deserialize,
            )
        return self._stubs["lookup_study"]

    @property
    def suggest_trials(
        self,
    ) -> Callable[
        [vizier_service.SuggestTrialsRequest], Awaitable[operations_pb2.Operation]
    ]:
        r"""Return a callable for the suggest trials method over gRPC.

        Adds one or more Trials to a Study, with parameter values
        suggested by Vertex AI Vizier. Returns a long-running operation
        associated with the generation of Trial suggestions. When this
        long-running operation succeeds, it will contain a
        [SuggestTrialsResponse][google.cloud.aiplatform.v1.SuggestTrialsResponse].

        Returns:
            Callable[[~.SuggestTrialsRequest],
                    Awaitable[~.Operation]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "suggest_trials" not in self._stubs:
            self._stubs["suggest_trials"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.VizierService/SuggestTrials",
                request_serializer=vizier_service.SuggestTrialsRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["suggest_trials"]

    @property
    def create_trial(
        self,
    ) -> Callable[[vizier_service.CreateTrialRequest], Awaitable[study.Trial]]:
        r"""Return a callable for the create trial method over gRPC.

        Adds a user provided Trial to a Study.

        Returns:
            Callable[[~.CreateTrialRequest],
                    Awaitable[~.Trial]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "create_trial" not in self._stubs:
            self._stubs["create_trial"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.VizierService/CreateTrial",
                request_serializer=vizier_service.CreateTrialRequest.serialize,
                response_deserializer=study.Trial.deserialize,
            )
        return self._stubs["create_trial"]

    @property
    def get_trial(
        self,
    ) -> Callable[[vizier_service.GetTrialRequest], Awaitable[study.Trial]]:
        r"""Return a callable for the get trial method over gRPC.

        Gets a Trial.

        Returns:
            Callable[[~.GetTrialRequest],
                    Awaitable[~.Trial]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_trial" not in self._stubs:
            self._stubs["get_trial"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.VizierService/GetTrial",
                request_serializer=vizier_service.GetTrialRequest.serialize,
                response_deserializer=study.Trial.deserialize,
            )
        return self._stubs["get_trial"]

    @property
    def list_trials(
        self,
    ) -> Callable[
        [vizier_service.ListTrialsRequest], Awaitable[vizier_service.ListTrialsResponse]
    ]:
        r"""Return a callable for the list trials method over gRPC.

        Lists the Trials associated with a Study.

        Returns:
            Callable[[~.ListTrialsRequest],
                    Awaitable[~.ListTrialsResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_trials" not in self._stubs:
            self._stubs["list_trials"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.VizierService/ListTrials",
                request_serializer=vizier_service.ListTrialsRequest.serialize,
                response_deserializer=vizier_service.ListTrialsResponse.deserialize,
            )
        return self._stubs["list_trials"]

    @property
    def add_trial_measurement(
        self,
    ) -> Callable[[vizier_service.AddTrialMeasurementRequest], Awaitable[study.Trial]]:
        r"""Return a callable for the add trial measurement method over gRPC.

        Adds a measurement of the objective metrics to a
        Trial. This measurement is assumed to have been taken
        before the Trial is complete.

        Returns:
            Callable[[~.AddTrialMeasurementRequest],
                    Awaitable[~.Trial]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "add_trial_measurement" not in self._stubs:
            self._stubs["add_trial_measurement"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.VizierService/AddTrialMeasurement",
                request_serializer=vizier_service.AddTrialMeasurementRequest.serialize,
                response_deserializer=study.Trial.deserialize,
            )
        return self._stubs["add_trial_measurement"]

    @property
    def complete_trial(
        self,
    ) -> Callable[[vizier_service.CompleteTrialRequest], Awaitable[study.Trial]]:
        r"""Return a callable for the complete trial method over gRPC.

        Marks a Trial as complete.

        Returns:
            Callable[[~.CompleteTrialRequest],
                    Awaitable[~.Trial]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "complete_trial" not in self._stubs:
            self._stubs["complete_trial"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.VizierService/CompleteTrial",
                request_serializer=vizier_service.CompleteTrialRequest.serialize,
                response_deserializer=study.Trial.deserialize,
            )
        return self._stubs["complete_trial"]

    @property
    def delete_trial(
        self,
    ) -> Callable[[vizier_service.DeleteTrialRequest], Awaitable[empty_pb2.Empty]]:
        r"""Return a callable for the delete trial method over gRPC.

        Deletes a Trial.

        Returns:
            Callable[[~.DeleteTrialRequest],
                    Awaitable[~.Empty]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_trial" not in self._stubs:
            self._stubs["delete_trial"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.VizierService/DeleteTrial",
                request_serializer=vizier_service.DeleteTrialRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs["delete_trial"]

    @property
    def check_trial_early_stopping_state(
        self,
    ) -> Callable[
        [vizier_service.CheckTrialEarlyStoppingStateRequest],
        Awaitable[operations_pb2.Operation],
    ]:
        r"""Return a callable for the check trial early stopping
        state method over gRPC.

        Checks whether a Trial should stop or not. Returns a
        long-running operation. When the operation is successful, it
        will contain a
        [CheckTrialEarlyStoppingStateResponse][google.cloud.aiplatform.v1.CheckTrialEarlyStoppingStateResponse].

        Returns:
            Callable[[~.CheckTrialEarlyStoppingStateRequest],
                    Awaitable[~.Operation]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "check_trial_early_stopping_state" not in self._stubs:
            self._stubs[
                "check_trial_early_stopping_state"
            ] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.VizierService/CheckTrialEarlyStoppingState",
                request_serializer=vizier_service.CheckTrialEarlyStoppingStateRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["check_trial_early_stopping_state"]

    @property
    def stop_trial(
        self,
    ) -> Callable[[vizier_service.StopTrialRequest], Awaitable[study.Trial]]:
        r"""Return a callable for the stop trial method over gRPC.

        Stops a Trial.

        Returns:
            Callable[[~.StopTrialRequest],
                    Awaitable[~.Trial]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "stop_trial" not in self._stubs:
            self._stubs["stop_trial"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.VizierService/StopTrial",
                request_serializer=vizier_service.StopTrialRequest.serialize,
                response_deserializer=study.Trial.deserialize,
            )
        return self._stubs["stop_trial"]

    @property
    def list_optimal_trials(
        self,
    ) -> Callable[
        [vizier_service.ListOptimalTrialsRequest],
        Awaitable[vizier_service.ListOptimalTrialsResponse],
    ]:
        r"""Return a callable for the list optimal trials method over gRPC.

        Lists the pareto-optimal Trials for multi-objective Study or the
        optimal Trials for single-objective Study. The definition of
        pareto-optimal can be checked in wiki page.
        https://en.wikipedia.org/wiki/Pareto_efficiency

        Returns:
            Callable[[~.ListOptimalTrialsRequest],
                    Awaitable[~.ListOptimalTrialsResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_optimal_trials" not in self._stubs:
            self._stubs["list_optimal_trials"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1.VizierService/ListOptimalTrials",
                request_serializer=vizier_service.ListOptimalTrialsRequest.serialize,
                response_deserializer=vizier_service.ListOptimalTrialsResponse.deserialize,
            )
        return self._stubs["list_optimal_trials"]

    def close(self):
        return self.grpc_channel.close()

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


__all__ = ("VizierServiceGrpcAsyncIOTransport",)
