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

from google.cloud.aiplatform_v1beta1.types import model
from google.cloud.aiplatform_v1beta1.types import model as gca_model
from google.cloud.aiplatform_v1beta1.types import model_evaluation
from google.cloud.aiplatform_v1beta1.types import (
    model_evaluation as gca_model_evaluation,
)
from google.cloud.aiplatform_v1beta1.types import model_evaluation_slice
from google.cloud.aiplatform_v1beta1.types import model_service
from google.cloud.location import locations_pb2  # type: ignore
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from .base import ModelServiceTransport, DEFAULT_CLIENT_INFO
from .grpc import ModelServiceGrpcTransport


class ModelServiceGrpcAsyncIOTransport(ModelServiceTransport):
    """gRPC AsyncIO backend transport for ModelService.

    A service for managing Vertex AI's machine learning Models.

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
    def upload_model(
        self,
    ) -> Callable[
        [model_service.UploadModelRequest], Awaitable[operations_pb2.Operation]
    ]:
        r"""Return a callable for the upload model method over gRPC.

        Uploads a Model artifact into Vertex AI.

        Returns:
            Callable[[~.UploadModelRequest],
                    Awaitable[~.Operation]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "upload_model" not in self._stubs:
            self._stubs["upload_model"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.ModelService/UploadModel",
                request_serializer=model_service.UploadModelRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["upload_model"]

    @property
    def get_model(
        self,
    ) -> Callable[[model_service.GetModelRequest], Awaitable[model.Model]]:
        r"""Return a callable for the get model method over gRPC.

        Gets a Model.

        Returns:
            Callable[[~.GetModelRequest],
                    Awaitable[~.Model]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_model" not in self._stubs:
            self._stubs["get_model"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.ModelService/GetModel",
                request_serializer=model_service.GetModelRequest.serialize,
                response_deserializer=model.Model.deserialize,
            )
        return self._stubs["get_model"]

    @property
    def list_models(
        self,
    ) -> Callable[
        [model_service.ListModelsRequest], Awaitable[model_service.ListModelsResponse]
    ]:
        r"""Return a callable for the list models method over gRPC.

        Lists Models in a Location.

        Returns:
            Callable[[~.ListModelsRequest],
                    Awaitable[~.ListModelsResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_models" not in self._stubs:
            self._stubs["list_models"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.ModelService/ListModels",
                request_serializer=model_service.ListModelsRequest.serialize,
                response_deserializer=model_service.ListModelsResponse.deserialize,
            )
        return self._stubs["list_models"]

    @property
    def list_model_versions(
        self,
    ) -> Callable[
        [model_service.ListModelVersionsRequest],
        Awaitable[model_service.ListModelVersionsResponse],
    ]:
        r"""Return a callable for the list model versions method over gRPC.

        Lists versions of the specified model.

        Returns:
            Callable[[~.ListModelVersionsRequest],
                    Awaitable[~.ListModelVersionsResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_model_versions" not in self._stubs:
            self._stubs["list_model_versions"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.ModelService/ListModelVersions",
                request_serializer=model_service.ListModelVersionsRequest.serialize,
                response_deserializer=model_service.ListModelVersionsResponse.deserialize,
            )
        return self._stubs["list_model_versions"]

    @property
    def update_model(
        self,
    ) -> Callable[[model_service.UpdateModelRequest], Awaitable[gca_model.Model]]:
        r"""Return a callable for the update model method over gRPC.

        Updates a Model.

        Returns:
            Callable[[~.UpdateModelRequest],
                    Awaitable[~.Model]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_model" not in self._stubs:
            self._stubs["update_model"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.ModelService/UpdateModel",
                request_serializer=model_service.UpdateModelRequest.serialize,
                response_deserializer=gca_model.Model.deserialize,
            )
        return self._stubs["update_model"]

    @property
    def update_explanation_dataset(
        self,
    ) -> Callable[
        [model_service.UpdateExplanationDatasetRequest],
        Awaitable[operations_pb2.Operation],
    ]:
        r"""Return a callable for the update explanation dataset method over gRPC.

        Incrementally update the dataset used for an examples
        model.

        Returns:
            Callable[[~.UpdateExplanationDatasetRequest],
                    Awaitable[~.Operation]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "update_explanation_dataset" not in self._stubs:
            self._stubs["update_explanation_dataset"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.ModelService/UpdateExplanationDataset",
                request_serializer=model_service.UpdateExplanationDatasetRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["update_explanation_dataset"]

    @property
    def delete_model(
        self,
    ) -> Callable[
        [model_service.DeleteModelRequest], Awaitable[operations_pb2.Operation]
    ]:
        r"""Return a callable for the delete model method over gRPC.

        Deletes a Model.

        A model cannot be deleted if any
        [Endpoint][google.cloud.aiplatform.v1beta1.Endpoint] resource
        has a
        [DeployedModel][google.cloud.aiplatform.v1beta1.DeployedModel]
        based on the model in its
        [deployed_models][google.cloud.aiplatform.v1beta1.Endpoint.deployed_models]
        field.

        Returns:
            Callable[[~.DeleteModelRequest],
                    Awaitable[~.Operation]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_model" not in self._stubs:
            self._stubs["delete_model"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.ModelService/DeleteModel",
                request_serializer=model_service.DeleteModelRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["delete_model"]

    @property
    def delete_model_version(
        self,
    ) -> Callable[
        [model_service.DeleteModelVersionRequest], Awaitable[operations_pb2.Operation]
    ]:
        r"""Return a callable for the delete model version method over gRPC.

        Deletes a Model version.

        Model version can only be deleted if there are no
        [DeployedModels][google.cloud.aiplatform.v1beta1.DeployedModel]
        created from it. Deleting the only version in the Model is not
        allowed. Use
        [DeleteModel][google.cloud.aiplatform.v1beta1.ModelService.DeleteModel]
        for deleting the Model instead.

        Returns:
            Callable[[~.DeleteModelVersionRequest],
                    Awaitable[~.Operation]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "delete_model_version" not in self._stubs:
            self._stubs["delete_model_version"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.ModelService/DeleteModelVersion",
                request_serializer=model_service.DeleteModelVersionRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["delete_model_version"]

    @property
    def merge_version_aliases(
        self,
    ) -> Callable[[model_service.MergeVersionAliasesRequest], Awaitable[model.Model]]:
        r"""Return a callable for the merge version aliases method over gRPC.

        Merges a set of aliases for a Model version.

        Returns:
            Callable[[~.MergeVersionAliasesRequest],
                    Awaitable[~.Model]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "merge_version_aliases" not in self._stubs:
            self._stubs["merge_version_aliases"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.ModelService/MergeVersionAliases",
                request_serializer=model_service.MergeVersionAliasesRequest.serialize,
                response_deserializer=model.Model.deserialize,
            )
        return self._stubs["merge_version_aliases"]

    @property
    def export_model(
        self,
    ) -> Callable[
        [model_service.ExportModelRequest], Awaitable[operations_pb2.Operation]
    ]:
        r"""Return a callable for the export model method over gRPC.

        Exports a trained, exportable Model to a location specified by
        the user. A Model is considered to be exportable if it has at
        least one [supported export
        format][google.cloud.aiplatform.v1beta1.Model.supported_export_formats].

        Returns:
            Callable[[~.ExportModelRequest],
                    Awaitable[~.Operation]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "export_model" not in self._stubs:
            self._stubs["export_model"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.ModelService/ExportModel",
                request_serializer=model_service.ExportModelRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["export_model"]

    @property
    def copy_model(
        self,
    ) -> Callable[
        [model_service.CopyModelRequest], Awaitable[operations_pb2.Operation]
    ]:
        r"""Return a callable for the copy model method over gRPC.

        Copies an already existing Vertex AI Model into the specified
        Location. The source Model must exist in the same Project. When
        copying custom Models, the users themselves are responsible for
        [Model.metadata][google.cloud.aiplatform.v1beta1.Model.metadata]
        content to be region-agnostic, as well as making sure that any
        resources (e.g. files) it depends on remain accessible.

        Returns:
            Callable[[~.CopyModelRequest],
                    Awaitable[~.Operation]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "copy_model" not in self._stubs:
            self._stubs["copy_model"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.ModelService/CopyModel",
                request_serializer=model_service.CopyModelRequest.serialize,
                response_deserializer=operations_pb2.Operation.FromString,
            )
        return self._stubs["copy_model"]

    @property
    def import_model_evaluation(
        self,
    ) -> Callable[
        [model_service.ImportModelEvaluationRequest],
        Awaitable[gca_model_evaluation.ModelEvaluation],
    ]:
        r"""Return a callable for the import model evaluation method over gRPC.

        Imports an externally generated ModelEvaluation.

        Returns:
            Callable[[~.ImportModelEvaluationRequest],
                    Awaitable[~.ModelEvaluation]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "import_model_evaluation" not in self._stubs:
            self._stubs["import_model_evaluation"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.ModelService/ImportModelEvaluation",
                request_serializer=model_service.ImportModelEvaluationRequest.serialize,
                response_deserializer=gca_model_evaluation.ModelEvaluation.deserialize,
            )
        return self._stubs["import_model_evaluation"]

    @property
    def batch_import_model_evaluation_slices(
        self,
    ) -> Callable[
        [model_service.BatchImportModelEvaluationSlicesRequest],
        Awaitable[model_service.BatchImportModelEvaluationSlicesResponse],
    ]:
        r"""Return a callable for the batch import model evaluation
        slices method over gRPC.

        Imports a list of externally generated
        ModelEvaluationSlice.

        Returns:
            Callable[[~.BatchImportModelEvaluationSlicesRequest],
                    Awaitable[~.BatchImportModelEvaluationSlicesResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "batch_import_model_evaluation_slices" not in self._stubs:
            self._stubs[
                "batch_import_model_evaluation_slices"
            ] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.ModelService/BatchImportModelEvaluationSlices",
                request_serializer=model_service.BatchImportModelEvaluationSlicesRequest.serialize,
                response_deserializer=model_service.BatchImportModelEvaluationSlicesResponse.deserialize,
            )
        return self._stubs["batch_import_model_evaluation_slices"]

    @property
    def batch_import_evaluated_annotations(
        self,
    ) -> Callable[
        [model_service.BatchImportEvaluatedAnnotationsRequest],
        Awaitable[model_service.BatchImportEvaluatedAnnotationsResponse],
    ]:
        r"""Return a callable for the batch import evaluated
        annotations method over gRPC.

        Imports a list of externally generated
        EvaluatedAnnotations.

        Returns:
            Callable[[~.BatchImportEvaluatedAnnotationsRequest],
                    Awaitable[~.BatchImportEvaluatedAnnotationsResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "batch_import_evaluated_annotations" not in self._stubs:
            self._stubs[
                "batch_import_evaluated_annotations"
            ] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.ModelService/BatchImportEvaluatedAnnotations",
                request_serializer=model_service.BatchImportEvaluatedAnnotationsRequest.serialize,
                response_deserializer=model_service.BatchImportEvaluatedAnnotationsResponse.deserialize,
            )
        return self._stubs["batch_import_evaluated_annotations"]

    @property
    def get_model_evaluation(
        self,
    ) -> Callable[
        [model_service.GetModelEvaluationRequest],
        Awaitable[model_evaluation.ModelEvaluation],
    ]:
        r"""Return a callable for the get model evaluation method over gRPC.

        Gets a ModelEvaluation.

        Returns:
            Callable[[~.GetModelEvaluationRequest],
                    Awaitable[~.ModelEvaluation]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_model_evaluation" not in self._stubs:
            self._stubs["get_model_evaluation"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.ModelService/GetModelEvaluation",
                request_serializer=model_service.GetModelEvaluationRequest.serialize,
                response_deserializer=model_evaluation.ModelEvaluation.deserialize,
            )
        return self._stubs["get_model_evaluation"]

    @property
    def list_model_evaluations(
        self,
    ) -> Callable[
        [model_service.ListModelEvaluationsRequest],
        Awaitable[model_service.ListModelEvaluationsResponse],
    ]:
        r"""Return a callable for the list model evaluations method over gRPC.

        Lists ModelEvaluations in a Model.

        Returns:
            Callable[[~.ListModelEvaluationsRequest],
                    Awaitable[~.ListModelEvaluationsResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_model_evaluations" not in self._stubs:
            self._stubs["list_model_evaluations"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.ModelService/ListModelEvaluations",
                request_serializer=model_service.ListModelEvaluationsRequest.serialize,
                response_deserializer=model_service.ListModelEvaluationsResponse.deserialize,
            )
        return self._stubs["list_model_evaluations"]

    @property
    def get_model_evaluation_slice(
        self,
    ) -> Callable[
        [model_service.GetModelEvaluationSliceRequest],
        Awaitable[model_evaluation_slice.ModelEvaluationSlice],
    ]:
        r"""Return a callable for the get model evaluation slice method over gRPC.

        Gets a ModelEvaluationSlice.

        Returns:
            Callable[[~.GetModelEvaluationSliceRequest],
                    Awaitable[~.ModelEvaluationSlice]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "get_model_evaluation_slice" not in self._stubs:
            self._stubs["get_model_evaluation_slice"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.ModelService/GetModelEvaluationSlice",
                request_serializer=model_service.GetModelEvaluationSliceRequest.serialize,
                response_deserializer=model_evaluation_slice.ModelEvaluationSlice.deserialize,
            )
        return self._stubs["get_model_evaluation_slice"]

    @property
    def list_model_evaluation_slices(
        self,
    ) -> Callable[
        [model_service.ListModelEvaluationSlicesRequest],
        Awaitable[model_service.ListModelEvaluationSlicesResponse],
    ]:
        r"""Return a callable for the list model evaluation slices method over gRPC.

        Lists ModelEvaluationSlices in a ModelEvaluation.

        Returns:
            Callable[[~.ListModelEvaluationSlicesRequest],
                    Awaitable[~.ListModelEvaluationSlicesResponse]]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "list_model_evaluation_slices" not in self._stubs:
            self._stubs["list_model_evaluation_slices"] = self.grpc_channel.unary_unary(
                "/google.cloud.aiplatform.v1beta1.ModelService/ListModelEvaluationSlices",
                request_serializer=model_service.ListModelEvaluationSlicesRequest.serialize,
                response_deserializer=model_service.ListModelEvaluationSlicesResponse.deserialize,
            )
        return self._stubs["list_model_evaluation_slices"]

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


__all__ = ("ModelServiceGrpcAsyncIOTransport",)
