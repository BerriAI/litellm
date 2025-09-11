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
import abc
from typing import Awaitable, Callable, Dict, Optional, Sequence, Union

from google.cloud.aiplatform_v1 import gapic_version as package_version

import google.auth  # type: ignore
import google.api_core
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.api_core import operations_v1
from google.auth import credentials as ga_credentials  # type: ignore
from google.oauth2 import service_account  # type: ignore

from google.cloud.aiplatform_v1.types import artifact
from google.cloud.aiplatform_v1.types import artifact as gca_artifact
from google.cloud.aiplatform_v1.types import context
from google.cloud.aiplatform_v1.types import context as gca_context
from google.cloud.aiplatform_v1.types import execution
from google.cloud.aiplatform_v1.types import execution as gca_execution
from google.cloud.aiplatform_v1.types import lineage_subgraph
from google.cloud.aiplatform_v1.types import metadata_schema
from google.cloud.aiplatform_v1.types import metadata_schema as gca_metadata_schema
from google.cloud.aiplatform_v1.types import metadata_service
from google.cloud.aiplatform_v1.types import metadata_store
from google.cloud.location import locations_pb2  # type: ignore
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=package_version.__version__
)


class MetadataServiceTransport(abc.ABC):
    """Abstract transport class for MetadataService."""

    AUTH_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)

    DEFAULT_HOST: str = "aiplatform.googleapis.com"

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        api_audience: Optional[str] = None,
        **kwargs,
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
            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is mutually exclusive with credentials.
            scopes (Optional[Sequence[str]]): A list of scopes.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.
        """

        scopes_kwargs = {"scopes": scopes, "default_scopes": self.AUTH_SCOPES}

        # Save the scopes.
        self._scopes = scopes

        # If no credentials are provided, then determine the appropriate
        # defaults.
        if credentials and credentials_file:
            raise core_exceptions.DuplicateCredentialArgs(
                "'credentials_file' and 'credentials' are mutually exclusive"
            )

        if credentials_file is not None:
            credentials, _ = google.auth.load_credentials_from_file(
                credentials_file, **scopes_kwargs, quota_project_id=quota_project_id
            )
        elif credentials is None:
            credentials, _ = google.auth.default(
                **scopes_kwargs, quota_project_id=quota_project_id
            )
            # Don't apply audience if the credentials file passed from user.
            if hasattr(credentials, "with_gdch_audience"):
                credentials = credentials.with_gdch_audience(
                    api_audience if api_audience else host
                )

        # If the credentials are service account credentials, then always try to use self signed JWT.
        if (
            always_use_jwt_access
            and isinstance(credentials, service_account.Credentials)
            and hasattr(service_account.Credentials, "with_always_use_jwt_access")
        ):
            credentials = credentials.with_always_use_jwt_access(True)

        # Save the credentials.
        self._credentials = credentials

        # Save the hostname. Default to port 443 (HTTPS) if none is specified.
        if ":" not in host:
            host += ":443"
        self._host = host

    @property
    def host(self):
        return self._host

    def _prep_wrapped_messages(self, client_info):
        # Precompute the wrapped methods.
        self._wrapped_methods = {
            self.create_metadata_store: gapic_v1.method.wrap_method(
                self.create_metadata_store,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_metadata_store: gapic_v1.method.wrap_method(
                self.get_metadata_store,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_metadata_stores: gapic_v1.method.wrap_method(
                self.list_metadata_stores,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_metadata_store: gapic_v1.method.wrap_method(
                self.delete_metadata_store,
                default_timeout=None,
                client_info=client_info,
            ),
            self.create_artifact: gapic_v1.method.wrap_method(
                self.create_artifact,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_artifact: gapic_v1.method.wrap_method(
                self.get_artifact,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_artifacts: gapic_v1.method.wrap_method(
                self.list_artifacts,
                default_timeout=None,
                client_info=client_info,
            ),
            self.update_artifact: gapic_v1.method.wrap_method(
                self.update_artifact,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_artifact: gapic_v1.method.wrap_method(
                self.delete_artifact,
                default_timeout=None,
                client_info=client_info,
            ),
            self.purge_artifacts: gapic_v1.method.wrap_method(
                self.purge_artifacts,
                default_timeout=None,
                client_info=client_info,
            ),
            self.create_context: gapic_v1.method.wrap_method(
                self.create_context,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_context: gapic_v1.method.wrap_method(
                self.get_context,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_contexts: gapic_v1.method.wrap_method(
                self.list_contexts,
                default_timeout=None,
                client_info=client_info,
            ),
            self.update_context: gapic_v1.method.wrap_method(
                self.update_context,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_context: gapic_v1.method.wrap_method(
                self.delete_context,
                default_timeout=None,
                client_info=client_info,
            ),
            self.purge_contexts: gapic_v1.method.wrap_method(
                self.purge_contexts,
                default_timeout=None,
                client_info=client_info,
            ),
            self.add_context_artifacts_and_executions: gapic_v1.method.wrap_method(
                self.add_context_artifacts_and_executions,
                default_timeout=None,
                client_info=client_info,
            ),
            self.add_context_children: gapic_v1.method.wrap_method(
                self.add_context_children,
                default_timeout=None,
                client_info=client_info,
            ),
            self.remove_context_children: gapic_v1.method.wrap_method(
                self.remove_context_children,
                default_timeout=None,
                client_info=client_info,
            ),
            self.query_context_lineage_subgraph: gapic_v1.method.wrap_method(
                self.query_context_lineage_subgraph,
                default_timeout=None,
                client_info=client_info,
            ),
            self.create_execution: gapic_v1.method.wrap_method(
                self.create_execution,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_execution: gapic_v1.method.wrap_method(
                self.get_execution,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_executions: gapic_v1.method.wrap_method(
                self.list_executions,
                default_timeout=None,
                client_info=client_info,
            ),
            self.update_execution: gapic_v1.method.wrap_method(
                self.update_execution,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_execution: gapic_v1.method.wrap_method(
                self.delete_execution,
                default_timeout=None,
                client_info=client_info,
            ),
            self.purge_executions: gapic_v1.method.wrap_method(
                self.purge_executions,
                default_timeout=None,
                client_info=client_info,
            ),
            self.add_execution_events: gapic_v1.method.wrap_method(
                self.add_execution_events,
                default_timeout=None,
                client_info=client_info,
            ),
            self.query_execution_inputs_and_outputs: gapic_v1.method.wrap_method(
                self.query_execution_inputs_and_outputs,
                default_timeout=None,
                client_info=client_info,
            ),
            self.create_metadata_schema: gapic_v1.method.wrap_method(
                self.create_metadata_schema,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_metadata_schema: gapic_v1.method.wrap_method(
                self.get_metadata_schema,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_metadata_schemas: gapic_v1.method.wrap_method(
                self.list_metadata_schemas,
                default_timeout=None,
                client_info=client_info,
            ),
            self.query_artifact_lineage_subgraph: gapic_v1.method.wrap_method(
                self.query_artifact_lineage_subgraph,
                default_timeout=None,
                client_info=client_info,
            ),
        }

    def close(self):
        """Closes resources associated with the transport.

        .. warning::
             Only call this method if the transport is NOT shared
             with other clients - this may cause errors in other clients!
        """
        raise NotImplementedError()

    @property
    def operations_client(self):
        """Return the client designed to process long-running operations."""
        raise NotImplementedError()

    @property
    def create_metadata_store(
        self,
    ) -> Callable[
        [metadata_service.CreateMetadataStoreRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def get_metadata_store(
        self,
    ) -> Callable[
        [metadata_service.GetMetadataStoreRequest],
        Union[metadata_store.MetadataStore, Awaitable[metadata_store.MetadataStore]],
    ]:
        raise NotImplementedError()

    @property
    def list_metadata_stores(
        self,
    ) -> Callable[
        [metadata_service.ListMetadataStoresRequest],
        Union[
            metadata_service.ListMetadataStoresResponse,
            Awaitable[metadata_service.ListMetadataStoresResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def delete_metadata_store(
        self,
    ) -> Callable[
        [metadata_service.DeleteMetadataStoreRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def create_artifact(
        self,
    ) -> Callable[
        [metadata_service.CreateArtifactRequest],
        Union[gca_artifact.Artifact, Awaitable[gca_artifact.Artifact]],
    ]:
        raise NotImplementedError()

    @property
    def get_artifact(
        self,
    ) -> Callable[
        [metadata_service.GetArtifactRequest],
        Union[artifact.Artifact, Awaitable[artifact.Artifact]],
    ]:
        raise NotImplementedError()

    @property
    def list_artifacts(
        self,
    ) -> Callable[
        [metadata_service.ListArtifactsRequest],
        Union[
            metadata_service.ListArtifactsResponse,
            Awaitable[metadata_service.ListArtifactsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def update_artifact(
        self,
    ) -> Callable[
        [metadata_service.UpdateArtifactRequest],
        Union[gca_artifact.Artifact, Awaitable[gca_artifact.Artifact]],
    ]:
        raise NotImplementedError()

    @property
    def delete_artifact(
        self,
    ) -> Callable[
        [metadata_service.DeleteArtifactRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def purge_artifacts(
        self,
    ) -> Callable[
        [metadata_service.PurgeArtifactsRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def create_context(
        self,
    ) -> Callable[
        [metadata_service.CreateContextRequest],
        Union[gca_context.Context, Awaitable[gca_context.Context]],
    ]:
        raise NotImplementedError()

    @property
    def get_context(
        self,
    ) -> Callable[
        [metadata_service.GetContextRequest],
        Union[context.Context, Awaitable[context.Context]],
    ]:
        raise NotImplementedError()

    @property
    def list_contexts(
        self,
    ) -> Callable[
        [metadata_service.ListContextsRequest],
        Union[
            metadata_service.ListContextsResponse,
            Awaitable[metadata_service.ListContextsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def update_context(
        self,
    ) -> Callable[
        [metadata_service.UpdateContextRequest],
        Union[gca_context.Context, Awaitable[gca_context.Context]],
    ]:
        raise NotImplementedError()

    @property
    def delete_context(
        self,
    ) -> Callable[
        [metadata_service.DeleteContextRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def purge_contexts(
        self,
    ) -> Callable[
        [metadata_service.PurgeContextsRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def add_context_artifacts_and_executions(
        self,
    ) -> Callable[
        [metadata_service.AddContextArtifactsAndExecutionsRequest],
        Union[
            metadata_service.AddContextArtifactsAndExecutionsResponse,
            Awaitable[metadata_service.AddContextArtifactsAndExecutionsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def add_context_children(
        self,
    ) -> Callable[
        [metadata_service.AddContextChildrenRequest],
        Union[
            metadata_service.AddContextChildrenResponse,
            Awaitable[metadata_service.AddContextChildrenResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def remove_context_children(
        self,
    ) -> Callable[
        [metadata_service.RemoveContextChildrenRequest],
        Union[
            metadata_service.RemoveContextChildrenResponse,
            Awaitable[metadata_service.RemoveContextChildrenResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def query_context_lineage_subgraph(
        self,
    ) -> Callable[
        [metadata_service.QueryContextLineageSubgraphRequest],
        Union[
            lineage_subgraph.LineageSubgraph,
            Awaitable[lineage_subgraph.LineageSubgraph],
        ],
    ]:
        raise NotImplementedError()

    @property
    def create_execution(
        self,
    ) -> Callable[
        [metadata_service.CreateExecutionRequest],
        Union[gca_execution.Execution, Awaitable[gca_execution.Execution]],
    ]:
        raise NotImplementedError()

    @property
    def get_execution(
        self,
    ) -> Callable[
        [metadata_service.GetExecutionRequest],
        Union[execution.Execution, Awaitable[execution.Execution]],
    ]:
        raise NotImplementedError()

    @property
    def list_executions(
        self,
    ) -> Callable[
        [metadata_service.ListExecutionsRequest],
        Union[
            metadata_service.ListExecutionsResponse,
            Awaitable[metadata_service.ListExecutionsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def update_execution(
        self,
    ) -> Callable[
        [metadata_service.UpdateExecutionRequest],
        Union[gca_execution.Execution, Awaitable[gca_execution.Execution]],
    ]:
        raise NotImplementedError()

    @property
    def delete_execution(
        self,
    ) -> Callable[
        [metadata_service.DeleteExecutionRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def purge_executions(
        self,
    ) -> Callable[
        [metadata_service.PurgeExecutionsRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def add_execution_events(
        self,
    ) -> Callable[
        [metadata_service.AddExecutionEventsRequest],
        Union[
            metadata_service.AddExecutionEventsResponse,
            Awaitable[metadata_service.AddExecutionEventsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def query_execution_inputs_and_outputs(
        self,
    ) -> Callable[
        [metadata_service.QueryExecutionInputsAndOutputsRequest],
        Union[
            lineage_subgraph.LineageSubgraph,
            Awaitable[lineage_subgraph.LineageSubgraph],
        ],
    ]:
        raise NotImplementedError()

    @property
    def create_metadata_schema(
        self,
    ) -> Callable[
        [metadata_service.CreateMetadataSchemaRequest],
        Union[
            gca_metadata_schema.MetadataSchema,
            Awaitable[gca_metadata_schema.MetadataSchema],
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_metadata_schema(
        self,
    ) -> Callable[
        [metadata_service.GetMetadataSchemaRequest],
        Union[
            metadata_schema.MetadataSchema, Awaitable[metadata_schema.MetadataSchema]
        ],
    ]:
        raise NotImplementedError()

    @property
    def list_metadata_schemas(
        self,
    ) -> Callable[
        [metadata_service.ListMetadataSchemasRequest],
        Union[
            metadata_service.ListMetadataSchemasResponse,
            Awaitable[metadata_service.ListMetadataSchemasResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def query_artifact_lineage_subgraph(
        self,
    ) -> Callable[
        [metadata_service.QueryArtifactLineageSubgraphRequest],
        Union[
            lineage_subgraph.LineageSubgraph,
            Awaitable[lineage_subgraph.LineageSubgraph],
        ],
    ]:
        raise NotImplementedError()

    @property
    def list_operations(
        self,
    ) -> Callable[
        [operations_pb2.ListOperationsRequest],
        Union[
            operations_pb2.ListOperationsResponse,
            Awaitable[operations_pb2.ListOperationsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_operation(
        self,
    ) -> Callable[
        [operations_pb2.GetOperationRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def cancel_operation(
        self,
    ) -> Callable[[operations_pb2.CancelOperationRequest], None,]:
        raise NotImplementedError()

    @property
    def delete_operation(
        self,
    ) -> Callable[[operations_pb2.DeleteOperationRequest], None,]:
        raise NotImplementedError()

    @property
    def wait_operation(
        self,
    ) -> Callable[
        [operations_pb2.WaitOperationRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def set_iam_policy(
        self,
    ) -> Callable[
        [iam_policy_pb2.SetIamPolicyRequest],
        Union[policy_pb2.Policy, Awaitable[policy_pb2.Policy]],
    ]:
        raise NotImplementedError()

    @property
    def get_iam_policy(
        self,
    ) -> Callable[
        [iam_policy_pb2.GetIamPolicyRequest],
        Union[policy_pb2.Policy, Awaitable[policy_pb2.Policy]],
    ]:
        raise NotImplementedError()

    @property
    def test_iam_permissions(
        self,
    ) -> Callable[
        [iam_policy_pb2.TestIamPermissionsRequest],
        Union[
            iam_policy_pb2.TestIamPermissionsResponse,
            Awaitable[iam_policy_pb2.TestIamPermissionsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_location(
        self,
    ) -> Callable[
        [locations_pb2.GetLocationRequest],
        Union[locations_pb2.Location, Awaitable[locations_pb2.Location]],
    ]:
        raise NotImplementedError()

    @property
    def list_locations(
        self,
    ) -> Callable[
        [locations_pb2.ListLocationsRequest],
        Union[
            locations_pb2.ListLocationsResponse,
            Awaitable[locations_pb2.ListLocationsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def kind(self) -> str:
        raise NotImplementedError()


__all__ = ("MetadataServiceTransport",)
