# -*- coding: utf-8 -*-

# Copyright 2023 Google LLC
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

from typing import Dict, List, Optional, Sequence

from google.auth import credentials as auth_credentials

from google.cloud.aiplatform.compat.types import context as gca_context
from google.cloud.aiplatform.compat.types import (
    lineage_subgraph as gca_lineage_subgraph,
)
from google.cloud.aiplatform.constants import base as base_constants
from google.cloud.aiplatform.metadata import constants
from google.cloud.aiplatform.metadata import context


class BaseContextSchema(context.Context):
    """Base class for Metadata Context schema."""

    @property
    @classmethod
    @abc.abstractmethod
    def schema_title(cls) -> str:
        """Identifies the Vertex Metadta schema title used by the resource."""
        pass

    def __init__(
        self,
        *,
        context_id: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        metadata: Optional[Dict] = None,
        description: Optional[str] = None,
    ):
        """Initializes the Context with the given name, URI and metadata.

        Args:
            context_id (str):
                Optional. The <resource_id> portion of the Context name with
                the following format, this is globally unique in a metadataStore.
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/Contexts/<resource_id>.
            display_name (str):
                Optional. The user-defined name of the Context.
            schema_version (str):
                Optional. schema_version specifies the version used by the Context.
                If not set, defaults to use the latest version.
            metadata (Dict):
                Optional. Contains the metadata information that will be stored in the Context.
            description (str):
                Optional. Describes the purpose of the Context to be created.
        """
        # initialize the exception to resolve the FutureManager exception.
        self._exception = None
        # resource_id is not stored in the proto. Create method uses the
        # resource_id along with project_id and location to construct an
        # resource_name which is stored in the proto message.
        self.context_id = context_id

        # Store all other attributes using the proto structure.
        self._gca_resource = gca_context.Context()
        self._gca_resource.display_name = display_name
        self._gca_resource.schema_version = (
            schema_version or constants._DEFAULT_SCHEMA_VERSION
        )
        # If metadata is None covert to {}
        metadata = metadata if metadata else {}
        self._nested_update_metadata(self._gca_resource, metadata)
        self._gca_resource.description = description

    # TODO() Switch to @singledispatchmethod constructor overload after py>=3.8
    def _init_with_resource_name(
        self,
        *,
        context_name: str,
    ):
        """Initializes the Artifact instance using an existing resource.
        Args:
            context_name (str):
                Context name with the following format, this is globally unique in a metadataStore:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/contexts/<resource_id>.
        """
        # Add User Agent Header for metrics tracking if one is not specified
        # If one is already specified this call was initiated by a sub class.
        if not base_constants.USER_AGENT_SDK_COMMAND:
            base_constants.USER_AGENT_SDK_COMMAND = "aiplatform.metadata.schema.base_context.BaseContextSchema._init_with_resource_name"

        super(BaseContextSchema, self).__init__(resource_name=context_name)

    def create(
        self,
        *,
        metadata_store_id: Optional[str] = "default",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "context.Context":
        """Creates a new Metadata Context.

        Args:
            metadata_store_id (str):
                Optional. The <metadata_store_id> portion of the resource name with
                the format:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/Contexts/<resource_id>
                If not provided, the MetadataStore's ID will be set to "default".
            project (str):
                Optional. Project used to create this Context. Overrides project set in
                aiplatform.init.
            location (str):
                Optional. Location used to create this Context. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials used to create this Context. Overrides
                credentials set in aiplatform.init.
        Returns:
            Context: Instantiated representation of the managed Metadata Context.

        """
        # Add User Agent Header for metrics tracking.
        base_constants.USER_AGENT_SDK_COMMAND = (
            "aiplatform.metadata.schema.base_context.BaseContextSchema.create"
        )

        # Check if metadata exists to avoid proto read error
        metadata = None
        if self._gca_resource.metadata:
            metadata = self.metadata

        new_context = context.Context.create(
            resource_id=self.context_id,
            schema_title=self.schema_title,
            display_name=self.display_name,
            schema_version=self.schema_version,
            description=self.description,
            metadata=metadata,
            metadata_store_id=metadata_store_id,
            project=project,
            location=location,
            credentials=credentials,
        )

        # Reinstantiate this class using the newly created resource.
        self._init_with_resource_name(context_name=new_context.resource_name)
        return self

    @classmethod
    def list(
        cls,
        filter: Optional[str] = None,  # pylint: disable=redefined-builtin
        metadata_store_id: str = "default",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        order_by: Optional[str] = None,
    ) -> List["BaseContextSchema"]:
        """List all the Context resources with a particular schema.

        Args:
            filter (str):
                Optional. A query to filter available resources for
                matching results.
            metadata_store_id (str):
                The <metadata_store_id> portion of the resource name with
                the format:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/<resource_noun>/<resource_id>
                If not provided, the MetadataStore's ID will be set to "default".
            project (str):
                Project used to create this resource. Overrides project set in
                aiplatform.init.
            location (str):
                Location used to create this resource. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Custom credentials used to create this resource. Overrides
                credentials set in aiplatform.init.
            order_by (str):
              Optional. How the list of messages is ordered.
              Specify the values to order by and an ordering operation. The
              default sorting order is ascending. To specify descending order
              for a field, users append a " desc" suffix; for example: "foo
              desc, bar". Subfields are specified with a ``.`` character, such
              as foo.bar. see https://google.aip.dev/132#ordering for more
              details.

        Returns:
            A list of context resources with a particular schema.

        """
        schema_filter = f'schema_title="{cls.schema_title}"'
        if filter:
            filter = f"{filter} AND {schema_filter}"
        else:
            filter = schema_filter

        return super().list(
            filter=filter,
            metadata_store_id=metadata_store_id,
            project=project,
            location=location,
            credentials=credentials,
        )

    def add_artifacts_and_executions(
        self,
        artifact_resource_names: Optional[Sequence[str]] = None,
        execution_resource_names: Optional[Sequence[str]] = None,
    ):
        """Associate Executions and attribute Artifacts to a given Context.

        Args:
            artifact_resource_names (Sequence[str]):
                Optional. The full resource name of Artifacts to attribute to
                the Context.
            execution_resource_names (Sequence[str]):
                Optional. The full resource name of Executions to associate with
                the Context.

        Raises:
            RuntimeError: if Context resource hasn't been created.
        """
        if self._gca_resource.name:
            super().add_artifacts_and_executions(
                artifact_resource_names=artifact_resource_names,
                execution_resource_names=execution_resource_names,
            )
        else:
            raise RuntimeError(
                f"{self.__class__.__name__} resource has not been created."
            )

    def add_context_children(self, contexts: List[context.Context]):
        """Adds the provided contexts as children of this context.

        Args:
            contexts (List[_Context]): Contexts to add as children.

        Raises:
            RuntimeError: if Context resource hasn't been created.
        """
        if self._gca_resource.name:
            super().add_context_children(contexts)
        else:
            raise RuntimeError(
                f"{self.__class__.__name__} resource has not been created."
            )

    def query_lineage_subgraph(self) -> gca_lineage_subgraph.LineageSubgraph:
        """Queries lineage subgraph of this context.

        Returns:
            lineage subgraph(gca_lineage_subgraph.LineageSubgraph):
            Lineage subgraph of this Context.

        Raises:
            RuntimeError: if Context resource hasn't been created.
        """
        if self._gca_resource.name:
            return super().query_lineage_subgraph()
        else:
            raise RuntimeError(
                f"{self.__class__.__name__} resource has not been created."
            )

    def __repr__(self) -> str:
        if self._gca_resource.name:
            return super().__repr__()
        else:
            return f"{object.__repr__(self)}\nschema_title: {self.schema_title}"
