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

from typing import Any, Optional, Dict, List

from google.auth import credentials as auth_credentials
from google.cloud.aiplatform.compat.types import artifact as gca_artifact
from google.cloud.aiplatform.metadata import artifact
from google.cloud.aiplatform.constants import base as base_constants
from google.cloud.aiplatform.metadata import constants


class BaseArtifactSchema(artifact.Artifact):
    """Base class for Metadata Artifact types."""

    @property
    @classmethod
    @abc.abstractmethod
    def schema_title(cls) -> str:
        """Identifies the Vertex Metadata schema title used by the resource."""
        pass

    def __init__(
        self,
        *,
        artifact_id: Optional[str] = None,
        uri: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        state: Optional[gca_artifact.Artifact.State] = gca_artifact.Artifact.State.LIVE,
    ):
        """Initializes the Artifact with the given name, URI and metadata.

        This is the base class for defining various artifact types, which can be
        passed to google.Artifact to create a corresponding resource.
        Artifacts carry a `metadata` field, which is a dictionary for storing
        metadata related to this artifact. Subclasses from ArtifactType can enforce
        various structure and field requirements for the metadata field.

        Args:
            artifact_id (str):
                Optional. The <resource_id> portion of the Artifact name with
                the following format, this is globally unique in a metadataStore:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/artifacts/<resource_id>.
            uri (str):
                Optional. The uniform resource identifier of the artifact file. May be empty if there is no actual
                artifact file.
            display_name (str):
                Optional. The user-defined name of the Artifact.
            schema_version (str):
                Optional. schema_version specifies the version used by the Artifact.
                If not set, defaults to use the latest version.
            description (str):
                Optional. Describes the purpose of the Artifact to be created.
            metadata (Dict):
                Optional. Contains the metadata information that will be stored in the Artifact.
            state (google.cloud.gapic.types.Artifact.State):
                Optional. The state of this Artifact. This is a
                property of the Artifact, and does not imply or
                capture any ongoing process. This property is
                managed by clients (such as Vertex AI
                Pipelines), and the system does not prescribe or
                check the validity of state transitions.
        """
        # initialize the exception to resolve the FutureManager exception.
        self._exception = None
        # resource_id is not stored in the proto. Create method uses the
        # resource_id along with project_id and location to construct an
        # resource_name which is stored in the proto message.
        self.artifact_id = artifact_id

        # Store all other attributes using the proto structure.
        self._gca_resource = gca_artifact.Artifact()
        self._gca_resource.uri = uri
        self._gca_resource.display_name = display_name
        self._gca_resource.schema_version = (
            schema_version or constants._DEFAULT_SCHEMA_VERSION
        )
        self._gca_resource.description = description

        # If metadata is None covert to {}
        metadata = metadata if metadata else {}
        self._nested_update_metadata(self._gca_resource, metadata)
        self._gca_resource.state = state

    # TODO() Switch to @singledispatchmethod constructor overload after py>=3.8
    def _init_with_resource_name(
        self,
        *,
        artifact_name: str,
        metadata_store_id: str = "default",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Initializes the Artifact instance using an existing resource.

        Args:
            artifact_name (str):
                Artifact name with the following format, this is globally unique in a metadataStore:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/artifacts/<resource_id>.
            metadata_store_id (str):
                Optional. MetadataStore to retrieve Artifact from. If not set, metadata_store_id is set to "default".
                If artifact_name is a fully-qualified resource, its metadata_store_id overrides this one.
            project (str):
                Optional. Project to retrieve the artifact from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve the Artifact from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve this Artifact. Overrides
                credentials set in aiplatform.init.
        """
        # Add User Agent Header for metrics tracking if one is not specified
        # If one is already specified this call was initiated by a sub class.
        if not base_constants.USER_AGENT_SDK_COMMAND:
            base_constants.USER_AGENT_SDK_COMMAND = "aiplatform.metadata.schema.base_artifact.BaseArtifactSchema._init_with_resource_name"

        super(BaseArtifactSchema, self).__init__(
            artifact_name=artifact_name,
            metadata_store_id=metadata_store_id,
            project=project,
            location=location,
            credentials=credentials,
        )

    def create(
        self,
        *,
        metadata_store_id: Optional[str] = "default",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "artifact.Artifact":
        """Creates a new Metadata Artifact.

        Args:
            metadata_store_id (str):
                Optional. The <metadata_store_id> portion of the resource name with
                the format:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/artifacts/<resource_id>
                If not provided, the MetadataStore's ID will be set to "default".
            project (str):
                Optional. Project used to create this Artifact. Overrides project set in
                aiplatform.init.
            location (str):
                Optional. Location used to create this Artifact. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials used to create this Artifact. Overrides
                credentials set in aiplatform.init.
        Returns:
            Artifact: Instantiated representation of the managed Metadata Artifact.
        """
        # Add User Agent Header for metrics tracking.
        base_constants.USER_AGENT_SDK_COMMAND = (
            "aiplatform.metadata.schema.base_artifact.BaseArtifactSchema.create"
        )

        # Check if metadata exists to avoid proto read error
        metadata = None
        if self._gca_resource.metadata:
            metadata = self.metadata

        new_artifact_instance = artifact.Artifact.create(
            resource_id=self.artifact_id,
            schema_title=self.schema_title,
            uri=self.uri,
            display_name=self.display_name,
            schema_version=self.schema_version,
            description=self.description,
            metadata=metadata,
            state=self.state,
            metadata_store_id=metadata_store_id,
            project=project,
            location=location,
            credentials=credentials,
        )

        # Reinstantiate this class using the newly created resource.
        self._init_with_resource_name(artifact_name=new_artifact_instance.resource_name)
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
    ) -> List["BaseArtifactSchema"]:
        """List all the Artifact resources with a particular schema.

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
            A list of artifact resources with a particular schema.

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

    def sync_resource(self):
        """Syncs local resource with the resource in metadata store.

        Raises:
            RuntimeError: if the artifact resource hasn't been created.
        """
        if self._gca_resource.name:
            super().sync_resource()
        else:
            raise RuntimeError(
                f"{self.__class__.__name__} resource has not been created."
            )

    def update(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Updates an existing Artifact resource with new metadata.

        Args:
            metadata (Dict):
                Optional. metadata contains the updated metadata information.
            description (str):
                Optional. Description describes the resource to be updated.
            credentials (auth_credentials.Credentials):
                Custom credentials to use to update this resource. Overrides
                credentials set in aiplatform.init.

        Raises:
            RuntimeError: if the artifact resource hasn't been created.
        """
        if self._gca_resource.name:
            super().update(
                metadata=metadata,
                description=description,
                credentials=credentials,
            )
        else:
            raise RuntimeError(
                f"{self.__class__.__name__} resource has not been created."
            )

    def __repr__(self) -> str:
        if self._gca_resource.name:
            return super().__repr__()
        else:
            return f"{object.__repr__(self)}\nschema_title: {self.schema_title}"
