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

from typing import Any, Dict, List, Optional, Union

from google.auth import credentials as auth_credentials

from google.cloud.aiplatform import models
from google.cloud.aiplatform.compat.types import execution as gca_execution
from google.cloud.aiplatform.constants import base as base_constants
from google.cloud.aiplatform.metadata import artifact
from google.cloud.aiplatform.metadata import constants
from google.cloud.aiplatform.metadata import execution
from google.cloud.aiplatform.metadata import metadata


class BaseExecutionSchema(execution.Execution):
    """Base class for Metadata Execution schema."""

    @property
    @classmethod
    @abc.abstractmethod
    def schema_title(cls) -> str:
        """Identifies the Vertex Metadta schema title used by the resource."""
        pass

    def __init__(
        self,
        *,
        state: Optional[
            gca_execution.Execution.State
        ] = gca_execution.Execution.State.RUNNING,
        execution_id: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        metadata: Optional[Dict] = None,
        description: Optional[str] = None,
    ):
        """Initializes the Execution with the given name, URI and metadata.

        Args:
            state (gca_execution.Execution.State.RUNNING):
                Optional. State of this Execution. Defaults to RUNNING.
            execution_id (str):
                Optional. The <resource_id> portion of the Execution name with
                the following format, this is globally unique in a metadataStore.
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/executions/<resource_id>.
            display_name (str):
                Optional. The user-defined name of the Execution.
            schema_version (str):
                Optional. schema_version specifies the version used by the Execution.
                If not set, defaults to use the latest version.
            metadata (Dict):
                Optional. Contains the metadata information that will be stored in the Execution.
            description (str):
                Optional. Describes the purpose of the Execution to be created.
        """
        # initialize the exception to resolve the FutureManager exception.
        self._exception = None
        # resource_id is not stored in the proto. Create method uses the
        # resource_id along with project_id and location to construct an
        # resource_name which is stored in the proto message.
        self.execution_id = execution_id

        # Store all other attributes using the proto structure.
        self._gca_resource = gca_execution.Execution()
        self._gca_resource.state = state
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
        execution_name: str,
    ):
        """Initializes the Execution instance using an existing resource.
        Args:
            execution_name (str):
                The Execution name with the following format, this is globally unique in a metadataStore.
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/executions/<resource_id>.
        """
        # Add User Agent Header for metrics tracking if one is not specified
        # If one is already specified this call was initiated by a sub class.
        if not base_constants.USER_AGENT_SDK_COMMAND:
            base_constants.USER_AGENT_SDK_COMMAND = "aiplatform.metadata.schema.base_execution.BaseExecutionSchema._init_with_resource_name"

        super(BaseExecutionSchema, self).__init__(execution_name=execution_name)

    def create(
        self,
        *,
        metadata_store_id: Optional[str] = "default",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "execution.Execution":
        """Creates a new Metadata Execution.

        Args:
            metadata_store_id (str):
                Optional. The <metadata_store_id> portion of the resource name with
                the format:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/executions/<resource_id>
                If not provided, the MetadataStore's ID will be set to "default".
            project (str):
                Optional. Project used to create this Execution. Overrides project set in
                aiplatform.init.
            location (str):
                Optional. Location used to create this Execution. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials used to create this Execution. Overrides
                credentials set in aiplatform.init.
        Returns:
            Execution: Instantiated representation of the managed Metadata Execution.

        """
        # Add User Agent Header for metrics tracking if one is not specified
        # If one is already specified this call was initiated by a sub class.
        base_constants.USER_AGENT_SDK_COMMAND = (
            "aiplatform.metadata.schema.base_execution.BaseExecutionSchema.create"
        )

        # Check if metadata exists to avoid proto read error
        metadata = None
        if self._gca_resource.metadata:
            metadata = self.metadata

        new_execution_instance = execution.Execution.create(
            resource_id=self.execution_id,
            schema_title=self.schema_title,
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
        self._init_with_resource_name(
            execution_name=new_execution_instance.resource_name
        )
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
    ) -> List["BaseExecutionSchema"]:
        """List all the Execution resources with a particular schema.

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
            A list of execution resources with a particular schema.

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

    def start_execution(
        self,
        *,
        metadata_store_id: Optional[str] = "default",
        resume: bool = False,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "execution.Execution":
        """Create and starts a new Metadata Execution or resumes a previously created Execution.

        This method is similar to create_execution with additional support for Experiments.
        If an Experiment is set prior to running this command, the Experiment will be
        associtaed with the created execution, otherwise this method behaves the same
        as create_execution.

        To start a new execution:
        ```
        instance_of_execution_schema = execution_schema.ContainerExecution(...)
        with instance_of_execution_schema.start_execution() as exc:
          exc.assign_input_artifacts([my_artifact])
          model = aiplatform.Artifact.create(uri='gs://my-uri', schema_title='system.Model')
          exc.assign_output_artifacts([model])
        ```

        To continue a previously created execution:
        ```
        with execution_schema.ContainerExecution(resource_id='my-exc', resume=True) as exc:
            ...
        ```
        Args:
            metadata_store_id (str):
                Optional. The <metadata_store_id> portion of the resource name with
                the format:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/executions/<executions_id>
                If not provided, the MetadataStore's ID will be set to "default". Currently only the 'default'
                MetadataStore ID is supported.
            resume (bool):
                Resume an existing execution.
            project (str):
                Optional. Project used to create this Execution. Overrides project set in
                aiplatform.init.
            location (str):
                Optional. Location used to create this Execution. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials used to create this Execution. Overrides
                credentials set in aiplatform.init.
        Returns:
            Execution: Instantiated representation of the managed Metadata Execution.
        Raises:
            ValueError: If metadata_store_id other than 'default' is provided.
        """
        # Add User Agent Header for metrics tracking if one is not specified
        # If one is already specified this call was initiated by a sub class.

        base_constants.USER_AGENT_SDK_COMMAND = "aiplatform.metadata.schema.base_execution.BaseExecutionSchema.start_execution"

        if metadata_store_id != "default":
            raise ValueError(
                f"metadata_store_id {metadata_store_id} is not supported. Only the default MetadataStore ID is supported."
            )

        new_execution_instance = metadata._ExperimentTracker().start_execution(
            schema_title=self.schema_title,
            display_name=self.display_name,
            resource_id=self.execution_id,
            metadata=self.metadata,
            schema_version=self.schema_version,
            description=self.description,
            # TODO: Add support for metadata_store_id once it is supported in experiment.
            resume=resume,
            project=project,
            location=location,
            credentials=credentials,
        )

        # Reinstantiate this class using the newly created resource.
        self._init_with_resource_name(
            execution_name=new_execution_instance.resource_name
        )
        return self

    def assign_input_artifacts(
        self, artifacts: List[Union[artifact.Artifact, models.Model]]
    ):
        """Assigns Artifacts as inputs to this Executions.

        Args:
            artifacts (List[Union[artifact.Artifact, models.Model]]):
                Required. Artifacts to assign as input.

        Raises:
            RuntimeError: if Execution resource hasn't been created.
        """
        if self._gca_resource.name:
            super().assign_input_artifacts(artifacts)
        else:
            raise RuntimeError(
                f"{self.__class__.__name__} resource has not been created."
            )

    def assign_output_artifacts(
        self, artifacts: List[Union[artifact.Artifact, models.Model]]
    ):
        """Assigns Artifacts as outputs to this Executions.

        Args:
            artifacts (List[Union[artifact.Artifact, models.Model]]):
                Required. Artifacts to assign as input.

        Raises:
            RuntimeError: if Execution resource hasn't been created.
        """
        if self._gca_resource.name:
            super().assign_output_artifacts(artifacts)
        else:
            raise RuntimeError(
                f"{self.__class__.__name__} resource has not been created."
            )

    def get_input_artifacts(self) -> List[artifact.Artifact]:
        """Get the input Artifacts of this Execution.

        Returns:
            List of input Artifacts.

        Raises:
            RuntimeError: if Execution resource hasn't been created.
        """
        if self._gca_resource.name:
            return super().get_input_artifacts()
        else:
            raise RuntimeError(
                f"{self.__class__.__name__} resource has not been created."
            )

    def get_output_artifacts(self) -> List[artifact.Artifact]:
        """Get the output Artifacts of this Execution.

        Returns:
            List of output Artifacts.

        Raises:
            RuntimeError: if Execution resource hasn't been created.
        """
        if self._gca_resource.name:
            return super().get_output_artifacts()
        else:
            raise RuntimeError(
                f"{self.__class__.__name__} resource has not been created."
            )

    def update(
        self,
        state: Optional[gca_execution.Execution.State] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Update this Execution.

        Args:
            state (gca_execution.Execution.State):
                    Optional. State of this Execution.
            description (str):
                Optional. Describes the purpose of the Execution to be created.
            metadata (Dict[str, Any):
                Optional. Contains the metadata information that will be stored
                in the Execution.

        Raises:
            RuntimeError: if Execution resource hasn't been created.
        """
        if self._gca_resource.name:
            super().update(
                state=state,
                description=description,
                metadata=metadata,
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
