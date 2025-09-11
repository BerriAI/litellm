# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
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

import copy
from typing import Optional, Dict

from google.cloud.aiplatform.compat.types import execution as gca_execution
from google.cloud.aiplatform.metadata.schema import base_execution


class ContainerExecution(base_execution.BaseExecutionSchema):
    """Execution schema for a container execution."""

    schema_title = "system.ContainerExecution"

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
        """Args:
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
        extended_metadata = copy.deepcopy(metadata) if metadata else {}
        super(ContainerExecution, self).__init__(
            execution_id=execution_id,
            state=state,
            display_name=display_name,
            schema_version=schema_version,
            description=description,
            metadata=extended_metadata,
        )


class CustomJobExecution(base_execution.BaseExecutionSchema):
    """Execution schema for a custom job execution."""

    schema_title = "system.CustomJobExecution"

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
        """Args:
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
        extended_metadata = copy.deepcopy(metadata) if metadata else {}
        super(CustomJobExecution, self).__init__(
            execution_id=execution_id,
            state=state,
            display_name=display_name,
            schema_version=schema_version,
            description=description,
            metadata=extended_metadata,
        )


class Run(base_execution.BaseExecutionSchema):
    """Execution schema for root run execution."""

    schema_title = "system.Run"

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
        """Args:
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
        extended_metadata = copy.deepcopy(metadata) if metadata else {}
        super(Run, self).__init__(
            execution_id=execution_id,
            state=state,
            display_name=display_name,
            schema_version=schema_version,
            description=description,
            metadata=extended_metadata,
        )
