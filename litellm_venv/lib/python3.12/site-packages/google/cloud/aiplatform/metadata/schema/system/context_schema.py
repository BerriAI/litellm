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

from google.cloud.aiplatform.metadata.schema import base_context


class Experiment(base_context.BaseContextSchema):
    """Context schema for a Experiment context."""

    schema_title = "system.Experiment"

    def __init__(
        self,
        *,
        context_id: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        metadata: Optional[Dict] = None,
        description: Optional[str] = None,
    ):
        """Args:
        context_id (str):
            Optional. The <resource_id> portion of the context name with
            the following format, this is globally unique in a metadataStore.
            projects/123/locations/us-central1/metadataStores/<metadata_store_id>/contexts/<resource_id>.
        display_name (str):
            Optional. The user-defined name of the context.
        schema_version (str):
            Optional. schema_version specifies the version used by the context.
            If not set, defaults to use the latest version.
        metadata (Dict):
            Optional. Contains the metadata information that will be stored in the context.
        description (str):
            Optional. Describes the purpose of the context to be created.
        """
        extended_metadata = copy.deepcopy(metadata) if metadata else {}
        super(Experiment, self).__init__(
            context_id=context_id,
            display_name=display_name,
            schema_version=schema_version,
            description=description,
            metadata=extended_metadata,
        )


class ExperimentRun(base_context.BaseContextSchema):
    """Context schema for a ExperimentRun context."""

    schema_title = "system.ExperimentRun"

    def __init__(
        self,
        *,
        experiment_id: Optional[str] = None,
        context_id: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        metadata: Optional[Dict] = None,
        description: Optional[str] = None,
    ):
        """Args:
        experiment_id (str):
            Optional. The experiment_id that this experiment_run belongs to.
        context_id (str):
            Optional. The <resource_id> portion of the context name with
            the following format, this is globally unique in a metadataStore.
            projects/123/locations/us-central1/metadataStores/<metadata_store_id>/contexts/<resource_id>.
        display_name (str):
            Optional. The user-defined name of the context.
        schema_version (str):
            Optional. schema_version specifies the version used by the context.
            If not set, defaults to use the latest version.
        metadata (Dict):
            Optional. Contains the metadata information that will be stored in the context.
        description (str):
            Optional. Describes the purpose of the context to be created.
        """
        extended_metadata = copy.deepcopy(metadata) if metadata else {}
        extended_metadata["experiment_id"] = experiment_id
        super(ExperimentRun, self).__init__(
            context_id=context_id,
            display_name=display_name,
            schema_version=schema_version,
            description=description,
            metadata=extended_metadata,
        )


class Pipeline(base_context.BaseContextSchema):
    """Context schema for a Pipeline context."""

    schema_title = "system.Pipeline"

    def __init__(
        self,
        *,
        context_id: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        metadata: Optional[Dict] = None,
        description: Optional[str] = None,
    ):
        """Args:
        context_id (str):
            Optional. The <resource_id> portion of the context name with
            the following format, this is globally unique in a metadataStore.
            projects/123/locations/us-central1/metadataStores/<metadata_store_id>/contexts/<resource_id>.
        display_name (str):
            Optional. The user-defined name of the context.
        schema_version (str):
            Optional. schema_version specifies the version used by the context.
            If not set, defaults to use the latest version.
        metadata (Dict):
            Optional. Contains the metadata information that will be stored in the context.
        description (str):
            Optional. Describes the purpose of the context to be created.
        """
        extended_metadata = copy.deepcopy(metadata) if metadata else {}
        super(Pipeline, self).__init__(
            context_id=context_id,
            display_name=display_name,
            schema_version=schema_version,
            description=description,
            metadata=extended_metadata,
        )


class PipelineRun(base_context.BaseContextSchema):
    """Context schema for a PipelineRun context."""

    schema_title = "system.PipelineRun"

    def __init__(
        self,
        *,
        pipeline_id: Optional[str] = None,
        context_id: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        metadata: Optional[Dict] = None,
        description: Optional[str] = None,
    ):
        """Args:
        pipeline_id (str):
            Optional. PipelineJob resource name corresponding to this run.
        context_id (str):
            Optional. The <resource_id> portion of the context name with
            the following format, this is globally unique in a metadataStore.
            projects/123/locations/us-central1/metadataStores/<metadata_store_id>/contexts/<resource_id>.
        display_name (str):
            Optional. The user-defined name of the context.
        schema_version (str):
            Optional. schema_version specifies the version used by the context.
            If not set, defaults to use the latest version.
        metadata (Dict):
            Optional. Contains the metadata information that will be stored in the context.
        description (str):
            Optional. Describes the purpose of the context to be created.
        """
        extended_metadata = copy.deepcopy(metadata) if metadata else {}
        extended_metadata["pipeline_id"] = pipeline_id
        super(PipelineRun, self).__init__(
            context_id=context_id,
            display_name=display_name,
            schema_version=schema_version,
            description=description,
            metadata=extended_metadata,
        )
