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

from google.cloud.aiplatform.compat.types import artifact as gca_artifact
from google.cloud.aiplatform.metadata.schema import base_artifact


class Model(base_artifact.BaseArtifactSchema):
    """Artifact type for model."""

    schema_title = "system.Model"

    def __init__(
        self,
        *,
        uri: Optional[str] = None,
        artifact_id: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        state: Optional[gca_artifact.Artifact.State] = gca_artifact.Artifact.State.LIVE,
    ):
        """Args:
        uri (str):
            Optional. The uniform resource identifier of the artifact file. May be empty if there is no actual
            artifact file.
        artifact_id (str):
            Optional. The <resource_id> portion of the Artifact name with
            the format. This is globally unique in a metadataStore:
            projects/123/locations/us-central1/metadataStores/<metadata_store_id>/artifacts/<resource_id>.
        display_name (str):
            Optional. The user-defined name of the base.
        schema_version (str):
            Optional. schema_version specifies the version used by the base.
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
        extended_metadata = copy.deepcopy(metadata) if metadata else {}
        super(Model, self).__init__(
            uri=uri,
            artifact_id=artifact_id,
            display_name=display_name,
            schema_version=schema_version,
            description=description,
            metadata=extended_metadata,
            state=state,
        )


class Artifact(base_artifact.BaseArtifactSchema):
    """A generic artifact."""

    schema_title = "system.Artifact"

    def __init__(
        self,
        *,
        uri: Optional[str] = None,
        artifact_id: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        state: Optional[gca_artifact.Artifact.State] = gca_artifact.Artifact.State.LIVE,
    ):
        """Args:
        uri (str):
            Optional. The uniform resource identifier of the artifact file. May be empty if there is no actual
            artifact file.
        artifact_id (str):
            Optional. The <resource_id> portion of the Artifact name with
            the format. This is globally unique in a metadataStore:
            projects/123/locations/us-central1/metadataStores/<metadata_store_id>/artifacts/<resource_id>.
        display_name (str):
            Optional. The user-defined name of the base.
        schema_version (str):
            Optional. schema_version specifies the version used by the base.
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
        extended_metadata = copy.deepcopy(metadata) if metadata else {}
        super(Artifact, self).__init__(
            uri=uri,
            artifact_id=artifact_id,
            display_name=display_name,
            schema_version=schema_version,
            description=description,
            metadata=extended_metadata,
            state=state,
        )


class Dataset(base_artifact.BaseArtifactSchema):
    """An artifact representing a system Dataset."""

    schema_title = "system.Dataset"

    def __init__(
        self,
        *,
        uri: Optional[str] = None,
        artifact_id: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        state: Optional[gca_artifact.Artifact.State] = gca_artifact.Artifact.State.LIVE,
    ):
        """Args:
        uri (str):
            Optional. The uniform resource identifier of the artifact file. May be empty if there is no actual
            artifact file.
        artifact_id (str):
            Optional. The <resource_id> portion of the Artifact name with
            the format. This is globally unique in a metadataStore:
            projects/123/locations/us-central1/metadataStores/<metadata_store_id>/artifacts/<resource_id>.
        display_name (str):
            Optional. The user-defined name of the base.
        schema_version (str):
            Optional. schema_version specifies the version used by the base.
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
        extended_metadata = copy.deepcopy(metadata) if metadata else {}
        super(Dataset, self).__init__(
            uri=uri,
            artifact_id=artifact_id,
            display_name=display_name,
            schema_version=schema_version,
            description=description,
            metadata=extended_metadata,
            state=state,
        )


class Metrics(base_artifact.BaseArtifactSchema):
    """Artifact schema for scalar metrics."""

    schema_title = "system.Metrics"

    def __init__(
        self,
        *,
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1score: Optional[float] = None,
        mean_absolute_error: Optional[float] = None,
        mean_squared_error: Optional[float] = None,
        uri: Optional[str] = None,
        artifact_id: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        state: Optional[gca_artifact.Artifact.State] = gca_artifact.Artifact.State.LIVE,
    ):
        """Args:
        accuracy (float):
            Optional.
        precision (float):
            Optional.
        recall (float):
            Optional.
        f1score (float):
            Optional.
        mean_absolute_error (float):
            Optional.
        mean_squared_error (float):
            Optional.
        uri (str):
            Optional. The uniform resource identifier of the artifact file. May be empty if there is no actual
            artifact file.
        artifact_id (str):
            Optional. The <resource_id> portion of the Artifact name with
            the format. This is globally unique in a metadataStore:
            projects/123/locations/us-central1/metadataStores/<metadata_store_id>/artifacts/<resource_id>.
        display_name (str):
            Optional. The user-defined name of the base.
        schema_version (str):
            Optional. schema_version specifies the version used by the base.
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
        extended_metadata = copy.deepcopy(metadata) if metadata else {}
        if accuracy:
            extended_metadata["accuracy"] = accuracy
        if precision:
            extended_metadata["precision"] = precision
        if recall:
            extended_metadata["recall"] = recall
        if f1score:
            extended_metadata["f1score"] = f1score
        if mean_absolute_error:
            extended_metadata["mean_absolute_error"] = mean_absolute_error
        if mean_squared_error:
            extended_metadata["mean_squared_error"] = mean_squared_error

        super(Metrics, self).__init__(
            uri=uri,
            artifact_id=artifact_id,
            display_name=display_name,
            schema_version=schema_version,
            description=description,
            metadata=extended_metadata,
            state=state,
        )
