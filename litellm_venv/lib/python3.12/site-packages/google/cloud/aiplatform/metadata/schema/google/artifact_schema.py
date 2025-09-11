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

import copy
from typing import Any, Dict, List, Optional, Sequence, Union

from google.auth import credentials as auth_credentials
from google.cloud.aiplatform import explain
from google.cloud.aiplatform.compat.types import artifact as gca_artifact
from google.cloud.aiplatform.metadata import _models
from google.cloud.aiplatform.metadata.schema import base_artifact
from google.cloud.aiplatform.metadata.schema import utils
from google.cloud.aiplatform.models import Model

# The artifact property key for the resource_name
_ARTIFACT_PROPERTY_KEY_RESOURCE_NAME = "resourceName"

_CLASSIFICATION_METRICS_AGGREGATION_TYPE = [
    "AGGREGATION_TYPE_UNSPECIFIED",
    "MACRO_AVERAGE",
    "MICRO_AVERAGE",
]


class VertexDataset(base_artifact.BaseArtifactSchema):
    """An artifact representing a Vertex Dataset."""

    schema_title = "google.VertexDataset"

    def __init__(
        self,
        *,
        vertex_dataset_name: str,
        artifact_id: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        state: Optional[gca_artifact.Artifact.State] = gca_artifact.Artifact.State.LIVE,
    ):
        """Args:
        vertex_dataset_name (str):
            The name of the Dataset resource, in a form of
            projects/{project}/locations/{location}/datasets/{dataset}. For
            more details, see
            https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.datasets/get
            This is used to generate the resource uri as follows:
            https://{service-endpoint}/v1/{dataset_name},
            where {service-endpoint} is one of the supported service endpoints at
            https://cloud.google.com/vertex-ai/docs/reference/rest#rest_endpoints
        artifact_id (str):
            Optional. The <resource_id> portion of the Artifact name with
            the format. This is globally unique in a metadataStore:
            projects/123/locations/us-central1/metadataStores/<metadata_store_id>/artifacts/<resource_id>.
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
        extended_metadata = copy.deepcopy(metadata) if metadata else {}
        extended_metadata[_ARTIFACT_PROPERTY_KEY_RESOURCE_NAME] = vertex_dataset_name

        super(VertexDataset, self).__init__(
            uri=utils.create_uri_from_resource_name(resource_name=vertex_dataset_name),
            artifact_id=artifact_id,
            display_name=display_name,
            schema_version=schema_version,
            description=description,
            metadata=extended_metadata,
            state=state,
        )


class VertexModel(base_artifact.BaseArtifactSchema):
    """An artifact representing a Vertex Model."""

    schema_title = "google.VertexModel"

    def __init__(
        self,
        *,
        vertex_model_name: str,
        artifact_id: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        state: Optional[gca_artifact.Artifact.State] = gca_artifact.Artifact.State.LIVE,
    ):
        """Args:
        vertex_model_name (str):
            The name of the Model resource, in a form of
            projects/{project}/locations/{location}/models/{model}. For
            more details, see
            https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.models/get
            This is used to generate the resource uri as follows:
            https://{service-endpoint}/v1/{vertex_model_name},
            where {service-endpoint} is one of the supported service endpoints at
            https://cloud.google.com/vertex-ai/docs/reference/rest#rest_endpoints
        artifact_id (str):
            Optional. The <resource_id> portion of the Artifact name with
            the format. This is globally unique in a metadataStore:
            projects/123/locations/us-central1/metadataStores/<metadata_store_id>/artifacts/<resource_id>.
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
        extended_metadata = copy.deepcopy(metadata) if metadata else {}
        extended_metadata[_ARTIFACT_PROPERTY_KEY_RESOURCE_NAME] = vertex_model_name

        super(VertexModel, self).__init__(
            uri=utils.create_uri_from_resource_name(resource_name=vertex_model_name),
            artifact_id=artifact_id,
            display_name=display_name,
            schema_version=schema_version,
            description=description,
            metadata=extended_metadata,
            state=state,
        )


class VertexEndpoint(base_artifact.BaseArtifactSchema):
    """An artifact representing a Vertex Endpoint."""

    schema_title = "google.VertexEndpoint"

    def __init__(
        self,
        *,
        vertex_endpoint_name: str,
        artifact_id: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        state: Optional[gca_artifact.Artifact.State] = gca_artifact.Artifact.State.LIVE,
    ):
        """Args:
        vertex_endpoint_name (str):
            The name of the Endpoint resource, in a form of
            projects/{project}/locations/{location}/endpoints/{endpoint}. For
            more details, see
            https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.endpoints/get
            This is used to generate the resource uri as follows:
            https://{service-endpoint}/v1/{vertex_endpoint_name},
            where {service-endpoint} is one of the supported service endpoints at
            https://cloud.google.com/vertex-ai/docs/reference/rest#rest_endpoints
        artifact_id (str):
            Optional. The <resource_id> portion of the Artifact name with
            the format. This is globally unique in a metadataStore:
            projects/123/locations/us-central1/metadataStores/<metadata_store_id>/artifacts/<resource_id>.
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
        extended_metadata = copy.deepcopy(metadata) if metadata else {}
        extended_metadata[_ARTIFACT_PROPERTY_KEY_RESOURCE_NAME] = vertex_endpoint_name

        super(VertexEndpoint, self).__init__(
            uri=utils.create_uri_from_resource_name(resource_name=vertex_endpoint_name),
            artifact_id=artifact_id,
            display_name=display_name,
            schema_version=schema_version,
            description=description,
            metadata=extended_metadata,
            state=state,
        )


class UnmanagedContainerModel(base_artifact.BaseArtifactSchema):
    """An artifact representing a Vertex Unmanaged Container Model."""

    schema_title = "google.UnmanagedContainerModel"

    def __init__(
        self,
        *,
        predict_schemata: utils.PredictSchemata,
        container_spec: utils.ContainerSpec,
        artifact_id: Optional[str] = None,
        uri: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        state: Optional[gca_artifact.Artifact.State] = gca_artifact.Artifact.State.LIVE,
    ):
        """Args:
        predict_schemata (PredictSchemata):
            An instance of PredictSchemata which holds instance, parameter and prediction schema uris.
        container_spec (ContainerSpec):
            An instance of ContainerSpec which holds the container configuration for the model.
        artifact_id (str):
            Optional. The <resource_id> portion of the Artifact name with
            the format. This is globally unique in a metadataStore:
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
        extended_metadata = copy.deepcopy(metadata) if metadata else {}
        extended_metadata["predictSchemata"] = predict_schemata.to_dict()
        extended_metadata["containerSpec"] = container_spec.to_dict()

        super(UnmanagedContainerModel, self).__init__(
            uri=uri,
            artifact_id=artifact_id,
            display_name=display_name,
            schema_version=schema_version,
            description=description,
            metadata=extended_metadata,
            state=state,
        )


class ClassificationMetrics(base_artifact.BaseArtifactSchema):
    """A Google artifact representing evaluation Classification Metrics."""

    schema_title = "google.ClassificationMetrics"

    def __init__(
        self,
        *,
        aggregation_type: Optional[str] = None,
        aggregation_threshold: Optional[float] = None,
        recall: Optional[float] = None,
        precision: Optional[float] = None,
        f1_score: Optional[float] = None,
        accuracy: Optional[float] = None,
        au_prc: Optional[float] = None,
        au_roc: Optional[float] = None,
        log_loss: Optional[float] = None,
        confusion_matrix: Optional[utils.ConfusionMatrix] = None,
        confidence_metrics: Optional[List[utils.ConfidenceMetric]] = None,
        artifact_id: Optional[str] = None,
        uri: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        state: Optional[gca_artifact.Artifact.State] = gca_artifact.Artifact.State.LIVE,
    ):
        """Args:
        aggregation_type (str):
            Optional. The way to generate the aggregated metrics. Choose from the following options:
            "AGGREGATION_TYPE_UNSPECIFIED": Indicating unset, used for per-class sliced metrics
            "MACRO_AVERAGE": The unweighted average, default behavior
            "MICRO_AVERAGE": The weighted average
        aggregation_threshold (float):
            Optional. The threshold used to generate aggregated metrics, default 0 for multi-class classification, 0.5 for binary classification.
        recall (float):
            Optional. Recall (True Positive Rate) for the given confidence threshold.
        precision (float):
            Optional. Precision for the given confidence threshold.
        f1_score (float):
            Optional. The harmonic mean of recall and precision.
        accuracy (float):
            Optional. Accuracy is the fraction of predictions given the correct label.
            For multiclass this is a micro-average metric.
        au_prc (float):
            Optional. The Area Under Precision-Recall Curve metric.
            Micro-averaged for the overall evaluation.
        au_roc (float):
            Optional. The Area Under Receiver Operating Characteristic curve metric.
            Micro-averaged for the overall evaluation.
        log_loss (float):
            Optional. The Log Loss metric.
        confusion_matrix (utils.ConfusionMatrix):
            Optional. Aggregated confusion matrix.
        confidence_metrics (List[utils.ConfidenceMetric]):
            Optional. List of metrics for different confidence thresholds.
        artifact_id (str):
            Optional. The <resource_id> portion of the Artifact name with
            the format. This is globally unique in a metadataStore:
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
        extended_metadata = copy.deepcopy(metadata) if metadata else {}
        if aggregation_type:
            if aggregation_type not in _CLASSIFICATION_METRICS_AGGREGATION_TYPE:
                raise ValueError(
                    "aggregation_type can only be 'AGGREGATION_TYPE_UNSPECIFIED', 'MACRO_AVERAGE', or 'MICRO_AVERAGE'."
                )
            extended_metadata["aggregationType"] = aggregation_type
        if aggregation_threshold is not None:
            extended_metadata["aggregationThreshold"] = aggregation_threshold
        if recall is not None:
            extended_metadata["recall"] = recall
        if precision is not None:
            extended_metadata["precision"] = precision
        if f1_score is not None:
            extended_metadata["f1Score"] = f1_score
        if accuracy is not None:
            extended_metadata["accuracy"] = accuracy
        if au_prc is not None:
            extended_metadata["auPrc"] = au_prc
        if au_roc is not None:
            extended_metadata["auRoc"] = au_roc
        if log_loss is not None:
            extended_metadata["logLoss"] = log_loss
        if confusion_matrix:
            extended_metadata["confusionMatrix"] = confusion_matrix.to_dict()
        if confidence_metrics:
            extended_metadata["confidenceMetrics"] = [
                confidence_metric.to_dict() for confidence_metric in confidence_metrics
            ]

        super(ClassificationMetrics, self).__init__(
            uri=uri,
            artifact_id=artifact_id,
            display_name=display_name,
            schema_version=schema_version,
            description=description,
            metadata=extended_metadata,
            state=state,
        )


class RegressionMetrics(base_artifact.BaseArtifactSchema):
    """A Google artifact representing evaluation Regression Metrics."""

    schema_title = "google.RegressionMetrics"

    def __init__(
        self,
        *,
        root_mean_squared_error: Optional[float] = None,
        mean_absolute_error: Optional[float] = None,
        mean_absolute_percentage_error: Optional[float] = None,
        r_squared: Optional[float] = None,
        root_mean_squared_log_error: Optional[float] = None,
        artifact_id: Optional[str] = None,
        uri: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        state: Optional[gca_artifact.Artifact.State] = gca_artifact.Artifact.State.LIVE,
    ):
        """Args:
        root_mean_squared_error (float):
            Optional. Root Mean Squared Error (RMSE).
        mean_absolute_error (float):
            Optional. Mean Absolute Error (MAE).
        mean_absolute_percentage_error (float):
            Optional. Mean absolute percentage error.
        r_squared (float):
            Optional. Coefficient of determination as Pearson correlation coefficient.
        root_mean_squared_log_error (float):
            Optional. Root mean squared log error.
        artifact_id (str):
            Optional. The <resource_id> portion of the Artifact name with
            the format. This is globally unique in a metadataStore:
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
        extended_metadata = copy.deepcopy(metadata) if metadata else {}
        if root_mean_squared_error:
            extended_metadata["rootMeanSquaredError"] = root_mean_squared_error
        if mean_absolute_error:
            extended_metadata["meanAbsoluteError"] = mean_absolute_error
        if mean_absolute_percentage_error:
            extended_metadata[
                "meanAbsolutePercentageError"
            ] = mean_absolute_percentage_error
        if r_squared:
            extended_metadata["rSquared"] = r_squared
        if root_mean_squared_log_error:
            extended_metadata["rootMeanSquaredLogError"] = root_mean_squared_log_error

        super(RegressionMetrics, self).__init__(
            uri=uri,
            artifact_id=artifact_id,
            display_name=display_name,
            schema_version=schema_version,
            description=description,
            metadata=extended_metadata,
            state=state,
        )


class ForecastingMetrics(base_artifact.BaseArtifactSchema):
    """A Google artifact representing evaluation Forecasting Metrics."""

    schema_title = "google.ForecastingMetrics"

    def __init__(
        self,
        *,
        root_mean_squared_error: Optional[float] = None,
        mean_absolute_error: Optional[float] = None,
        mean_absolute_percentage_error: Optional[float] = None,
        r_squared: Optional[float] = None,
        root_mean_squared_log_error: Optional[float] = None,
        weighted_absolute_percentage_error: Optional[float] = None,
        root_mean_squared_percentage_error: Optional[float] = None,
        symmetric_mean_absolute_percentage_error: Optional[float] = None,
        artifact_id: Optional[str] = None,
        uri: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        state: Optional[gca_artifact.Artifact.State] = gca_artifact.Artifact.State.LIVE,
    ):
        """Args:
        root_mean_squared_error (float):
            Optional. Root Mean Squared Error (RMSE).
        mean_absolute_error (float):
            Optional. Mean Absolute Error (MAE).
        mean_absolute_percentage_error (float):
            Optional. Mean absolute percentage error.
        r_squared (float):
            Optional. Coefficient of determination as Pearson correlation coefficient.
        root_mean_squared_log_error (float):
            Optional. Root mean squared log error.
        weighted_absolute_percentage_error (float):
            Optional. Weighted Absolute Percentage Error.
            Does not use weights, this is just what the metric is called.
            Undefined if actual values sum to zero.
            Will be very large if actual values sum to a very small number.
        root_mean_squared_percentage_error (float):
            Optional. Root Mean Square Percentage Error. Square root of MSPE.
            Undefined/imaginary when MSPE is negative.
        symmetric_mean_absolute_percentage_error (float):
            Optional. Symmetric Mean Absolute Percentage Error.
        artifact_id (str):
            Optional. The <resource_id> portion of the Artifact name with
            the format. This is globally unique in a metadataStore:
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
        extended_metadata = copy.deepcopy(metadata) if metadata else {}
        if root_mean_squared_error:
            extended_metadata["rootMeanSquaredError"] = root_mean_squared_error
        if mean_absolute_error:
            extended_metadata["meanAbsoluteError"] = mean_absolute_error
        if mean_absolute_percentage_error:
            extended_metadata[
                "meanAbsolutePercentageError"
            ] = mean_absolute_percentage_error
        if r_squared:
            extended_metadata["rSquared"] = r_squared
        if root_mean_squared_log_error:
            extended_metadata["rootMeanSquaredLogError"] = root_mean_squared_log_error
        if weighted_absolute_percentage_error:
            extended_metadata[
                "weightedAbsolutePercentageError"
            ] = weighted_absolute_percentage_error
        if root_mean_squared_percentage_error:
            extended_metadata[
                "rootMeanSquaredPercentageError"
            ] = root_mean_squared_percentage_error
        if symmetric_mean_absolute_percentage_error:
            extended_metadata[
                "symmetricMeanAbsolutePercentageError"
            ] = symmetric_mean_absolute_percentage_error

        super(ForecastingMetrics, self).__init__(
            uri=uri,
            artifact_id=artifact_id,
            display_name=display_name,
            schema_version=schema_version,
            description=description,
            metadata=extended_metadata,
            state=state,
        )


class ExperimentModel(base_artifact.BaseArtifactSchema):
    """An artifact representing a Vertex Experiment Model."""

    schema_title = "google.ExperimentModel"

    RESERVED_METADATA_KEYS = [
        "frameworkName",
        "frameworkVersion",
        "modelFile",
        "modelClass",
        "predictSchemata",
    ]

    def __init__(
        self,
        *,
        framework_name: str,
        framework_version: str,
        model_file: str,
        uri: str,
        model_class: Optional[str] = None,
        predict_schemata: Optional[utils.PredictSchemata] = None,
        artifact_id: Optional[str] = None,
        display_name: Optional[str] = None,
        schema_version: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        state: Optional[gca_artifact.Artifact.State] = gca_artifact.Artifact.State.LIVE,
    ):
        """Instantiates an ExperimentModel that represents a saved ML model.

        Args:
            framework_name (str):
                Required. The name of the model's framework. E.g., 'sklearn'
            framework_version (str):
                Required. The version of the model's framework. E.g., '1.1.0'
            model_file (str):
                Required. The file name of the model. E.g., 'model.pkl'
            uri (str):
                Required. The uniform resource identifier of the model artifact directory.
            model_class (str):
                Optional. The class name of the model. E.g., 'sklearn.linear_model._base.LinearRegression'
            predict_schemata (PredictSchemata):
                Optional. An instance of PredictSchemata which holds instance, parameter and prediction schema uris.
            artifact_id (str):
                Optional. The <resource_id> portion of the Artifact name with
                the format. This is globally unique in a metadataStore:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/artifacts/<resource_id>.
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
                apture any ongoing process. This property is
                managed by clients (such as Vertex AI
                Pipelines), and the system does not prescribe or
                check the validity of state transitions.
        """
        if metadata:
            for k in metadata:
                if k in self.RESERVED_METADATA_KEYS:
                    raise ValueError(f"'{k}' is a system reserved key in metadata.")
            extended_metadata = copy.deepcopy(metadata)
        else:
            extended_metadata = {}
        extended_metadata["frameworkName"] = framework_name
        extended_metadata["frameworkVersion"] = framework_version
        extended_metadata["modelFile"] = model_file
        if model_class is not None:
            extended_metadata["modelClass"] = model_class
        if predict_schemata is not None:
            extended_metadata["predictSchemata"] = predict_schemata.to_dict()

        super().__init__(
            uri=uri,
            artifact_id=artifact_id,
            display_name=display_name,
            schema_version=schema_version,
            description=description,
            metadata=extended_metadata,
            state=state,
        )

    @classmethod
    def get(
        cls,
        artifact_id: str,
        *,
        metadata_store_id: str = "default",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "ExperimentModel":
        """Retrieves an existing ExperimentModel artifact given an artifact id.

        Args:
            artifact_id (str):
                Required. An artifact id of the ExperimentModel artifact.
            metadata_store_id (str):
                Optional. MetadataStore to retrieve Artifact from. If not set, metadata_store_id is set to "default".
                If artifact_id is a fully-qualified resource name, its metadata_store_id overrides this one.
            project (str):
                Optional. Project to retrieve the artifact from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve the Artifact from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve this Artifact. Overrides
                credentials set in aiplatform.init.

        Returns:
            An ExperimentModel class that represents an Artifact resource.

        Raises:
            ValueError: if artifact's schema title is not 'google.ExperimentModel'.
        """
        experiment_model = ExperimentModel(
            framework_name="",
            framework_version="",
            model_file="",
            uri="",
        )
        experiment_model._init_with_resource_name(
            artifact_name=artifact_id,
            metadata_store_id=metadata_store_id,
            project=project,
            location=location,
            credentials=credentials,
        )
        if experiment_model.schema_title != cls.schema_title:
            raise ValueError(
                f"The schema title of the artifact must be {cls.schema_title}."
                f"Got {experiment_model.schema_title}."
            )
        return experiment_model

    @property
    def framework_name(self) -> Optional[str]:
        """The framework name of the saved ML model."""
        return self.metadata.get("frameworkName")

    @property
    def framework_version(self) -> Optional[str]:
        """The framework version of the saved ML model."""
        return self.metadata.get("frameworkVersion")

    @property
    def model_class(self) -> Optional[str]:
        "The class name of the saved ML model."
        return self.metadata.get("modelClass")

    def get_model_info(self) -> Dict[str, Any]:
        """Get the model's info from an experiment model artifact.

        Returns:
            A dict of model's info. This includes model's class name, framework name,
            framework version, and input example.
        """
        return _models.get_experiment_model_info(self)

    def load_model(
        self,
    ) -> Union["sklearn.base.BaseEstimator", "xgb.Booster", "tf.Module"]:  # noqa: F821
        """Retrieves the original ML model from an ExperimentModel.

        Example Usage:
        ```
        experiment_model = aiplatform.get_experiment_model("my-sklearn-model")
        sk_model = experiment_model.load_model()
        pred_y = model.predict(test_X)
        ```

        Returns:
            The original ML model.

        Raises:
            ValueError: if model type is not supported.
        """
        return _models.load_model(self)

    def register_model(
        self,
        *,
        model_id: Optional[str] = None,
        parent_model: Optional[str] = None,
        use_gpu: bool = False,
        is_default_version: bool = True,
        version_aliases: Optional[Sequence[str]] = None,
        version_description: Optional[str] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        serving_container_image_uri: Optional[str] = None,
        serving_container_predict_route: Optional[str] = None,
        serving_container_health_route: Optional[str] = None,
        serving_container_command: Optional[Sequence[str]] = None,
        serving_container_args: Optional[Sequence[str]] = None,
        serving_container_environment_variables: Optional[Dict[str, str]] = None,
        serving_container_ports: Optional[Sequence[int]] = None,
        instance_schema_uri: Optional[str] = None,
        parameters_schema_uri: Optional[str] = None,
        prediction_schema_uri: Optional[str] = None,
        explanation_metadata: Optional[explain.ExplanationMetadata] = None,
        explanation_parameters: Optional[explain.ExplanationParameters] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        encryption_spec_key_name: Optional[str] = None,
        staging_bucket: Optional[str] = None,
        sync: Optional[bool] = True,
        upload_request_timeout: Optional[float] = None,
    ) -> Model:
        """Register an ExperimentModel to Model Registry and returns a Model representing the registered Model resource.

        Example Usage:
        ```
        experiment_model = aiplatform.get_experiment_model("my-sklearn-model")
        registered_model = experiment_model.register_model()
        registered_model.deploy(endpoint=my_endpoint)
        ```

        Args:
            model_id (str):
                Optional. The ID to use for the registered Model, which will
                become the final component of the model resource name.
                This value may be up to 63 characters, and valid characters
                are `[a-z0-9_-]`. The first character cannot be a number or hyphen.
            parent_model (str):
                Optional. The resource name or model ID of an existing model that the
                newly-registered model will be a version of.
                Only set this field when uploading a new version of an existing model.
            use_gpu (str):
                Optional. Whether or not to use GPUs for the serving container. Only
                specify this argument when registering a Tensorflow model and
                'serving_container_image_uri' is not specified.
            is_default_version (bool):
                Optional. When set to True, the newly registered model version will
                automatically have alias "default" included. Subsequent uses of
                this model without a version specified will use this "default" version.

                When set to False, the "default" alias will not be moved.
                Actions targeting the newly-registered model version will need
                to specifically reference this version by ID or alias.

                New model uploads, i.e. version 1, will always be "default" aliased.
            version_aliases (Sequence[str]):
                Optional. User provided version aliases so that a model version
                can be referenced via alias instead of auto-generated version ID.
                A default version alias will be created for the first version of the model.

                The format is [a-z][a-zA-Z0-9-]{0,126}[a-z0-9]
            version_description (str):
                Optional. The description of the model version being uploaded.
            display_name (str):
                Optional. The display name of the Model. The name can be up to 128
                characters long and can be consist of any UTF-8 characters.
            description (str):
                Optional. The description of the model.
            labels (Dict[str, str]):
                Optional. The labels with user-defined metadata to
                organize your Models.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                and examples of labels.
            serving_container_image_uri (str):
                Optional. The URI of the Model serving container. A pre-built container
                <https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers>
                is automatically chosen based on the model's framwork. Set this field to
                override the default pre-built container.
            serving_container_predict_route (str):
                Optional. An HTTP path to send prediction requests to the container, and
                which must be supported by it. If not specified a default HTTP path will
                be used by Vertex AI.
            serving_container_health_route (str):
                Optional. An HTTP path to send health check requests to the container, and which
                must be supported by it. If not specified a standard HTTP path will be
                used by Vertex AI.
            serving_container_command (Sequence[str]):
                Optional. The command with which the container is run. Not executed within a
                shell. The Docker image's ENTRYPOINT is used if this is not provided.
                Variable references $(VAR_NAME) are expanded using the container's
                environment. If a variable cannot be resolved, the reference in the
                input string will be unchanged. The $(VAR_NAME) syntax can be escaped
                with a double $$, ie: $$(VAR_NAME). Escaped references will never be
                expanded, regardless of whether the variable exists or not.
            serving_container_args (Sequence[str]):
                Optional. The arguments to the command. The Docker image's CMD is used if this is
                not provided. Variable references $(VAR_NAME) are expanded using the
                container's environment. If a variable cannot be resolved, the reference
                in the input string will be unchanged. The $(VAR_NAME) syntax can be
                escaped with a double $$, ie: $$(VAR_NAME). Escaped references will
                never be expanded, regardless of whether the variable exists or not.
            serving_container_environment_variables (Dict[str, str]):
                Optional. The environment variables that are to be present in the container.
                Should be a dictionary where keys are environment variable names
                and values are environment variable values for those names.
            serving_container_ports (Sequence[int]):
                Optional. Declaration of ports that are exposed by the container. This field is
                primarily informational, it gives Vertex AI information about the
                network connections the container uses. Listing or not a port here has
                no impact on whether the port is actually exposed, any port listening on
                the default "0.0.0.0" address inside a container will be accessible from
                the network.
            instance_schema_uri (str):
                Optional. Points to a YAML file stored on Google Cloud
                Storage describing the format of a single instance, which
                are used in
                ``PredictRequest.instances``,
                ``ExplainRequest.instances``
                and
                ``BatchPredictionJob.input_config``.
                The schema is defined as an OpenAPI 3.0.2 `Schema
                Object <https://tinyurl.com/y538mdwt#schema-object>`__.
                AutoML Models always have this field populated by AI
                Platform. Note: The URI given on output will be immutable
                and probably different, including the URI scheme, than the
                one given on input. The output URI will point to a location
                where the user only has a read access.
            parameters_schema_uri (str):
                Optional. Points to a YAML file stored on Google Cloud
                Storage describing the parameters of prediction and
                explanation via
                ``PredictRequest.parameters``,
                ``ExplainRequest.parameters``
                and
                ``BatchPredictionJob.model_parameters``.
                The schema is defined as an OpenAPI 3.0.2 `Schema
                Object <https://tinyurl.com/y538mdwt#schema-object>`__.
                AutoML Models always have this field populated by AI
                Platform, if no parameters are supported it is set to an
                empty string. Note: The URI given on output will be
                immutable and probably different, including the URI scheme,
                than the one given on input. The output URI will point to a
                location where the user only has a read access.
            prediction_schema_uri (str):
                Optional. Points to a YAML file stored on Google Cloud
                Storage describing the format of a single prediction
                produced by this Model, which are returned via
                ``PredictResponse.predictions``,
                ``ExplainResponse.explanations``,
                and
                ``BatchPredictionJob.output_config``.
                The schema is defined as an OpenAPI 3.0.2 `Schema
                Object <https://tinyurl.com/y538mdwt#schema-object>`__.
                AutoML Models always have this field populated by AI
                Platform. Note: The URI given on output will be immutable
                and probably different, including the URI scheme, than the
                one given on input. The output URI will point to a location
                where the user only has a read access.
            explanation_metadata (aiplatform.explain.ExplanationMetadata):
                Optional. Metadata describing the Model's input and output for explanation.
                `explanation_metadata` is optional while `explanation_parameters` must be
                specified when used.
                For more details, see `Ref docs <http://tinyurl.com/1igh60kt>`
            explanation_parameters (aiplatform.explain.ExplanationParameters):
                Optional. Parameters to configure explaining for Model's predictions.
                For more details, see `Ref docs <http://tinyurl.com/1an4zake>`
            project: Optional[str]=None,
                Project to upload this model to. Overrides project set in
                aiplatform.init.
            location: Optional[str]=None,
                Location to upload this model to. Overrides location set in
                aiplatform.init.
            credentials: Optional[auth_credentials.Credentials]=None,
                Custom credentials to use to upload this model. Overrides credentials
                set in aiplatform.init.
            encryption_spec_key_name (Optional[str]):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the model. Has the
                form
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this Model and all sub-resources of this Model will be secured by this key.

                Overrides encryption_spec_key_name set in aiplatform.init.
            staging_bucket (str):
                Optional. Bucket to stage local model artifacts. Overrides
                staging_bucket set in aiplatform.init.
            sync (bool):
                Optional. Whether to execute this method synchronously. If False,
                this method will unblock and it will be executed in a concurrent Future.
            upload_request_timeout (float):
                Optional. The timeout for the upload request in seconds.

        Returns:
            model (aiplatform.Model):
                Instantiated representation of the registered model resource.

        Raises:
            ValueError: If the model doesn't have a pre-built container that is
                        suitable for its framework and 'serving_container_image_uri'
                        is not set.
        """
        return _models.register_model(
            model=self,
            model_id=model_id,
            parent_model=parent_model,
            use_gpu=use_gpu,
            is_default_version=is_default_version,
            version_aliases=version_aliases,
            version_description=version_description,
            display_name=display_name,
            description=description,
            labels=labels,
            serving_container_image_uri=serving_container_image_uri,
            serving_container_predict_route=serving_container_predict_route,
            serving_container_health_route=serving_container_health_route,
            serving_container_command=serving_container_command,
            serving_container_args=serving_container_args,
            serving_container_environment_variables=serving_container_environment_variables,
            serving_container_ports=serving_container_ports,
            instance_schema_uri=instance_schema_uri,
            parameters_schema_uri=parameters_schema_uri,
            prediction_schema_uri=prediction_schema_uri,
            explanation_metadata=explanation_metadata,
            explanation_parameters=explanation_parameters,
            project=project,
            location=location,
            credentials=credentials,
            encryption_spec_key_name=encryption_spec_key_name,
            staging_bucket=staging_bucket,
            sync=sync,
            upload_request_timeout=upload_request_timeout,
        )
