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
import re

from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class PredictSchemata:
    """A class holding instance, parameter and prediction schema uris.

    Args:
        instance_schema_uri (str):
            Required. Points to a YAML file stored on Google Cloud Storage
            describing the format of a single instance, which are used in
            PredictRequest.instances, ExplainRequest.instances and
            BatchPredictionJob.input_config. The schema is defined as an
            OpenAPI 3.0.2 `Schema Object.
        parameters_schema_uri (str):
            Required. Points to a YAML file stored on Google Cloud Storage
            describing the parameters of prediction and explanation via
            PredictRequest.parameters, ExplainRequest.parameters and
            BatchPredictionJob.model_parameters. The schema is defined as an
            OpenAPI 3.0.2 `Schema Object.
        prediction_schema_uri (str):
            Required. Points to a YAML file stored on Google Cloud Storage
            describing the format of a single prediction produced by this Model
            , which are returned via PredictResponse.predictions,
            ExplainResponse.explanations, and BatchPredictionJob.output_config.
            The schema is defined as an OpenAPI 3.0.2 `Schema Object.
    """

    instance_schema_uri: Optional[str] = None
    parameters_schema_uri: Optional[str] = None
    prediction_schema_uri: Optional[str] = None

    def to_dict(self):
        """ML metadata schema dictionary representation of this DataClass.


        Returns:
            A dictionary that represents the PredictSchemata class.
        """
        results = {}
        if self.instance_schema_uri:
            results["instanceSchemaUri"] = self.instance_schema_uri
        if self.parameters_schema_uri:
            results["parametersSchemaUri"] = self.parameters_schema_uri
        if self.prediction_schema_uri:
            results["predictionSchemaUri"] = self.prediction_schema_uri

        return results


@dataclass
class ContainerSpec:
    """Container configuration for the model.

    Args:
        image_uri (str):
            Required. URI of the Docker image to be used as the custom
            container for serving predictions. This URI must identify an image
            in Artifact Registry or Container Registry.
        command (Sequence[str]):
            Optional. Specifies the command that runs when the container
            starts. This overrides the container's `ENTRYPOINT`.
        args (Sequence[str]):
            Optional. Specifies arguments for the command that runs when the
            container starts. This overrides the container's `CMD`
        env (Sequence[google.cloud.aiplatform_v1.types.EnvVar]):
            Optional. List of environment variables to set in the container.
            After the container starts running, code running in the container
            can read these environment variables. Additionally, the command
            and args fields can reference these variables. Later entries in
            this list can also reference earlier entries. For example, the
            following example sets the variable ``VAR_2`` to have the value
            ``foo bar``: .. code:: json [ { "name": "VAR_1", "value": "foo" },
            { "name": "VAR_2", "value": "$(VAR_1) bar" } ] If you switch the
            order of the variables in the example, then the expansion does not
            occur. This field corresponds to the ``env`` field of the
            Kubernetes Containers `v1 core API.
        ports (Sequence[google.cloud.aiplatform_v1.types.Port]):
            Optional. List of ports to expose from the container. Vertex AI
            sends any prediction requests that it receives to the first port on
            this list. Vertex AI also sends `liveness and health checks.
        predict_route (str):
            Optional. HTTP path on the container to send prediction requests
            to. Vertex AI forwards requests sent using
            projects.locations.endpoints.predict to this path on the
            container's IP address and port. Vertex AI then returns the
            container's response in the API response. For example, if you set
            this field to ``/foo``, then when Vertex AI receives a prediction
            request, it forwards the request body in a POST request to the
            ``/foo`` path on the port of your container specified by the first
            value of this ``ModelContainerSpec``'s ports field. If you don't
            specify this field, it defaults to the following value when you
            deploy this Model to an Endpoint
            /v1/endpoints/ENDPOINT/deployedModels/DEPLOYED_MODEL:predict
            The placeholders in this value are replaced as follows:
            - ENDPOINT: The last segment (following ``endpoints/``)of the
              Endpoint.name][] field of the Endpoint where this Model has
              been deployed. (Vertex AI makes this value available to your
              container code as the ```AIP_ENDPOINT_ID`` environment variable
        health_route (str):
            Optional. HTTP path on the container to send health checks to.
            Vertex AI intermittently sends GET requests to this path on the
            container's IP address and port to check that the container is
            healthy. Read more about `health checks
        display_name (str):
    """

    image_uri: str
    command: Optional[List[str]] = None
    args: Optional[List[str]] = None
    env: Optional[List[Dict[str, str]]] = None
    ports: Optional[List[int]] = None
    predict_route: Optional[str] = None
    health_route: Optional[str] = None

    def to_dict(self):
        """ML metadata schema dictionary representation of this DataClass.


        Returns:
            A dictionary that represents the ContainerSpec class.
        """
        results = {}
        results["imageUri"] = self.image_uri
        if self.command:
            results["command"] = self.command
        if self.args:
            results["args"] = self.args
        if self.env:
            results["env"] = self.env
        if self.ports:
            results["ports"] = self.ports
        if self.predict_route:
            results["predictRoute"] = self.predict_route
        if self.health_route:
            results["healthRoute"] = self.health_route

        return results


@dataclass
class AnnotationSpec:
    """A class that represents the annotation spec of a Confusion Matrix.

    Args:
        display_name (str):
            Optional. Display name for a column of a confusion matrix.
        id (str):
            Optional. Id for a column of a confusion matrix.
    """

    display_name: Optional[str] = None
    id: Optional[str] = None

    def to_dict(self):
        """ML metadata schema dictionary representation of this DataClass.


        Returns:
            A dictionary that represents the AnnotationSpec class.
        """
        results = {}
        if self.display_name:
            results["displayName"] = self.display_name
        if self.id:
            results["id"] = self.id

        return results


@dataclass
class ConfusionMatrix:
    """A class that represents a Confusion Matrix.

    Args:
        matrix (List[List[int]]):
            Required. A 2D array of integers that represets the values for the confusion matrix.
        annotation_specs: (List(AnnotationSpec)):
            Optional. List of column annotation specs which contains display_name (str) and id (str)
    """

    matrix: List[List[int]]
    annotation_specs: Optional[List[AnnotationSpec]] = None

    def to_dict(self):
        """ML metadata schema dictionary representation of this DataClass.

        Returns:
            A dictionary that represents the ConfusionMatrix class.

        Raises:
            ValueError: if annotation_specs and matrix have different length.
        """
        results = {}
        if self.annotation_specs:
            if len(self.annotation_specs) != len(self.matrix):
                raise ValueError(
                    "Length of annotation_specs and matrix must be the same. "
                    "Got lengths {} and {} respectively.".format(
                        len(self.annotation_specs), len(self.matrix)
                    )
                )
            results["annotationSpecs"] = [
                annotation_spec.to_dict() for annotation_spec in self.annotation_specs
            ]
        if self.matrix:
            results["rows"] = self.matrix

        return results


@dataclass
class ConfidenceMetric:
    """A class that represents a Confidence Metric.
    Args:
        confidence_threshold (float):
            Required. Metrics are computed with an assumption that the Model never returns predictions with a score lower than this value.
            For binary classification this is the positive class threshold. For multi-class classification this is the confidence threshold.
        recall (float):
            Optional. Recall (True Positive Rate) for the given confidence threshold.
        precision (float):
            Optional. Precision for the given confidence threshold.
        f1_score (float):
            Optional. The harmonic mean of recall and precision.
        max_predictions (int):
            Optional. Metrics are computed with an assumption that the Model always returns at most this many predictions (ordered by their score, descendingly).
            But they all still need to meet the `confidence_threshold`.
        false_positive_rate (float):
            Optional. False Positive Rate for the given confidence threshold.
        accuracy (float):
            Optional. Accuracy is the fraction of predictions given the correct label. For multiclass this is a micro-average metric.
        true_positive_count (int):
            Optional. The number of Model created labels that match a ground truth label.
        false_positive_count (int):
            Optional. The number of Model created labels that do not match a ground truth label.
        false_negative_count (int):
            Optional. The number of ground truth labels that are not matched by a Model created label.
        true_negative_count (int):
            Optional. The number of labels that were not created by the Model, but if they would, they would not match a ground truth label.
        recall_at_1 (float):
            Optional. The Recall (True Positive Rate) when only considering the label that has the highest prediction score
            and not below the confidence threshold for each DataItem.
        precision_at_1 (float):
            Optional. The precision when only considering the label that has the highest prediction score
            and not below the confidence threshold for each DataItem.
        false_positive_rate_at_1 (float):
            Optional. The False Positive Rate when only considering the label that has the highest prediction score
            and not below the confidence threshold for each DataItem.
        f1_score_at_1 (float):
            Optional. The harmonic mean of recallAt1 and precisionAt1.
        confusion_matrix (ConfusionMatrix):
            Optional. Confusion matrix for the given confidence threshold.
    """

    confidence_threshold: float
    recall: Optional[float] = None
    precision: Optional[float] = None
    f1_score: Optional[float] = None
    max_predictions: Optional[int] = None
    false_positive_rate: Optional[float] = None
    accuracy: Optional[float] = None
    true_positive_count: Optional[int] = None
    false_positive_count: Optional[int] = None
    false_negative_count: Optional[int] = None
    true_negative_count: Optional[int] = None
    recall_at_1: Optional[float] = None
    precision_at_1: Optional[float] = None
    false_positive_rate_at_1: Optional[float] = None
    f1_score_at_1: Optional[float] = None
    confusion_matrix: Optional[ConfusionMatrix] = None

    def to_dict(self):
        """ML metadata schema dictionary representation of this DataClass.


        Returns:
            A dictionary that represents the ConfidenceMetric class.
        """
        results = {}
        results["confidenceThreshold"] = self.confidence_threshold
        if self.recall is not None:
            results["recall"] = self.recall
        if self.precision is not None:
            results["precision"] = self.precision
        if self.f1_score is not None:
            results["f1Score"] = self.f1_score
        if self.max_predictions is not None:
            results["maxPredictions"] = self.max_predictions
        if self.false_positive_rate is not None:
            results["falsePositiveRate"] = self.false_positive_rate
        if self.accuracy is not None:
            results["accuracy"] = self.accuracy
        if self.true_positive_count is not None:
            results["truePositiveCount"] = self.true_positive_count
        if self.false_positive_count is not None:
            results["falsePositiveCount"] = self.false_positive_count
        if self.false_negative_count is not None:
            results["falseNegativeCount"] = self.false_negative_count
        if self.true_negative_count is not None:
            results["trueNegativeCount"] = self.true_negative_count
        if self.recall_at_1 is not None:
            results["recallAt1"] = self.recall_at_1
        if self.precision_at_1 is not None:
            results["precisionAt1"] = self.precision_at_1
        if self.false_positive_rate_at_1 is not None:
            results["falsePositiveRateAt1"] = self.false_positive_rate_at_1
        if self.f1_score_at_1 is not None:
            results["f1ScoreAt1"] = self.f1_score_at_1
        if self.confusion_matrix:
            results["confusionMatrix"] = self.confusion_matrix.to_dict()

        return results


def create_uri_from_resource_name(resource_name: str) -> str:
    """Construct the service URI for a given resource_name.
    Args:
        resource_name (str):
            The name of the Vertex resource, in one of the forms:
            projects/{project}/locations/{location}/{resource_type}/{resource_id}
            projects/{project}/locations/{location}/{resource_type}/{resource_id}@{version}
            projects/{project}/locations/{location}/metadataStores/{store_id}/{resource_type}/{resource_id}
            projects/{project}/locations/{location}/metadataStores/{store_id}/{resource_type}/{resource_id}@{version}
    Returns:
        The resource URI in the form of:
        https://{service-endpoint}/v1/{resource_name},
        where {service-endpoint} is one of the supported service endpoints at
        https://cloud.google.com/vertex-ai/docs/reference/rest#rest_endpoints
    Raises:
        ValueError: If resource_name does not match the specified format.
    """
    # TODO: support nested resource names such as models/123/evaluations/456
    match_results = re.match(
        r"^projects\/(?P<project>[\w-]+)\/locations\/(?P<location>[\w-]+)(\/metadataStores\/(?P<store>[\w-]+))?\/[\w-]+\/(?P<id>[\w-]+)(?P<version>@[\w-]+)?$",
        resource_name,
    )
    if not match_results:
        raise ValueError(f"Invalid resource_name format for {resource_name}.")

    location = match_results["location"]
    return f"https://{location}-aiplatform.googleapis.com/v1/{resource_name}"
