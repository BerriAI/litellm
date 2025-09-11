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
from __future__ import annotations

from typing import MutableMapping, MutableSequence

import proto  # type: ignore

from google.cloud.aiplatform_v1.types import encryption_spec as gca_encryption_spec
from google.cloud.aiplatform_v1.types import explanation
from google.cloud.aiplatform_v1.types import io
from google.cloud.aiplatform_v1.types import machine_resources
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "Endpoint",
        "DeployedModel",
        "PrivateEndpoints",
        "PredictRequestResponseLoggingConfig",
    },
)


class Endpoint(proto.Message):
    r"""Models are deployed into it, and afterwards Endpoint is
    called to obtain predictions and explanations.

    Attributes:
        name (str):
            Output only. The resource name of the
            Endpoint.
        display_name (str):
            Required. The display name of the Endpoint.
            The name can be up to 128 characters long and
            can consist of any UTF-8 characters.
        description (str):
            The description of the Endpoint.
        deployed_models (MutableSequence[google.cloud.aiplatform_v1.types.DeployedModel]):
            Output only. The models deployed in this Endpoint. To add or
            remove DeployedModels use
            [EndpointService.DeployModel][google.cloud.aiplatform.v1.EndpointService.DeployModel]
            and
            [EndpointService.UndeployModel][google.cloud.aiplatform.v1.EndpointService.UndeployModel]
            respectively.
        traffic_split (MutableMapping[str, int]):
            A map from a DeployedModel's ID to the
            percentage of this Endpoint's traffic that
            should be forwarded to that DeployedModel.

            If a DeployedModel's ID is not listed in this
            map, then it receives no traffic.

            The traffic percentage values must add up to
            100, or map must be empty if the Endpoint is to
            not accept any traffic at a moment.
        etag (str):
            Used to perform consistent read-modify-write
            updates. If not set, a blind "overwrite" update
            happens.
        labels (MutableMapping[str, str]):
            The labels with user-defined metadata to
            organize your Endpoints.
            Label keys and values can be no longer than 64
            characters (Unicode codepoints), can only
            contain lowercase letters, numeric characters,
            underscores and dashes. International characters
            are allowed.

            See https://goo.gl/xmQnxf for more information
            and examples of labels.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Endpoint was
            created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Endpoint was
            last updated.
        encryption_spec (google.cloud.aiplatform_v1.types.EncryptionSpec):
            Customer-managed encryption key spec for an
            Endpoint. If set, this Endpoint and all
            sub-resources of this Endpoint will be secured
            by this key.
        network (str):
            Optional. The full name of the Google Compute Engine
            `network <https://cloud.google.com//compute/docs/networks-and-firewalls#networks>`__
            to which the Endpoint should be peered.

            Private services access must already be configured for the
            network. If left unspecified, the Endpoint is not peered
            with any network.

            Only one of the fields,
            [network][google.cloud.aiplatform.v1.Endpoint.network] or
            [enable_private_service_connect][google.cloud.aiplatform.v1.Endpoint.enable_private_service_connect],
            can be set.

            `Format <https://cloud.google.com/compute/docs/reference/rest/v1/networks/insert>`__:
            ``projects/{project}/global/networks/{network}``. Where
            ``{project}`` is a project number, as in ``12345``, and
            ``{network}`` is network name.
        enable_private_service_connect (bool):
            Deprecated: If true, expose the Endpoint via private service
            connect.

            Only one of the fields,
            [network][google.cloud.aiplatform.v1.Endpoint.network] or
            [enable_private_service_connect][google.cloud.aiplatform.v1.Endpoint.enable_private_service_connect],
            can be set.
        model_deployment_monitoring_job (str):
            Output only. Resource name of the Model Monitoring job
            associated with this Endpoint if monitoring is enabled by
            [JobService.CreateModelDeploymentMonitoringJob][google.cloud.aiplatform.v1.JobService.CreateModelDeploymentMonitoringJob].
            Format:
            ``projects/{project}/locations/{location}/modelDeploymentMonitoringJobs/{model_deployment_monitoring_job}``
        predict_request_response_logging_config (google.cloud.aiplatform_v1.types.PredictRequestResponseLoggingConfig):
            Configures the request-response logging for
            online prediction.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    description: str = proto.Field(
        proto.STRING,
        number=3,
    )
    deployed_models: MutableSequence["DeployedModel"] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message="DeployedModel",
    )
    traffic_split: MutableMapping[str, int] = proto.MapField(
        proto.STRING,
        proto.INT32,
        number=5,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=6,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=7,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=8,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=9,
        message=timestamp_pb2.Timestamp,
    )
    encryption_spec: gca_encryption_spec.EncryptionSpec = proto.Field(
        proto.MESSAGE,
        number=10,
        message=gca_encryption_spec.EncryptionSpec,
    )
    network: str = proto.Field(
        proto.STRING,
        number=13,
    )
    enable_private_service_connect: bool = proto.Field(
        proto.BOOL,
        number=17,
    )
    model_deployment_monitoring_job: str = proto.Field(
        proto.STRING,
        number=14,
    )
    predict_request_response_logging_config: "PredictRequestResponseLoggingConfig" = (
        proto.Field(
            proto.MESSAGE,
            number=18,
            message="PredictRequestResponseLoggingConfig",
        )
    )


class DeployedModel(proto.Message):
    r"""A deployment of a Model. Endpoints contain one or more
    DeployedModels.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        dedicated_resources (google.cloud.aiplatform_v1.types.DedicatedResources):
            A description of resources that are dedicated
            to the DeployedModel, and that need a higher
            degree of manual configuration.

            This field is a member of `oneof`_ ``prediction_resources``.
        automatic_resources (google.cloud.aiplatform_v1.types.AutomaticResources):
            A description of resources that to large
            degree are decided by Vertex AI, and require
            only a modest additional configuration.

            This field is a member of `oneof`_ ``prediction_resources``.
        shared_resources (str):
            The resource name of the shared DeploymentResourcePool to
            deploy on. Format:
            ``projects/{project}/locations/{location}/deploymentResourcePools/{deployment_resource_pool}``

            This field is a member of `oneof`_ ``prediction_resources``.
        id (str):
            Immutable. The ID of the DeployedModel. If not provided upon
            deployment, Vertex AI will generate a value for this ID.

            This value should be 1-10 characters, and valid characters
            are ``/[0-9]/``.
        model (str):
            Required. The resource name of the Model that this is the
            deployment of. Note that the Model may be in a different
            location than the DeployedModel's Endpoint.

            The resource name may contain version id or version alias to
            specify the version. Example:
            ``projects/{project}/locations/{location}/models/{model}@2``
            or
            ``projects/{project}/locations/{location}/models/{model}@golden``
            if no version is specified, the default version will be
            deployed.
        model_version_id (str):
            Output only. The version ID of the model that
            is deployed.
        display_name (str):
            The display name of the DeployedModel. If not provided upon
            creation, the Model's display_name is used.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when the DeployedModel
            was created.
        explanation_spec (google.cloud.aiplatform_v1.types.ExplanationSpec):
            Explanation configuration for this DeployedModel.

            When deploying a Model using
            [EndpointService.DeployModel][google.cloud.aiplatform.v1.EndpointService.DeployModel],
            this value overrides the value of
            [Model.explanation_spec][google.cloud.aiplatform.v1.Model.explanation_spec].
            All fields of
            [explanation_spec][google.cloud.aiplatform.v1.DeployedModel.explanation_spec]
            are optional in the request. If a field of
            [explanation_spec][google.cloud.aiplatform.v1.DeployedModel.explanation_spec]
            is not populated, the value of the same field of
            [Model.explanation_spec][google.cloud.aiplatform.v1.Model.explanation_spec]
            is inherited. If the corresponding
            [Model.explanation_spec][google.cloud.aiplatform.v1.Model.explanation_spec]
            is not populated, all fields of the
            [explanation_spec][google.cloud.aiplatform.v1.DeployedModel.explanation_spec]
            will be used for the explanation configuration.
        disable_explanations (bool):
            If true, deploy the model without explainable feature,
            regardless the existence of
            [Model.explanation_spec][google.cloud.aiplatform.v1.Model.explanation_spec]
            or
            [explanation_spec][google.cloud.aiplatform.v1.DeployedModel.explanation_spec].
        service_account (str):
            The service account that the DeployedModel's container runs
            as. Specify the email address of the service account. If
            this service account is not specified, the container runs as
            a service account that doesn't have access to the resource
            project.

            Users deploying the Model must have the
            ``iam.serviceAccounts.actAs`` permission on this service
            account.
        disable_container_logging (bool):
            For custom-trained Models and AutoML Tabular Models, the
            container of the DeployedModel instances will send
            ``stderr`` and ``stdout`` streams to Cloud Logging by
            default. Please note that the logs incur cost, which are
            subject to `Cloud Logging
            pricing <https://cloud.google.com/logging/pricing>`__.

            User can disable container logging by setting this flag to
            true.
        enable_access_logging (bool):
            If true, online prediction access logs are
            sent to Cloud Logging.
            These logs are like standard server access logs,
            containing information like timestamp and
            latency for each prediction request.

            Note that logs may incur a cost, especially if
            your project receives prediction requests at a
            high queries per second rate (QPS). Estimate
            your costs before enabling this option.
        private_endpoints (google.cloud.aiplatform_v1.types.PrivateEndpoints):
            Output only. Provide paths for users to send
            predict/explain/health requests directly to the deployed
            model services running on Cloud via private services access.
            This field is populated if
            [network][google.cloud.aiplatform.v1.Endpoint.network] is
            configured.
    """

    dedicated_resources: machine_resources.DedicatedResources = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="prediction_resources",
        message=machine_resources.DedicatedResources,
    )
    automatic_resources: machine_resources.AutomaticResources = proto.Field(
        proto.MESSAGE,
        number=8,
        oneof="prediction_resources",
        message=machine_resources.AutomaticResources,
    )
    shared_resources: str = proto.Field(
        proto.STRING,
        number=17,
        oneof="prediction_resources",
    )
    id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    model: str = proto.Field(
        proto.STRING,
        number=2,
    )
    model_version_id: str = proto.Field(
        proto.STRING,
        number=18,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=3,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    explanation_spec: explanation.ExplanationSpec = proto.Field(
        proto.MESSAGE,
        number=9,
        message=explanation.ExplanationSpec,
    )
    disable_explanations: bool = proto.Field(
        proto.BOOL,
        number=19,
    )
    service_account: str = proto.Field(
        proto.STRING,
        number=11,
    )
    disable_container_logging: bool = proto.Field(
        proto.BOOL,
        number=15,
    )
    enable_access_logging: bool = proto.Field(
        proto.BOOL,
        number=13,
    )
    private_endpoints: "PrivateEndpoints" = proto.Field(
        proto.MESSAGE,
        number=14,
        message="PrivateEndpoints",
    )


class PrivateEndpoints(proto.Message):
    r"""PrivateEndpoints proto is used to provide paths for users to send
    requests privately. To send request via private service access, use
    predict_http_uri, explain_http_uri or health_http_uri. To send
    request via private service connect, use service_attachment.

    Attributes:
        predict_http_uri (str):
            Output only. Http(s) path to send prediction
            requests.
        explain_http_uri (str):
            Output only. Http(s) path to send explain
            requests.
        health_http_uri (str):
            Output only. Http(s) path to send health
            check requests.
        service_attachment (str):
            Output only. The name of the service
            attachment resource. Populated if private
            service connect is enabled.
    """

    predict_http_uri: str = proto.Field(
        proto.STRING,
        number=1,
    )
    explain_http_uri: str = proto.Field(
        proto.STRING,
        number=2,
    )
    health_http_uri: str = proto.Field(
        proto.STRING,
        number=3,
    )
    service_attachment: str = proto.Field(
        proto.STRING,
        number=4,
    )


class PredictRequestResponseLoggingConfig(proto.Message):
    r"""Configuration for logging request-response to a BigQuery
    table.

    Attributes:
        enabled (bool):
            If logging is enabled or not.
        sampling_rate (float):
            Percentage of requests to be logged, expressed as a fraction
            in range(0,1].
        bigquery_destination (google.cloud.aiplatform_v1.types.BigQueryDestination):
            BigQuery table for logging. If only given a project, a new
            dataset will be created with name
            ``logging_<endpoint-display-name>_<endpoint-id>`` where will
            be made BigQuery-dataset-name compatible (e.g. most special
            characters will become underscores). If no table name is
            given, a new table will be created with name
            ``request_response_logging``
    """

    enabled: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    sampling_rate: float = proto.Field(
        proto.DOUBLE,
        number=2,
    )
    bigquery_destination: io.BigQueryDestination = proto.Field(
        proto.MESSAGE,
        number=3,
        message=io.BigQueryDestination,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
