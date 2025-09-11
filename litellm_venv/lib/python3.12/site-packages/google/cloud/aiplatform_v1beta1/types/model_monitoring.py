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

from google.cloud.aiplatform_v1beta1.types import io


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "ModelMonitoringConfig",
        "ModelMonitoringObjectiveConfig",
        "ModelMonitoringAlertConfig",
        "ThresholdConfig",
        "SamplingStrategy",
    },
)


class ModelMonitoringConfig(proto.Message):
    r"""The model monitoring configuration used for Batch Prediction
    Job.

    Attributes:
        objective_configs (MutableSequence[google.cloud.aiplatform_v1beta1.types.ModelMonitoringObjectiveConfig]):
            Model monitoring objective config.
        alert_config (google.cloud.aiplatform_v1beta1.types.ModelMonitoringAlertConfig):
            Model monitoring alert config.
        analysis_instance_schema_uri (str):
            YAML schema file uri in Cloud Storage
            describing the format of a single instance that
            you want Tensorflow Data Validation (TFDV) to
            analyze.

            If there are any data type differences between
            predict instance and TFDV instance, this field
            can be used to override the schema. For models
            trained with Vertex AI, this field must be set
            as all the fields in predict instance formatted
            as string.
        stats_anomalies_base_directory (google.cloud.aiplatform_v1beta1.types.GcsDestination):
            A Google Cloud Storage location for batch
            prediction model monitoring to dump statistics
            and anomalies. If not provided, a folder will be
            created in customer project to hold statistics
            and anomalies.
    """

    objective_configs: MutableSequence[
        "ModelMonitoringObjectiveConfig"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message="ModelMonitoringObjectiveConfig",
    )
    alert_config: "ModelMonitoringAlertConfig" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="ModelMonitoringAlertConfig",
    )
    analysis_instance_schema_uri: str = proto.Field(
        proto.STRING,
        number=4,
    )
    stats_anomalies_base_directory: io.GcsDestination = proto.Field(
        proto.MESSAGE,
        number=5,
        message=io.GcsDestination,
    )


class ModelMonitoringObjectiveConfig(proto.Message):
    r"""The objective configuration for model monitoring, including
    the information needed to detect anomalies for one particular
    model.

    Attributes:
        training_dataset (google.cloud.aiplatform_v1beta1.types.ModelMonitoringObjectiveConfig.TrainingDataset):
            Training dataset for models. This field has
            to be set only if
            TrainingPredictionSkewDetectionConfig is
            specified.
        training_prediction_skew_detection_config (google.cloud.aiplatform_v1beta1.types.ModelMonitoringObjectiveConfig.TrainingPredictionSkewDetectionConfig):
            The config for skew between training data and
            prediction data.
        prediction_drift_detection_config (google.cloud.aiplatform_v1beta1.types.ModelMonitoringObjectiveConfig.PredictionDriftDetectionConfig):
            The config for drift of prediction data.
        explanation_config (google.cloud.aiplatform_v1beta1.types.ModelMonitoringObjectiveConfig.ExplanationConfig):
            The config for integrating with Vertex
            Explainable AI.
    """

    class TrainingDataset(proto.Message):
        r"""Training Dataset information.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            dataset (str):
                The resource name of the Dataset used to
                train this Model.

                This field is a member of `oneof`_ ``data_source``.
            gcs_source (google.cloud.aiplatform_v1beta1.types.GcsSource):
                The Google Cloud Storage uri of the unmanaged
                Dataset used to train this Model.

                This field is a member of `oneof`_ ``data_source``.
            bigquery_source (google.cloud.aiplatform_v1beta1.types.BigQuerySource):
                The BigQuery table of the unmanaged Dataset
                used to train this Model.

                This field is a member of `oneof`_ ``data_source``.
            data_format (str):
                Data format of the dataset, only applicable
                if the input is from Google Cloud Storage.
                The possible formats are:

                "tf-record"
                The source file is a TFRecord file.

                "csv"
                The source file is a CSV file.
                "jsonl"
                The source file is a JSONL file.
            target_field (str):
                The target field name the model is to
                predict. This field will be excluded when doing
                Predict and (or) Explain for the training data.
            logging_sampling_strategy (google.cloud.aiplatform_v1beta1.types.SamplingStrategy):
                Strategy to sample data from Training
                Dataset. If not set, we process the whole
                dataset.
        """

        dataset: str = proto.Field(
            proto.STRING,
            number=3,
            oneof="data_source",
        )
        gcs_source: io.GcsSource = proto.Field(
            proto.MESSAGE,
            number=4,
            oneof="data_source",
            message=io.GcsSource,
        )
        bigquery_source: io.BigQuerySource = proto.Field(
            proto.MESSAGE,
            number=5,
            oneof="data_source",
            message=io.BigQuerySource,
        )
        data_format: str = proto.Field(
            proto.STRING,
            number=2,
        )
        target_field: str = proto.Field(
            proto.STRING,
            number=6,
        )
        logging_sampling_strategy: "SamplingStrategy" = proto.Field(
            proto.MESSAGE,
            number=7,
            message="SamplingStrategy",
        )

    class TrainingPredictionSkewDetectionConfig(proto.Message):
        r"""The config for Training & Prediction data skew detection. It
        specifies the training dataset sources and the skew detection
        parameters.

        Attributes:
            skew_thresholds (MutableMapping[str, google.cloud.aiplatform_v1beta1.types.ThresholdConfig]):
                Key is the feature name and value is the
                threshold. If a feature needs to be monitored
                for skew, a value threshold must be configured
                for that feature. The threshold here is against
                feature distribution distance between the
                training and prediction feature.
            attribution_score_skew_thresholds (MutableMapping[str, google.cloud.aiplatform_v1beta1.types.ThresholdConfig]):
                Key is the feature name and value is the
                threshold. The threshold here is against
                attribution score distance between the training
                and prediction feature.
            default_skew_threshold (google.cloud.aiplatform_v1beta1.types.ThresholdConfig):
                Skew anomaly detection threshold used by all
                features. When the per-feature thresholds are
                not set, this field can be used to specify a
                threshold for all features.
        """

        skew_thresholds: MutableMapping[str, "ThresholdConfig"] = proto.MapField(
            proto.STRING,
            proto.MESSAGE,
            number=1,
            message="ThresholdConfig",
        )
        attribution_score_skew_thresholds: MutableMapping[
            str, "ThresholdConfig"
        ] = proto.MapField(
            proto.STRING,
            proto.MESSAGE,
            number=2,
            message="ThresholdConfig",
        )
        default_skew_threshold: "ThresholdConfig" = proto.Field(
            proto.MESSAGE,
            number=6,
            message="ThresholdConfig",
        )

    class PredictionDriftDetectionConfig(proto.Message):
        r"""The config for Prediction data drift detection.

        Attributes:
            drift_thresholds (MutableMapping[str, google.cloud.aiplatform_v1beta1.types.ThresholdConfig]):
                Key is the feature name and value is the
                threshold. If a feature needs to be monitored
                for drift, a value threshold must be configured
                for that feature. The threshold here is against
                feature distribution distance between different
                time windws.
            attribution_score_drift_thresholds (MutableMapping[str, google.cloud.aiplatform_v1beta1.types.ThresholdConfig]):
                Key is the feature name and value is the
                threshold. The threshold here is against
                attribution score distance between different
                time windows.
            default_drift_threshold (google.cloud.aiplatform_v1beta1.types.ThresholdConfig):
                Drift anomaly detection threshold used by all
                features. When the per-feature thresholds are
                not set, this field can be used to specify a
                threshold for all features.
        """

        drift_thresholds: MutableMapping[str, "ThresholdConfig"] = proto.MapField(
            proto.STRING,
            proto.MESSAGE,
            number=1,
            message="ThresholdConfig",
        )
        attribution_score_drift_thresholds: MutableMapping[
            str, "ThresholdConfig"
        ] = proto.MapField(
            proto.STRING,
            proto.MESSAGE,
            number=2,
            message="ThresholdConfig",
        )
        default_drift_threshold: "ThresholdConfig" = proto.Field(
            proto.MESSAGE,
            number=5,
            message="ThresholdConfig",
        )

    class ExplanationConfig(proto.Message):
        r"""The config for integrating with Vertex Explainable AI. Only
        applicable if the Model has explanation_spec populated.

        Attributes:
            enable_feature_attributes (bool):
                If want to analyze the Vertex Explainable AI
                feature attribute scores or not. If set to true,
                Vertex AI will log the feature attributions from
                explain response and do the skew/drift detection
                for them.
            explanation_baseline (google.cloud.aiplatform_v1beta1.types.ModelMonitoringObjectiveConfig.ExplanationConfig.ExplanationBaseline):
                Predictions generated by the
                BatchPredictionJob using baseline dataset.
        """

        class ExplanationBaseline(proto.Message):
            r"""Output from
            [BatchPredictionJob][google.cloud.aiplatform.v1beta1.BatchPredictionJob]
            for Model Monitoring baseline dataset, which can be used to generate
            baseline attribution scores.

            This message has `oneof`_ fields (mutually exclusive fields).
            For each oneof, at most one member field can be set at the same time.
            Setting any member of the oneof automatically clears all other
            members.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                gcs (google.cloud.aiplatform_v1beta1.types.GcsDestination):
                    Cloud Storage location for BatchExplain
                    output.

                    This field is a member of `oneof`_ ``destination``.
                bigquery (google.cloud.aiplatform_v1beta1.types.BigQueryDestination):
                    BigQuery location for BatchExplain output.

                    This field is a member of `oneof`_ ``destination``.
                prediction_format (google.cloud.aiplatform_v1beta1.types.ModelMonitoringObjectiveConfig.ExplanationConfig.ExplanationBaseline.PredictionFormat):
                    The storage format of the predictions
                    generated BatchPrediction job.
            """

            class PredictionFormat(proto.Enum):
                r"""The storage format of the predictions generated
                BatchPrediction job.

                Values:
                    PREDICTION_FORMAT_UNSPECIFIED (0):
                        Should not be set.
                    JSONL (2):
                        Predictions are in JSONL files.
                    BIGQUERY (3):
                        Predictions are in BigQuery.
                """
                PREDICTION_FORMAT_UNSPECIFIED = 0
                JSONL = 2
                BIGQUERY = 3

            gcs: io.GcsDestination = proto.Field(
                proto.MESSAGE,
                number=2,
                oneof="destination",
                message=io.GcsDestination,
            )
            bigquery: io.BigQueryDestination = proto.Field(
                proto.MESSAGE,
                number=3,
                oneof="destination",
                message=io.BigQueryDestination,
            )
            prediction_format: "ModelMonitoringObjectiveConfig.ExplanationConfig.ExplanationBaseline.PredictionFormat" = proto.Field(
                proto.ENUM,
                number=1,
                enum="ModelMonitoringObjectiveConfig.ExplanationConfig.ExplanationBaseline.PredictionFormat",
            )

        enable_feature_attributes: bool = proto.Field(
            proto.BOOL,
            number=1,
        )
        explanation_baseline: "ModelMonitoringObjectiveConfig.ExplanationConfig.ExplanationBaseline" = proto.Field(
            proto.MESSAGE,
            number=2,
            message="ModelMonitoringObjectiveConfig.ExplanationConfig.ExplanationBaseline",
        )

    training_dataset: TrainingDataset = proto.Field(
        proto.MESSAGE,
        number=1,
        message=TrainingDataset,
    )
    training_prediction_skew_detection_config: TrainingPredictionSkewDetectionConfig = (
        proto.Field(
            proto.MESSAGE,
            number=2,
            message=TrainingPredictionSkewDetectionConfig,
        )
    )
    prediction_drift_detection_config: PredictionDriftDetectionConfig = proto.Field(
        proto.MESSAGE,
        number=3,
        message=PredictionDriftDetectionConfig,
    )
    explanation_config: ExplanationConfig = proto.Field(
        proto.MESSAGE,
        number=5,
        message=ExplanationConfig,
    )


class ModelMonitoringAlertConfig(proto.Message):
    r"""The alert config for model monitoring.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        email_alert_config (google.cloud.aiplatform_v1beta1.types.ModelMonitoringAlertConfig.EmailAlertConfig):
            Email alert config.

            This field is a member of `oneof`_ ``alert``.
        enable_logging (bool):
            Dump the anomalies to Cloud Logging. The anomalies will be
            put to json payload encoded from proto
            [google.cloud.aiplatform.logging.ModelMonitoringAnomaliesLogEntry][].
            This can be further sinked to Pub/Sub or any other services
            supported by Cloud Logging.
        notification_channels (MutableSequence[str]):
            Resource names of the NotificationChannels to send alert.
            Must be of the format
            ``projects/<project_id_or_number>/notificationChannels/<channel_id>``
    """

    class EmailAlertConfig(proto.Message):
        r"""The config for email alert.

        Attributes:
            user_emails (MutableSequence[str]):
                The email addresses to send the alert.
        """

        user_emails: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=1,
        )

    email_alert_config: EmailAlertConfig = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="alert",
        message=EmailAlertConfig,
    )
    enable_logging: bool = proto.Field(
        proto.BOOL,
        number=2,
    )
    notification_channels: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=3,
    )


class ThresholdConfig(proto.Message):
    r"""The config for feature monitoring threshold.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        value (float):
            Specify a threshold value that can trigger
            the alert. If this threshold config is for
            feature distribution distance:

              1. For categorical feature, the distribution
                distance is calculated by      L-inifinity
                norm.
              2. For numerical feature, the distribution
                distance is calculated by
                Jensen–Shannon divergence.
            Each feature must have a non-zero threshold if
            they need to be monitored. Otherwise no alert
            will be triggered for that feature.

            This field is a member of `oneof`_ ``threshold``.
    """

    value: float = proto.Field(
        proto.DOUBLE,
        number=1,
        oneof="threshold",
    )


class SamplingStrategy(proto.Message):
    r"""Sampling Strategy for logging, can be for both training and
    prediction dataset.

    Attributes:
        random_sample_config (google.cloud.aiplatform_v1beta1.types.SamplingStrategy.RandomSampleConfig):
            Random sample config. Will support more
            sampling strategies later.
    """

    class RandomSampleConfig(proto.Message):
        r"""Requests are randomly selected.

        Attributes:
            sample_rate (float):
                Sample rate (0, 1]
        """

        sample_rate: float = proto.Field(
            proto.DOUBLE,
            number=1,
        )

    random_sample_config: RandomSampleConfig = proto.Field(
        proto.MESSAGE,
        number=1,
        message=RandomSampleConfig,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
