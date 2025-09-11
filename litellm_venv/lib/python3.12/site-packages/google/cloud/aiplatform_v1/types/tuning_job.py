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

from google.cloud.aiplatform_v1.types import content
from google.cloud.aiplatform_v1.types import job_state
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "TuningJob",
        "TunedModel",
        "SupervisedTuningDatasetDistribution",
        "SupervisedTuningDataStats",
        "TuningDataStats",
        "SupervisedHyperParameters",
        "SupervisedTuningSpec",
    },
)


class TuningJob(proto.Message):
    r"""Represents a TuningJob that runs with Google owned models.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        base_model (str):
            Model name for tuning, e.g.,
            "gemini-1.0-pro-002".

            This field is a member of `oneof`_ ``source_model``.
        supervised_tuning_spec (google.cloud.aiplatform_v1.types.SupervisedTuningSpec):
            Tuning Spec for Supervised Fine Tuning.

            This field is a member of `oneof`_ ``tuning_spec``.
        name (str):
            Output only. Identifier. Resource name of a TuningJob.
            Format:
            ``projects/{project}/locations/{location}/tuningJobs/{tuning_job}``
        tuned_model_display_name (str):
            Optional. The display name of the
            [TunedModel][google.cloud.aiplatform.v1.Model]. The name can
            be up to 128 characters long and can consist of any UTF-8
            characters.
        description (str):
            Optional. The description of the
            [TuningJob][google.cloud.aiplatform.v1.TuningJob].
        state (google.cloud.aiplatform_v1.types.JobState):
            Output only. The detailed state of the job.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the
            [TuningJob][google.cloud.aiplatform.v1.TuningJob] was
            created.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the
            [TuningJob][google.cloud.aiplatform.v1.TuningJob] for the
            first time entered the ``JOB_STATE_RUNNING`` state.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the TuningJob entered any of the
            following [JobStates][google.cloud.aiplatform.v1.JobState]:
            ``JOB_STATE_SUCCEEDED``, ``JOB_STATE_FAILED``,
            ``JOB_STATE_CANCELLED``, ``JOB_STATE_EXPIRED``.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the
            [TuningJob][google.cloud.aiplatform.v1.TuningJob] was most
            recently updated.
        error (google.rpc.status_pb2.Status):
            Output only. Only populated when job's state is
            ``JOB_STATE_FAILED`` or ``JOB_STATE_CANCELLED``.
        labels (MutableMapping[str, str]):
            Optional. The labels with user-defined metadata to organize
            [TuningJob][google.cloud.aiplatform.v1.TuningJob] and
            generated resources such as
            [Model][google.cloud.aiplatform.v1.Model] and
            [Endpoint][google.cloud.aiplatform.v1.Endpoint].

            Label keys and values can be no longer than 64 characters
            (Unicode codepoints), can only contain lowercase letters,
            numeric characters, underscores and dashes. International
            characters are allowed.

            See https://goo.gl/xmQnxf for more information and examples
            of labels.
        experiment (str):
            Output only. The Experiment associated with this
            [TuningJob][google.cloud.aiplatform.v1.TuningJob].
        tuned_model (google.cloud.aiplatform_v1.types.TunedModel):
            Output only. The tuned model resources assiociated with this
            [TuningJob][google.cloud.aiplatform.v1.TuningJob].
        tuning_data_stats (google.cloud.aiplatform_v1.types.TuningDataStats):
            Output only. The tuning data statistics associated with this
            [TuningJob][google.cloud.aiplatform.v1.TuningJob].
    """

    base_model: str = proto.Field(
        proto.STRING,
        number=4,
        oneof="source_model",
    )
    supervised_tuning_spec: "SupervisedTuningSpec" = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="tuning_spec",
        message="SupervisedTuningSpec",
    )
    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    tuned_model_display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    description: str = proto.Field(
        proto.STRING,
        number=3,
    )
    state: job_state.JobState = proto.Field(
        proto.ENUM,
        number=6,
        enum=job_state.JobState,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=8,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=9,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=10,
        message=timestamp_pb2.Timestamp,
    )
    error: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=11,
        message=status_pb2.Status,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=12,
    )
    experiment: str = proto.Field(
        proto.STRING,
        number=13,
    )
    tuned_model: "TunedModel" = proto.Field(
        proto.MESSAGE,
        number=14,
        message="TunedModel",
    )
    tuning_data_stats: "TuningDataStats" = proto.Field(
        proto.MESSAGE,
        number=15,
        message="TuningDataStats",
    )


class TunedModel(proto.Message):
    r"""The Model Registry Model and Online Prediction Endpoint assiociated
    with this [TuningJob][google.cloud.aiplatform.v1.TuningJob].

    Attributes:
        model (str):
            Output only. The resource name of the TunedModel. Format:
            ``projects/{project}/locations/{location}/models/{model}``.
        endpoint (str):
            Output only. A resource name of an Endpoint. Format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``.
    """

    model: str = proto.Field(
        proto.STRING,
        number=1,
    )
    endpoint: str = proto.Field(
        proto.STRING,
        number=2,
    )


class SupervisedTuningDatasetDistribution(proto.Message):
    r"""Dataset distribution for Supervised Tuning.

    Attributes:
        sum (int):
            Output only. Sum of a given population of
            values.
        min_ (float):
            Output only. The minimum of the population
            values.
        max_ (float):
            Output only. The maximum of the population
            values.
        mean (float):
            Output only. The arithmetic mean of the
            values in the population.
        median (float):
            Output only. The median of the values in the
            population.
        p5 (float):
            Output only. The 5th percentile of the values
            in the population.
        p95 (float):
            Output only. The 95th percentile of the
            values in the population.
        buckets (MutableSequence[google.cloud.aiplatform_v1.types.SupervisedTuningDatasetDistribution.DatasetBucket]):
            Output only. Defines the histogram bucket.
    """

    class DatasetBucket(proto.Message):
        r"""Dataset bucket used to create a histogram for the
        distribution given a population of values.

        Attributes:
            count (float):
                Output only. Number of values in the bucket.
            left (float):
                Output only. Left bound of the bucket.
            right (float):
                Output only. Right bound of the bucket.
        """

        count: float = proto.Field(
            proto.DOUBLE,
            number=1,
        )
        left: float = proto.Field(
            proto.DOUBLE,
            number=2,
        )
        right: float = proto.Field(
            proto.DOUBLE,
            number=3,
        )

    sum: int = proto.Field(
        proto.INT64,
        number=1,
    )
    min_: float = proto.Field(
        proto.DOUBLE,
        number=2,
    )
    max_: float = proto.Field(
        proto.DOUBLE,
        number=3,
    )
    mean: float = proto.Field(
        proto.DOUBLE,
        number=4,
    )
    median: float = proto.Field(
        proto.DOUBLE,
        number=5,
    )
    p5: float = proto.Field(
        proto.DOUBLE,
        number=6,
    )
    p95: float = proto.Field(
        proto.DOUBLE,
        number=7,
    )
    buckets: MutableSequence[DatasetBucket] = proto.RepeatedField(
        proto.MESSAGE,
        number=8,
        message=DatasetBucket,
    )


class SupervisedTuningDataStats(proto.Message):
    r"""Tuning data statistics for Supervised Tuning.

    Attributes:
        tuning_dataset_example_count (int):
            Output only. Number of examples in the tuning
            dataset.
        total_tuning_character_count (int):
            Output only. Number of tuning characters in
            the tuning dataset.
        total_billable_character_count (int):
            Output only. Number of billable characters in
            the tuning dataset.
        tuning_step_count (int):
            Output only. Number of tuning steps for this
            Tuning Job.
        user_input_token_distribution (google.cloud.aiplatform_v1.types.SupervisedTuningDatasetDistribution):
            Output only. Dataset distributions for the
            user input tokens.
        user_output_token_distribution (google.cloud.aiplatform_v1.types.SupervisedTuningDatasetDistribution):
            Output only. Dataset distributions for the
            user output tokens.
        user_message_per_example_distribution (google.cloud.aiplatform_v1.types.SupervisedTuningDatasetDistribution):
            Output only. Dataset distributions for the
            messages per example.
        user_dataset_examples (MutableSequence[google.cloud.aiplatform_v1.types.Content]):
            Output only. Sample user messages in the
            training dataset uri.
    """

    tuning_dataset_example_count: int = proto.Field(
        proto.INT64,
        number=1,
    )
    total_tuning_character_count: int = proto.Field(
        proto.INT64,
        number=2,
    )
    total_billable_character_count: int = proto.Field(
        proto.INT64,
        number=3,
    )
    tuning_step_count: int = proto.Field(
        proto.INT64,
        number=4,
    )
    user_input_token_distribution: "SupervisedTuningDatasetDistribution" = proto.Field(
        proto.MESSAGE,
        number=5,
        message="SupervisedTuningDatasetDistribution",
    )
    user_output_token_distribution: "SupervisedTuningDatasetDistribution" = proto.Field(
        proto.MESSAGE,
        number=6,
        message="SupervisedTuningDatasetDistribution",
    )
    user_message_per_example_distribution: "SupervisedTuningDatasetDistribution" = (
        proto.Field(
            proto.MESSAGE,
            number=7,
            message="SupervisedTuningDatasetDistribution",
        )
    )
    user_dataset_examples: MutableSequence[content.Content] = proto.RepeatedField(
        proto.MESSAGE,
        number=8,
        message=content.Content,
    )


class TuningDataStats(proto.Message):
    r"""The tuning data statistic values for
    [TuningJob][google.cloud.aiplatform.v1.TuningJob].


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        supervised_tuning_data_stats (google.cloud.aiplatform_v1.types.SupervisedTuningDataStats):
            The SFT Tuning data stats.

            This field is a member of `oneof`_ ``tuning_data_stats``.
    """

    supervised_tuning_data_stats: "SupervisedTuningDataStats" = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="tuning_data_stats",
        message="SupervisedTuningDataStats",
    )


class SupervisedHyperParameters(proto.Message):
    r"""Hyperparameters for SFT.

    Attributes:
        epoch_count (int):
            Optional. Number of training epoches for this
            tuning job.
        learning_rate_multiplier (float):
            Optional. Learning rate multiplier for
            tuning.
        adapter_size (google.cloud.aiplatform_v1.types.SupervisedHyperParameters.AdapterSize):
            Optional. Adapter size for tuning.
    """

    class AdapterSize(proto.Enum):
        r"""Supported adapter sizes for tuning.

        Values:
            ADAPTER_SIZE_UNSPECIFIED (0):
                Adapter size is unspecified.
            ADAPTER_SIZE_ONE (1):
                Adapter size 1.
            ADAPTER_SIZE_FOUR (2):
                Adapter size 4.
            ADAPTER_SIZE_EIGHT (3):
                Adapter size 8.
            ADAPTER_SIZE_SIXTEEN (4):
                Adapter size 16.
        """
        ADAPTER_SIZE_UNSPECIFIED = 0
        ADAPTER_SIZE_ONE = 1
        ADAPTER_SIZE_FOUR = 2
        ADAPTER_SIZE_EIGHT = 3
        ADAPTER_SIZE_SIXTEEN = 4

    epoch_count: int = proto.Field(
        proto.INT64,
        number=1,
    )
    learning_rate_multiplier: float = proto.Field(
        proto.DOUBLE,
        number=2,
    )
    adapter_size: AdapterSize = proto.Field(
        proto.ENUM,
        number=3,
        enum=AdapterSize,
    )


class SupervisedTuningSpec(proto.Message):
    r"""Tuning Spec for Supervised Tuning.

    Attributes:
        training_dataset_uri (str):
            Required. Cloud Storage path to file
            containing training dataset for tuning.
        validation_dataset_uri (str):
            Optional. Cloud Storage path to file
            containing validation dataset for tuning.
        hyper_parameters (google.cloud.aiplatform_v1.types.SupervisedHyperParameters):
            Optional. Hyperparameters for SFT.
    """

    training_dataset_uri: str = proto.Field(
        proto.STRING,
        number=1,
    )
    validation_dataset_uri: str = proto.Field(
        proto.STRING,
        number=2,
    )
    hyper_parameters: "SupervisedHyperParameters" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="SupervisedHyperParameters",
    )


__all__ = tuple(sorted(__protobuf__.manifest))
