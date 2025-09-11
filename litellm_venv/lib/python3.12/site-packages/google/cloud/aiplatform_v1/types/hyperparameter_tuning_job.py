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

from google.cloud.aiplatform_v1.types import custom_job
from google.cloud.aiplatform_v1.types import encryption_spec as gca_encryption_spec
from google.cloud.aiplatform_v1.types import job_state
from google.cloud.aiplatform_v1.types import study
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "HyperparameterTuningJob",
    },
)


class HyperparameterTuningJob(proto.Message):
    r"""Represents a HyperparameterTuningJob. A
    HyperparameterTuningJob has a Study specification and multiple
    CustomJobs with identical CustomJob specification.

    Attributes:
        name (str):
            Output only. Resource name of the
            HyperparameterTuningJob.
        display_name (str):
            Required. The display name of the
            HyperparameterTuningJob. The name can be up to
            128 characters long and can consist of any UTF-8
            characters.
        study_spec (google.cloud.aiplatform_v1.types.StudySpec):
            Required. Study configuration of the
            HyperparameterTuningJob.
        max_trial_count (int):
            Required. The desired total number of Trials.
        parallel_trial_count (int):
            Required. The desired number of Trials to run
            in parallel.
        max_failed_trial_count (int):
            The number of failed Trials that need to be
            seen before failing the HyperparameterTuningJob.

            If set to 0, Vertex AI decides how many Trials
            must fail before the whole job fails.
        trial_job_spec (google.cloud.aiplatform_v1.types.CustomJobSpec):
            Required. The spec of a trial job. The same
            spec applies to the CustomJobs created in all
            the trials.
        trials (MutableSequence[google.cloud.aiplatform_v1.types.Trial]):
            Output only. Trials of the
            HyperparameterTuningJob.
        state (google.cloud.aiplatform_v1.types.JobState):
            Output only. The detailed state of the job.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the
            HyperparameterTuningJob was created.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the HyperparameterTuningJob for the
            first time entered the ``JOB_STATE_RUNNING`` state.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the HyperparameterTuningJob entered
            any of the following states: ``JOB_STATE_SUCCEEDED``,
            ``JOB_STATE_FAILED``, ``JOB_STATE_CANCELLED``.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the
            HyperparameterTuningJob was most recently
            updated.
        error (google.rpc.status_pb2.Status):
            Output only. Only populated when job's state is
            JOB_STATE_FAILED or JOB_STATE_CANCELLED.
        labels (MutableMapping[str, str]):
            The labels with user-defined metadata to
            organize HyperparameterTuningJobs.
            Label keys and values can be no longer than 64
            characters (Unicode codepoints), can only
            contain lowercase letters, numeric characters,
            underscores and dashes. International characters
            are allowed.

            See https://goo.gl/xmQnxf for more information
            and examples of labels.
        encryption_spec (google.cloud.aiplatform_v1.types.EncryptionSpec):
            Customer-managed encryption key options for a
            HyperparameterTuningJob. If this is set, then
            all resources created by the
            HyperparameterTuningJob will be encrypted with
            the provided encryption key.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    study_spec: study.StudySpec = proto.Field(
        proto.MESSAGE,
        number=4,
        message=study.StudySpec,
    )
    max_trial_count: int = proto.Field(
        proto.INT32,
        number=5,
    )
    parallel_trial_count: int = proto.Field(
        proto.INT32,
        number=6,
    )
    max_failed_trial_count: int = proto.Field(
        proto.INT32,
        number=7,
    )
    trial_job_spec: custom_job.CustomJobSpec = proto.Field(
        proto.MESSAGE,
        number=8,
        message=custom_job.CustomJobSpec,
    )
    trials: MutableSequence[study.Trial] = proto.RepeatedField(
        proto.MESSAGE,
        number=9,
        message=study.Trial,
    )
    state: job_state.JobState = proto.Field(
        proto.ENUM,
        number=10,
        enum=job_state.JobState,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=11,
        message=timestamp_pb2.Timestamp,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=12,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=13,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=14,
        message=timestamp_pb2.Timestamp,
    )
    error: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=15,
        message=status_pb2.Status,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=16,
    )
    encryption_spec: gca_encryption_spec.EncryptionSpec = proto.Field(
        proto.MESSAGE,
        number=17,
        message=gca_encryption_spec.EncryptionSpec,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
