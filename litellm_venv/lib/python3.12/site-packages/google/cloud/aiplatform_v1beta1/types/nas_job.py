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

from google.cloud.aiplatform_v1beta1.types import custom_job
from google.cloud.aiplatform_v1beta1.types import encryption_spec as gca_encryption_spec
from google.cloud.aiplatform_v1beta1.types import job_state
from google.cloud.aiplatform_v1beta1.types import study
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "NasJob",
        "NasTrialDetail",
        "NasJobSpec",
        "NasJobOutput",
        "NasTrial",
    },
)


class NasJob(proto.Message):
    r"""Represents a Neural Architecture Search (NAS) job.

    Attributes:
        name (str):
            Output only. Resource name of the NasJob.
        display_name (str):
            Required. The display name of the NasJob.
            The name can be up to 128 characters long and
            can consist of any UTF-8 characters.
        nas_job_spec (google.cloud.aiplatform_v1beta1.types.NasJobSpec):
            Required. The specification of a NasJob.
        nas_job_output (google.cloud.aiplatform_v1beta1.types.NasJobOutput):
            Output only. Output of the NasJob.
        state (google.cloud.aiplatform_v1beta1.types.JobState):
            Output only. The detailed state of the job.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the NasJob was
            created.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the NasJob for the first time entered
            the ``JOB_STATE_RUNNING`` state.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the NasJob entered any of the
            following states: ``JOB_STATE_SUCCEEDED``,
            ``JOB_STATE_FAILED``, ``JOB_STATE_CANCELLED``.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the NasJob was most
            recently updated.
        error (google.rpc.status_pb2.Status):
            Output only. Only populated when job's state is
            JOB_STATE_FAILED or JOB_STATE_CANCELLED.
        labels (MutableMapping[str, str]):
            The labels with user-defined metadata to
            organize NasJobs.
            Label keys and values can be no longer than 64
            characters (Unicode codepoints), can only
            contain lowercase letters, numeric characters,
            underscores and dashes. International characters
            are allowed.

            See https://goo.gl/xmQnxf for more information
            and examples of labels.
        encryption_spec (google.cloud.aiplatform_v1beta1.types.EncryptionSpec):
            Customer-managed encryption key options for a
            NasJob. If this is set, then all resources
            created by the NasJob will be encrypted with the
            provided encryption key.
        enable_restricted_image_training (bool):
            Optional. Enable a separation of Custom model
            training and restricted image training for
            tenant project.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    nas_job_spec: "NasJobSpec" = proto.Field(
        proto.MESSAGE,
        number=4,
        message="NasJobSpec",
    )
    nas_job_output: "NasJobOutput" = proto.Field(
        proto.MESSAGE,
        number=5,
        message="NasJobOutput",
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
    encryption_spec: gca_encryption_spec.EncryptionSpec = proto.Field(
        proto.MESSAGE,
        number=13,
        message=gca_encryption_spec.EncryptionSpec,
    )
    enable_restricted_image_training: bool = proto.Field(
        proto.BOOL,
        number=14,
    )


class NasTrialDetail(proto.Message):
    r"""Represents a NasTrial details along with its parameters. If
    there is a corresponding train NasTrial, the train NasTrial is
    also returned.

    Attributes:
        name (str):
            Output only. Resource name of the
            NasTrialDetail.
        parameters (str):
            The parameters for the NasJob NasTrial.
        search_trial (google.cloud.aiplatform_v1beta1.types.NasTrial):
            The requested search NasTrial.
        train_trial (google.cloud.aiplatform_v1beta1.types.NasTrial):
            The train NasTrial corresponding to
            [search_trial][google.cloud.aiplatform.v1beta1.NasTrialDetail.search_trial].
            Only populated if
            [search_trial][google.cloud.aiplatform.v1beta1.NasTrialDetail.search_trial]
            is used for training.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    parameters: str = proto.Field(
        proto.STRING,
        number=2,
    )
    search_trial: "NasTrial" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="NasTrial",
    )
    train_trial: "NasTrial" = proto.Field(
        proto.MESSAGE,
        number=4,
        message="NasTrial",
    )


class NasJobSpec(proto.Message):
    r"""Represents the spec of a NasJob.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        multi_trial_algorithm_spec (google.cloud.aiplatform_v1beta1.types.NasJobSpec.MultiTrialAlgorithmSpec):
            The spec of multi-trial algorithms.

            This field is a member of `oneof`_ ``nas_algorithm_spec``.
        resume_nas_job_id (str):
            The ID of the existing NasJob in the same Project and
            Location which will be used to resume search.
            search_space_spec and nas_algorithm_spec are obtained from
            previous NasJob hence should not provide them again for this
            NasJob.
        search_space_spec (str):
            It defines the search space for Neural
            Architecture Search (NAS).
    """

    class MultiTrialAlgorithmSpec(proto.Message):
        r"""The spec of multi-trial Neural Architecture Search (NAS).

        Attributes:
            multi_trial_algorithm (google.cloud.aiplatform_v1beta1.types.NasJobSpec.MultiTrialAlgorithmSpec.MultiTrialAlgorithm):
                The multi-trial Neural Architecture Search (NAS) algorithm
                type. Defaults to ``REINFORCEMENT_LEARNING``.
            metric (google.cloud.aiplatform_v1beta1.types.NasJobSpec.MultiTrialAlgorithmSpec.MetricSpec):
                Metric specs for the NAS job. Validation for this field is
                done at ``multi_trial_algorithm_spec`` field.
            search_trial_spec (google.cloud.aiplatform_v1beta1.types.NasJobSpec.MultiTrialAlgorithmSpec.SearchTrialSpec):
                Required. Spec for search trials.
            train_trial_spec (google.cloud.aiplatform_v1beta1.types.NasJobSpec.MultiTrialAlgorithmSpec.TrainTrialSpec):
                Spec for train trials. Top N
                [TrainTrialSpec.max_parallel_trial_count] search trials will
                be trained for every M [TrainTrialSpec.frequency] trials
                searched.
        """

        class MultiTrialAlgorithm(proto.Enum):
            r"""The available types of multi-trial algorithms.

            Values:
                MULTI_TRIAL_ALGORITHM_UNSPECIFIED (0):
                    Defaults to ``REINFORCEMENT_LEARNING``.
                REINFORCEMENT_LEARNING (1):
                    The Reinforcement Learning Algorithm for
                    Multi-trial Neural Architecture Search (NAS).
                GRID_SEARCH (2):
                    The Grid Search Algorithm for Multi-trial
                    Neural Architecture Search (NAS).
            """
            MULTI_TRIAL_ALGORITHM_UNSPECIFIED = 0
            REINFORCEMENT_LEARNING = 1
            GRID_SEARCH = 2

        class MetricSpec(proto.Message):
            r"""Represents a metric to optimize.

            Attributes:
                metric_id (str):
                    Required. The ID of the metric. Must not
                    contain whitespaces.
                goal (google.cloud.aiplatform_v1beta1.types.NasJobSpec.MultiTrialAlgorithmSpec.MetricSpec.GoalType):
                    Required. The optimization goal of the
                    metric.
            """

            class GoalType(proto.Enum):
                r"""The available types of optimization goals.

                Values:
                    GOAL_TYPE_UNSPECIFIED (0):
                        Goal Type will default to maximize.
                    MAXIMIZE (1):
                        Maximize the goal metric.
                    MINIMIZE (2):
                        Minimize the goal metric.
                """
                GOAL_TYPE_UNSPECIFIED = 0
                MAXIMIZE = 1
                MINIMIZE = 2

            metric_id: str = proto.Field(
                proto.STRING,
                number=1,
            )
            goal: "NasJobSpec.MultiTrialAlgorithmSpec.MetricSpec.GoalType" = (
                proto.Field(
                    proto.ENUM,
                    number=2,
                    enum="NasJobSpec.MultiTrialAlgorithmSpec.MetricSpec.GoalType",
                )
            )

        class SearchTrialSpec(proto.Message):
            r"""Represent spec for search trials.

            Attributes:
                search_trial_job_spec (google.cloud.aiplatform_v1beta1.types.CustomJobSpec):
                    Required. The spec of a search trial job. The
                    same spec applies to all search trials.
                max_trial_count (int):
                    Required. The maximum number of Neural
                    Architecture Search (NAS) trials to run.
                max_parallel_trial_count (int):
                    Required. The maximum number of trials to run
                    in parallel.
                max_failed_trial_count (int):
                    The number of failed trials that need to be
                    seen before failing the NasJob.

                    If set to 0, Vertex AI decides how many trials
                    must fail before the whole job fails.
            """

            search_trial_job_spec: custom_job.CustomJobSpec = proto.Field(
                proto.MESSAGE,
                number=1,
                message=custom_job.CustomJobSpec,
            )
            max_trial_count: int = proto.Field(
                proto.INT32,
                number=2,
            )
            max_parallel_trial_count: int = proto.Field(
                proto.INT32,
                number=3,
            )
            max_failed_trial_count: int = proto.Field(
                proto.INT32,
                number=4,
            )

        class TrainTrialSpec(proto.Message):
            r"""Represent spec for train trials.

            Attributes:
                train_trial_job_spec (google.cloud.aiplatform_v1beta1.types.CustomJobSpec):
                    Required. The spec of a train trial job. The
                    same spec applies to all train trials.
                max_parallel_trial_count (int):
                    Required. The maximum number of trials to run
                    in parallel.
                frequency (int):
                    Required. Frequency of search trials to start train stage.
                    Top N [TrainTrialSpec.max_parallel_trial_count] search
                    trials will be trained for every M
                    [TrainTrialSpec.frequency] trials searched.
            """

            train_trial_job_spec: custom_job.CustomJobSpec = proto.Field(
                proto.MESSAGE,
                number=1,
                message=custom_job.CustomJobSpec,
            )
            max_parallel_trial_count: int = proto.Field(
                proto.INT32,
                number=2,
            )
            frequency: int = proto.Field(
                proto.INT32,
                number=3,
            )

        multi_trial_algorithm: "NasJobSpec.MultiTrialAlgorithmSpec.MultiTrialAlgorithm" = proto.Field(
            proto.ENUM,
            number=1,
            enum="NasJobSpec.MultiTrialAlgorithmSpec.MultiTrialAlgorithm",
        )
        metric: "NasJobSpec.MultiTrialAlgorithmSpec.MetricSpec" = proto.Field(
            proto.MESSAGE,
            number=2,
            message="NasJobSpec.MultiTrialAlgorithmSpec.MetricSpec",
        )
        search_trial_spec: "NasJobSpec.MultiTrialAlgorithmSpec.SearchTrialSpec" = (
            proto.Field(
                proto.MESSAGE,
                number=3,
                message="NasJobSpec.MultiTrialAlgorithmSpec.SearchTrialSpec",
            )
        )
        train_trial_spec: "NasJobSpec.MultiTrialAlgorithmSpec.TrainTrialSpec" = (
            proto.Field(
                proto.MESSAGE,
                number=4,
                message="NasJobSpec.MultiTrialAlgorithmSpec.TrainTrialSpec",
            )
        )

    multi_trial_algorithm_spec: MultiTrialAlgorithmSpec = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="nas_algorithm_spec",
        message=MultiTrialAlgorithmSpec,
    )
    resume_nas_job_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    search_space_spec: str = proto.Field(
        proto.STRING,
        number=1,
    )


class NasJobOutput(proto.Message):
    r"""Represents a uCAIP NasJob output.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        multi_trial_job_output (google.cloud.aiplatform_v1beta1.types.NasJobOutput.MultiTrialJobOutput):
            Output only. The output of this multi-trial
            Neural Architecture Search (NAS) job.

            This field is a member of `oneof`_ ``output``.
    """

    class MultiTrialJobOutput(proto.Message):
        r"""The output of a multi-trial Neural Architecture Search (NAS)
        jobs.

        Attributes:
            search_trials (MutableSequence[google.cloud.aiplatform_v1beta1.types.NasTrial]):
                Output only. List of NasTrials that were
                started as part of search stage.
            train_trials (MutableSequence[google.cloud.aiplatform_v1beta1.types.NasTrial]):
                Output only. List of NasTrials that were
                started as part of train stage.
        """

        search_trials: MutableSequence["NasTrial"] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="NasTrial",
        )
        train_trials: MutableSequence["NasTrial"] = proto.RepeatedField(
            proto.MESSAGE,
            number=2,
            message="NasTrial",
        )

    multi_trial_job_output: MultiTrialJobOutput = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="output",
        message=MultiTrialJobOutput,
    )


class NasTrial(proto.Message):
    r"""Represents a uCAIP NasJob trial.

    Attributes:
        id (str):
            Output only. The identifier of the NasTrial
            assigned by the service.
        state (google.cloud.aiplatform_v1beta1.types.NasTrial.State):
            Output only. The detailed state of the
            NasTrial.
        final_measurement (google.cloud.aiplatform_v1beta1.types.Measurement):
            Output only. The final measurement containing
            the objective value.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the NasTrial was
            started.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the NasTrial's status changed to
            ``SUCCEEDED`` or ``INFEASIBLE``.
    """

    class State(proto.Enum):
        r"""Describes a NasTrial state.

        Values:
            STATE_UNSPECIFIED (0):
                The NasTrial state is unspecified.
            REQUESTED (1):
                Indicates that a specific NasTrial has been
                requested, but it has not yet been suggested by
                the service.
            ACTIVE (2):
                Indicates that the NasTrial has been
                suggested.
            STOPPING (3):
                Indicates that the NasTrial should stop
                according to the service.
            SUCCEEDED (4):
                Indicates that the NasTrial is completed
                successfully.
            INFEASIBLE (5):
                Indicates that the NasTrial should not be attempted again.
                The service will set a NasTrial to INFEASIBLE when it's done
                but missing the final_measurement.
        """
        STATE_UNSPECIFIED = 0
        REQUESTED = 1
        ACTIVE = 2
        STOPPING = 3
        SUCCEEDED = 4
        INFEASIBLE = 5

    id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=2,
        enum=State,
    )
    final_measurement: study.Measurement = proto.Field(
        proto.MESSAGE,
        number=3,
        message=study.Measurement,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
