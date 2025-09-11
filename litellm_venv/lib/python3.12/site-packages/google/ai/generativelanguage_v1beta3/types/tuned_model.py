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

from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.ai.generativelanguage.v1beta3",
    manifest={
        "TunedModel",
        "TunedModelSource",
        "TuningTask",
        "Hyperparameters",
        "Dataset",
        "TuningExamples",
        "TuningExample",
        "TuningSnapshot",
    },
)


class TunedModel(proto.Message):
    r"""A fine-tuned model created using
    ModelService.CreateTunedModel.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        tuned_model_source (google.ai.generativelanguage_v1beta3.types.TunedModelSource):
            Optional. TunedModel to use as the starting
            point for training the new model.

            This field is a member of `oneof`_ ``source_model``.
        base_model (str):
            Immutable. The name of the ``Model`` to tune. Example:
            ``models/text-bison-001``

            This field is a member of `oneof`_ ``source_model``.
        name (str):
            Output only. The tuned model name. A unique name will be
            generated on create. Example: ``tunedModels/az2mb0bpw6i`` If
            display_name is set on create, the id portion of the name
            will be set by concatenating the words of the display_name
            with hyphens and adding a random portion for uniqueness.
            Example: display_name = "Sentence Translator" name =
            "tunedModels/sentence-translator-u3b7m".
        display_name (str):
            Optional. The name to display for this model
            in user interfaces. The display name must be up
            to 40 characters including spaces.
        description (str):
            Optional. A short description of this model.
        temperature (float):
            Optional. Controls the randomness of the output.

            Values can range over ``[0.0,1.0]``, inclusive. A value
            closer to ``1.0`` will produce responses that are more
            varied, while a value closer to ``0.0`` will typically
            result in less surprising responses from the model.

            This value specifies default to be the one used by the base
            model while creating the model.

            This field is a member of `oneof`_ ``_temperature``.
        top_p (float):
            Optional. For Nucleus sampling.

            Nucleus sampling considers the smallest set of tokens whose
            probability sum is at least ``top_p``.

            This value specifies default to be the one used by the base
            model while creating the model.

            This field is a member of `oneof`_ ``_top_p``.
        top_k (int):
            Optional. For Top-k sampling.

            Top-k sampling considers the set of ``top_k`` most probable
            tokens. This value specifies default to be used by the
            backend while making the call to the model.

            This value specifies default to be the one used by the base
            model while creating the model.

            This field is a member of `oneof`_ ``_top_k``.
        state (google.ai.generativelanguage_v1beta3.types.TunedModel.State):
            Output only. The state of the tuned model.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The timestamp when this model
            was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The timestamp when this model
            was updated.
        tuning_task (google.ai.generativelanguage_v1beta3.types.TuningTask):
            Required. The tuning task that creates the
            tuned model.
    """

    class State(proto.Enum):
        r"""The state of the tuned model.

        Values:
            STATE_UNSPECIFIED (0):
                The default value. This value is unused.
            CREATING (1):
                The model is being created.
            ACTIVE (2):
                The model is ready to be used.
            FAILED (3):
                The model failed to be created.
        """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        FAILED = 3

    tuned_model_source: "TunedModelSource" = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="source_model",
        message="TunedModelSource",
    )
    base_model: str = proto.Field(
        proto.STRING,
        number=4,
        oneof="source_model",
    )
    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=5,
    )
    description: str = proto.Field(
        proto.STRING,
        number=6,
    )
    temperature: float = proto.Field(
        proto.FLOAT,
        number=11,
        optional=True,
    )
    top_p: float = proto.Field(
        proto.FLOAT,
        number=12,
        optional=True,
    )
    top_k: int = proto.Field(
        proto.INT32,
        number=13,
        optional=True,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=7,
        enum=State,
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
    tuning_task: "TuningTask" = proto.Field(
        proto.MESSAGE,
        number=10,
        message="TuningTask",
    )


class TunedModelSource(proto.Message):
    r"""Tuned model as a source for training a new model.

    Attributes:
        tuned_model (str):
            Immutable. The name of the ``TunedModel`` to use as the
            starting point for training the new model. Example:
            ``tunedModels/my-tuned-model``
        base_model (str):
            Output only. The name of the base ``Model`` this
            ``TunedModel`` was tuned from. Example:
            ``models/text-bison-001``
    """

    tuned_model: str = proto.Field(
        proto.STRING,
        number=1,
    )
    base_model: str = proto.Field(
        proto.STRING,
        number=2,
    )


class TuningTask(proto.Message):
    r"""Tuning tasks that create tuned models.

    Attributes:
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The timestamp when tuning this
            model started.
        complete_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The timestamp when tuning this
            model completed.
        snapshots (MutableSequence[google.ai.generativelanguage_v1beta3.types.TuningSnapshot]):
            Output only. Metrics collected during tuning.
        training_data (google.ai.generativelanguage_v1beta3.types.Dataset):
            Required. Input only. Immutable. The model
            training data.
        hyperparameters (google.ai.generativelanguage_v1beta3.types.Hyperparameters):
            Immutable. Hyperparameters controlling the
            tuning process. If not provided, default values
            will be used.
    """

    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )
    complete_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    snapshots: MutableSequence["TuningSnapshot"] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message="TuningSnapshot",
    )
    training_data: "Dataset" = proto.Field(
        proto.MESSAGE,
        number=4,
        message="Dataset",
    )
    hyperparameters: "Hyperparameters" = proto.Field(
        proto.MESSAGE,
        number=5,
        message="Hyperparameters",
    )


class Hyperparameters(proto.Message):
    r"""Hyperparameters controlling the tuning process.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        epoch_count (int):
            Immutable. The number of training epochs. An
            epoch is one pass through the training data. If
            not set, a default of 10 will be used.

            This field is a member of `oneof`_ ``_epoch_count``.
        batch_size (int):
            Immutable. The batch size hyperparameter for
            tuning. If not set, a default of 16 or 64 will
            be used based on the number of training
            examples.

            This field is a member of `oneof`_ ``_batch_size``.
        learning_rate (float):
            Immutable. The learning rate hyperparameter
            for tuning. If not set, a default of 0.0002 or
            0.002 will be calculated based on the number of
            training examples.

            This field is a member of `oneof`_ ``_learning_rate``.
    """

    epoch_count: int = proto.Field(
        proto.INT32,
        number=14,
        optional=True,
    )
    batch_size: int = proto.Field(
        proto.INT32,
        number=15,
        optional=True,
    )
    learning_rate: float = proto.Field(
        proto.FLOAT,
        number=16,
        optional=True,
    )


class Dataset(proto.Message):
    r"""Dataset for training or validation.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        examples (google.ai.generativelanguage_v1beta3.types.TuningExamples):
            Optional. Inline examples.

            This field is a member of `oneof`_ ``dataset``.
    """

    examples: "TuningExamples" = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="dataset",
        message="TuningExamples",
    )


class TuningExamples(proto.Message):
    r"""A set of tuning examples. Can be training or validatation
    data.

    Attributes:
        examples (MutableSequence[google.ai.generativelanguage_v1beta3.types.TuningExample]):
            Required. The examples. Example input can be
            for text or discuss, but all examples in a set
            must be of the same type.
    """

    examples: MutableSequence["TuningExample"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="TuningExample",
    )


class TuningExample(proto.Message):
    r"""A single example for tuning.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        text_input (str):
            Optional. Text model input.

            This field is a member of `oneof`_ ``model_input``.
        output (str):
            Required. The expected model output.
    """

    text_input: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="model_input",
    )
    output: str = proto.Field(
        proto.STRING,
        number=3,
    )


class TuningSnapshot(proto.Message):
    r"""Record for a single tuning step.

    Attributes:
        step (int):
            Output only. The tuning step.
        epoch (int):
            Output only. The epoch this step was part of.
        mean_loss (float):
            Output only. The mean loss of the training
            examples for this step.
        compute_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The timestamp when this metric
            was computed.
    """

    step: int = proto.Field(
        proto.INT32,
        number=1,
    )
    epoch: int = proto.Field(
        proto.INT32,
        number=2,
    )
    mean_loss: float = proto.Field(
        proto.FLOAT,
        number=3,
    )
    compute_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
