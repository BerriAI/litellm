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

from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "TensorboardExperiment",
    },
)


class TensorboardExperiment(proto.Message):
    r"""A TensorboardExperiment is a group of TensorboardRuns, that
    are typically the results of a training job run, in a
    Tensorboard.

    Attributes:
        name (str):
            Output only. Name of the TensorboardExperiment. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}``
        display_name (str):
            User provided name of this
            TensorboardExperiment.
        description (str):
            Description of this TensorboardExperiment.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            TensorboardExperiment was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            TensorboardExperiment was last updated.
        labels (MutableMapping[str, str]):
            The labels with user-defined metadata to organize your
            TensorboardExperiment.

            Label keys and values cannot be longer than 64 characters
            (Unicode codepoints), can only contain lowercase letters,
            numeric characters, underscores and dashes. International
            characters are allowed. No more than 64 user labels can be
            associated with one Dataset (System labels are excluded).

            See https://goo.gl/xmQnxf for more information and examples
            of labels. System reserved label keys are prefixed with
            ``aiplatform.googleapis.com/`` and are immutable. The
            following system labels exist for each Dataset:

            -  ``aiplatform.googleapis.com/dataset_metadata_schema``:
               output only. Its value is the
               [metadata_schema's][google.cloud.aiplatform.v1.Dataset.metadata_schema_uri]
               title.
        etag (str):
            Used to perform consistent read-modify-write
            updates. If not set, a blind "overwrite" update
            happens.
        source (str):
            Immutable. Source of the
            TensorboardExperiment. Example: a custom
            training job.
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
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=6,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=7,
    )
    source: str = proto.Field(
        proto.STRING,
        number=8,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
