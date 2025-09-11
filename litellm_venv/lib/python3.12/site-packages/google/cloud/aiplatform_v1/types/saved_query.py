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

from google.protobuf import struct_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "SavedQuery",
    },
)


class SavedQuery(proto.Message):
    r"""A SavedQuery is a view of the dataset. It references a subset
    of annotations by problem type and filters.

    Attributes:
        name (str):
            Output only. Resource name of the SavedQuery.
        display_name (str):
            Required. The user-defined name of the
            SavedQuery. The name can be up to 128 characters
            long and can consist of any UTF-8 characters.
        metadata (google.protobuf.struct_pb2.Value):
            Some additional information about the
            SavedQuery.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this SavedQuery
            was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when SavedQuery was
            last updated.
        annotation_filter (str):
            Output only. Filters on the Annotations in
            the dataset.
        problem_type (str):
            Required. Problem type of the SavedQuery. Allowed values:

            -  IMAGE_CLASSIFICATION_SINGLE_LABEL
            -  IMAGE_CLASSIFICATION_MULTI_LABEL
            -  IMAGE_BOUNDING_POLY
            -  IMAGE_BOUNDING_BOX
            -  TEXT_CLASSIFICATION_SINGLE_LABEL
            -  TEXT_CLASSIFICATION_MULTI_LABEL
            -  TEXT_EXTRACTION
            -  TEXT_SENTIMENT
            -  VIDEO_CLASSIFICATION
            -  VIDEO_OBJECT_TRACKING
        annotation_spec_count (int):
            Output only. Number of AnnotationSpecs in the
            context of the SavedQuery.
        etag (str):
            Used to perform a consistent
            read-modify-write update. If not set, a blind
            "overwrite" update happens.
        support_automl_training (bool):
            Output only. If the Annotations belonging to
            the SavedQuery can be used for AutoML training.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    metadata: struct_pb2.Value = proto.Field(
        proto.MESSAGE,
        number=12,
        message=struct_pb2.Value,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    annotation_filter: str = proto.Field(
        proto.STRING,
        number=5,
    )
    problem_type: str = proto.Field(
        proto.STRING,
        number=6,
    )
    annotation_spec_count: int = proto.Field(
        proto.INT32,
        number=10,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=8,
    )
    support_automl_training: bool = proto.Field(
        proto.BOOL,
        number=9,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
