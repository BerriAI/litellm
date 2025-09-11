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
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "ReasoningEngineSpec",
        "ReasoningEngine",
    },
)


class ReasoningEngineSpec(proto.Message):
    r"""ReasoningEngine configurations

    Attributes:
        package_spec (google.cloud.aiplatform_v1beta1.types.ReasoningEngineSpec.PackageSpec):
            Required. User provided package spec of the
            ReasoningEngine.
        class_methods (MutableSequence[google.protobuf.struct_pb2.Struct]):
            Optional. Declarations for object class
            methods.
    """

    class PackageSpec(proto.Message):
        r"""User provided package spec like pickled object and package
        requirements.

        Attributes:
            pickle_object_gcs_uri (str):
                Optional. The Cloud Storage URI of the
                pickled python object.
            dependency_files_gcs_uri (str):
                Optional. The Cloud Storage URI of the
                dependency files in tar.gz format.
            requirements_gcs_uri (str):
                Optional. The Cloud Storage URI of the ``requirements.txt``
                file
            python_version (str):
                Optional. The Python version. Currently
                support 3.8, 3.9, 3.10, 3.11. If not specified,
                default value is 3.10.
        """

        pickle_object_gcs_uri: str = proto.Field(
            proto.STRING,
            number=1,
        )
        dependency_files_gcs_uri: str = proto.Field(
            proto.STRING,
            number=2,
        )
        requirements_gcs_uri: str = proto.Field(
            proto.STRING,
            number=3,
        )
        python_version: str = proto.Field(
            proto.STRING,
            number=4,
        )

    package_spec: PackageSpec = proto.Field(
        proto.MESSAGE,
        number=2,
        message=PackageSpec,
    )
    class_methods: MutableSequence[struct_pb2.Struct] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=struct_pb2.Struct,
    )


class ReasoningEngine(proto.Message):
    r"""ReasoningEngine provides a customizable runtime for models to
    determine which actions to take and in which order.

    Attributes:
        name (str):
            Identifier. The resource name of the
            ReasoningEngine.
        display_name (str):
            Required. The display name of the
            ReasoningEngine.
        description (str):
            Optional. The description of the
            ReasoningEngine.
        spec (google.cloud.aiplatform_v1beta1.types.ReasoningEngineSpec):
            Required. Configurations of the
            ReasoningEngine
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            ReasoningEngine was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            ReasoningEngine was most recently updated.
        etag (str):
            Optional. Used to perform consistent
            read-modify-write updates. If not set, a blind
            "overwrite" update happens.
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
        number=7,
    )
    spec: "ReasoningEngineSpec" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="ReasoningEngineSpec",
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
    etag: str = proto.Field(
        proto.STRING,
        number=6,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
