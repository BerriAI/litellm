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


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "UserActionReference",
    },
)


class UserActionReference(proto.Message):
    r"""References an API call. It contains more information about
    long running operation and Jobs that are triggered by the API
    call.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        operation (str):
            For API calls that return a long running operation. Resource
            name of the long running operation. Format:
            ``projects/{project}/locations/{location}/operations/{operation}``

            This field is a member of `oneof`_ ``reference``.
        data_labeling_job (str):
            For API calls that start a LabelingJob. Resource name of the
            LabelingJob. Format:
            ``projects/{project}/locations/{location}/dataLabelingJobs/{data_labeling_job}``

            This field is a member of `oneof`_ ``reference``.
        method (str):
            The method name of the API RPC call. For
            example,
            "/google.cloud.aiplatform.{apiVersion}.DatasetService.CreateDataset".
    """

    operation: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="reference",
    )
    data_labeling_job: str = proto.Field(
        proto.STRING,
        number=2,
        oneof="reference",
    )
    method: str = proto.Field(
        proto.STRING,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
