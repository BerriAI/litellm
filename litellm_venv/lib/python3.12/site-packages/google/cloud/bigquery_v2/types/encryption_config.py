# -*- coding: utf-8 -*-
# Copyright 2022 Google LLC
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
import proto  # type: ignore

from google.protobuf import wrappers_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.bigquery.v2",
    manifest={
        "EncryptionConfiguration",
    },
)


class EncryptionConfiguration(proto.Message):
    r"""

    Attributes:
        kms_key_name (google.protobuf.wrappers_pb2.StringValue):
            Optional. Describes the Cloud KMS encryption
            key that will be used to protect destination
            BigQuery table. The BigQuery Service Account
            associated with your project requires access to
            this encryption key.
    """

    kms_key_name = proto.Field(
        proto.MESSAGE,
        number=1,
        message=wrappers_pb2.StringValue,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
