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


__protobuf__ = proto.module(
    package="google.cloud.bigquery.v2",
    manifest={
        "TableReference",
    },
)


class TableReference(proto.Message):
    r"""

    Attributes:
        project_id (str):
            Required. The ID of the project containing
            this table.
        dataset_id (str):
            Required. The ID of the dataset containing
            this table.
        table_id (str):
            Required. The ID of the table. The ID must contain only
            letters (a-z, A-Z), numbers (0-9), or underscores (_). The
            maximum length is 1,024 characters. Certain operations allow
            suffixing of the table ID with a partition decorator, such
            as ``sample_table$20190123``.
        project_id_alternative (Sequence[str]):
            The alternative field that will be used when ESF is not able
            to translate the received data to the project_id field.
        dataset_id_alternative (Sequence[str]):
            The alternative field that will be used when ESF is not able
            to translate the received data to the project_id field.
        table_id_alternative (Sequence[str]):
            The alternative field that will be used when ESF is not able
            to translate the received data to the project_id field.
    """

    project_id = proto.Field(
        proto.STRING,
        number=1,
    )
    dataset_id = proto.Field(
        proto.STRING,
        number=2,
    )
    table_id = proto.Field(
        proto.STRING,
        number=3,
    )
    project_id_alternative = proto.RepeatedField(
        proto.STRING,
        number=4,
    )
    dataset_id_alternative = proto.RepeatedField(
        proto.STRING,
        number=5,
    )
    table_id_alternative = proto.RepeatedField(
        proto.STRING,
        number=6,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
