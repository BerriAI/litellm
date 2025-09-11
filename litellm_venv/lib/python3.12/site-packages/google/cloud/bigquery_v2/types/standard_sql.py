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
        "StandardSqlDataType",
        "StandardSqlField",
        "StandardSqlStructType",
        "StandardSqlTableType",
    },
)


class StandardSqlDataType(proto.Message):
    r"""The type of a variable, e.g., a function argument. Examples: INT64:
    {type_kind="INT64"} ARRAY: {type_kind="ARRAY",
    array_element_type="STRING"} STRUCT<x STRING, y ARRAY>:
    {type_kind="STRUCT", struct_type={fields=[ {name="x",
    type={type_kind="STRING"}}, {name="y", type={type_kind="ARRAY",
    array_element_type="DATE"}} ]}}

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        type_kind (google.cloud.bigquery_v2.types.StandardSqlDataType.TypeKind):
            Required. The top level type of this field.
            Can be any standard SQL data type (e.g.,
            "INT64", "DATE", "ARRAY").
        array_element_type (google.cloud.bigquery_v2.types.StandardSqlDataType):
            The type of the array's elements, if type_kind = "ARRAY".

            This field is a member of `oneof`_ ``sub_type``.
        struct_type (google.cloud.bigquery_v2.types.StandardSqlStructType):
            The fields of this struct, in order, if type_kind =
            "STRUCT".

            This field is a member of `oneof`_ ``sub_type``.
    """

    class TypeKind(proto.Enum):
        r""""""
        TYPE_KIND_UNSPECIFIED = 0
        INT64 = 2
        BOOL = 5
        FLOAT64 = 7
        STRING = 8
        BYTES = 9
        TIMESTAMP = 19
        DATE = 10
        TIME = 20
        DATETIME = 21
        INTERVAL = 26
        GEOGRAPHY = 22
        NUMERIC = 23
        BIGNUMERIC = 24
        JSON = 25
        ARRAY = 16
        STRUCT = 17

    type_kind = proto.Field(
        proto.ENUM,
        number=1,
        enum=TypeKind,
    )
    array_element_type = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="sub_type",
        message="StandardSqlDataType",
    )
    struct_type = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="sub_type",
        message="StandardSqlStructType",
    )


class StandardSqlField(proto.Message):
    r"""A field or a column.

    Attributes:
        name (str):
            Optional. The name of this field. Can be
            absent for struct fields.
        type (google.cloud.bigquery_v2.types.StandardSqlDataType):
            Optional. The type of this parameter. Absent
            if not explicitly specified (e.g., CREATE
            FUNCTION statement can omit the return type; in
            this case the output parameter does not have
            this "type" field).
    """

    name = proto.Field(
        proto.STRING,
        number=1,
    )
    type = proto.Field(
        proto.MESSAGE,
        number=2,
        message="StandardSqlDataType",
    )


class StandardSqlStructType(proto.Message):
    r"""

    Attributes:
        fields (Sequence[google.cloud.bigquery_v2.types.StandardSqlField]):

    """

    fields = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="StandardSqlField",
    )


class StandardSqlTableType(proto.Message):
    r"""A table type

    Attributes:
        columns (Sequence[google.cloud.bigquery_v2.types.StandardSqlField]):
            The columns in this table type
    """

    columns = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="StandardSqlField",
    )


__all__ = tuple(sorted(__protobuf__.manifest))
