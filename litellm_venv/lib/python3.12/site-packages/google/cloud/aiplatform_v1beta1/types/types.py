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
        "BoolArray",
        "DoubleArray",
        "Int64Array",
        "StringArray",
        "Tensor",
    },
)


class BoolArray(proto.Message):
    r"""A list of boolean values.

    Attributes:
        values (MutableSequence[bool]):
            A list of bool values.
    """

    values: MutableSequence[bool] = proto.RepeatedField(
        proto.BOOL,
        number=1,
    )


class DoubleArray(proto.Message):
    r"""A list of double values.

    Attributes:
        values (MutableSequence[float]):
            A list of double values.
    """

    values: MutableSequence[float] = proto.RepeatedField(
        proto.DOUBLE,
        number=1,
    )


class Int64Array(proto.Message):
    r"""A list of int64 values.

    Attributes:
        values (MutableSequence[int]):
            A list of int64 values.
    """

    values: MutableSequence[int] = proto.RepeatedField(
        proto.INT64,
        number=1,
    )


class StringArray(proto.Message):
    r"""A list of string values.

    Attributes:
        values (MutableSequence[str]):
            A list of string values.
    """

    values: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=1,
    )


class Tensor(proto.Message):
    r"""A tensor value type.

    Attributes:
        dtype (google.cloud.aiplatform_v1beta1.types.Tensor.DataType):
            The data type of tensor.
        shape (MutableSequence[int]):
            Shape of the tensor.
        bool_val (MutableSequence[bool]):
            Type specific representations that make it easy to create
            tensor protos in all languages. Only the representation
            corresponding to "dtype" can be set. The values hold the
            flattened representation of the tensor in row major order.

            [BOOL][google.aiplatform.master.Tensor.DataType.BOOL]
        string_val (MutableSequence[str]):
            [STRING][google.aiplatform.master.Tensor.DataType.STRING]
        bytes_val (MutableSequence[bytes]):
            [STRING][google.aiplatform.master.Tensor.DataType.STRING]
        float_val (MutableSequence[float]):
            [FLOAT][google.aiplatform.master.Tensor.DataType.FLOAT]
        double_val (MutableSequence[float]):
            [DOUBLE][google.aiplatform.master.Tensor.DataType.DOUBLE]
        int_val (MutableSequence[int]):
            [INT_8][google.aiplatform.master.Tensor.DataType.INT8]
            [INT_16][google.aiplatform.master.Tensor.DataType.INT16]
            [INT_32][google.aiplatform.master.Tensor.DataType.INT32]
        int64_val (MutableSequence[int]):
            [INT64][google.aiplatform.master.Tensor.DataType.INT64]
        uint_val (MutableSequence[int]):
            [UINT8][google.aiplatform.master.Tensor.DataType.UINT8]
            [UINT16][google.aiplatform.master.Tensor.DataType.UINT16]
            [UINT32][google.aiplatform.master.Tensor.DataType.UINT32]
        uint64_val (MutableSequence[int]):
            [UINT64][google.aiplatform.master.Tensor.DataType.UINT64]
        list_val (MutableSequence[google.cloud.aiplatform_v1beta1.types.Tensor]):
            A list of tensor values.
        struct_val (MutableMapping[str, google.cloud.aiplatform_v1beta1.types.Tensor]):
            A map of string to tensor.
        tensor_val (bytes):
            Serialized raw tensor content.
    """

    class DataType(proto.Enum):
        r"""Data type of the tensor.

        Values:
            DATA_TYPE_UNSPECIFIED (0):
                Not a legal value for DataType. Used to
                indicate a DataType field has not been set.
            BOOL (1):
                Data types that all computation devices are
                expected to be capable to support.
            STRING (2):
                No description available.
            FLOAT (3):
                No description available.
            DOUBLE (4):
                No description available.
            INT8 (5):
                No description available.
            INT16 (6):
                No description available.
            INT32 (7):
                No description available.
            INT64 (8):
                No description available.
            UINT8 (9):
                No description available.
            UINT16 (10):
                No description available.
            UINT32 (11):
                No description available.
            UINT64 (12):
                No description available.
        """
        DATA_TYPE_UNSPECIFIED = 0
        BOOL = 1
        STRING = 2
        FLOAT = 3
        DOUBLE = 4
        INT8 = 5
        INT16 = 6
        INT32 = 7
        INT64 = 8
        UINT8 = 9
        UINT16 = 10
        UINT32 = 11
        UINT64 = 12

    dtype: DataType = proto.Field(
        proto.ENUM,
        number=1,
        enum=DataType,
    )
    shape: MutableSequence[int] = proto.RepeatedField(
        proto.INT64,
        number=2,
    )
    bool_val: MutableSequence[bool] = proto.RepeatedField(
        proto.BOOL,
        number=3,
    )
    string_val: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=14,
    )
    bytes_val: MutableSequence[bytes] = proto.RepeatedField(
        proto.BYTES,
        number=15,
    )
    float_val: MutableSequence[float] = proto.RepeatedField(
        proto.FLOAT,
        number=5,
    )
    double_val: MutableSequence[float] = proto.RepeatedField(
        proto.DOUBLE,
        number=6,
    )
    int_val: MutableSequence[int] = proto.RepeatedField(
        proto.INT32,
        number=7,
    )
    int64_val: MutableSequence[int] = proto.RepeatedField(
        proto.INT64,
        number=8,
    )
    uint_val: MutableSequence[int] = proto.RepeatedField(
        proto.UINT32,
        number=9,
    )
    uint64_val: MutableSequence[int] = proto.RepeatedField(
        proto.UINT64,
        number=10,
    )
    list_val: MutableSequence["Tensor"] = proto.RepeatedField(
        proto.MESSAGE,
        number=11,
        message="Tensor",
    )
    struct_val: MutableMapping[str, "Tensor"] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=12,
        message="Tensor",
    )
    tensor_val: bytes = proto.Field(
        proto.BYTES,
        number=13,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
