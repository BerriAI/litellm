# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import typing
from typing import Any, Dict, Iterable, List, Optional

from google.cloud.bigquery.enums import StandardSqlTypeNames


class StandardSqlDataType:
    """The type of a variable, e.g., a function argument.

    See:
    https://cloud.google.com/bigquery/docs/reference/rest/v2/StandardSqlDataType

    Examples:

    .. code-block:: text

        INT64: {type_kind="INT64"}
        ARRAY: {type_kind="ARRAY", array_element_type="STRING"}
        STRUCT<x STRING, y ARRAY>: {
            type_kind="STRUCT",
            struct_type={
                fields=[
                    {name="x", type={type_kind="STRING"}},
                    {
                        name="y",
                        type={type_kind="ARRAY", array_element_type="DATE"}
                    }
                ]
            }
        }
        RANGE: {type_kind="RANGE", range_element_type="DATETIME"}

    Args:
        type_kind:
            The top level type of this field. Can be any standard SQL data type,
            e.g. INT64, DATE, ARRAY.
        array_element_type:
            The type of the array's elements, if type_kind is ARRAY.
        struct_type:
            The fields of this struct, in order, if type_kind is STRUCT.
        range_element_type:
            The type of the range's elements, if type_kind is RANGE.
    """

    def __init__(
        self,
        type_kind: Optional[
            StandardSqlTypeNames
        ] = StandardSqlTypeNames.TYPE_KIND_UNSPECIFIED,
        array_element_type: Optional["StandardSqlDataType"] = None,
        struct_type: Optional["StandardSqlStructType"] = None,
        range_element_type: Optional["StandardSqlDataType"] = None,
    ):
        self._properties: Dict[str, Any] = {}

        self.type_kind = type_kind
        self.array_element_type = array_element_type
        self.struct_type = struct_type
        self.range_element_type = range_element_type

    @property
    def type_kind(self) -> Optional[StandardSqlTypeNames]:
        """The top level type of this field.

        Can be any standard SQL data type, e.g. INT64, DATE, ARRAY.
        """
        kind = self._properties["typeKind"]
        return StandardSqlTypeNames[kind]  # pytype: disable=missing-parameter

    @type_kind.setter
    def type_kind(self, value: Optional[StandardSqlTypeNames]):
        if not value:
            kind = StandardSqlTypeNames.TYPE_KIND_UNSPECIFIED.value
        else:
            kind = value.value
        self._properties["typeKind"] = kind

    @property
    def array_element_type(self) -> Optional["StandardSqlDataType"]:
        """The type of the array's elements, if type_kind is ARRAY."""
        element_type = self._properties.get("arrayElementType")

        if element_type is None:
            return None

        result = StandardSqlDataType()
        result._properties = element_type  # We do not use a copy on purpose.
        return result

    @array_element_type.setter
    def array_element_type(self, value: Optional["StandardSqlDataType"]):
        element_type = None if value is None else value.to_api_repr()

        if element_type is None:
            self._properties.pop("arrayElementType", None)
        else:
            self._properties["arrayElementType"] = element_type

    @property
    def struct_type(self) -> Optional["StandardSqlStructType"]:
        """The fields of this struct, in order, if type_kind is STRUCT."""
        struct_info = self._properties.get("structType")

        if struct_info is None:
            return None

        result = StandardSqlStructType()
        result._properties = struct_info  # We do not use a copy on purpose.
        return result

    @struct_type.setter
    def struct_type(self, value: Optional["StandardSqlStructType"]):
        struct_type = None if value is None else value.to_api_repr()

        if struct_type is None:
            self._properties.pop("structType", None)
        else:
            self._properties["structType"] = struct_type

    @property
    def range_element_type(self) -> Optional["StandardSqlDataType"]:
        """The type of the range's elements, if type_kind = "RANGE". Must be
        one of DATETIME, DATE, or TIMESTAMP."""
        range_element_info = self._properties.get("rangeElementType")

        if range_element_info is None:
            return None

        result = StandardSqlDataType()
        result._properties = range_element_info  # We do not use a copy on purpose.
        return result

    @range_element_type.setter
    def range_element_type(self, value: Optional["StandardSqlDataType"]):
        range_element_type = None if value is None else value.to_api_repr()

        if range_element_type is None:
            self._properties.pop("rangeElementType", None)
        else:
            self._properties["rangeElementType"] = range_element_type

    def to_api_repr(self) -> Dict[str, Any]:
        """Construct the API resource representation of this SQL data type."""
        return copy.deepcopy(self._properties)

    @classmethod
    def from_api_repr(cls, resource: Dict[str, Any]):
        """Construct an SQL data type instance given its API representation."""
        type_kind = resource.get("typeKind")
        if type_kind not in StandardSqlTypeNames.__members__:
            type_kind = StandardSqlTypeNames.TYPE_KIND_UNSPECIFIED
        else:
            # Convert string to an enum member.
            type_kind = StandardSqlTypeNames[  # pytype: disable=missing-parameter
                typing.cast(str, type_kind)
            ]

        array_element_type = None
        if type_kind == StandardSqlTypeNames.ARRAY:
            element_type = resource.get("arrayElementType")
            if element_type:
                array_element_type = cls.from_api_repr(element_type)

        struct_type = None
        if type_kind == StandardSqlTypeNames.STRUCT:
            struct_info = resource.get("structType")
            if struct_info:
                struct_type = StandardSqlStructType.from_api_repr(struct_info)

        range_element_type = None
        if type_kind == StandardSqlTypeNames.RANGE:
            range_element_info = resource.get("rangeElementType")
            if range_element_info:
                range_element_type = cls.from_api_repr(range_element_info)

        return cls(type_kind, array_element_type, struct_type, range_element_type)

    def __eq__(self, other):
        if not isinstance(other, StandardSqlDataType):
            return NotImplemented
        else:
            return (
                self.type_kind == other.type_kind
                and self.array_element_type == other.array_element_type
                and self.struct_type == other.struct_type
                and self.range_element_type == other.range_element_type
            )

    def __str__(self):
        result = f"{self.__class__.__name__}(type_kind={self.type_kind!r}, ...)"
        return result


class StandardSqlField:
    """A field or a column.

    See:
    https://cloud.google.com/bigquery/docs/reference/rest/v2/StandardSqlField

    Args:
        name:
            The name of this field. Can be absent for struct fields.
        type:
            The type of this parameter. Absent if not explicitly specified.

            For example, CREATE FUNCTION statement can omit the return type; in this
            case the output parameter does not have this "type" field).
    """

    def __init__(
        self, name: Optional[str] = None, type: Optional[StandardSqlDataType] = None
    ):
        type_repr = None if type is None else type.to_api_repr()
        self._properties = {"name": name, "type": type_repr}

    @property
    def name(self) -> Optional[str]:
        """The name of this field. Can be absent for struct fields."""
        return typing.cast(Optional[str], self._properties["name"])

    @name.setter
    def name(self, value: Optional[str]):
        self._properties["name"] = value

    @property
    def type(self) -> Optional[StandardSqlDataType]:
        """The type of this parameter. Absent if not explicitly specified.

        For example, CREATE FUNCTION statement can omit the return type; in this
        case the output parameter does not have this "type" field).
        """
        type_info = self._properties["type"]

        if type_info is None:
            return None

        result = StandardSqlDataType()
        # We do not use a properties copy on purpose.
        result._properties = typing.cast(Dict[str, Any], type_info)

        return result

    @type.setter
    def type(self, value: Optional[StandardSqlDataType]):
        value_repr = None if value is None else value.to_api_repr()
        self._properties["type"] = value_repr

    def to_api_repr(self) -> Dict[str, Any]:
        """Construct the API resource representation of this SQL field."""
        return copy.deepcopy(self._properties)

    @classmethod
    def from_api_repr(cls, resource: Dict[str, Any]):
        """Construct an SQL field instance given its API representation."""
        result = cls(
            name=resource.get("name"),
            type=StandardSqlDataType.from_api_repr(resource.get("type", {})),
        )
        return result

    def __eq__(self, other):
        if not isinstance(other, StandardSqlField):
            return NotImplemented
        else:
            return self.name == other.name and self.type == other.type


class StandardSqlStructType:
    """Type of a struct field.

    See:
    https://cloud.google.com/bigquery/docs/reference/rest/v2/StandardSqlDataType#StandardSqlStructType

    Args:
        fields: The fields in this struct.
    """

    def __init__(self, fields: Optional[Iterable[StandardSqlField]] = None):
        if fields is None:
            fields = []
        self._properties = {"fields": [field.to_api_repr() for field in fields]}

    @property
    def fields(self) -> List[StandardSqlField]:
        """The fields in this struct."""
        result = []

        for field_resource in self._properties.get("fields", []):
            field = StandardSqlField()
            field._properties = field_resource  # We do not use a copy on purpose.
            result.append(field)

        return result

    @fields.setter
    def fields(self, value: Iterable[StandardSqlField]):
        self._properties["fields"] = [field.to_api_repr() for field in value]

    def to_api_repr(self) -> Dict[str, Any]:
        """Construct the API resource representation of this SQL struct type."""
        return copy.deepcopy(self._properties)

    @classmethod
    def from_api_repr(cls, resource: Dict[str, Any]) -> "StandardSqlStructType":
        """Construct an SQL struct type instance given its API representation."""
        fields = (
            StandardSqlField.from_api_repr(field_resource)
            for field_resource in resource.get("fields", [])
        )
        return cls(fields=fields)

    def __eq__(self, other):
        if not isinstance(other, StandardSqlStructType):
            return NotImplemented
        else:
            return self.fields == other.fields


class StandardSqlTableType:
    """A table type.

    See:
    https://cloud.google.com/workflows/docs/reference/googleapis/bigquery/v2/Overview#StandardSqlTableType

    Args:
        columns: The columns in this table type.
    """

    def __init__(self, columns: Iterable[StandardSqlField]):
        self._properties = {"columns": [col.to_api_repr() for col in columns]}

    @property
    def columns(self) -> List[StandardSqlField]:
        """The columns in this table type."""
        result = []

        for column_resource in self._properties.get("columns", []):
            column = StandardSqlField()
            column._properties = column_resource  # We do not use a copy on purpose.
            result.append(column)

        return result

    @columns.setter
    def columns(self, value: Iterable[StandardSqlField]):
        self._properties["columns"] = [col.to_api_repr() for col in value]

    def to_api_repr(self) -> Dict[str, Any]:
        """Construct the API resource representation of this SQL table type."""
        return copy.deepcopy(self._properties)

    @classmethod
    def from_api_repr(cls, resource: Dict[str, Any]) -> "StandardSqlTableType":
        """Construct an SQL table type instance given its API representation."""
        columns = []

        for column_resource in resource.get("columns", []):
            type_ = column_resource.get("type")
            if type_ is None:
                type_ = {}

            column = StandardSqlField(
                name=column_resource.get("name"),
                type=StandardSqlDataType.from_api_repr(type_),
            )
            columns.append(column)

        return cls(columns=columns)

    def __eq__(self, other):
        if not isinstance(other, StandardSqlTableType):
            return NotImplemented
        else:
            return self.columns == other.columns
