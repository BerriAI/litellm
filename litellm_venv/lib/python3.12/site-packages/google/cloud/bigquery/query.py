# Copyright 2015 Google LLC
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

"""BigQuery query processing."""

from collections import OrderedDict
import copy
import datetime
import decimal
from typing import Any, cast, Optional, Dict, Union

from google.cloud.bigquery.table import _parse_schema_resource
from google.cloud.bigquery import _helpers
from google.cloud.bigquery._helpers import _rows_from_json
from google.cloud.bigquery._helpers import _SCALAR_VALUE_TO_JSON_PARAM
from google.cloud.bigquery._helpers import _SUPPORTED_RANGE_ELEMENTS


_SCALAR_VALUE_TYPE = Optional[
    Union[str, int, float, decimal.Decimal, bool, datetime.datetime, datetime.date]
]


class ConnectionProperty:
    """A connection-level property to customize query behavior.

    See
    https://cloud.google.com/bigquery/docs/reference/rest/v2/ConnectionProperty

    Args:
        key:
            The key of the property to set, for example, ``'time_zone'`` or
            ``'session_id'``.
        value: The value of the property to set.
    """

    def __init__(self, key: str = "", value: str = ""):
        self._properties = {
            "key": key,
            "value": value,
        }

    @property
    def key(self) -> str:
        """Name of the property.

        For example:

        * ``time_zone``
        * ``session_id``
        """
        return self._properties["key"]

    @property
    def value(self) -> str:
        """Value of the property."""
        return self._properties["value"]

    @classmethod
    def from_api_repr(cls, resource) -> "ConnectionProperty":
        """Construct :class:`~google.cloud.bigquery.query.ConnectionProperty`
        from JSON resource.

        Args:
            resource: JSON representation.

        Returns:
            A connection property.
        """
        value = cls()
        value._properties = resource
        return value

    def to_api_repr(self) -> Dict[str, Any]:
        """Construct JSON API representation for the connection property.

        Returns:
            JSON mapping
        """
        return self._properties


class UDFResource(object):
    """Describe a single user-defined function (UDF) resource.

    Args:
        udf_type (str): The type of the resource ('inlineCode' or 'resourceUri')

        value (str): The inline code or resource URI.

    See:
    https://cloud.google.com/bigquery/user-defined-functions#api
    """

    def __init__(self, udf_type, value):
        self.udf_type = udf_type
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, UDFResource):
            return NotImplemented
        return self.udf_type == other.udf_type and self.value == other.value

    def __ne__(self, other):
        return not self == other


class _AbstractQueryParameterType:
    """Base class for representing query parameter types.

    https://cloud.google.com/bigquery/docs/reference/rest/v2/QueryParameter#queryparametertype
    """

    @classmethod
    def from_api_repr(cls, resource):
        """Factory: construct parameter type from JSON resource.

        Args:
            resource (Dict): JSON mapping of parameter

        Returns:
            google.cloud.bigquery.query.QueryParameterType: Instance
        """
        raise NotImplementedError

    def to_api_repr(self):
        """Construct JSON API representation for the parameter type.

        Returns:
            Dict: JSON mapping
        """
        raise NotImplementedError


class ScalarQueryParameterType(_AbstractQueryParameterType):
    """Type representation for scalar query parameters.

    Args:
        type_ (str):
            One of 'STRING', 'INT64', 'FLOAT64', 'NUMERIC', 'BOOL', 'TIMESTAMP',
            'DATETIME', or 'DATE'.
        name (Optional[str]):
            The name of the query parameter. Primarily used if the type is
            one of the subfields in ``StructQueryParameterType`` instance.
        description (Optional[str]):
            The query parameter description. Primarily used if the type is
            one of the subfields in ``StructQueryParameterType`` instance.
    """

    def __init__(self, type_, *, name=None, description=None):
        self._type = type_
        self.name = name
        self.description = description

    @classmethod
    def from_api_repr(cls, resource):
        """Factory: construct parameter type from JSON resource.

        Args:
            resource (Dict): JSON mapping of parameter

        Returns:
            google.cloud.bigquery.query.ScalarQueryParameterType: Instance
        """
        type_ = resource["type"]
        return cls(type_)

    def to_api_repr(self):
        """Construct JSON API representation for the parameter type.

        Returns:
            Dict: JSON mapping
        """
        # Name and description are only used if the type is a field inside a struct
        # type, but it's StructQueryParameterType's responsibilty to use these two
        # attributes in the API representation when needed. Here we omit them.
        return {"type": self._type}

    def with_name(self, new_name: Union[str, None]):
        """Return a copy of the instance with ``name`` set to ``new_name``.

        Args:
            name (Union[str, None]):
                The new name of the query parameter type. If ``None``, the existing
                name is cleared.

        Returns:
            google.cloud.bigquery.query.ScalarQueryParameterType:
               A new instance with updated name.
        """
        return type(self)(self._type, name=new_name, description=self.description)

    def __repr__(self):
        name = f", name={self.name!r}" if self.name is not None else ""
        description = (
            f", description={self.description!r}"
            if self.description is not None
            else ""
        )
        return f"{self.__class__.__name__}({self._type!r}{name}{description})"


class ArrayQueryParameterType(_AbstractQueryParameterType):
    """Type representation for array query parameters.

    Args:
        array_type (Union[ScalarQueryParameterType, StructQueryParameterType]):
            The type of array elements.
        name (Optional[str]):
            The name of the query parameter. Primarily used if the type is
            one of the subfields in ``StructQueryParameterType`` instance.
        description (Optional[str]):
            The query parameter description. Primarily used if the type is
            one of the subfields in ``StructQueryParameterType`` instance.
    """

    def __init__(self, array_type, *, name=None, description=None):
        self._array_type = array_type
        self.name = name
        self.description = description

    @classmethod
    def from_api_repr(cls, resource):
        """Factory: construct parameter type from JSON resource.

        Args:
            resource (Dict): JSON mapping of parameter

        Returns:
            google.cloud.bigquery.query.ArrayQueryParameterType: Instance
        """
        array_item_type = resource["arrayType"]["type"]

        if array_item_type in {"STRUCT", "RECORD"}:
            klass = StructQueryParameterType
        else:
            klass = ScalarQueryParameterType

        item_type_instance = klass.from_api_repr(resource["arrayType"])
        return cls(item_type_instance)

    def to_api_repr(self):
        """Construct JSON API representation for the parameter type.

        Returns:
            Dict: JSON mapping
        """
        # Name and description are only used if the type is a field inside a struct
        # type, but it's StructQueryParameterType's responsibilty to use these two
        # attributes in the API representation when needed. Here we omit them.
        return {
            "type": "ARRAY",
            "arrayType": self._array_type.to_api_repr(),
        }

    def __repr__(self):
        name = f", name={self.name!r}" if self.name is not None else ""
        description = (
            f", description={self.description!r}"
            if self.description is not None
            else ""
        )
        return f"{self.__class__.__name__}({self._array_type!r}{name}{description})"


class StructQueryParameterType(_AbstractQueryParameterType):
    """Type representation for struct query parameters.

    Args:
        fields (Iterable[Union[ \
            ArrayQueryParameterType, ScalarQueryParameterType, StructQueryParameterType \
        ]]):
            An non-empty iterable describing the struct's field types.
        name (Optional[str]):
            The name of the query parameter. Primarily used if the type is
            one of the subfields in ``StructQueryParameterType`` instance.
        description (Optional[str]):
            The query parameter description. Primarily used if the type is
            one of the subfields in ``StructQueryParameterType`` instance.
    """

    def __init__(self, *fields, name=None, description=None):
        if not fields:
            raise ValueError("Struct type must have at least one field defined.")

        self._fields = fields  # fields is a tuple (immutable), no shallow copy needed
        self.name = name
        self.description = description

    @property
    def fields(self):
        return self._fields  # no copy needed, self._fields is an immutable sequence

    @classmethod
    def from_api_repr(cls, resource):
        """Factory: construct parameter type from JSON resource.

        Args:
            resource (Dict): JSON mapping of parameter

        Returns:
            google.cloud.bigquery.query.StructQueryParameterType: Instance
        """
        fields = []

        for struct_field in resource["structTypes"]:
            type_repr = struct_field["type"]
            if type_repr["type"] in {"STRUCT", "RECORD"}:
                klass = StructQueryParameterType
            elif type_repr["type"] == "ARRAY":
                klass = ArrayQueryParameterType
            else:
                klass = ScalarQueryParameterType

            type_instance = klass.from_api_repr(type_repr)
            type_instance.name = struct_field.get("name")
            type_instance.description = struct_field.get("description")
            fields.append(type_instance)

        return cls(*fields)

    def to_api_repr(self):
        """Construct JSON API representation for the parameter type.

        Returns:
            Dict: JSON mapping
        """
        fields = []

        for field in self._fields:
            item = {"type": field.to_api_repr()}
            if field.name is not None:
                item["name"] = field.name
            if field.description is not None:
                item["description"] = field.description

            fields.append(item)

        return {
            "type": "STRUCT",
            "structTypes": fields,
        }

    def __repr__(self):
        name = f", name={self.name!r}" if self.name is not None else ""
        description = (
            f", description={self.description!r}"
            if self.description is not None
            else ""
        )
        items = ", ".join(repr(field) for field in self._fields)
        return f"{self.__class__.__name__}({items}{name}{description})"


class RangeQueryParameterType(_AbstractQueryParameterType):
    """Type representation for range query parameters.

    Args:
        type_ (Union[ScalarQueryParameterType, str]):
            Type of range element, must be one of 'TIMESTAMP', 'DATETIME', or
            'DATE'.
        name (Optional[str]):
            The name of the query parameter. Primarily used if the type is
            one of the subfields in ``StructQueryParameterType`` instance.
        description (Optional[str]):
            The query parameter description. Primarily used if the type is
            one of the subfields in ``StructQueryParameterType`` instance.
    """

    @classmethod
    def _parse_range_element_type(self, type_):
        """Helper method that parses the input range element type, which may
        be a string, or a ScalarQueryParameterType object.

        Returns:
            google.cloud.bigquery.query.ScalarQueryParameterType: Instance
        """
        if isinstance(type_, str):
            if type_ not in _SUPPORTED_RANGE_ELEMENTS:
                raise ValueError(
                    "If given as a string, range element type must be one of "
                    "'TIMESTAMP', 'DATE', or 'DATETIME'."
                )
            return ScalarQueryParameterType(type_)
        elif isinstance(type_, ScalarQueryParameterType):
            if type_._type not in _SUPPORTED_RANGE_ELEMENTS:
                raise ValueError(
                    "If given as a ScalarQueryParameter object, range element "
                    "type must be one of 'TIMESTAMP', 'DATE', or 'DATETIME' "
                    "type."
                )
            return type_
        else:
            raise ValueError(
                "range_type must be a string or ScalarQueryParameter object, "
                "of 'TIMESTAMP', 'DATE', or 'DATETIME' type."
            )

    def __init__(self, type_, *, name=None, description=None):
        self.type_ = self._parse_range_element_type(type_)
        self.name = name
        self.description = description

    @classmethod
    def from_api_repr(cls, resource):
        """Factory: construct parameter type from JSON resource.

        Args:
            resource (Dict): JSON mapping of parameter

        Returns:
            google.cloud.bigquery.query.RangeQueryParameterType: Instance
        """
        type_ = resource["rangeElementType"]["type"]
        name = resource.get("name")
        description = resource.get("description")

        return cls(type_, name=name, description=description)

    def to_api_repr(self):
        """Construct JSON API representation for the parameter type.

        Returns:
            Dict: JSON mapping
        """
        # Name and description are only used if the type is a field inside a struct
        # type, but it's StructQueryParameterType's responsibilty to use these two
        # attributes in the API representation when needed. Here we omit them.
        return {
            "type": "RANGE",
            "rangeElementType": self.type_.to_api_repr(),
        }

    def with_name(self, new_name: Union[str, None]):
        """Return a copy of the instance with ``name`` set to ``new_name``.

        Args:
            name (Union[str, None]):
                The new name of the range query parameter type. If ``None``,
                the existing name is cleared.

        Returns:
            google.cloud.bigquery.query.RangeQueryParameterType:
               A new instance with updated name.
        """
        return type(self)(self.type_, name=new_name, description=self.description)

    def __repr__(self):
        name = f", name={self.name!r}" if self.name is not None else ""
        description = (
            f", description={self.description!r}"
            if self.description is not None
            else ""
        )
        return f"{self.__class__.__name__}({self.type_!r}{name}{description})"

    def _key(self):
        """A tuple key that uniquely describes this field.

        Used to compute this instance's hashcode and evaluate equality.

        Returns:
            Tuple: The contents of this
            :class:`~google.cloud.bigquery.query.RangeQueryParameterType`.
        """
        type_ = self.type_.to_api_repr()
        return (self.name, type_, self.description)

    def __eq__(self, other):
        if not isinstance(other, RangeQueryParameterType):
            return NotImplemented
        return self._key() == other._key()

    def __ne__(self, other):
        return not self == other


class _AbstractQueryParameter(object):
    """Base class for named / positional query parameters."""

    @classmethod
    def from_api_repr(cls, resource: dict) -> "_AbstractQueryParameter":
        """Factory: construct parameter from JSON resource.

        Args:
            resource (Dict): JSON mapping of parameter

        Returns:
            A new instance of _AbstractQueryParameter subclass.
        """
        raise NotImplementedError

    def to_api_repr(self) -> dict:
        """Construct JSON API representation for the parameter.

        Returns:
            Dict: JSON representation for the parameter.
        """
        raise NotImplementedError


class ScalarQueryParameter(_AbstractQueryParameter):
    """Named / positional query parameters for scalar values.

    Args:
        name:
            Parameter name, used via ``@foo`` syntax.  If None, the
            parameter can only be addressed via position (``?``).

        type_:
            Name of parameter type. See
            :class:`google.cloud.bigquery.enums.SqlTypeNames` and
            :class:`google.cloud.bigquery.query.SqlParameterScalarTypes` for
            supported types.

        value:
            The scalar parameter value.
    """

    def __init__(
        self,
        name: Optional[str],
        type_: Optional[Union[str, ScalarQueryParameterType]],
        value: _SCALAR_VALUE_TYPE,
    ):
        self.name = name
        if isinstance(type_, ScalarQueryParameterType):
            self.type_ = type_._type
        else:
            self.type_ = type_
        self.value = value

    @classmethod
    def positional(
        cls, type_: Union[str, ScalarQueryParameterType], value: _SCALAR_VALUE_TYPE
    ) -> "ScalarQueryParameter":
        """Factory for positional paramater.

        Args:
            type_:
                Name of parameter type.  One of 'STRING', 'INT64',
                'FLOAT64', 'NUMERIC', 'BIGNUMERIC', 'BOOL', 'TIMESTAMP', 'DATETIME', or
                'DATE'.

            value:
                The scalar parameter value.

        Returns:
            google.cloud.bigquery.query.ScalarQueryParameter: Instance without name
        """
        return cls(None, type_, value)

    @classmethod
    def from_api_repr(cls, resource: dict) -> "ScalarQueryParameter":
        """Factory: construct parameter from JSON resource.

        Args:
            resource (Dict): JSON mapping of parameter

        Returns:
            google.cloud.bigquery.query.ScalarQueryParameter: Instance
        """
        # Import here to avoid circular imports.
        from google.cloud.bigquery import schema

        name = resource.get("name")
        type_ = resource["parameterType"]["type"]

        # parameterValue might not be present if JSON resource originates
        # from the back-end - the latter omits it for None values.
        value = resource.get("parameterValue", {}).get("value")
        if value is not None:
            converted = _helpers.SCALAR_QUERY_PARAM_PARSER.to_py(
                value, schema.SchemaField(cast(str, name), type_)
            )
        else:
            converted = None

        return cls(name, type_, converted)

    def to_api_repr(self) -> dict:
        """Construct JSON API representation for the parameter.

        Returns:
            Dict: JSON mapping
        """
        value = self.value
        converter = _SCALAR_VALUE_TO_JSON_PARAM.get(self.type_, lambda value: value)
        value = converter(value)  # type: ignore
        resource: Dict[str, Any] = {
            "parameterType": {"type": self.type_},
            "parameterValue": {"value": value},
        }
        if self.name is not None:
            resource["name"] = self.name
        return resource

    def _key(self):
        """A tuple key that uniquely describes this field.

        Used to compute this instance's hashcode and evaluate equality.

        Returns:
            Tuple: The contents of this :class:`~google.cloud.bigquery.query.ScalarQueryParameter`.
        """
        return (self.name, self.type_.upper(), self.value)

    def __eq__(self, other):
        if not isinstance(other, ScalarQueryParameter):
            return NotImplemented
        return self._key() == other._key()

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "ScalarQueryParameter{}".format(self._key())


class ArrayQueryParameter(_AbstractQueryParameter):
    """Named / positional query parameters for array values.

    Args:
        name (Optional[str]):
            Parameter name, used via ``@foo`` syntax.  If None, the
            parameter can only be addressed via position (``?``).

        array_type (Union[str, ScalarQueryParameterType, StructQueryParameterType]):
            The type of array elements. If given as a string, it must be one of
            `'STRING'`, `'INT64'`, `'FLOAT64'`, `'NUMERIC'`, `'BIGNUMERIC'`, `'BOOL'`,
            `'TIMESTAMP'`, `'DATE'`, or `'STRUCT'`/`'RECORD'`.
            If the type is ``'STRUCT'``/``'RECORD'`` and ``values`` is empty,
            the exact item type cannot be deduced, thus a ``StructQueryParameterType``
            instance needs to be passed in.

        values (List[appropriate type]): The parameter array values.
    """

    def __init__(self, name, array_type, values) -> None:
        self.name = name
        self.values = values

        if isinstance(array_type, str):
            if not values and array_type in {"RECORD", "STRUCT"}:
                raise ValueError(
                    "Missing detailed struct item type info for an empty array, "
                    "please provide a StructQueryParameterType instance."
                )
        self.array_type = array_type

    @classmethod
    def positional(cls, array_type: str, values: list) -> "ArrayQueryParameter":
        """Factory for positional parameters.

        Args:
            array_type (Union[str, ScalarQueryParameterType, StructQueryParameterType]):
                The type of array elements. If given as a string, it must be one of
                `'STRING'`, `'INT64'`, `'FLOAT64'`, `'NUMERIC'`, `'BIGNUMERIC'`,
                `'BOOL'`, `'TIMESTAMP'`, `'DATE'`, or `'STRUCT'`/`'RECORD'`.
                If the type is ``'STRUCT'``/``'RECORD'`` and ``values`` is empty,
                the exact item type cannot be deduced, thus a ``StructQueryParameterType``
                instance needs to be passed in.

            values (List[appropriate type]): The parameter array values.

        Returns:
            google.cloud.bigquery.query.ArrayQueryParameter: Instance without name
        """
        return cls(None, array_type, values)

    @classmethod
    def _from_api_repr_struct(cls, resource):
        name = resource.get("name")
        converted = []
        # We need to flatten the array to use the StructQueryParameter
        # parse code.
        resource_template = {
            # The arrayType includes all the types of the fields of the STRUCT
            "parameterType": resource["parameterType"]["arrayType"]
        }
        for array_value in resource["parameterValue"]["arrayValues"]:
            struct_resource = copy.deepcopy(resource_template)
            struct_resource["parameterValue"] = array_value
            struct_value = StructQueryParameter.from_api_repr(struct_resource)
            converted.append(struct_value)
        return cls(name, "STRUCT", converted)

    @classmethod
    def _from_api_repr_scalar(cls, resource):
        """Converts REST resource into a list of scalar values."""
        # Import here to avoid circular imports.
        from google.cloud.bigquery import schema

        name = resource.get("name")
        array_type = resource["parameterType"]["arrayType"]["type"]
        parameter_value = resource.get("parameterValue", {})
        array_values = parameter_value.get("arrayValues", ())
        values = [value["value"] for value in array_values]
        converted = [
            _helpers.SCALAR_QUERY_PARAM_PARSER.to_py(
                value, schema.SchemaField(name, array_type)
            )
            for value in values
        ]
        return cls(name, array_type, converted)

    @classmethod
    def from_api_repr(cls, resource: dict) -> "ArrayQueryParameter":
        """Factory: construct parameter from JSON resource.

        Args:
            resource (Dict): JSON mapping of parameter

        Returns:
            google.cloud.bigquery.query.ArrayQueryParameter: Instance
        """
        array_type = resource["parameterType"]["arrayType"]["type"]
        if array_type == "STRUCT":
            return cls._from_api_repr_struct(resource)
        return cls._from_api_repr_scalar(resource)

    def to_api_repr(self) -> dict:
        """Construct JSON API representation for the parameter.

        Returns:
            Dict: JSON mapping
        """
        values = self.values

        if self.array_type in {"RECORD", "STRUCT"} or isinstance(
            self.array_type, StructQueryParameterType
        ):
            reprs = [value.to_api_repr() for value in values]
            a_values = [repr_["parameterValue"] for repr_ in reprs]

            if reprs:
                a_type = reprs[0]["parameterType"]
            else:
                # This assertion always evaluates to True because the
                # constructor disallows STRUCT/RECORD type defined as a
                # string with empty values.
                assert isinstance(self.array_type, StructQueryParameterType)
                a_type = self.array_type.to_api_repr()
        else:
            # Scalar array item type.
            if isinstance(self.array_type, str):
                a_type = {"type": self.array_type}
            else:
                a_type = self.array_type.to_api_repr()

            converter = _SCALAR_VALUE_TO_JSON_PARAM.get(
                a_type["type"], lambda value: value
            )
            values = [converter(value) for value in values]  # type: ignore
            a_values = [{"value": value} for value in values]

        resource = {
            "parameterType": {"type": "ARRAY", "arrayType": a_type},
            "parameterValue": {"arrayValues": a_values},
        }
        if self.name is not None:
            resource["name"] = self.name

        return resource

    def _key(self):
        """A tuple key that uniquely describes this field.

        Used to compute this instance's hashcode and evaluate equality.

        Returns:
            Tuple: The contents of this :class:`~google.cloud.bigquery.query.ArrayQueryParameter`.
        """
        if isinstance(self.array_type, str):
            item_type = self.array_type
        elif isinstance(self.array_type, ScalarQueryParameterType):
            item_type = self.array_type._type
        else:
            item_type = "STRUCT"

        return (self.name, item_type.upper(), self.values)

    def __eq__(self, other):
        if not isinstance(other, ArrayQueryParameter):
            return NotImplemented
        return self._key() == other._key()

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "ArrayQueryParameter{}".format(self._key())


class StructQueryParameter(_AbstractQueryParameter):
    """Name / positional query parameters for struct values.

    Args:
        name (Optional[str]):
            Parameter name, used via ``@foo`` syntax.  If None, the
            parameter can only be addressed via position (``?``).

        sub_params (Union[Tuple[
            google.cloud.bigquery.query.ScalarQueryParameter,
            google.cloud.bigquery.query.ArrayQueryParameter,
            google.cloud.bigquery.query.StructQueryParameter
        ]]): The sub-parameters for the struct
    """

    def __init__(self, name, *sub_params) -> None:
        self.name = name
        self.struct_types: Dict[str, Any] = OrderedDict()
        self.struct_values: Dict[str, Any] = {}

        types = self.struct_types
        values = self.struct_values
        for sub in sub_params:
            if isinstance(sub, self.__class__):
                types[sub.name] = "STRUCT"
                values[sub.name] = sub
            elif isinstance(sub, ArrayQueryParameter):
                types[sub.name] = "ARRAY"
                values[sub.name] = sub
            else:
                types[sub.name] = sub.type_
                values[sub.name] = sub.value

    @classmethod
    def positional(cls, *sub_params):
        """Factory for positional parameters.

        Args:
            sub_params (Union[Tuple[
                google.cloud.bigquery.query.ScalarQueryParameter,
                google.cloud.bigquery.query.ArrayQueryParameter,
                google.cloud.bigquery.query.StructQueryParameter
            ]]): The sub-parameters for the struct

        Returns:
            google.cloud.bigquery.query.StructQueryParameter: Instance without name
        """
        return cls(None, *sub_params)

    @classmethod
    def from_api_repr(cls, resource: dict) -> "StructQueryParameter":
        """Factory: construct parameter from JSON resource.

        Args:
            resource (Dict): JSON mapping of parameter

        Returns:
            google.cloud.bigquery.query.StructQueryParameter: Instance
        """
        # Import here to avoid circular imports.
        from google.cloud.bigquery import schema

        name = resource.get("name")
        instance = cls(name)
        type_resources = {}
        types = instance.struct_types
        for item in resource["parameterType"]["structTypes"]:
            types[item["name"]] = item["type"]["type"]
            type_resources[item["name"]] = item["type"]
        struct_values = resource["parameterValue"]["structValues"]
        for key, value in struct_values.items():
            type_ = types[key]
            converted: Optional[Union[ArrayQueryParameter, StructQueryParameter]] = None
            if type_ == "STRUCT":
                struct_resource = {
                    "name": key,
                    "parameterType": type_resources[key],
                    "parameterValue": value,
                }
                converted = StructQueryParameter.from_api_repr(struct_resource)
            elif type_ == "ARRAY":
                struct_resource = {
                    "name": key,
                    "parameterType": type_resources[key],
                    "parameterValue": value,
                }
                converted = ArrayQueryParameter.from_api_repr(struct_resource)
            else:
                value = value["value"]
                converted = _helpers.SCALAR_QUERY_PARAM_PARSER.to_py(
                    value, schema.SchemaField(cast(str, name), type_)
                )
            instance.struct_values[key] = converted
        return instance

    def to_api_repr(self) -> dict:
        """Construct JSON API representation for the parameter.

        Returns:
            Dict: JSON mapping
        """
        s_types = {}
        values = {}
        for name, value in self.struct_values.items():
            type_ = self.struct_types[name]
            if type_ in ("STRUCT", "ARRAY"):
                repr_ = value.to_api_repr()
                s_types[name] = {"name": name, "type": repr_["parameterType"]}
                values[name] = repr_["parameterValue"]
            else:
                s_types[name] = {"name": name, "type": {"type": type_}}
                converter = _SCALAR_VALUE_TO_JSON_PARAM.get(type_, lambda value: value)
                values[name] = {"value": converter(value)}

        resource = {
            "parameterType": {
                "type": "STRUCT",
                "structTypes": [s_types[key] for key in self.struct_types],
            },
            "parameterValue": {"structValues": values},
        }
        if self.name is not None:
            resource["name"] = self.name
        return resource

    def _key(self):
        """A tuple key that uniquely describes this field.

        Used to compute this instance's hashcode and evaluate equality.

        Returns:
            Tuple: The contents of this :class:`~google.cloud.bigquery.ArrayQueryParameter`.
        """
        return (self.name, self.struct_types, self.struct_values)

    def __eq__(self, other):
        if not isinstance(other, StructQueryParameter):
            return NotImplemented
        return self._key() == other._key()

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "StructQueryParameter{}".format(self._key())


class RangeQueryParameter(_AbstractQueryParameter):
    """Named / positional query parameters for range values.

    Args:
        range_element_type (Union[str, RangeQueryParameterType]):
            The type of range elements. It must be one of 'TIMESTAMP',
            'DATE', or 'DATETIME'.

        start (Optional[Union[ScalarQueryParameter, str]]):
            The start of the range value. Must be the same type as
            range_element_type. If not provided, it's interpreted as UNBOUNDED.

        end (Optional[Union[ScalarQueryParameter, str]]):
            The end of the range value. Must be the same type as
            range_element_type. If not provided, it's interpreted as UNBOUNDED.

        name (Optional[str]):
            Parameter name, used via ``@foo`` syntax.  If None, the
            parameter can only be addressed via position (``?``).
    """

    @classmethod
    def _parse_range_element_type(self, range_element_type):
        if isinstance(range_element_type, str):
            if range_element_type not in _SUPPORTED_RANGE_ELEMENTS:
                raise ValueError(
                    "If given as a string, range_element_type must be one of "
                    f"'TIMESTAMP', 'DATE', or 'DATETIME'. Got {range_element_type}."
                )
            return RangeQueryParameterType(range_element_type)
        elif isinstance(range_element_type, RangeQueryParameterType):
            if range_element_type.type_._type not in _SUPPORTED_RANGE_ELEMENTS:
                raise ValueError(
                    "If given as a RangeQueryParameterType object, "
                    "range_element_type must be one of 'TIMESTAMP', 'DATE', "
                    "or 'DATETIME' type."
                )
            return range_element_type
        else:
            raise ValueError(
                "range_element_type must be a string or "
                "RangeQueryParameterType object, of 'TIMESTAMP', 'DATE', "
                "or 'DATETIME' type. Got "
                f"{type(range_element_type)}:{range_element_type}"
            )

    @classmethod
    def _serialize_range_element_value(self, value, type_):
        if value is None or isinstance(value, str):
            return value
        else:
            converter = _SCALAR_VALUE_TO_JSON_PARAM.get(type_)
            if converter is not None:
                return converter(value)  # type: ignore
            else:
                raise ValueError(
                    f"Cannot convert range element value from type {type_}, "
                    "must be one of the strings 'TIMESTAMP', 'DATE' "
                    "'DATETIME' or a RangeQueryParameterType object."
                )

    def __init__(
        self,
        range_element_type,
        start=None,
        end=None,
        name=None,
    ):
        self.name = name
        self.range_element_type = self._parse_range_element_type(range_element_type)
        print(self.range_element_type.type_._type)
        self.start = start
        self.end = end

    @classmethod
    def positional(
        cls, range_element_type, start=None, end=None
    ) -> "RangeQueryParameter":
        """Factory for positional parameters.

        Args:
            range_element_type (Union[str, RangeQueryParameterType]):
                The type of range elements. It must be one of `'TIMESTAMP'`,
                `'DATE'`, or `'DATETIME'`.

            start (Optional[Union[ScalarQueryParameter, str]]):
                The start of the range value. Must be the same type as
                range_element_type. If not provided, it's interpreted as
                UNBOUNDED.

            end (Optional[Union[ScalarQueryParameter, str]]):
                The end of the range value. Must be the same type as
                range_element_type. If not provided, it's interpreted as
                UNBOUNDED.

        Returns:
            google.cloud.bigquery.query.RangeQueryParameter: Instance without
            name.
        """
        return cls(range_element_type, start, end)

    @classmethod
    def from_api_repr(cls, resource: dict) -> "RangeQueryParameter":
        """Factory: construct parameter from JSON resource.

        Args:
            resource (Dict): JSON mapping of parameter

        Returns:
            google.cloud.bigquery.query.RangeQueryParameter: Instance
        """
        name = resource.get("name")
        range_element_type = (
            resource.get("parameterType", {}).get("rangeElementType", {}).get("type")
        )
        range_value = resource.get("parameterValue", {}).get("rangeValue", {})
        start = range_value.get("start", {}).get("value")
        end = range_value.get("end", {}).get("value")

        return cls(range_element_type, start=start, end=end, name=name)

    def to_api_repr(self) -> dict:
        """Construct JSON API representation for the parameter.

        Returns:
            Dict: JSON mapping
        """
        range_element_type = self.range_element_type.to_api_repr()
        type_ = self.range_element_type.type_._type
        start = self._serialize_range_element_value(self.start, type_)
        end = self._serialize_range_element_value(self.end, type_)
        resource = {
            "parameterType": range_element_type,
            "parameterValue": {
                "rangeValue": {
                    "start": {"value": start},
                    "end": {"value": end},
                },
            },
        }

        # distinguish between name not provided vs. name being empty string
        if self.name is not None:
            resource["name"] = self.name

        return resource

    def _key(self):
        """A tuple key that uniquely describes this field.

        Used to compute this instance's hashcode and evaluate equality.

        Returns:
            Tuple: The contents of this
            :class:`~google.cloud.bigquery.query.RangeQueryParameter`.
        """

        range_element_type = self.range_element_type.to_api_repr()
        return (self.name, range_element_type, self.start, self.end)

    def __eq__(self, other):
        if not isinstance(other, RangeQueryParameter):
            return NotImplemented
        return self._key() == other._key()

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "RangeQueryParameter{}".format(self._key())


class SqlParameterScalarTypes:
    """Supported scalar SQL query parameter types as type objects."""

    BOOL = ScalarQueryParameterType("BOOL")
    BOOLEAN = ScalarQueryParameterType("BOOL")
    BIGDECIMAL = ScalarQueryParameterType("BIGNUMERIC")
    BIGNUMERIC = ScalarQueryParameterType("BIGNUMERIC")
    BYTES = ScalarQueryParameterType("BYTES")
    DATE = ScalarQueryParameterType("DATE")
    DATETIME = ScalarQueryParameterType("DATETIME")
    DECIMAL = ScalarQueryParameterType("NUMERIC")
    FLOAT = ScalarQueryParameterType("FLOAT64")
    FLOAT64 = ScalarQueryParameterType("FLOAT64")
    GEOGRAPHY = ScalarQueryParameterType("GEOGRAPHY")
    INT64 = ScalarQueryParameterType("INT64")
    INTEGER = ScalarQueryParameterType("INT64")
    NUMERIC = ScalarQueryParameterType("NUMERIC")
    STRING = ScalarQueryParameterType("STRING")
    TIME = ScalarQueryParameterType("TIME")
    TIMESTAMP = ScalarQueryParameterType("TIMESTAMP")


class _QueryResults(object):
    """Results of a query.

    See:
    https://g.co/cloud/bigquery/docs/reference/rest/v2/jobs/getQueryResults
    """

    def __init__(self, properties):
        self._properties = {}
        self._set_properties(properties)

    @classmethod
    def from_api_repr(cls, api_response):
        return cls(api_response)

    @property
    def project(self):
        """Project bound to the query job.

        Returns:
            str: The project that the query job is associated with.
        """
        return self._properties.get("jobReference", {}).get("projectId")

    @property
    def cache_hit(self):
        """Query results served from cache.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query#body.QueryResponse.FIELDS.cache_hit

        Returns:
            Optional[bool]:
                True if the query results were served from cache (None
                until set by the server).
        """
        return self._properties.get("cacheHit")

    @property
    def complete(self):
        """Server completed query.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query#body.QueryResponse.FIELDS.job_complete

        Returns:
            Optional[bool]:
                True if the query completed on the server (None
                until set by the server).
        """
        return self._properties.get("jobComplete")

    @property
    def errors(self):
        """Errors generated by the query.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query#body.QueryResponse.FIELDS.errors

        Returns:
            Optional[List[Mapping]]:
                Mappings describing errors generated on the server (None
                until set by the server).
        """
        return self._properties.get("errors")

    @property
    def job_id(self):
        """Job ID of the query job these results are from.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query#body.QueryResponse.FIELDS.job_reference

        Returns:
            str: Job ID of the query job.
        """
        return self._properties.get("jobReference", {}).get("jobId")

    @property
    def location(self):
        """Location of the query job these results are from.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query#body.QueryResponse.FIELDS.job_reference
        or https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query#body.QueryResponse.FIELDS.location

        Returns:
            str: Job ID of the query job.
        """
        location = self._properties.get("jobReference", {}).get("location")

        # Sometimes there's no job, but we still want to get the location
        # information. Prefer the value from job for backwards compatibilitity.
        if not location:
            location = self._properties.get("location")
        return location

    @property
    def query_id(self) -> Optional[str]:
        """[Preview] ID of a completed query.

        This ID is auto-generated and not guaranteed to be populated.
        """
        return self._properties.get("queryId")

    @property
    def page_token(self):
        """Token for fetching next bach of results.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query#body.QueryResponse.FIELDS.page_token

        Returns:
            Optional[str]: Token generated on the server (None until set by the server).
        """
        return self._properties.get("pageToken")

    @property
    def total_rows(self):
        """Total number of rows returned by the query.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query#body.QueryResponse.FIELDS.total_rows

        Returns:
            Optional[int]: Count generated on the server (None until set by the server).
        """
        total_rows = self._properties.get("totalRows")
        if total_rows is not None:
            return int(total_rows)

    @property
    def total_bytes_processed(self):
        """Total number of bytes processed by the query.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query#body.QueryResponse.FIELDS.total_bytes_processed

        Returns:
            Optional[int]: Count generated on the server (None until set by the server).
        """
        total_bytes_processed = self._properties.get("totalBytesProcessed")
        if total_bytes_processed is not None:
            return int(total_bytes_processed)

    @property
    def slot_millis(self):
        """Total number of slot ms the user is actually billed for.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query#body.QueryResponse.FIELDS.total_slot_ms

        Returns:
            Optional[int]: Count generated on the server (None until set by the server).
        """
        slot_millis = self._properties.get("totalSlotMs")
        if slot_millis is not None:
            return int(slot_millis)

    @property
    def num_dml_affected_rows(self):
        """Total number of rows affected by a DML query.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query#body.QueryResponse.FIELDS.num_dml_affected_rows

        Returns:
            Optional[int]: Count generated on the server (None until set by the server).
        """
        num_dml_affected_rows = self._properties.get("numDmlAffectedRows")
        if num_dml_affected_rows is not None:
            return int(num_dml_affected_rows)

    @property
    def created(self):
        """Creation time of this query.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query#body.QueryResponse.FIELDS.creation_time

        Returns:
            Optional[datetime.datetime]:
                the creation time (None until set from the server).
        """
        millis = self._properties.get("creationTime")
        if millis is not None:
            return _helpers._datetime_from_microseconds(int(millis) * 1000.0)

    @property
    def started(self):
        """Start time of this query.

        This field will be present when the query transitions from the
        PENDING state to either RUNNING or DONE.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query#body.QueryResponse.FIELDS.start_time

        Returns:
            Optional[datetime.datetime]:
                the start time (None until set from the server).
        """
        millis = self._properties.get("startTime")
        if millis is not None:
            return _helpers._datetime_from_microseconds(int(millis) * 1000.0)

    @property
    def ended(self):
        """End time of this query.

        This field will be present whenever a query is in the DONE state.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query#body.QueryResponse.FIELDS.end_time

        Returns:
            Optional[datetime.datetime]:
                the end time (None until set from the server).
        """
        millis = self._properties.get("endTime")
        if millis is not None:
            return _helpers._datetime_from_microseconds(int(millis) * 1000.0)

    @property
    def rows(self):
        """Query results.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query#body.QueryResponse.FIELDS.rows

        Returns:
            Optional[List[google.cloud.bigquery.table.Row]]:
                Rows containing the results of the query.
        """
        return _rows_from_json(self._properties.get("rows", ()), self.schema)

    @property
    def schema(self):
        """Schema for query results.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query#body.QueryResponse.FIELDS.schema

        Returns:
            Optional[List[SchemaField]]:
                Fields describing the schema (None until set by the server).
        """
        return _parse_schema_resource(self._properties.get("schema", {}))

    def _set_properties(self, api_response):
        """Update properties from resource in body of ``api_response``

        Args:
            api_response (Dict): Response returned from an API call
        """
        self._properties.clear()
        self._properties.update(api_response)


def _query_param_from_api_repr(resource):
    """Helper:  Construct concrete query parameter from JSON resource."""
    qp_type = resource["parameterType"]
    if "arrayType" in qp_type:
        klass = ArrayQueryParameter
    elif "structTypes" in qp_type:
        klass = StructQueryParameter
    else:
        klass = ScalarQueryParameter
    return klass.from_api_repr(resource)
