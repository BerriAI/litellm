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

"""Schemas for BigQuery tables / queries."""

from __future__ import annotations
import enum
import typing
from typing import Any, cast, Dict, Iterable, Optional, Union, Sequence

from google.cloud.bigquery import _helpers
from google.cloud.bigquery import standard_sql
from google.cloud.bigquery import enums
from google.cloud.bigquery.enums import StandardSqlTypeNames


_STRUCT_TYPES = ("RECORD", "STRUCT")

# SQL types reference:
# LEGACY SQL: https://cloud.google.com/bigquery/data-types#legacy_sql_data_types
# GoogleSQL: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types
LEGACY_TO_STANDARD_TYPES = {
    "STRING": StandardSqlTypeNames.STRING,
    "BYTES": StandardSqlTypeNames.BYTES,
    "INTEGER": StandardSqlTypeNames.INT64,
    "INT64": StandardSqlTypeNames.INT64,
    "FLOAT": StandardSqlTypeNames.FLOAT64,
    "FLOAT64": StandardSqlTypeNames.FLOAT64,
    "NUMERIC": StandardSqlTypeNames.NUMERIC,
    "BIGNUMERIC": StandardSqlTypeNames.BIGNUMERIC,
    "BOOLEAN": StandardSqlTypeNames.BOOL,
    "BOOL": StandardSqlTypeNames.BOOL,
    "GEOGRAPHY": StandardSqlTypeNames.GEOGRAPHY,
    "RECORD": StandardSqlTypeNames.STRUCT,
    "STRUCT": StandardSqlTypeNames.STRUCT,
    "TIMESTAMP": StandardSqlTypeNames.TIMESTAMP,
    "DATE": StandardSqlTypeNames.DATE,
    "TIME": StandardSqlTypeNames.TIME,
    "DATETIME": StandardSqlTypeNames.DATETIME,
    "FOREIGN": StandardSqlTypeNames.FOREIGN,
    # no direct conversion from ARRAY, the latter is represented by mode="REPEATED"
}
"""String names of the legacy SQL types to integer codes of Standard SQL standard_sql."""


class _DefaultSentinel(enum.Enum):
    """Object used as 'sentinel' indicating default value should be used.

    Uses enum so that pytype/mypy knows that this is the only possible value.
    https://stackoverflow.com/a/60605919/101923

    Literal[_DEFAULT_VALUE] is an alternative, but only added in Python 3.8.
    https://docs.python.org/3/library/typing.html#typing.Literal
    """

    DEFAULT_VALUE = object()


_DEFAULT_VALUE = _DefaultSentinel.DEFAULT_VALUE


class FieldElementType(object):
    """Represents the type of a field element.

    Args:
        element_type (str): The type of a field element.
    """

    def __init__(self, element_type: str):
        self._properties = {}
        self._properties["type"] = element_type.upper()

    @property
    def element_type(self):
        return self._properties.get("type")

    @classmethod
    def from_api_repr(cls, api_repr: Optional[dict]) -> Optional["FieldElementType"]:
        """Factory: construct a FieldElementType given its API representation.

        Args:
            api_repr (Dict[str, str]): field element type as returned from
            the API.

        Returns:
            google.cloud.bigquery.FieldElementType:
                Python object, as parsed from ``api_repr``.
        """
        if not api_repr:
            return None
        return cls(api_repr["type"].upper())

    def to_api_repr(self) -> dict:
        """Construct the API resource representation of this field element type.

        Returns:
            Dict[str, str]: Field element type represented as an API resource.
        """
        return self._properties


class SchemaField(object):
    """Describe a single field within a table schema.

    Args:
        name: The name of the field.

        field_type:
            The type of the field. See
            https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#TableFieldSchema.FIELDS.type

        mode:
            Defaults to ``'NULLABLE'``. The mode of the field. See
            https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#TableFieldSchema.FIELDS.mode

        description: Description for the field.

        fields: Subfields (requires ``field_type`` of 'RECORD').

        policy_tags: The policy tag list for the field.

        precision:
            Precison (number of digits) of fields with NUMERIC or BIGNUMERIC type.

        scale:
            Scale (digits after decimal) of fields with NUMERIC or BIGNUMERIC type.

        max_length: Maximum length of fields with STRING or BYTES type.

        default_value_expression: str, Optional
            Used to specify the default value of a field using a SQL expression. It can only be set for
            top level fields (columns).

            You can use a struct or array expression to specify default value for the entire struct or
            array. The valid SQL expressions are:

            - Literals for all data types, including STRUCT and ARRAY.

            - The following functions:

                `CURRENT_TIMESTAMP`
                `CURRENT_TIME`
                `CURRENT_DATE`
                `CURRENT_DATETIME`
                `GENERATE_UUID`
                `RAND`
                `SESSION_USER`
                `ST_GEOPOINT`

            - Struct or array composed with the above allowed functions, for example:

                "[CURRENT_DATE(), DATE '2020-01-01'"]

        range_element_type: FieldElementType, str, Optional
            The subtype of the RANGE, if the type of this field is RANGE. If
            the type is RANGE, this field is required. Possible values for the
            field element type of a RANGE include `DATE`, `DATETIME` and
            `TIMESTAMP`.

        rounding_mode: Union[enums.RoundingMode, str, None]
            Specifies the rounding mode to be used when storing values of
            NUMERIC and BIGNUMERIC type.

            Unspecified will default to using ROUND_HALF_AWAY_FROM_ZERO.
            ROUND_HALF_AWAY_FROM_ZERO rounds half values away from zero
            when applying precision and scale upon writing of NUMERIC and BIGNUMERIC
            values.

            For Scale: 0
            1.1, 1.2, 1.3, 1.4 => 1
            1.5, 1.6, 1.7, 1.8, 1.9 => 2

            ROUND_HALF_EVEN rounds half values to the nearest even value
            when applying precision and scale upon writing of NUMERIC and BIGNUMERIC
            values.

            For Scale: 0
            1.1, 1.2, 1.3, 1.4 => 1
            1.5 => 2
            1.6, 1.7, 1.8, 1.9 => 2
            2.5 => 2

        foreign_type_definition: Optional[str]
            Definition of the foreign data type.

            Only valid for top-level schema fields (not nested fields).
            If the type is FOREIGN, this field is required.
    """

    def __init__(
        self,
        name: str,
        field_type: str,
        mode: str = "NULLABLE",
        default_value_expression: Optional[str] = None,
        description: Union[str, _DefaultSentinel] = _DEFAULT_VALUE,
        fields: Iterable["SchemaField"] = (),
        policy_tags: Union["PolicyTagList", None, _DefaultSentinel] = _DEFAULT_VALUE,
        precision: Union[int, _DefaultSentinel] = _DEFAULT_VALUE,
        scale: Union[int, _DefaultSentinel] = _DEFAULT_VALUE,
        max_length: Union[int, _DefaultSentinel] = _DEFAULT_VALUE,
        range_element_type: Union[FieldElementType, str, None] = None,
        rounding_mode: Union[enums.RoundingMode, str, None] = None,
        foreign_type_definition: Optional[str] = None,
    ):
        self._properties: Dict[str, Any] = {
            "name": name,
            "type": field_type,
        }
        self._properties["name"] = name
        if mode is not None:
            self._properties["mode"] = mode.upper()
        if description is not _DEFAULT_VALUE:
            self._properties["description"] = description
        if default_value_expression is not None:
            self._properties["defaultValueExpression"] = default_value_expression
        if precision is not _DEFAULT_VALUE:
            self._properties["precision"] = precision
        if scale is not _DEFAULT_VALUE:
            self._properties["scale"] = scale
        if max_length is not _DEFAULT_VALUE:
            self._properties["maxLength"] = max_length
        if policy_tags is not _DEFAULT_VALUE:
            self._properties["policyTags"] = (
                policy_tags.to_api_repr()
                if isinstance(policy_tags, PolicyTagList)
                else None
            )
        if isinstance(range_element_type, str):
            self._properties["rangeElementType"] = {"type": range_element_type}
        if isinstance(range_element_type, FieldElementType):
            self._properties["rangeElementType"] = range_element_type.to_api_repr()
        if rounding_mode is not None:
            self._properties["roundingMode"] = rounding_mode
        if foreign_type_definition is not None:
            self._properties["foreignTypeDefinition"] = foreign_type_definition

        if fields:  # Don't set the property if it's not set.
            self._properties["fields"] = [field.to_api_repr() for field in fields]

    @classmethod
    def from_api_repr(cls, api_repr: dict) -> "SchemaField":
        """Return a ``SchemaField`` object deserialized from a dictionary.

        Args:
            api_repr (Mapping[str, str]): The serialized representation
                of the SchemaField, such as what is output by
                :meth:`to_api_repr`.

        Returns:
            google.cloud.bigquery.schema.SchemaField: The ``SchemaField`` object.
        """
        placeholder = cls("this_will_be_replaced", "PLACEHOLDER")

        # Note: we don't make a copy of api_repr because this can cause
        # unnecessary slowdowns, especially on deeply nested STRUCT / RECORD
        # fields. See https://github.com/googleapis/python-bigquery/issues/6
        placeholder._properties = api_repr

        # Add the field `mode` with default value if it does not exist. Fixes
        # an incompatibility issue with pandas-gbq:
        # https://github.com/googleapis/python-bigquery-pandas/issues/854
        if "mode" not in placeholder._properties:
            placeholder._properties["mode"] = "NULLABLE"

        return placeholder

    @property
    def name(self):
        """str: The name of the field."""
        return self._properties.get("name", "")

    @property
    def field_type(self) -> str:
        """str: The type of the field.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#TableFieldSchema.FIELDS.type
        """
        type_ = self._properties.get("type")
        return cast(str, type_).upper()

    @property
    def mode(self):
        """Optional[str]: The mode of the field.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#TableFieldSchema.FIELDS.mode
        """
        return cast(str, self._properties.get("mode", "NULLABLE")).upper()

    @property
    def is_nullable(self):
        """bool: whether 'mode' is 'nullable'."""
        return self.mode == "NULLABLE"

    @property
    def default_value_expression(self):
        """Optional[str] default value of a field, using an SQL expression"""
        return self._properties.get("defaultValueExpression")

    @property
    def description(self):
        """Optional[str]: description for the field."""
        return self._properties.get("description")

    @property
    def precision(self):
        """Optional[int]: Precision (number of digits) for the NUMERIC field."""
        return _helpers._int_or_none(self._properties.get("precision"))

    @property
    def scale(self):
        """Optional[int]: Scale (digits after decimal) for the NUMERIC field."""
        return _helpers._int_or_none(self._properties.get("scale"))

    @property
    def max_length(self):
        """Optional[int]: Maximum length for the STRING or BYTES field."""
        return _helpers._int_or_none(self._properties.get("maxLength"))

    @property
    def range_element_type(self):
        """Optional[FieldElementType]: The subtype of the RANGE, if the
        type of this field is RANGE.

        Must be set when ``type`` is `"RANGE"`. Must be one of `"DATE"`,
        `"DATETIME"` or `"TIMESTAMP"`.
        """
        if self._properties.get("rangeElementType"):
            ret = self._properties.get("rangeElementType")
            return FieldElementType.from_api_repr(ret)

    @property
    def rounding_mode(self):
        """Enum that specifies the rounding mode to be used when storing values of
        NUMERIC and BIGNUMERIC type.
        """
        return self._properties.get("roundingMode")

    @property
    def foreign_type_definition(self):
        """Definition of the foreign data type.

        Only valid for top-level schema fields (not nested fields).
        If the type is FOREIGN, this field is required.
        """
        return self._properties.get("foreignTypeDefinition")

    @property
    def fields(self):
        """Optional[tuple]: Subfields contained in this field.

        Must be empty unset if ``field_type`` is not 'RECORD'.
        """
        return tuple(_to_schema_fields(self._properties.get("fields", [])))

    @property
    def policy_tags(self):
        """Optional[google.cloud.bigquery.schema.PolicyTagList]: Policy tag list
        definition for this field.
        """
        resource = self._properties.get("policyTags")
        return PolicyTagList.from_api_repr(resource) if resource is not None else None

    def to_api_repr(self) -> dict:
        """Return a dictionary representing this schema field.

        Returns:
            Dict: A dictionary representing the SchemaField in a serialized form.
        """
        # Note: we don't make a copy of _properties because this can cause
        # unnecessary slowdowns, especially on deeply nested STRUCT / RECORD
        # fields. See https://github.com/googleapis/python-bigquery/issues/6
        return self._properties

    def _key(self):
        """A tuple key that uniquely describes this field.

        Used to compute this instance's hashcode and evaluate equality.

        Returns:
            Tuple: The contents of this :class:`~google.cloud.bigquery.schema.SchemaField`.
        """
        field_type = self.field_type
        if field_type == "STRING" or field_type == "BYTES":
            if self.max_length is not None:
                field_type = f"{field_type}({self.max_length})"
        elif field_type.endswith("NUMERIC"):
            if self.precision is not None:
                if self.scale is not None:
                    field_type = f"{field_type}({self.precision}, {self.scale})"
                else:
                    field_type = f"{field_type}({self.precision})"

        policy_tags = (
            None if self.policy_tags is None else tuple(sorted(self.policy_tags.names))
        )

        return (
            self.name,
            field_type,
            # Mode is always str, if not given it defaults to a str value
            self.mode.upper(),  # pytype: disable=attribute-error
            self.default_value_expression,
            self.description,
            self.fields,
            policy_tags,
        )

    def to_standard_sql(self) -> standard_sql.StandardSqlField:
        """Return the field as the standard SQL field representation object."""
        sql_type = standard_sql.StandardSqlDataType()

        if self.mode == "REPEATED":
            sql_type.type_kind = StandardSqlTypeNames.ARRAY
        else:
            sql_type.type_kind = LEGACY_TO_STANDARD_TYPES.get(
                self.field_type,
                StandardSqlTypeNames.TYPE_KIND_UNSPECIFIED,
            )

        if sql_type.type_kind == StandardSqlTypeNames.ARRAY:  # noqa: E721
            array_element_type = LEGACY_TO_STANDARD_TYPES.get(
                self.field_type,
                StandardSqlTypeNames.TYPE_KIND_UNSPECIFIED,
            )
            sql_type.array_element_type = standard_sql.StandardSqlDataType(
                type_kind=array_element_type
            )

            # ARRAY cannot directly contain other arrays, only scalar types and STRUCTs
            # https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#array-type
            if array_element_type == StandardSqlTypeNames.STRUCT:  # noqa: E721
                sql_type.array_element_type.struct_type = (
                    standard_sql.StandardSqlStructType(
                        fields=(field.to_standard_sql() for field in self.fields)
                    )
                )
        elif sql_type.type_kind == StandardSqlTypeNames.STRUCT:  # noqa: E721
            sql_type.struct_type = standard_sql.StandardSqlStructType(
                fields=(field.to_standard_sql() for field in self.fields)
            )

        return standard_sql.StandardSqlField(name=self.name, type=sql_type)

    def __eq__(self, other):
        if not isinstance(other, SchemaField):
            return NotImplemented
        return self._key() == other._key()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        key = self._key()
        policy_tags = key[-1]
        policy_tags_inst = None if policy_tags is None else PolicyTagList(policy_tags)
        adjusted_key = key[:-1] + (policy_tags_inst,)
        return f"{self.__class__.__name__}{adjusted_key}"


def _parse_schema_resource(info):
    """Parse a resource fragment into a schema field.

    Args:
        info: (Mapping[str, Dict]): should contain a "fields" key to be parsed

    Returns:
        Optional[Sequence[google.cloud.bigquery.schema.SchemaField`]:
            A list of parsed fields, or ``None`` if no "fields" key found.
    """
    if isinstance(info, list):
        return [SchemaField.from_api_repr(f) for f in info]
    return [SchemaField.from_api_repr(f) for f in info.get("fields", ())]


def _build_schema_resource(fields):
    """Generate a resource fragment for a schema.

    Args:
        fields (Sequence[google.cloud.bigquery.schema.SchemaField): schema to be dumped.

    Returns:
        Sequence[Dict]: Mappings describing the schema of the supplied fields.
    """
    if isinstance(fields, Sequence):
        # Input is a Sequence (e.g. a list): Process and return a list of SchemaFields
        return [field.to_api_repr() for field in fields]

    else:
        raise TypeError("Schema must be a Sequence (e.g. a list) or None.")


def _to_schema_fields(schema):
    """Coerces schema to a list of SchemaField instances while
    preserving the original structure as much as possible.

    Args:
        schema (Sequence[Union[ \
                   :class:`~google.cloud.bigquery.schema.SchemaField`, \
                   Mapping[str, Any] \
                       ]
                   ]
               )::
            Table schema to convert. Can be a list of SchemaField
            objects or mappings.

    Returns:
        A list of SchemaField objects.

    Raises:
        TypeError: If schema is not a Sequence.
    """

    if isinstance(schema, Sequence):
        # Input is a Sequence (e.g. a list): Process and return a list of SchemaFields
        return [
            field
            if isinstance(field, SchemaField)
            else SchemaField.from_api_repr(field)
            for field in schema
        ]

    else:
        raise TypeError("Schema must be a Sequence (e.g. a list) or None.")


class PolicyTagList(object):
    """Define Policy Tags for a column.

    Args:
        names (
            Optional[Tuple[str]]): list of policy tags to associate with
            the column.  Policy tag identifiers are of the form
            `projects/*/locations/*/taxonomies/*/policyTags/*`.
    """

    def __init__(self, names: Iterable[str] = ()):
        self._properties = {}
        self._properties["names"] = tuple(names)

    @property
    def names(self):
        """Tuple[str]: Policy tags associated with this definition."""
        return self._properties.get("names", ())

    def _key(self):
        """A tuple key that uniquely describes this PolicyTagList.

        Used to compute this instance's hashcode and evaluate equality.

        Returns:
            Tuple: The contents of this :class:`~google.cloud.bigquery.schema.PolicyTagList`.
        """
        return tuple(sorted(self._properties.get("names", ())))

    def __eq__(self, other):
        if not isinstance(other, PolicyTagList):
            return NotImplemented
        return self._key() == other._key()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        return f"{self.__class__.__name__}(names={self._key()})"

    @classmethod
    def from_api_repr(cls, api_repr: dict) -> "PolicyTagList":
        """Return a :class:`PolicyTagList` object deserialized from a dict.

        This method creates a new ``PolicyTagList`` instance that points to
        the ``api_repr`` parameter as its internal properties dict. This means
        that when a ``PolicyTagList`` instance is stored as a property of
        another object, any changes made at the higher level will also appear
        here.

        Args:
            api_repr (Mapping[str, str]):
                The serialized representation of the PolicyTagList, such as
                what is output by :meth:`to_api_repr`.

        Returns:
            Optional[google.cloud.bigquery.schema.PolicyTagList]:
                The ``PolicyTagList`` object or None.
        """
        if api_repr is None:
            return None
        names = api_repr.get("names", ())
        return cls(names=names)

    def to_api_repr(self) -> dict:
        """Return a dictionary representing this object.

        This method returns the properties dict of the ``PolicyTagList``
        instance rather than making a copy. This means that when a
        ``PolicyTagList`` instance is stored as a property of another
        object, any changes made at the higher level will also appear here.

        Returns:
            dict:
                A dictionary representing the PolicyTagList object in
                serialized form.
        """
        answer = {"names": list(self.names)}
        return answer


class ForeignTypeInfo:
    """Metadata about the foreign data type definition such as the system in which the
    type is defined.

    Args:
        type_system (str): Required. Specifies the system which defines the
            foreign data type.

            TypeSystem enum currently includes:
            * "TYPE_SYSTEM_UNSPECIFIED"
            * "HIVE"
    """

    def __init__(self, type_system: Optional[str] = None):
        self._properties: Dict[str, Any] = {}
        self.type_system = type_system

    @property
    def type_system(self) -> Optional[str]:
        """Required. Specifies the system which defines the foreign data
        type."""

        return self._properties.get("typeSystem")

    @type_system.setter
    def type_system(self, value: Optional[str]):
        value = _helpers._isinstance_or_raise(value, str, none_allowed=True)
        self._properties["typeSystem"] = value

    def to_api_repr(self) -> dict:
        """Build an API representation of this object.

        Returns:
            Dict[str, Any]:
                A dictionary in the format used by the BigQuery API.
        """

        return self._properties

    @classmethod
    def from_api_repr(cls, api_repr: Dict[str, Any]) -> "ForeignTypeInfo":
        """Factory: constructs an instance of the class (cls)
        given its API representation.

        Args:
            api_repr (Dict[str, Any]):
                API representation of the object to be instantiated.

        Returns:
            An instance of the class initialized with data from 'api_repr'.
        """

        config = cls()
        config._properties = api_repr
        return config


class SerDeInfo:
    """Serializer and deserializer information.

    Args:
        serialization_library (str): Required. Specifies a fully-qualified class
            name of the serialization library that is responsible for the
            translation of data between table representation and the underlying
            low-level input and output format structures. The maximum length is
            256 characters.
        name (Optional[str]): Name of the SerDe. The maximum length is 256
            characters.
        parameters: (Optional[dict[str, str]]): Key-value pairs that define the initialization
            parameters for the serialization library. Maximum size 10 Kib.
    """

    def __init__(
        self,
        serialization_library: str,
        name: Optional[str] = None,
        parameters: Optional[dict[str, str]] = None,
    ):
        self._properties: Dict[str, Any] = {}
        self.serialization_library = serialization_library
        self.name = name
        self.parameters = parameters

    @property
    def serialization_library(self) -> str:
        """Required. Specifies a fully-qualified class name of the serialization
        library that is responsible for the translation of data between table
        representation and the underlying low-level input and output format
        structures. The maximum length is 256 characters."""

        return typing.cast(str, self._properties.get("serializationLibrary"))

    @serialization_library.setter
    def serialization_library(self, value: str):
        value = _helpers._isinstance_or_raise(value, str, none_allowed=False)
        self._properties["serializationLibrary"] = value

    @property
    def name(self) -> Optional[str]:
        """Optional. Name of the SerDe. The maximum length is 256 characters."""

        return self._properties.get("name")

    @name.setter
    def name(self, value: Optional[str] = None):
        value = _helpers._isinstance_or_raise(value, str, none_allowed=True)
        self._properties["name"] = value

    @property
    def parameters(self) -> Optional[dict[str, str]]:
        """Optional. Key-value pairs that define the initialization parameters
        for the serialization library. Maximum size 10 Kib."""

        return self._properties.get("parameters")

    @parameters.setter
    def parameters(self, value: Optional[dict[str, str]] = None):
        value = _helpers._isinstance_or_raise(value, dict, none_allowed=True)
        self._properties["parameters"] = value

    def to_api_repr(self) -> dict:
        """Build an API representation of this object.

        Returns:
            Dict[str, Any]:
                A dictionary in the format used by the BigQuery API.
        """
        return self._properties

    @classmethod
    def from_api_repr(cls, api_repr: dict) -> SerDeInfo:
        """Factory: constructs an instance of the class (cls)
        given its API representation.

        Args:
            api_repr (Dict[str, Any]):
                API representation of the object to be instantiated.

        Returns:
            An instance of the class initialized with data from 'api_repr'.
        """
        config = cls("PLACEHOLDER")
        config._properties = api_repr
        return config


class StorageDescriptor:
    """Contains information about how a table's data is stored and accessed by open
    source query engines.

    Args:
        input_format (Optional[str]): Specifies the fully qualified class name of
            the InputFormat (e.g.
            "org.apache.hadoop.hive.ql.io.orc.OrcInputFormat"). The maximum
            length is 128 characters.
        location_uri (Optional[str]): The physical location of the table (e.g.
            'gs://spark-dataproc-data/pangea-data/case_sensitive/' or
            'gs://spark-dataproc-data/pangea-data/'). The maximum length is
            2056 bytes.
        output_format (Optional[str]): Specifies the fully qualified class name
            of the OutputFormat (e.g.
            "org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat"). The maximum
            length is 128 characters.
        serde_info (Union[SerDeInfo, dict, None]): Serializer and deserializer information.
    """

    def __init__(
        self,
        input_format: Optional[str] = None,
        location_uri: Optional[str] = None,
        output_format: Optional[str] = None,
        serde_info: Union[SerDeInfo, dict, None] = None,
    ):
        self._properties: Dict[str, Any] = {}
        self.input_format = input_format
        self.location_uri = location_uri
        self.output_format = output_format
        # Using typing.cast() because mypy cannot wrap it's head around the fact that:
        # the setter can accept Union[SerDeInfo, dict, None]
        # but the getter will only ever return Optional[SerDeInfo].
        self.serde_info = typing.cast(Optional[SerDeInfo], serde_info)

    @property
    def input_format(self) -> Optional[str]:
        """Optional. Specifies the fully qualified class name of the InputFormat
        (e.g. "org.apache.hadoop.hive.ql.io.orc.OrcInputFormat"). The maximum
        length is 128 characters."""

        return self._properties.get("inputFormat")

    @input_format.setter
    def input_format(self, value: Optional[str]):
        value = _helpers._isinstance_or_raise(value, str, none_allowed=True)
        self._properties["inputFormat"] = value

    @property
    def location_uri(self) -> Optional[str]:
        """Optional. The physical location of the table (e.g. 'gs://spark-
        dataproc-data/pangea-data/case_sensitive/' or 'gs://spark-dataproc-
        data/pangea-data/'). The maximum length is 2056 bytes."""

        return self._properties.get("locationUri")

    @location_uri.setter
    def location_uri(self, value: Optional[str]):
        value = _helpers._isinstance_or_raise(value, str, none_allowed=True)
        self._properties["locationUri"] = value

    @property
    def output_format(self) -> Optional[str]:
        """Optional. Specifies the fully qualified class name of the
        OutputFormat (e.g. "org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat").
        The maximum length is 128 characters."""

        return self._properties.get("outputFormat")

    @output_format.setter
    def output_format(self, value: Optional[str]):
        value = _helpers._isinstance_or_raise(value, str, none_allowed=True)
        self._properties["outputFormat"] = value

    @property
    def serde_info(self) -> Optional[SerDeInfo]:
        """Optional. Serializer and deserializer information."""

        prop = _helpers._get_sub_prop(self._properties, ["serDeInfo"])
        if prop is not None:
            return typing.cast(SerDeInfo, SerDeInfo.from_api_repr(prop))
        return None

    @serde_info.setter
    def serde_info(self, value: Union[SerDeInfo, dict, None]):
        value = _helpers._isinstance_or_raise(
            value, (SerDeInfo, dict), none_allowed=True
        )

        if isinstance(value, SerDeInfo):
            self._properties["serDeInfo"] = value.to_api_repr()
        else:
            self._properties["serDeInfo"] = value

    def to_api_repr(self) -> dict:
        """Build an API representation of this object.
        Returns:
            Dict[str, Any]:
                A dictionary in the format used by the BigQuery API.
        """
        return self._properties

    @classmethod
    def from_api_repr(cls, resource: dict) -> StorageDescriptor:
        """Factory: constructs an instance of the class (cls)
        given its API representation.
        Args:
            resource (Dict[str, Any]):
                API representation of the object to be instantiated.
        Returns:
            An instance of the class initialized with data from 'resource'.
        """
        config = cls()
        config._properties = resource
        return config
