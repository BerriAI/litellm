# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from collections import abc as collections_abc
import datetime
import decimal
import functools
import numbers
import re
import typing

from google.cloud import bigquery
from google.cloud.bigquery import table, query
from google.cloud.bigquery.dbapi import exceptions


_NUMERIC_SERVER_MIN = decimal.Decimal("-9.9999999999999999999999999999999999999E+28")
_NUMERIC_SERVER_MAX = decimal.Decimal("9.9999999999999999999999999999999999999E+28")

type_parameters_re = re.compile(
    r"""
    \(
    \s*[0-9]+\s*
    (,
    \s*[0-9]+\s*
    )*
    \)
    """,
    re.VERBOSE,
)


def _parameter_type(name, value, query_parameter_type=None, value_doc=""):
    if query_parameter_type:
        # Strip type parameters
        query_parameter_type = type_parameters_re.sub("", query_parameter_type)
        try:
            parameter_type = getattr(
                query.SqlParameterScalarTypes, query_parameter_type.upper()
            )._type
        except AttributeError:
            raise exceptions.ProgrammingError(
                f"The given parameter type, {query_parameter_type},"
                f" for {name} is not a valid BigQuery scalar type."
            )
    else:
        parameter_type = bigquery_scalar_type(value)
        if parameter_type is None:
            raise exceptions.ProgrammingError(
                f"Encountered parameter {name} with "
                f"{value_doc} value {value} of unexpected type."
            )
    return parameter_type


def scalar_to_query_parameter(value, name=None, query_parameter_type=None):
    """Convert a scalar value into a query parameter.

    Args:
        value (Any):
            A scalar value to convert into a query parameter.

        name (str):
            (Optional) Name of the query parameter.
        query_parameter_type (Optional[str]): Given type for the parameter.

    Returns:
        google.cloud.bigquery.ScalarQueryParameter:
            A query parameter corresponding with the type and value of the plain
            Python object.

    Raises:
        google.cloud.bigquery.dbapi.exceptions.ProgrammingError:
            if the type cannot be determined.
    """
    return bigquery.ScalarQueryParameter(
        name, _parameter_type(name, value, query_parameter_type), value
    )


def array_to_query_parameter(value, name=None, query_parameter_type=None):
    """Convert an array-like value into a query parameter.

    Args:
        value (Sequence[Any]): The elements of the array (should not be a
            string-like Sequence).
        name (Optional[str]): Name of the query parameter.
        query_parameter_type (Optional[str]): Given type for the parameter.

    Returns:
        A query parameter corresponding with the type and value of the plain
        Python object.

    Raises:
        google.cloud.bigquery.dbapi.exceptions.ProgrammingError:
            if the type of array elements cannot be determined.
    """
    if not array_like(value):
        raise exceptions.ProgrammingError(
            "The value of parameter {} must be a sequence that is "
            "not string-like.".format(name)
        )

    if query_parameter_type or value:
        array_type = _parameter_type(
            name,
            value[0] if value else None,
            query_parameter_type,
            value_doc="array element ",
        )
    else:
        raise exceptions.ProgrammingError(
            "Encountered an empty array-like value of parameter {}, cannot "
            "determine array elements type.".format(name)
        )

    return bigquery.ArrayQueryParameter(name, array_type, value)


def _parse_struct_fields(
    fields,
    base,
    parse_struct_field=re.compile(
        r"""
        (?:(\w+)\s+)    # field name
        ([A-Z0-9<> ,()]+)  # Field type
        $""",
        re.VERBOSE | re.IGNORECASE,
    ).match,
):
    # Split a string of struct fields.  They're defined by commas, but
    # we have to avoid splitting on commas internal to fields.  For
    # example:
    # name string, children array<struct<name string, bdate date>>
    #
    # only has 2 top-level fields.
    fields = fields.split(",")
    fields = list(reversed(fields))  # in the off chance that there are very many
    while fields:
        field = fields.pop()
        while fields and field.count("<") != field.count(">"):
            field += "," + fields.pop()

        m = parse_struct_field(field.strip())
        if not m:
            raise exceptions.ProgrammingError(
                f"Invalid struct field, {field}, in {base}"
            )
        yield m.group(1, 2)


SCALAR, ARRAY, STRUCT = ("s", "a", "r")


def _parse_type(
    type_,
    name,
    base,
    complex_query_parameter_parse=re.compile(
        r"""
        \s*
        (ARRAY|STRUCT|RECORD)  # Type
        \s*
        <([A-Z0-9_<> ,()]+)>   # Subtype(s)
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    ).match,
):
    if "<" not in type_:
        # Scalar

        # Strip type parameters
        type_ = type_parameters_re.sub("", type_).strip()
        try:
            type_ = getattr(query.SqlParameterScalarTypes, type_.upper())
        except AttributeError:
            raise exceptions.ProgrammingError(
                f"The given parameter type, {type_},"
                f"{' for ' + name if name else ''}"
                f" is not a valid BigQuery scalar type, in {base}."
            )
        if name:
            type_ = type_.with_name(name)
        return SCALAR, type_

    m = complex_query_parameter_parse(type_)
    if not m:
        raise exceptions.ProgrammingError(f"Invalid parameter type, {type_}")
    tname, sub = m.group(1, 2)
    if tname.upper() == "ARRAY":
        sub_type = complex_query_parameter_type(None, sub, base)
        if isinstance(sub_type, query.ArrayQueryParameterType):
            raise exceptions.ProgrammingError(f"Array can't contain an array in {base}")
        sub_type._complex__src = sub
        return ARRAY, sub_type
    else:
        return STRUCT, _parse_struct_fields(sub, base)


def complex_query_parameter_type(name: typing.Optional[str], type_: str, base: str):
    """Construct a parameter type (`StructQueryParameterType`) for a complex type

    or a non-complex type that's part of a complex type.

    Examples:

    array<struct<x float64, y float64>>

    struct<name string, children array<struct<name string, bdate date>>>

    This is used for computing array types.
    """

    type_type, sub_type = _parse_type(type_, name, base)
    if type_type == SCALAR:
        result_type = sub_type
    elif type_type == ARRAY:
        result_type = query.ArrayQueryParameterType(sub_type, name=name)
    elif type_type == STRUCT:
        fields = [
            complex_query_parameter_type(field_name, field_type, base)
            for field_name, field_type in sub_type
        ]
        result_type = query.StructQueryParameterType(*fields, name=name)
    else:  # pragma: NO COVER
        raise AssertionError("Bad type_type", type_type)  # Can't happen :)

    return result_type


def complex_query_parameter(
    name: typing.Optional[str], value, type_: str, base: typing.Optional[str] = None
):
    """
    Construct a query parameter for a complex type (array or struct record)

    or for a subtype, which may not be complex

    Examples:

    array<struct<x float64, y float64>>

    struct<name string, children array<struct<name string, bdate date>>>

    """
    param: typing.Union[
        query.ScalarQueryParameter,
        query.ArrayQueryParameter,
        query.StructQueryParameter,
    ]

    base = base or type_

    type_type, sub_type = _parse_type(type_, name, base)

    if type_type == SCALAR:
        param = query.ScalarQueryParameter(name, sub_type._type, value)
    elif type_type == ARRAY:
        if not array_like(value):
            raise exceptions.ProgrammingError(
                f"Array type with non-array-like value"
                f" with type {type(value).__name__}"
            )
        param = query.ArrayQueryParameter(
            name,
            sub_type,
            (
                value
                if isinstance(sub_type, query.ScalarQueryParameterType)
                else [
                    complex_query_parameter(None, v, sub_type._complex__src, base)
                    for v in value
                ]
            ),
        )
    elif type_type == STRUCT:
        if not isinstance(value, collections_abc.Mapping):
            raise exceptions.ProgrammingError(f"Non-mapping value for type {type_}")
        value_keys = set(value)
        fields = []
        for field_name, field_type in sub_type:
            if field_name not in value:
                raise exceptions.ProgrammingError(
                    f"No field value for {field_name} in {type_}"
                )
            value_keys.remove(field_name)
            fields.append(
                complex_query_parameter(field_name, value[field_name], field_type, base)
            )
        if value_keys:
            raise exceptions.ProgrammingError(f"Extra data keys for {type_}")

        param = query.StructQueryParameter(name, *fields)
    else:  # pragma: NO COVER
        raise AssertionError("Bad type_type", type_type)  # Can't happen :)

    return param


def _dispatch_parameter(type_, value, name=None):
    if type_ is not None and "<" in type_:
        param = complex_query_parameter(name, value, type_)
    elif isinstance(value, collections_abc.Mapping):
        raise NotImplementedError(
            f"STRUCT-like parameter values are not supported"
            f"{' (parameter ' + name + ')' if name else ''},"
            f" unless an explicit type is give in the parameter placeholder"
            f" (e.g. '%({name if name else ''}:struct<...>)s')."
        )
    elif array_like(value):
        param = array_to_query_parameter(value, name, type_)
    else:
        param = scalar_to_query_parameter(value, name, type_)

    return param


def to_query_parameters_list(parameters, parameter_types):
    """Converts a sequence of parameter values into query parameters.

    Args:
        parameters (Sequence[Any]): Sequence of query parameter values.
        parameter_types:
            A list of parameter types, one for each parameter.
            Unknown types are provided as None.

    Returns:
        List[google.cloud.bigquery.query._AbstractQueryParameter]:
            A list of query parameters.
    """
    return [
        _dispatch_parameter(type_, value)
        for value, type_ in zip(parameters, parameter_types)
    ]


def to_query_parameters_dict(parameters, query_parameter_types):
    """Converts a dictionary of parameter values into query parameters.

    Args:
        parameters (Mapping[str, Any]): Dictionary of query parameter values.
        parameter_types:
            A dictionary of parameter types. It needn't have a key for each
            parameter.

    Returns:
        List[google.cloud.bigquery.query._AbstractQueryParameter]:
            A list of named query parameters.
    """
    return [
        _dispatch_parameter(query_parameter_types.get(name), value, name)
        for name, value in parameters.items()
    ]


def to_query_parameters(parameters, parameter_types):
    """Converts DB-API parameter values into query parameters.

    Args:
        parameters (Union[Mapping[str, Any], Sequence[Any]]):
            A dictionary or sequence of query parameter values.
        parameter_types (Union[Mapping[str, str], Sequence[str]]):
            A dictionary or list of parameter types.

            If parameters is a mapping, then this must be a dictionary
            of parameter types.  It needn't have a key for each
            parameter.

            If parameters is a sequence, then this must be a list of
            parameter types, one for each paramater.  Unknown types
            are provided as None.

    Returns:
        List[google.cloud.bigquery.query._AbstractQueryParameter]:
            A list of query parameters.
    """
    if parameters is None:
        return []

    if isinstance(parameters, collections_abc.Mapping):
        return to_query_parameters_dict(parameters, parameter_types)
    else:
        return to_query_parameters_list(parameters, parameter_types)


def bigquery_scalar_type(value):
    """Return a BigQuery name of the scalar type that matches the given value.

    If the scalar type name could not be determined (e.g. for non-scalar
    values), ``None`` is returned.

    Args:
        value (Any)

    Returns:
        Optional[str]: The BigQuery scalar type name.
    """
    if isinstance(value, bool):
        return "BOOL"
    elif isinstance(value, numbers.Integral):
        return "INT64"
    elif isinstance(value, numbers.Real):
        return "FLOAT64"
    elif isinstance(value, decimal.Decimal):
        vtuple = value.as_tuple()
        # NUMERIC values have precision of 38 (number of digits) and scale of 9 (number
        # of fractional digits), and their max absolute value must be strictly smaller
        # than 1.0E+29.
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#decimal_types
        if (
            len(vtuple.digits) <= 38  # max precision: 38
            and vtuple.exponent >= -9  # max scale: 9
            and _NUMERIC_SERVER_MIN <= value <= _NUMERIC_SERVER_MAX
        ):
            return "NUMERIC"
        else:
            return "BIGNUMERIC"

    elif isinstance(value, str):
        return "STRING"
    elif isinstance(value, bytes):
        return "BYTES"
    elif isinstance(value, datetime.datetime):
        return "DATETIME" if value.tzinfo is None else "TIMESTAMP"
    elif isinstance(value, datetime.date):
        return "DATE"
    elif isinstance(value, datetime.time):
        return "TIME"

    return None


def array_like(value):
    """Determine if the given value is array-like.

    Examples of array-like values (as interpreted by this function) are
    sequences such as ``list`` and ``tuple``, but not strings and other
    iterables such as sets.

    Args:
        value (Any)

    Returns:
        bool: ``True`` if the value is considered array-like, ``False`` otherwise.
    """
    return isinstance(value, collections_abc.Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def to_bq_table_rows(rows_iterable):
    """Convert table rows to BigQuery table Row instances.

    Args:
        rows_iterable (Iterable[Mapping]):
            An iterable of row data items to convert to ``Row`` instances.

    Returns:
        Iterable[google.cloud.bigquery.table.Row]
    """

    def to_table_row(row):
        # NOTE: We fetch ARROW values, thus we need to convert them to Python
        # objects with as_py().
        values = tuple(value.as_py() for value in row.values())
        keys_to_index = {key: i for i, key in enumerate(row.keys())}
        return table.Row(values, keys_to_index)

    return (to_table_row(row_data) for row_data in rows_iterable)


def raise_on_closed(
    exc_msg, exc_class=exceptions.ProgrammingError, closed_attr_name="_closed"
):
    """Make public instance methods raise an error if the instance is closed."""

    def _raise_on_closed(method):
        """Make a non-static method raise an error if its containing instance is closed."""

        def with_closed_check(self, *args, **kwargs):
            if getattr(self, closed_attr_name):
                raise exc_class(exc_msg)
            return method(self, *args, **kwargs)

        functools.update_wrapper(with_closed_check, method)
        return with_closed_check

    def decorate_public_methods(klass):
        """Apply ``_raise_on_closed()`` decorator to public instance methods."""
        for name in dir(klass):
            if name.startswith("_") and name != "__iter__":
                continue

            member = getattr(klass, name)
            if not callable(member):
                continue

            # We need to check for class/static methods directly in the instance
            # __dict__, not via the retrieved attribute (`member`), as the
            # latter is already a callable *produced* by one of these descriptors.
            if isinstance(klass.__dict__[name], (staticmethod, classmethod)):
                continue

            member = _raise_on_closed(member)
            setattr(klass, name, member)

        return klass

    return decorate_public_methods
