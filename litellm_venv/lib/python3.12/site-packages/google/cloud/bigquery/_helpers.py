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

"""Shared helper functions for BigQuery API classes."""

import base64
import datetime
import decimal
import json
import math
import re
import os
import textwrap
import warnings
from typing import Any, Optional, Tuple, Type, Union

from dateutil import relativedelta
from google.cloud._helpers import UTC  # type: ignore
from google.cloud._helpers import _date_from_iso8601_date
from google.cloud._helpers import _datetime_from_microseconds
from google.cloud._helpers import _RFC3339_MICROS
from google.cloud._helpers import _RFC3339_NO_FRACTION
from google.cloud._helpers import _to_bytes
from google.auth import credentials as ga_credentials  # type: ignore
from google.api_core import client_options as client_options_lib

TimeoutType = Union[float, None]

_RFC3339_MICROS_NO_ZULU = "%Y-%m-%dT%H:%M:%S.%f"
_TIMEONLY_WO_MICROS = "%H:%M:%S"
_TIMEONLY_W_MICROS = "%H:%M:%S.%f"
_PROJECT_PREFIX_PATTERN = re.compile(
    r"""
    (?P<project_id>\S+\:[^.]+)\.(?P<dataset_id>[^.]+)(?:$|\.(?P<custom_id>[^.]+)$)
""",
    re.VERBOSE,
)

# BigQuery sends INTERVAL data in "canonical format"
# https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#interval_type
_INTERVAL_PATTERN = re.compile(
    r"(?P<calendar_sign>-?)(?P<years>\d+)-(?P<months>\d+) "
    r"(?P<days>-?\d+) "
    r"(?P<time_sign>-?)(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)\.?(?P<fraction>\d*)?$"
)
_RANGE_PATTERN = re.compile(r"\[.*, .*\)")

BIGQUERY_EMULATOR_HOST = "BIGQUERY_EMULATOR_HOST"
"""Environment variable defining host for emulator."""

_DEFAULT_HOST = "https://bigquery.googleapis.com"
"""Default host for JSON API."""

_DEFAULT_HOST_TEMPLATE = "https://bigquery.{UNIVERSE_DOMAIN}"
""" Templatized endpoint format. """

_DEFAULT_UNIVERSE = "googleapis.com"
"""Default universe for the JSON API."""

_UNIVERSE_DOMAIN_ENV = "GOOGLE_CLOUD_UNIVERSE_DOMAIN"
"""Environment variable for setting universe domain."""

_SUPPORTED_RANGE_ELEMENTS = {"TIMESTAMP", "DATETIME", "DATE"}


def _get_client_universe(
    client_options: Optional[Union[client_options_lib.ClientOptions, dict]]
) -> str:
    """Retrieves the specified universe setting.

    Args:
        client_options: specified client options.
    Returns:
        str: resolved universe setting.

    """
    if isinstance(client_options, dict):
        client_options = client_options_lib.from_dict(client_options)
    universe = _DEFAULT_UNIVERSE
    options_universe = getattr(client_options, "universe_domain", None)
    if (
        options_universe
        and isinstance(options_universe, str)
        and len(options_universe) > 0
    ):
        universe = options_universe
    else:
        env_universe = os.getenv(_UNIVERSE_DOMAIN_ENV)
        if isinstance(env_universe, str) and len(env_universe) > 0:
            universe = env_universe
    return universe


def _validate_universe(client_universe: str, credentials: ga_credentials.Credentials):
    """Validates that client provided universe and universe embedded in credentials match.

    Args:
        client_universe (str): The universe domain configured via the client options.
        credentials (ga_credentials.Credentials): The credentials being used in the client.

    Raises:
        ValueError: when client_universe does not match the universe in credentials.
    """
    if hasattr(credentials, "universe_domain"):
        cred_universe = getattr(credentials, "universe_domain")
        if isinstance(cred_universe, str):
            if client_universe != cred_universe:
                raise ValueError(
                    "The configured universe domain "
                    f"({client_universe}) does not match the universe domain "
                    f"found in the credentials ({cred_universe}). "
                    "If you haven't configured the universe domain explicitly, "
                    f"`{_DEFAULT_UNIVERSE}` is the default."
                )


def _get_bigquery_host():
    return os.environ.get(BIGQUERY_EMULATOR_HOST, _DEFAULT_HOST)


def _not_null(value, field):
    """Check whether 'value' should be coerced to 'field' type."""
    return value is not None or (field is not None and field.mode != "NULLABLE")


class CellDataParser:
    """Converter from BigQuery REST resource to Python value for RowIterator and similar classes.

    See: "rows" field of
    https://cloud.google.com/bigquery/docs/reference/rest/v2/tabledata/list and
    https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/getQueryResults.
    """

    def to_py(self, resource, field):
        def default_converter(value, field):
            _warn_unknown_field_type(field)
            return value

        converter = getattr(
            self, f"{field.field_type.lower()}_to_py", default_converter
        )
        if field.mode == "REPEATED":
            return [converter(item["v"], field) for item in resource]
        else:
            return converter(resource, field)

    def bool_to_py(self, value, field):
        """Coerce 'value' to a bool, if set or not nullable."""
        if _not_null(value, field):
            # TODO(tswast): Why does _not_null care if the field is NULLABLE or
            # REQUIRED? Do we actually need such client-side validation?
            if value is None:
                raise TypeError(f"got None for required boolean field {field}")
            return value.lower() in ("t", "true", "1")

    def boolean_to_py(self, value, field):
        """Coerce 'value' to a bool, if set or not nullable."""
        return self.bool_to_py(value, field)

    def integer_to_py(self, value, field):
        """Coerce 'value' to an int, if set or not nullable."""
        if _not_null(value, field):
            return int(value)

    def int64_to_py(self, value, field):
        """Coerce 'value' to an int, if set or not nullable."""
        return self.integer_to_py(value, field)

    def interval_to_py(
        self, value: Optional[str], field
    ) -> Optional[relativedelta.relativedelta]:
        """Coerce 'value' to an interval, if set or not nullable."""
        if not _not_null(value, field):
            return None
        if value is None:
            raise TypeError(f"got {value} for REQUIRED field: {repr(field)}")

        parsed = _INTERVAL_PATTERN.match(value)
        if parsed is None:
            raise ValueError(
                textwrap.dedent(
                    f"""
                    Got interval: '{value}' with unexpected format.
                    Expected interval in canonical format of "[sign]Y-M [sign]D [sign]H:M:S[.F]".
                    See:
                    https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#interval_type
                    for more information.
                    """
                ),
            )

        calendar_sign = -1 if parsed.group("calendar_sign") == "-" else 1
        years = calendar_sign * int(parsed.group("years"))
        months = calendar_sign * int(parsed.group("months"))
        days = int(parsed.group("days"))
        time_sign = -1 if parsed.group("time_sign") == "-" else 1
        hours = time_sign * int(parsed.group("hours"))
        minutes = time_sign * int(parsed.group("minutes"))
        seconds = time_sign * int(parsed.group("seconds"))
        fraction = parsed.group("fraction")
        microseconds = time_sign * int(fraction.ljust(6, "0")[:6]) if fraction else 0

        return relativedelta.relativedelta(
            years=years,
            months=months,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            microseconds=microseconds,
        )

    def float_to_py(self, value, field):
        """Coerce 'value' to a float, if set or not nullable."""
        if _not_null(value, field):
            return float(value)

    def float64_to_py(self, value, field):
        """Coerce 'value' to a float, if set or not nullable."""
        return self.float_to_py(value, field)

    def numeric_to_py(self, value, field):
        """Coerce 'value' to a Decimal, if set or not nullable."""
        if _not_null(value, field):
            return decimal.Decimal(value)

    def bignumeric_to_py(self, value, field):
        """Coerce 'value' to a Decimal, if set or not nullable."""
        return self.numeric_to_py(value, field)

    def string_to_py(self, value, _):
        """NOOP string -> string coercion"""
        return value

    def geography_to_py(self, value, _):
        """NOOP string -> string coercion"""
        return value

    def bytes_to_py(self, value, field):
        """Base64-decode value"""
        if _not_null(value, field):
            return base64.standard_b64decode(_to_bytes(value))

    def timestamp_to_py(self, value, field):
        """Coerce 'value' to a datetime, if set or not nullable."""
        if _not_null(value, field):
            # value will be a integer in seconds, to microsecond precision, in UTC.
            return _datetime_from_microseconds(int(value))

    def datetime_to_py(self, value, field):
        """Coerce 'value' to a datetime, if set or not nullable.

        Args:
            value (str): The timestamp.
            field (google.cloud.bigquery.schema.SchemaField):
                The field corresponding to the value.

        Returns:
            Optional[datetime.datetime]:
                The parsed datetime object from
                ``value`` if the ``field`` is not null (otherwise it is
                :data:`None`).
        """
        if _not_null(value, field):
            if "." in value:
                # YYYY-MM-DDTHH:MM:SS.ffffff
                return datetime.datetime.strptime(value, _RFC3339_MICROS_NO_ZULU)
            else:
                # YYYY-MM-DDTHH:MM:SS
                return datetime.datetime.strptime(value, _RFC3339_NO_FRACTION)
        else:
            return None

    def date_to_py(self, value, field):
        """Coerce 'value' to a datetime date, if set or not nullable"""
        if _not_null(value, field):
            # value will be a string, in YYYY-MM-DD form.
            return _date_from_iso8601_date(value)

    def time_to_py(self, value, field):
        """Coerce 'value' to a datetime date, if set or not nullable"""
        if _not_null(value, field):
            if len(value) == 8:  # HH:MM:SS
                fmt = _TIMEONLY_WO_MICROS
            elif len(value) == 15:  # HH:MM:SS.micros
                fmt = _TIMEONLY_W_MICROS
            else:
                raise ValueError(
                    textwrap.dedent(
                        f"""
                        Got {repr(value)} with unknown time format.
                        Expected HH:MM:SS or HH:MM:SS.micros. See
                        https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#time_type
                        for more information.
                        """
                    ),
                )
            return datetime.datetime.strptime(value, fmt).time()

    def record_to_py(self, value, field):
        """Coerce 'value' to a mapping, if set or not nullable."""
        if _not_null(value, field):
            record = {}
            record_iter = zip(field.fields, value["f"])
            for subfield, cell in record_iter:
                record[subfield.name] = self.to_py(cell["v"], subfield)
            return record

    def struct_to_py(self, value, field):
        """Coerce 'value' to a mapping, if set or not nullable."""
        return self.record_to_py(value, field)

    def json_to_py(self, value, field):
        """Coerce 'value' to a Pythonic JSON representation."""
        if _not_null(value, field):
            return json.loads(value)
        else:
            return None

    def _range_element_to_py(self, value, field_element_type):
        """Coerce 'value' to a range element value."""
        # Avoid circular imports by importing here.
        from google.cloud.bigquery import schema

        if value == "UNBOUNDED":
            return None
        if field_element_type.element_type in _SUPPORTED_RANGE_ELEMENTS:
            return self.to_py(
                value,
                schema.SchemaField("placeholder", field_element_type.element_type),
            )
        else:
            raise ValueError(
                textwrap.dedent(
                    f"""
                    Got unsupported range element type: {field_element_type.element_type}.
                    Exptected one of {repr(_SUPPORTED_RANGE_ELEMENTS)}. See:
                    https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#declare_a_range_type
                    for more information.
                    """
                ),
            )

    def range_to_py(self, value, field):
        """Coerce 'value' to a range, if set or not nullable.

        Args:
            value (str): The literal representation of the range.
            field (google.cloud.bigquery.schema.SchemaField):
                The field corresponding to the value.

        Returns:
            Optional[dict]:
                The parsed range object from ``value`` if the ``field`` is not
                null (otherwise it is :data:`None`).
        """
        if _not_null(value, field):
            if _RANGE_PATTERN.match(value):
                start, end = value[1:-1].split(", ")
                start = self._range_element_to_py(start, field.range_element_type)
                end = self._range_element_to_py(end, field.range_element_type)
                return {"start": start, "end": end}
            else:
                raise ValueError(
                    textwrap.dedent(
                        f"""
                        Got unknown format for range value: {value}.
                        Expected format '[lower_bound, upper_bound)'. See:
                        https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#range_with_literal
                        for more information.
                        """
                    ),
                )


CELL_DATA_PARSER = CellDataParser()


class DataFrameCellDataParser(CellDataParser):
    """Override of CellDataParser to handle differences in expression of values in DataFrame-like outputs.

    This is used to turn the output of the REST API into a pyarrow Table,
    emulating the serialized arrow from the BigQuery Storage Read API.
    """

    def json_to_py(self, value, _):
        """No-op because DataFrame expects string for JSON output."""
        return value


DATA_FRAME_CELL_DATA_PARSER = DataFrameCellDataParser()


class ScalarQueryParamParser(CellDataParser):
    """Override of CellDataParser to handle the differences in the response from query params.

    See: "value" field of
    https://cloud.google.com/bigquery/docs/reference/rest/v2/QueryParameter#QueryParameterValue
    """

    def timestamp_to_py(self, value, field):
        """Coerce 'value' to a datetime, if set or not nullable.

        Args:
            value (str): The timestamp.

            field (google.cloud.bigquery.schema.SchemaField):
                The field corresponding to the value.

        Returns:
            Optional[datetime.datetime]:
                The parsed datetime object from
                ``value`` if the ``field`` is not null (otherwise it is
                :data:`None`).
        """
        if _not_null(value, field):
            # Canonical formats for timestamps in BigQuery are flexible. See:
            # g.co/cloud/bigquery/docs/reference/standard-sql/data-types#timestamp-type
            # The separator between the date and time can be 'T' or ' '.
            value = value.replace(" ", "T", 1)
            # The UTC timezone may be formatted as Z or +00:00.
            value = value.replace("Z", "")
            value = value.replace("+00:00", "")

            if "." in value:
                # YYYY-MM-DDTHH:MM:SS.ffffff
                return datetime.datetime.strptime(
                    value, _RFC3339_MICROS_NO_ZULU
                ).replace(tzinfo=UTC)
            else:
                # YYYY-MM-DDTHH:MM:SS
                return datetime.datetime.strptime(value, _RFC3339_NO_FRACTION).replace(
                    tzinfo=UTC
                )
        else:
            return None


SCALAR_QUERY_PARAM_PARSER = ScalarQueryParamParser()


def _field_to_index_mapping(schema):
    """Create a mapping from schema field name to index of field."""
    return {f.name: i for i, f in enumerate(schema)}


def _row_tuple_from_json(row, schema):
    """Convert JSON row data to row with appropriate types.

    Note:  ``row['f']`` and ``schema`` are presumed to be of the same length.

    Args:
        row (Dict): A JSON response row to be converted.
        schema (Sequence[Union[ \
                :class:`~google.cloud.bigquery.schema.SchemaField`, \
                Mapping[str, Any] \
        ]]):  Specification of the field types in ``row``.

    Returns:
        Tuple: A tuple of data converted to native types.
    """
    from google.cloud.bigquery.schema import _to_schema_fields

    schema = _to_schema_fields(schema)

    row_data = []
    for field, cell in zip(schema, row["f"]):
        row_data.append(CELL_DATA_PARSER.to_py(cell["v"], field))
    return tuple(row_data)


def _rows_from_json(values, schema):
    """Convert JSON row data to rows with appropriate types.

    Args:
        values (Sequence[Dict]): The list of responses (JSON rows) to convert.
        schema (Sequence[Union[ \
                :class:`~google.cloud.bigquery.schema.SchemaField`, \
                Mapping[str, Any] \
        ]]):
            The table's schema. If any item is a mapping, its content must be
            compatible with
            :meth:`~google.cloud.bigquery.schema.SchemaField.from_api_repr`.

    Returns:
        List[:class:`~google.cloud.bigquery.Row`]
    """
    from google.cloud.bigquery import Row
    from google.cloud.bigquery.schema import _to_schema_fields

    schema = _to_schema_fields(schema)
    field_to_index = _field_to_index_mapping(schema)
    return [Row(_row_tuple_from_json(r, schema), field_to_index) for r in values]


def _int_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, int):
        value = str(value)
    return value


def _float_to_json(value) -> Union[None, str, float]:
    """Coerce 'value' to an JSON-compatible representation."""
    if value is None:
        return None

    if isinstance(value, str):
        value = float(value)

    return str(value) if (math.isnan(value) or math.isinf(value)) else float(value)


def _decimal_to_json(value):
    """Coerce 'value' to a JSON-compatible representation."""
    if isinstance(value, decimal.Decimal):
        value = str(value)
    return value


def _bool_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, bool):
        value = "true" if value else "false"
    return value


def _bytes_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, bytes):
        value = base64.standard_b64encode(value).decode("ascii")
    return value


def _json_to_json(value):
    """Coerce 'value' to a BigQuery REST API representation."""
    if value is None:
        return None
    return json.dumps(value)


def _string_to_json(value):
    """NOOP string -> string coercion"""
    return value


def _timestamp_to_json_parameter(value):
    """Coerce 'value' to an JSON-compatible representation.

    This version returns the string representation used in query parameters.
    """
    if isinstance(value, datetime.datetime):
        if value.tzinfo not in (None, UTC):
            # Convert to UTC and remove the time zone info.
            value = value.replace(tzinfo=None) - value.utcoffset()
        value = "%s %s+00:00" % (value.date().isoformat(), value.time().isoformat())
    return value


def _timestamp_to_json_row(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, datetime.datetime):
        # For naive datetime objects UTC timezone is assumed, thus we format
        # those to string directly without conversion.
        if value.tzinfo is not None:
            value = value.astimezone(UTC)
        value = value.strftime(_RFC3339_MICROS)
    return value


def _datetime_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, datetime.datetime):
        # For naive datetime objects UTC timezone is assumed, thus we format
        # those to string directly without conversion.
        if value.tzinfo is not None:
            value = value.astimezone(UTC)
        value = value.strftime(_RFC3339_MICROS_NO_ZULU)
    return value


def _date_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, datetime.date):
        value = value.isoformat()
    return value


def _time_to_json(value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, datetime.time):
        value = value.isoformat()
    return value


def _range_element_to_json(value, element_type=None):
    """Coerce 'value' to an JSON-compatible representation."""
    if value is None:
        return None
    elif isinstance(value, str):
        if value.upper() in ("UNBOUNDED", "NULL"):
            return None
        else:
            # We do not enforce range element value to be valid to reduce
            # redundancy with backend.
            return value
    elif (
        element_type and element_type.element_type.upper() in _SUPPORTED_RANGE_ELEMENTS
    ):
        converter = _SCALAR_VALUE_TO_JSON_ROW.get(element_type.element_type.upper())
        return converter(value)
    else:
        raise ValueError(
            f"Unsupported RANGE element type {element_type}, or "
            "element type is empty. Must be DATE, DATETIME, or "
            "TIMESTAMP"
        )


def _range_field_to_json(range_element_type, value):
    """Coerce 'value' to an JSON-compatible representation."""
    if isinstance(value, str):
        # string literal
        if _RANGE_PATTERN.match(value):
            start, end = value[1:-1].split(", ")
        else:
            raise ValueError(f"RANGE literal {value} has incorrect format")
    elif isinstance(value, dict):
        # dictionary
        start = value.get("start")
        end = value.get("end")
    else:
        raise ValueError(
            f"Unsupported type of RANGE value {value}, must be " "string or dict"
        )

    start = _range_element_to_json(start, range_element_type)
    end = _range_element_to_json(end, range_element_type)
    return {"start": start, "end": end}


# Converters used for scalar values marshalled to the BigQuery API, such as in
# query parameters or the tabledata.insert API.
_SCALAR_VALUE_TO_JSON_ROW = {
    "INTEGER": _int_to_json,
    "INT64": _int_to_json,
    "FLOAT": _float_to_json,
    "FLOAT64": _float_to_json,
    "NUMERIC": _decimal_to_json,
    "BIGNUMERIC": _decimal_to_json,
    "BOOLEAN": _bool_to_json,
    "BOOL": _bool_to_json,
    "BYTES": _bytes_to_json,
    "TIMESTAMP": _timestamp_to_json_row,
    "DATETIME": _datetime_to_json,
    "DATE": _date_to_json,
    "TIME": _time_to_json,
    "JSON": _json_to_json,
    "STRING": _string_to_json,
    # Make sure DECIMAL and BIGDECIMAL are handled, even though
    # requests for them should be converted to NUMERIC.  Better safe
    # than sorry.
    "DECIMAL": _decimal_to_json,
    "BIGDECIMAL": _decimal_to_json,
}


# Converters used for scalar values marshalled as query parameters.
_SCALAR_VALUE_TO_JSON_PARAM = _SCALAR_VALUE_TO_JSON_ROW.copy()
_SCALAR_VALUE_TO_JSON_PARAM["TIMESTAMP"] = _timestamp_to_json_parameter


def _warn_unknown_field_type(field):
    warnings.warn(
        "Unknown type '{}' for field '{}'. Behavior reading and writing this type is not officially supported and may change in the future.".format(
            field.field_type, field.name
        ),
        FutureWarning,
    )


def _scalar_field_to_json(field, row_value):
    """Maps a field and value to a JSON-safe value.

    Args:
        field (google.cloud.bigquery.schema.SchemaField):
            The SchemaField to use for type conversion and field name.
        row_value (Any):
            Value to be converted, based on the field's type.

    Returns:
        Any: A JSON-serializable object.
    """

    def default_converter(value):
        _warn_unknown_field_type(field)
        return value

    converter = _SCALAR_VALUE_TO_JSON_ROW.get(field.field_type, default_converter)
    return converter(row_value)


def _repeated_field_to_json(field, row_value):
    """Convert a repeated/array field to its JSON representation.

    Args:
        field (google.cloud.bigquery.schema.SchemaField):
            The SchemaField to use for type conversion and field name. The
            field mode must equal ``REPEATED``.
        row_value (Sequence[Any]):
            A sequence of values to convert to JSON-serializable values.

    Returns:
        List[Any]: A list of JSON-serializable objects.
    """
    values = []
    for item in row_value:
        values.append(_single_field_to_json(field, item))
    return values


def _record_field_to_json(fields, row_value):
    """Convert a record/struct field to its JSON representation.

    Args:
        fields (Sequence[google.cloud.bigquery.schema.SchemaField]):
            The :class:`~google.cloud.bigquery.schema.SchemaField`s of the
            record's subfields to use for type conversion and field names.
        row_value (Union[Tuple[Any], Mapping[str, Any]):
            A tuple or dictionary to convert to JSON-serializable values.

    Returns:
        Mapping[str, Any]: A JSON-serializable dictionary.
    """
    isdict = isinstance(row_value, dict)

    # If row is passed as a tuple, make the length sanity check to avoid either
    # uninformative index errors a few lines below or silently omitting some of
    # the values from the result (we cannot know exactly which fields are missing
    # or redundant, since we don't have their names).
    if not isdict and len(row_value) != len(fields):
        msg = "The number of row fields ({}) does not match schema length ({}).".format(
            len(row_value), len(fields)
        )
        raise ValueError(msg)

    record = {}

    if isdict:
        processed_fields = set()

    for subindex, subfield in enumerate(fields):
        subname = subfield.name
        subvalue = row_value.get(subname) if isdict else row_value[subindex]

        # None values are unconditionally omitted
        if subvalue is not None:
            record[subname] = _field_to_json(subfield, subvalue)

        if isdict:
            processed_fields.add(subname)

    # Unknown fields should not be silently dropped, include them. Since there
    # is no schema information available for them, include them as strings
    # to make them JSON-serializable.
    if isdict:
        not_processed = set(row_value.keys()) - processed_fields

        for field_name in not_processed:
            value = row_value[field_name]
            if value is not None:
                record[field_name] = str(value)

    return record


def _single_field_to_json(field, row_value):
    """Convert a single field into JSON-serializable values.

    Ignores mode so that this can function for ARRAY / REPEATING fields
    without requiring a deepcopy of the field. See:
    https://github.com/googleapis/python-bigquery/issues/6

    Args:
        field (google.cloud.bigquery.schema.SchemaField):
            The SchemaField to use for type conversion and field name.

        row_value (Any):
            Scalar or Struct to be inserted. The type
            is inferred from the SchemaField's field_type.

    Returns:
        Any: A JSON-serializable object.
    """
    if row_value is None:
        return None

    if field.field_type == "RECORD":
        return _record_field_to_json(field.fields, row_value)
    if field.field_type == "RANGE":
        return _range_field_to_json(field.range_element_type, row_value)

    return _scalar_field_to_json(field, row_value)


def _field_to_json(field, row_value):
    """Convert a field into JSON-serializable values.

    Args:
        field (google.cloud.bigquery.schema.SchemaField):
            The SchemaField to use for type conversion and field name.

        row_value (Union[Sequence[List], Any]):
            Row data to be inserted. If the SchemaField's mode is
            REPEATED, assume this is a list. If not, the type
            is inferred from the SchemaField's field_type.

    Returns:
        Any: A JSON-serializable object.
    """
    if row_value is None:
        return None

    if field.mode == "REPEATED":
        return _repeated_field_to_json(field, row_value)

    return _single_field_to_json(field, row_value)


def _snake_to_camel_case(value):
    """Convert snake case string to camel case."""
    words = value.split("_")
    return words[0] + "".join(map(str.capitalize, words[1:]))


def _get_sub_prop(container, keys, default=None):
    """Get a nested value from a dictionary.

    This method works like ``dict.get(key)``, but for nested values.

    Args:
        container (Dict):
            A dictionary which may contain other dictionaries as values.
        keys (Iterable):
            A sequence of keys to attempt to get the value for. If ``keys`` is a
            string, it is treated as sequence containing a single string key. Each item
            in the sequence represents a deeper nesting. The first key is for
            the top level. If there is a dictionary there, the second key
            attempts to get the value within that, and so on.
        default (Optional[object]):
            Value to returned if any of the keys are not found.
            Defaults to ``None``.

    Examples:
        Get a top-level value (equivalent to ``container.get('key')``).

        >>> _get_sub_prop({'key': 'value'}, ['key'])
        'value'

        Get a top-level value, providing a default (equivalent to
        ``container.get('key', default='default')``).

        >>> _get_sub_prop({'nothere': 123}, ['key'], default='not found')
        'not found'

        Get a nested value.

        >>> _get_sub_prop({'key': {'subkey': 'value'}}, ['key', 'subkey'])
        'value'

    Returns:
        object: The value if present or the default.
    """
    if isinstance(keys, str):
        keys = [keys]

    sub_val = container
    for key in keys:
        if key not in sub_val:
            return default
        sub_val = sub_val[key]
    return sub_val


def _set_sub_prop(container, keys, value):
    """Set a nested value in a dictionary.

    Args:
        container (Dict):
            A dictionary which may contain other dictionaries as values.
        keys (Iterable):
            A sequence of keys to attempt to set the value for. If ``keys`` is a
            string, it is treated as sequence containing a single string key. Each item
            in the sequence represents a deeper nesting. The first key is for
            the top level. If there is a dictionary there, the second key
            attempts to get the value within that, and so on.
        value (object): Value to set within the container.

    Examples:
        Set a top-level value (equivalent to ``container['key'] = 'value'``).

        >>> container = {}
        >>> _set_sub_prop(container, ['key'], 'value')
        >>> container
        {'key': 'value'}

        Set a nested value.

        >>> container = {}
        >>> _set_sub_prop(container, ['key', 'subkey'], 'value')
        >>> container
        {'key': {'subkey': 'value'}}

        Replace a nested value.

        >>> container = {'key': {'subkey': 'prev'}}
        >>> _set_sub_prop(container, ['key', 'subkey'], 'new')
        >>> container
        {'key': {'subkey': 'new'}}
    """
    if isinstance(keys, str):
        keys = [keys]

    sub_val = container
    for key in keys[:-1]:
        if key not in sub_val:
            sub_val[key] = {}
        sub_val = sub_val[key]
    sub_val[keys[-1]] = value


def _del_sub_prop(container, keys):
    """Remove a nested key fro a dictionary.

    Args:
        container (Dict):
            A dictionary which may contain other dictionaries as values.
        keys (Iterable):
            A sequence of keys to attempt to clear the value for. Each item in
            the sequence represents a deeper nesting. The first key is for
            the top level. If there is a dictionary there, the second key
            attempts to get the value within that, and so on.

    Examples:
        Remove a top-level value (equivalent to ``del container['key']``).

        >>> container = {'key': 'value'}
        >>> _del_sub_prop(container, ['key'])
        >>> container
        {}

        Remove a nested value.

        >>> container = {'key': {'subkey': 'value'}}
        >>> _del_sub_prop(container, ['key', 'subkey'])
        >>> container
        {'key': {}}
    """
    sub_val = container
    for key in keys[:-1]:
        if key not in sub_val:
            sub_val[key] = {}
        sub_val = sub_val[key]
    if keys[-1] in sub_val:
        del sub_val[keys[-1]]


def _int_or_none(value):
    """Helper: deserialize int value from JSON string."""
    if isinstance(value, int):
        return value
    if value is not None:
        return int(value)


def _str_or_none(value):
    """Helper: serialize value to JSON string."""
    if value is not None:
        return str(value)


def _split_id(full_id):
    """Helper: split full_id into composite parts.

    Args:
        full_id (str): Fully-qualified ID in standard SQL format.

    Returns:
        List[str]: ID's parts separated into components.
    """
    with_prefix = _PROJECT_PREFIX_PATTERN.match(full_id)
    if with_prefix is None:
        parts = full_id.split(".")
    else:
        parts = with_prefix.groups()
        parts = [part for part in parts if part]
    return parts


def _parse_3_part_id(full_id, default_project=None, property_name="table_id"):
    output_project_id = default_project
    output_dataset_id = None
    output_resource_id = None
    parts = _split_id(full_id)

    if len(parts) != 2 and len(parts) != 3:
        raise ValueError(
            "{property_name} must be a fully-qualified ID in "
            'standard SQL format, e.g., "project.dataset.{property_name}", '
            "got {}".format(full_id, property_name=property_name)
        )

    if len(parts) == 2 and not default_project:
        raise ValueError(
            "When default_project is not set, {property_name} must be a "
            "fully-qualified ID in standard SQL format, "
            'e.g., "project.dataset_id.{property_name}", got {}'.format(
                full_id, property_name=property_name
            )
        )

    if len(parts) == 2:
        output_dataset_id, output_resource_id = parts
    else:
        output_project_id, output_dataset_id, output_resource_id = parts

    return output_project_id, output_dataset_id, output_resource_id


def _build_resource_from_properties(obj, filter_fields):
    """Build a resource based on a ``_properties`` dictionary, filtered by
    ``filter_fields``, which follow the name of the Python object.
    """
    partial = {}
    for filter_field in filter_fields:
        api_field = _get_sub_prop(obj._PROPERTY_TO_API_FIELD, filter_field)
        if api_field is None and filter_field not in obj._properties:
            raise ValueError("No property %s" % filter_field)
        elif api_field is not None:
            _set_sub_prop(partial, api_field, _get_sub_prop(obj._properties, api_field))
        else:
            # allows properties that are not defined in the library
            # and properties that have the same name as API resource key
            partial[filter_field] = obj._properties[filter_field]

    return partial


def _verify_job_config_type(job_config, expected_type, param_name="job_config"):
    if not isinstance(job_config, expected_type):
        msg = (
            "Expected an instance of {expected_type} class for the {param_name} parameter, "
            "but received {param_name} = {job_config}"
        )
        raise TypeError(
            msg.format(
                expected_type=expected_type.__name__,
                param_name=param_name,
                job_config=job_config,
            )
        )


def _isinstance_or_raise(
    value: Any,
    dtype: Union[Type, Tuple[Type, ...]],
    none_allowed: Optional[bool] = False,
) -> Any:
    """Determine whether a value type matches a given datatype or None.
    Args:
        value (Any): Value to be checked.
        dtype (type): Expected data type or tuple of data types.
        none_allowed Optional(bool): whether value is allowed to be None. Default
           is False.
    Returns:
        Any: Returns the input value if the type check is successful.
    Raises:
        TypeError: If the input value's type does not match the expected data type(s).
    """
    if none_allowed and value is None:
        return value

    if isinstance(value, dtype):
        return value

    or_none = ""
    if none_allowed:
        or_none = " (or None)"

    msg = f"Pass {value} as a '{dtype}'{or_none}. Got {type(value)}."
    raise TypeError(msg)
