# Copyright 2023 Google LLC
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

"""Shared helper functions for connecting BigQuery and pyarrow.

NOTE: This module is DEPRECATED. Please make updates in the pandas-gbq package,
instead. See: go/pandas-gbq-and-bigframes-redundancy,
https://github.com/googleapis/python-bigquery-pandas/blob/main/pandas_gbq/schema/bigquery_to_pyarrow.py
and
https://github.com/googleapis/python-bigquery-pandas/blob/main/pandas_gbq/schema/pyarrow_to_bigquery.py
"""

from typing import Any

try:
    import pyarrow  # type: ignore
except ImportError:
    pyarrow = None

try:
    import db_dtypes  # type: ignore

    db_dtypes_import_exception = None
except ImportError as exc:
    db_dtypes = None
    db_dtypes_import_exception = exc


def pyarrow_datetime():
    return pyarrow.timestamp("us", tz=None)


def pyarrow_numeric():
    return pyarrow.decimal128(38, 9)


def pyarrow_bignumeric():
    # 77th digit is partial.
    # https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#decimal_types
    return pyarrow.decimal256(76, 38)


def pyarrow_time():
    return pyarrow.time64("us")


def pyarrow_timestamp():
    return pyarrow.timestamp("us", tz="UTC")


_BQ_TO_ARROW_SCALARS = {}
_ARROW_SCALAR_IDS_TO_BQ = {}

if pyarrow:
    # This dictionary is duplicated in bigquery_storage/test/unite/test_reader.py
    # When modifying it be sure to update it there as well.
    # Note(todo!!): type "BIGNUMERIC"'s matching pyarrow type is added in _pandas_helpers.py
    _BQ_TO_ARROW_SCALARS = {
        "BOOL": pyarrow.bool_,
        "BOOLEAN": pyarrow.bool_,
        "BYTES": pyarrow.binary,
        "DATE": pyarrow.date32,
        "DATETIME": pyarrow_datetime,
        "FLOAT": pyarrow.float64,
        "FLOAT64": pyarrow.float64,
        "GEOGRAPHY": pyarrow.string,
        "INT64": pyarrow.int64,
        "INTEGER": pyarrow.int64,
        # Normally, we'd prefer JSON type built-in to pyarrow (added in 19.0.0),
        # but we'd like this to map as closely to the BQ Storage API as
        # possible, which uses the string() dtype, as JSON support in Arrow
        # predates JSON support in BigQuery by several years.
        "JSON": pyarrow.string,
        "NUMERIC": pyarrow_numeric,
        "STRING": pyarrow.string,
        "TIME": pyarrow_time,
        "TIMESTAMP": pyarrow_timestamp,
    }

    # DEPRECATED: update pandas_gbq.schema.pyarrow_to_bigquery, instead.
    _ARROW_SCALAR_IDS_TO_BQ = {
        # https://arrow.apache.org/docs/python/api/datatypes.html#type-classes
        pyarrow.bool_().id: "BOOL",
        pyarrow.int8().id: "INT64",
        pyarrow.int16().id: "INT64",
        pyarrow.int32().id: "INT64",
        pyarrow.int64().id: "INT64",
        pyarrow.uint8().id: "INT64",
        pyarrow.uint16().id: "INT64",
        pyarrow.uint32().id: "INT64",
        pyarrow.uint64().id: "INT64",
        pyarrow.float16().id: "FLOAT64",
        pyarrow.float32().id: "FLOAT64",
        pyarrow.float64().id: "FLOAT64",
        pyarrow.time32("ms").id: "TIME",
        pyarrow.time64("ns").id: "TIME",
        pyarrow.timestamp("ns").id: "TIMESTAMP",
        pyarrow.date32().id: "DATE",
        pyarrow.date64().id: "DATETIME",  # because millisecond resolution
        pyarrow.binary().id: "BYTES",
        pyarrow.string().id: "STRING",  # also alias for pyarrow.utf8()
        pyarrow.large_string().id: "STRING",
        # The exact scale and precision don't matter, see below.
        pyarrow.decimal128(38, scale=9).id: "NUMERIC",
        # NOTE: all extension types (e.g. json_, uuid, db_dtypes.JSONArrowType)
        # have the same id (31 as of version 19.0.1), so these should not be
        # matched by id.
    }

    _BQ_TO_ARROW_SCALARS["BIGNUMERIC"] = pyarrow_bignumeric
    # The exact decimal's scale and precision are not important, as only
    # the type ID matters, and it's the same for all decimal256 instances.
    _ARROW_SCALAR_IDS_TO_BQ[pyarrow.decimal256(76, scale=38).id] = "BIGNUMERIC"


def bq_to_arrow_scalars(bq_scalar: str):
    """
    DEPRECATED: update pandas_gbq.schema.bigquery_to_pyarrow, instead, which is
    to be added in https://github.com/googleapis/python-bigquery-pandas/pull/893.

    Returns:
        The Arrow scalar type that the input BigQuery scalar type maps to.
        If it cannot find the BigQuery scalar, return None.
    """
    return _BQ_TO_ARROW_SCALARS.get(bq_scalar)


def arrow_scalar_ids_to_bq(arrow_scalar: Any):
    """
    DEPRECATED: update pandas_gbq.schema.pyarrow_to_bigquery, instead.

    Returns:
        The BigQuery scalar type that the input arrow scalar type maps to.
        If it cannot find the arrow scalar, return None.
    """
    return _ARROW_SCALAR_IDS_TO_BQ.get(arrow_scalar)
