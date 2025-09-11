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

"""Google BigQuery API wrapper.

The main concepts with this API are:

- :class:`~google.cloud.bigquery.client.Client` manages connections to the
  BigQuery API. Use the client methods to run jobs (such as a
  :class:`~google.cloud.bigquery.job.QueryJob` via
  :meth:`~google.cloud.bigquery.client.Client.query`) and manage resources.

- :class:`~google.cloud.bigquery.dataset.Dataset` represents a
  collection of tables.

- :class:`~google.cloud.bigquery.table.Table` represents a single "relation".
"""

import warnings

from google.cloud.bigquery import version as bigquery_version

__version__ = bigquery_version.__version__

from google.cloud.bigquery.client import Client
from google.cloud.bigquery.dataset import AccessEntry
from google.cloud.bigquery.dataset import Dataset
from google.cloud.bigquery.dataset import DatasetReference
from google.cloud.bigquery import enums
from google.cloud.bigquery.enums import AutoRowIDs
from google.cloud.bigquery.enums import DecimalTargetType
from google.cloud.bigquery.enums import KeyResultStatementKind
from google.cloud.bigquery.enums import SqlTypeNames
from google.cloud.bigquery.enums import StandardSqlTypeNames
from google.cloud.bigquery.exceptions import LegacyBigQueryStorageError
from google.cloud.bigquery.exceptions import LegacyPandasError
from google.cloud.bigquery.exceptions import LegacyPyarrowError
from google.cloud.bigquery.external_config import ExternalConfig
from google.cloud.bigquery.external_config import BigtableOptions
from google.cloud.bigquery.external_config import BigtableColumnFamily
from google.cloud.bigquery.external_config import BigtableColumn
from google.cloud.bigquery.external_config import CSVOptions
from google.cloud.bigquery.external_config import GoogleSheetsOptions
from google.cloud.bigquery.external_config import ExternalSourceFormat
from google.cloud.bigquery.external_config import HivePartitioningOptions
from google.cloud.bigquery.format_options import AvroOptions
from google.cloud.bigquery.format_options import ParquetOptions
from google.cloud.bigquery.job.base import SessionInfo
from google.cloud.bigquery.job import Compression
from google.cloud.bigquery.job import CopyJob
from google.cloud.bigquery.job import CopyJobConfig
from google.cloud.bigquery.job import CreateDisposition
from google.cloud.bigquery.job import DestinationFormat
from google.cloud.bigquery.job import DmlStats
from google.cloud.bigquery.job import Encoding
from google.cloud.bigquery.job import ExtractJob
from google.cloud.bigquery.job import ExtractJobConfig
from google.cloud.bigquery.job import LoadJob
from google.cloud.bigquery.job import LoadJobConfig
from google.cloud.bigquery.job import OperationType
from google.cloud.bigquery.job import QueryJob
from google.cloud.bigquery.job import QueryJobConfig
from google.cloud.bigquery.job import QueryPriority
from google.cloud.bigquery.job import SchemaUpdateOption
from google.cloud.bigquery.job import ScriptOptions
from google.cloud.bigquery.job import SourceFormat
from google.cloud.bigquery.job import UnknownJob
from google.cloud.bigquery.job import TransactionInfo
from google.cloud.bigquery.job import WriteDisposition
from google.cloud.bigquery.model import Model
from google.cloud.bigquery.model import ModelReference
from google.cloud.bigquery.query import ArrayQueryParameter
from google.cloud.bigquery.query import ArrayQueryParameterType
from google.cloud.bigquery.query import ConnectionProperty
from google.cloud.bigquery.query import ScalarQueryParameter
from google.cloud.bigquery.query import ScalarQueryParameterType
from google.cloud.bigquery.query import RangeQueryParameter
from google.cloud.bigquery.query import RangeQueryParameterType
from google.cloud.bigquery.query import SqlParameterScalarTypes
from google.cloud.bigquery.query import StructQueryParameter
from google.cloud.bigquery.query import StructQueryParameterType
from google.cloud.bigquery.query import UDFResource
from google.cloud.bigquery.retry import DEFAULT_RETRY
from google.cloud.bigquery.routine import DeterminismLevel
from google.cloud.bigquery.routine import Routine
from google.cloud.bigquery.routine import RoutineArgument
from google.cloud.bigquery.routine import RoutineReference
from google.cloud.bigquery.routine import RoutineType
from google.cloud.bigquery.routine import RemoteFunctionOptions
from google.cloud.bigquery.schema import PolicyTagList
from google.cloud.bigquery.schema import SchemaField
from google.cloud.bigquery.schema import FieldElementType
from google.cloud.bigquery.standard_sql import StandardSqlDataType
from google.cloud.bigquery.standard_sql import StandardSqlField
from google.cloud.bigquery.standard_sql import StandardSqlStructType
from google.cloud.bigquery.standard_sql import StandardSqlTableType
from google.cloud.bigquery.table import PartitionRange
from google.cloud.bigquery.table import RangePartitioning
from google.cloud.bigquery.table import Row
from google.cloud.bigquery.table import SnapshotDefinition
from google.cloud.bigquery.table import CloneDefinition
from google.cloud.bigquery.table import Table
from google.cloud.bigquery.table import TableReference
from google.cloud.bigquery.table import TimePartitioningType
from google.cloud.bigquery.table import TimePartitioning
from google.cloud.bigquery.encryption_configuration import EncryptionConfiguration
from google.cloud.bigquery import _versions_helpers

try:
    import bigquery_magics  # type: ignore
except ImportError:
    bigquery_magics = None

sys_major, sys_minor, sys_micro = _versions_helpers.extract_runtime_version()

if sys_major == 3 and sys_minor in (7, 8):
    warnings.warn(
        "The python-bigquery library no longer supports Python 3.7 "
        "and Python 3.8. "
        f"Your Python version is {sys_major}.{sys_minor}.{sys_micro}. We "
        "recommend that you update soon to ensure ongoing support. For "
        "more details, see: [Google Cloud Client Libraries Supported Python Versions policy](https://cloud.google.com/python/docs/supported-python-versions)",
        FutureWarning,
    )

__all__ = [
    "__version__",
    "Client",
    # Queries
    "ConnectionProperty",
    "QueryJob",
    "QueryJobConfig",
    "ArrayQueryParameter",
    "ScalarQueryParameter",
    "StructQueryParameter",
    "RangeQueryParameter",
    "ArrayQueryParameterType",
    "ScalarQueryParameterType",
    "SqlParameterScalarTypes",
    "StructQueryParameterType",
    "RangeQueryParameterType",
    # Datasets
    "Dataset",
    "DatasetReference",
    "AccessEntry",
    # Tables
    "Table",
    "TableReference",
    "PartitionRange",
    "RangePartitioning",
    "Row",
    "SnapshotDefinition",
    "CloneDefinition",
    "TimePartitioning",
    "TimePartitioningType",
    # Jobs
    "CopyJob",
    "CopyJobConfig",
    "ExtractJob",
    "ExtractJobConfig",
    "LoadJob",
    "LoadJobConfig",
    "SessionInfo",
    "UnknownJob",
    # Models
    "Model",
    "ModelReference",
    # Routines
    "Routine",
    "RoutineArgument",
    "RoutineReference",
    "RemoteFunctionOptions",
    # Shared helpers
    "SchemaField",
    "FieldElementType",
    "PolicyTagList",
    "UDFResource",
    "ExternalConfig",
    "AvroOptions",
    "BigtableOptions",
    "BigtableColumnFamily",
    "BigtableColumn",
    "DmlStats",
    "CSVOptions",
    "GoogleSheetsOptions",
    "HivePartitioningOptions",
    "ParquetOptions",
    "ScriptOptions",
    "TransactionInfo",
    "DEFAULT_RETRY",
    # Standard SQL types
    "StandardSqlDataType",
    "StandardSqlField",
    "StandardSqlStructType",
    "StandardSqlTableType",
    # Enum Constants
    "enums",
    "AutoRowIDs",
    "Compression",
    "CreateDisposition",
    "DecimalTargetType",
    "DestinationFormat",
    "DeterminismLevel",
    "ExternalSourceFormat",
    "Encoding",
    "KeyResultStatementKind",
    "OperationType",
    "QueryPriority",
    "RoutineType",
    "SchemaUpdateOption",
    "SourceFormat",
    "SqlTypeNames",
    "StandardSqlTypeNames",
    "WriteDisposition",
    # EncryptionConfiguration
    "EncryptionConfiguration",
    # Custom exceptions
    "LegacyBigQueryStorageError",
    "LegacyPyarrowError",
    "LegacyPandasError",
]


def load_ipython_extension(ipython):
    """Called by IPython when this module is loaded as an IPython extension."""
    warnings.warn(
        "%load_ext google.cloud.bigquery is deprecated. Install bigquery-magics package and use `%load_ext bigquery_magics`, instead.",
        category=FutureWarning,
    )

    if bigquery_magics is not None:
        bigquery_magics.load_ipython_extension(ipython)
    else:
        from google.cloud.bigquery.magics.magics import _cell_magic

        ipython.register_magic_function(
            _cell_magic, magic_kind="cell", magic_name="bigquery"
        )
