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

"""Define API Jobs."""

from google.cloud.bigquery.job.base import _AsyncJob
from google.cloud.bigquery.job.base import _error_result_to_exception
from google.cloud.bigquery.job.base import _DONE_STATE
from google.cloud.bigquery.job.base import _JobConfig
from google.cloud.bigquery.job.base import _JobReference
from google.cloud.bigquery.job.base import ReservationUsage
from google.cloud.bigquery.job.base import ScriptStatistics
from google.cloud.bigquery.job.base import ScriptStackFrame
from google.cloud.bigquery.job.base import TransactionInfo
from google.cloud.bigquery.job.base import UnknownJob
from google.cloud.bigquery.job.copy_ import CopyJob
from google.cloud.bigquery.job.copy_ import CopyJobConfig
from google.cloud.bigquery.job.copy_ import OperationType
from google.cloud.bigquery.job.extract import ExtractJob
from google.cloud.bigquery.job.extract import ExtractJobConfig
from google.cloud.bigquery.job.load import LoadJob
from google.cloud.bigquery.job.load import LoadJobConfig
from google.cloud.bigquery.job.query import _contains_order_by
from google.cloud.bigquery.job.query import DmlStats
from google.cloud.bigquery.job.query import QueryJob
from google.cloud.bigquery.job.query import QueryJobConfig
from google.cloud.bigquery.job.query import QueryPlanEntry
from google.cloud.bigquery.job.query import QueryPlanEntryStep
from google.cloud.bigquery.job.query import ScriptOptions
from google.cloud.bigquery.job.query import TimelineEntry
from google.cloud.bigquery.enums import Compression
from google.cloud.bigquery.enums import CreateDisposition
from google.cloud.bigquery.enums import DestinationFormat
from google.cloud.bigquery.enums import Encoding
from google.cloud.bigquery.enums import QueryPriority
from google.cloud.bigquery.enums import SchemaUpdateOption
from google.cloud.bigquery.enums import SourceFormat
from google.cloud.bigquery.enums import WriteDisposition


# Include classes previously in job.py for backwards compatibility.
__all__ = [
    "_AsyncJob",
    "_error_result_to_exception",
    "_DONE_STATE",
    "_JobConfig",
    "_JobReference",
    "ReservationUsage",
    "ScriptStatistics",
    "ScriptStackFrame",
    "UnknownJob",
    "CopyJob",
    "CopyJobConfig",
    "OperationType",
    "ExtractJob",
    "ExtractJobConfig",
    "LoadJob",
    "LoadJobConfig",
    "_contains_order_by",
    "DmlStats",
    "QueryJob",
    "QueryJobConfig",
    "QueryPlanEntry",
    "QueryPlanEntryStep",
    "ScriptOptions",
    "TimelineEntry",
    "Compression",
    "CreateDisposition",
    "DestinationFormat",
    "Encoding",
    "QueryPriority",
    "SchemaUpdateOption",
    "SourceFormat",
    "TransactionInfo",
    "WriteDisposition",
]
