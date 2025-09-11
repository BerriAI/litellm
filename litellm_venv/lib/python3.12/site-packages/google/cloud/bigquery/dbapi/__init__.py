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

"""Google BigQuery implementation of the Database API Specification v2.0.

This module implements the `Python Database API Specification v2.0 (DB-API)`_
for Google BigQuery.

.. _Python Database API Specification v2.0 (DB-API):
   https://www.python.org/dev/peps/pep-0249/
"""

from google.cloud.bigquery.dbapi.connection import connect
from google.cloud.bigquery.dbapi.connection import Connection
from google.cloud.bigquery.dbapi.cursor import Cursor
from google.cloud.bigquery.dbapi.exceptions import Warning
from google.cloud.bigquery.dbapi.exceptions import Error
from google.cloud.bigquery.dbapi.exceptions import InterfaceError
from google.cloud.bigquery.dbapi.exceptions import DatabaseError
from google.cloud.bigquery.dbapi.exceptions import DataError
from google.cloud.bigquery.dbapi.exceptions import OperationalError
from google.cloud.bigquery.dbapi.exceptions import IntegrityError
from google.cloud.bigquery.dbapi.exceptions import InternalError
from google.cloud.bigquery.dbapi.exceptions import ProgrammingError
from google.cloud.bigquery.dbapi.exceptions import NotSupportedError
from google.cloud.bigquery.dbapi.types import Binary
from google.cloud.bigquery.dbapi.types import Date
from google.cloud.bigquery.dbapi.types import DateFromTicks
from google.cloud.bigquery.dbapi.types import Time
from google.cloud.bigquery.dbapi.types import TimeFromTicks
from google.cloud.bigquery.dbapi.types import Timestamp
from google.cloud.bigquery.dbapi.types import TimestampFromTicks
from google.cloud.bigquery.dbapi.types import BINARY
from google.cloud.bigquery.dbapi.types import DATETIME
from google.cloud.bigquery.dbapi.types import NUMBER
from google.cloud.bigquery.dbapi.types import ROWID
from google.cloud.bigquery.dbapi.types import STRING


apilevel = "2.0"

# Threads may share the module and connections, but not cursors.
threadsafety = 2

paramstyle = "pyformat"

__all__ = [
    "apilevel",
    "threadsafety",
    "paramstyle",
    "connect",
    "Connection",
    "Cursor",
    "Warning",
    "Error",
    "InterfaceError",
    "DatabaseError",
    "DataError",
    "OperationalError",
    "IntegrityError",
    "InternalError",
    "ProgrammingError",
    "NotSupportedError",
    "Binary",
    "Date",
    "DateFromTicks",
    "Time",
    "TimeFromTicks",
    "Timestamp",
    "TimestampFromTicks",
    "BINARY",
    "DATETIME",
    "NUMBER",
    "ROWID",
    "STRING",
]
