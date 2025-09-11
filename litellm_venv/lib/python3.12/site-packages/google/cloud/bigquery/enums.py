# Copyright 2019 Google LLC
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

import enum


class AutoRowIDs(enum.Enum):
    """How to handle automatic insert IDs when inserting rows as a stream."""

    DISABLED = enum.auto()
    GENERATE_UUID = enum.auto()


class Compression(str, enum.Enum):
    """The compression type to use for exported files. The default value is
    :attr:`NONE`.

    :attr:`DEFLATE` and :attr:`SNAPPY` are
    only supported for Avro.
    """

    GZIP = "GZIP"
    """Specifies GZIP format."""

    DEFLATE = "DEFLATE"
    """Specifies DEFLATE format."""

    SNAPPY = "SNAPPY"
    """Specifies SNAPPY format."""

    ZSTD = "ZSTD"
    """Specifies ZSTD format."""

    NONE = "NONE"
    """Specifies no compression."""


class DecimalTargetType:
    """The data types that could be used as a target type when converting decimal values.

    https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#DecimalTargetType

    .. versionadded:: 2.21.0
    """

    NUMERIC = "NUMERIC"
    """Decimal values could be converted to NUMERIC type."""

    BIGNUMERIC = "BIGNUMERIC"
    """Decimal values could be converted to BIGNUMERIC type."""

    STRING = "STRING"
    """Decimal values could be converted to STRING type."""


class CreateDisposition(object):
    """Specifies whether the job is allowed to create new tables. The default
    value is :attr:`CREATE_IF_NEEDED`.

    Creation, truncation and append actions occur as one atomic update
    upon job completion.
    """

    CREATE_IF_NEEDED = "CREATE_IF_NEEDED"
    """If the table does not exist, BigQuery creates the table."""

    CREATE_NEVER = "CREATE_NEVER"
    """The table must already exist. If it does not, a 'notFound' error is
    returned in the job result."""


class DatasetView(enum.Enum):
    """DatasetView specifies which dataset information is returned."""

    DATASET_VIEW_UNSPECIFIED = "DATASET_VIEW_UNSPECIFIED"
    """The default value. Currently maps to the FULL view."""

    METADATA = "METADATA"
    """View metadata information for the dataset, such as friendlyName,
    description, labels, etc."""

    ACL = "ACL"
    """View ACL information for the dataset, which defines dataset access
    for one or more entities."""

    FULL = "FULL"
    """View both dataset metadata and ACL information."""


class DefaultPandasDTypes(enum.Enum):
    """Default Pandas DataFrem DTypes to convert BigQuery data. These
    Sentinel values are used instead of None to maintain backward compatibility,
    and allow Pandas package is not available. For more information:
    https://stackoverflow.com/a/60605919/101923
    """

    BOOL_DTYPE = object()
    """Specifies default bool dtype"""

    INT_DTYPE = object()
    """Specifies default integer dtype"""

    DATE_DTYPE = object()
    """Specifies default date dtype"""

    TIME_DTYPE = object()
    """Specifies default time dtype"""

    RANGE_DATE_DTYPE = object()
    """Specifies default range date dtype"""

    RANGE_DATETIME_DTYPE = object()
    """Specifies default range datetime dtype"""

    RANGE_TIMESTAMP_DTYPE = object()
    """Specifies default range timestamp dtype"""


class DestinationFormat(object):
    """The exported file format. The default value is :attr:`CSV`.

    Tables with nested or repeated fields cannot be exported as CSV.
    """

    CSV = "CSV"
    """Specifies CSV format."""

    NEWLINE_DELIMITED_JSON = "NEWLINE_DELIMITED_JSON"
    """Specifies newline delimited JSON format."""

    AVRO = "AVRO"
    """Specifies Avro format."""

    PARQUET = "PARQUET"
    """Specifies Parquet format."""


class Encoding(object):
    """The character encoding of the data. The default is :attr:`UTF_8`.

    BigQuery decodes the data after the raw, binary data has been
    split using the values of the quote and fieldDelimiter properties.
    """

    UTF_8 = "UTF-8"
    """Specifies UTF-8 encoding."""

    ISO_8859_1 = "ISO-8859-1"
    """Specifies ISO-8859-1 encoding."""


class QueryPriority(object):
    """Specifies a priority for the query. The default value is
    :attr:`INTERACTIVE`.
    """

    INTERACTIVE = "INTERACTIVE"
    """Specifies interactive priority."""

    BATCH = "BATCH"
    """Specifies batch priority."""


class QueryApiMethod(str, enum.Enum):
    """API method used to start the query. The default value is
    :attr:`INSERT`.
    """

    INSERT = "INSERT"
    """Submit a query job by using the `jobs.insert REST API method
    <https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/insert>`_.

    This supports all job configuration options.
    """

    QUERY = "QUERY"
    """Submit a query job by using the `jobs.query REST API method
    <https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query>`_.

    Differences from ``INSERT``:

    * Many parameters and job configuration options, including job ID and
      destination table, cannot be used
      with this API method. See the `jobs.query REST API documentation
      <https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query>`_ for
      the complete list of supported configuration options.

    * API blocks up to a specified timeout, waiting for the query to
      finish.

    * The full job resource (including job statistics) may not be available.
      Call :meth:`~google.cloud.bigquery.job.QueryJob.reload` or
      :meth:`~google.cloud.bigquery.client.Client.get_job` to get full job
      statistics and configuration.

    * :meth:`~google.cloud.bigquery.Client.query` can raise API exceptions if
      the query fails, whereas the same errors don't appear until calling
      :meth:`~google.cloud.bigquery.job.QueryJob.result` when the ``INSERT``
      API method is used.
    """


class SchemaUpdateOption(object):
    """Specifies an update to the destination table schema as a side effect of
    a load job.
    """

    ALLOW_FIELD_ADDITION = "ALLOW_FIELD_ADDITION"
    """Allow adding a nullable field to the schema."""

    ALLOW_FIELD_RELAXATION = "ALLOW_FIELD_RELAXATION"
    """Allow relaxing a required field in the original schema to nullable."""


class SourceFormat(object):
    """The format of the data files. The default value is :attr:`CSV`.

    Note that the set of allowed values for loading data is different
    than the set used for external data sources (see
    :class:`~google.cloud.bigquery.external_config.ExternalSourceFormat`).
    """

    CSV = "CSV"
    """Specifies CSV format."""

    DATASTORE_BACKUP = "DATASTORE_BACKUP"
    """Specifies datastore backup format"""

    NEWLINE_DELIMITED_JSON = "NEWLINE_DELIMITED_JSON"
    """Specifies newline delimited JSON format."""

    AVRO = "AVRO"
    """Specifies Avro format."""

    PARQUET = "PARQUET"
    """Specifies Parquet format."""

    ORC = "ORC"
    """Specifies Orc format."""


class KeyResultStatementKind:
    """Determines which statement in the script represents the "key result".

    The "key result" is used to populate the schema and query results of the script job.

    https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#keyresultstatementkind
    """

    KEY_RESULT_STATEMENT_KIND_UNSPECIFIED = "KEY_RESULT_STATEMENT_KIND_UNSPECIFIED"
    LAST = "LAST"
    FIRST_SELECT = "FIRST_SELECT"


class StandardSqlTypeNames(str, enum.Enum):
    """Enum of allowed SQL type names in schema.SchemaField.

    Datatype used in GoogleSQL.
    """

    def _generate_next_value_(name, start, count, last_values):
        return name

    TYPE_KIND_UNSPECIFIED = enum.auto()
    INT64 = enum.auto()
    BOOL = enum.auto()
    FLOAT64 = enum.auto()
    STRING = enum.auto()
    BYTES = enum.auto()
    TIMESTAMP = enum.auto()
    DATE = enum.auto()
    TIME = enum.auto()
    DATETIME = enum.auto()
    INTERVAL = enum.auto()
    GEOGRAPHY = enum.auto()
    NUMERIC = enum.auto()
    BIGNUMERIC = enum.auto()
    JSON = enum.auto()
    ARRAY = enum.auto()
    STRUCT = enum.auto()
    RANGE = enum.auto()
    # NOTE: FOREIGN acts as a wrapper for data types
    # not natively understood by BigQuery unless translated
    FOREIGN = enum.auto()


class EntityTypes(str, enum.Enum):
    """Enum of allowed entity type names in AccessEntry"""

    USER_BY_EMAIL = "userByEmail"
    GROUP_BY_EMAIL = "groupByEmail"
    DOMAIN = "domain"
    DATASET = "dataset"
    SPECIAL_GROUP = "specialGroup"
    VIEW = "view"
    IAM_MEMBER = "iamMember"
    ROUTINE = "routine"


# See also: https://cloud.google.com/bigquery/data-types#legacy_sql_data_types
# and https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types
class SqlTypeNames(str, enum.Enum):
    """Enum of allowed SQL type names in schema.SchemaField.

    Datatype used in Legacy SQL.
    """

    STRING = "STRING"
    BYTES = "BYTES"
    INTEGER = "INTEGER"
    INT64 = "INTEGER"
    FLOAT = "FLOAT"
    FLOAT64 = "FLOAT"
    DECIMAL = NUMERIC = "NUMERIC"
    BIGDECIMAL = BIGNUMERIC = "BIGNUMERIC"
    BOOLEAN = "BOOLEAN"
    BOOL = "BOOLEAN"
    GEOGRAPHY = "GEOGRAPHY"  # NOTE: not available in legacy types
    RECORD = "RECORD"
    STRUCT = "RECORD"
    TIMESTAMP = "TIMESTAMP"
    DATE = "DATE"
    TIME = "TIME"
    DATETIME = "DATETIME"
    INTERVAL = "INTERVAL"  # NOTE: not available in legacy types
    RANGE = "RANGE"  # NOTE: not available in legacy types
    # NOTE: FOREIGN acts as a wrapper for data types
    # not natively understood by BigQuery unless translated
    FOREIGN = "FOREIGN"


class WriteDisposition(object):
    """Specifies the action that occurs if destination table already exists.

    The default value is :attr:`WRITE_APPEND`.

    Each action is atomic and only occurs if BigQuery is able to complete
    the job successfully. Creation, truncation and append actions occur as one
    atomic update upon job completion.
    """

    WRITE_APPEND = "WRITE_APPEND"
    """If the table already exists, BigQuery appends the data to the table."""

    WRITE_TRUNCATE = "WRITE_TRUNCATE"
    """If the table already exists, BigQuery overwrites the table data."""

    WRITE_TRUNCATE_DATA = "WRITE_TRUNCATE_DATA"
    """For existing tables, truncate data but preserve existing schema
    and constraints."""

    WRITE_EMPTY = "WRITE_EMPTY"
    """If the table already exists and contains data, a 'duplicate' error is
    returned in the job result."""


class DeterminismLevel:
    """Specifies determinism level for JavaScript user-defined functions (UDFs).

    https://cloud.google.com/bigquery/docs/reference/rest/v2/routines#DeterminismLevel
    """

    DETERMINISM_LEVEL_UNSPECIFIED = "DETERMINISM_LEVEL_UNSPECIFIED"
    """The determinism of the UDF is unspecified."""

    DETERMINISTIC = "DETERMINISTIC"
    """The UDF is deterministic, meaning that 2 function calls with the same inputs
    always produce the same result, even across 2 query runs."""

    NOT_DETERMINISTIC = "NOT_DETERMINISTIC"
    """The UDF is not deterministic."""


class RoundingMode(str, enum.Enum):
    """Rounding mode options that can be used when storing NUMERIC or BIGNUMERIC
    values.

    ROUNDING_MODE_UNSPECIFIED: will default to using ROUND_HALF_AWAY_FROM_ZERO.

    ROUND_HALF_AWAY_FROM_ZERO: rounds half values away from zero when applying
    precision and scale upon writing of NUMERIC and BIGNUMERIC values.
    For Scale: 0
    * 1.1, 1.2, 1.3, 1.4 => 1
    * 1.5, 1.6, 1.7, 1.8, 1.9 => 2

    ROUND_HALF_EVEN: rounds half values to the nearest even value when applying
    precision and scale upon writing of NUMERIC and BIGNUMERIC values.
    For Scale: 0
    * 1.1, 1.2, 1.3, 1.4 => 1
    * 1.5 => 2
    * 1.6, 1.7, 1.8, 1.9 => 2
    * 2.5 => 2
    """

    def _generate_next_value_(name, start, count, last_values):
        return name

    ROUNDING_MODE_UNSPECIFIED = enum.auto()
    ROUND_HALF_AWAY_FROM_ZERO = enum.auto()
    ROUND_HALF_EVEN = enum.auto()


class BigLakeFileFormat(object):
    FILE_FORMAT_UNSPECIFIED = "FILE_FORMAT_UNSPECIFIED"
    """The default unspecified value."""

    PARQUET = "PARQUET"
    """Apache Parquet format."""


class BigLakeTableFormat(object):
    TABLE_FORMAT_UNSPECIFIED = "TABLE_FORMAT_UNSPECIFIED"
    """The default unspecified value."""

    ICEBERG = "ICEBERG"
    """Apache Iceberg format."""


class UpdateMode(enum.Enum):
    """Specifies the kind of information to update in a dataset."""

    UPDATE_MODE_UNSPECIFIED = "UPDATE_MODE_UNSPECIFIED"
    """The default value. Behavior defaults to UPDATE_FULL."""

    UPDATE_METADATA = "UPDATE_METADATA"
    """Includes metadata information for the dataset, such as friendlyName,
    description, labels, etc."""

    UPDATE_ACL = "UPDATE_ACL"
    """Includes ACL information for the dataset, which defines dataset access
    for one or more entities."""

    UPDATE_FULL = "UPDATE_FULL"
    """Includes both dataset metadata and ACL information."""


class JobCreationMode(object):
    """Documented values for Job Creation Mode."""

    JOB_CREATION_MODE_UNSPECIFIED = "JOB_CREATION_MODE_UNSPECIFIED"
    """Job creation mode is unspecified."""

    JOB_CREATION_REQUIRED = "JOB_CREATION_REQUIRED"
    """Job creation is always required."""

    JOB_CREATION_OPTIONAL = "JOB_CREATION_OPTIONAL"
    """Job creation is optional.

    Returning immediate results is prioritized.
    BigQuery will automatically determine if a Job needs to be created.
    The conditions under which BigQuery can decide to not create a Job are
    subject to change.
    """


class SourceColumnMatch(str, enum.Enum):
    """Uses sensible defaults based on how the schema is provided.
    If autodetect is used, then columns are matched by name. Otherwise, columns
    are matched by position. This is done to keep the behavior backward-compatible.
    """

    SOURCE_COLUMN_MATCH_UNSPECIFIED = "SOURCE_COLUMN_MATCH_UNSPECIFIED"
    """Unspecified column name match option."""

    POSITION = "POSITION"
    """Matches by position. This assumes that the columns are ordered the same
    way as the schema."""

    NAME = "NAME"
    """Matches by name. This reads the header row as column names and reorders
    columns to match the field names in the schema."""
