# Copyright 2017 Google LLC
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

"""Define classes that describe external data sources.

   These are used for both Table.externalDataConfiguration and
   Job.configuration.query.tableDefinitions.
"""

from __future__ import absolute_import, annotations

import base64
import copy
import typing
from typing import Any, Dict, FrozenSet, Iterable, Optional, Union

from google.cloud.bigquery._helpers import _to_bytes
from google.cloud.bigquery._helpers import _bytes_to_json
from google.cloud.bigquery._helpers import _int_or_none
from google.cloud.bigquery._helpers import _str_or_none
from google.cloud.bigquery import _helpers
from google.cloud.bigquery.enums import SourceColumnMatch
from google.cloud.bigquery.format_options import AvroOptions, ParquetOptions
from google.cloud.bigquery import schema
from google.cloud.bigquery.schema import SchemaField


class ExternalSourceFormat(object):
    """The format for external data files.

    Note that the set of allowed values for external data sources is different
    than the set used for loading data (see
    :class:`~google.cloud.bigquery.job.SourceFormat`).
    """

    CSV = "CSV"
    """Specifies CSV format."""

    GOOGLE_SHEETS = "GOOGLE_SHEETS"
    """Specifies Google Sheets format."""

    NEWLINE_DELIMITED_JSON = "NEWLINE_DELIMITED_JSON"
    """Specifies newline delimited JSON format."""

    AVRO = "AVRO"
    """Specifies Avro format."""

    DATASTORE_BACKUP = "DATASTORE_BACKUP"
    """Specifies datastore backup format"""

    ORC = "ORC"
    """Specifies ORC format."""

    PARQUET = "PARQUET"
    """Specifies Parquet format."""

    BIGTABLE = "BIGTABLE"
    """Specifies Bigtable format."""


class BigtableColumn(object):
    """Options for a Bigtable column."""

    def __init__(self):
        self._properties = {}

    @property
    def encoding(self):
        """str: The encoding of the values when the type is not `STRING`

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#BigtableColumn.FIELDS.encoding
        """
        return self._properties.get("encoding")

    @encoding.setter
    def encoding(self, value):
        self._properties["encoding"] = value

    @property
    def field_name(self):
        """str: An identifier to use if the qualifier is not a valid BigQuery
        field identifier

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#BigtableColumn.FIELDS.field_name
        """
        return self._properties.get("fieldName")

    @field_name.setter
    def field_name(self, value):
        self._properties["fieldName"] = value

    @property
    def only_read_latest(self):
        """bool: If this is set, only the latest version of value in this
        column are exposed.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#BigtableColumn.FIELDS.only_read_latest
        """
        return self._properties.get("onlyReadLatest")

    @only_read_latest.setter
    def only_read_latest(self, value):
        self._properties["onlyReadLatest"] = value

    @property
    def qualifier_encoded(self):
        """Union[str, bytes]: The qualifier encoded in binary.

        The type is ``str`` (Python 2.x) or ``bytes`` (Python 3.x). The module
        will handle base64 encoding for you.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#BigtableColumn.FIELDS.qualifier_encoded
        """
        prop = self._properties.get("qualifierEncoded")
        if prop is None:
            return None
        return base64.standard_b64decode(_to_bytes(prop))

    @qualifier_encoded.setter
    def qualifier_encoded(self, value):
        self._properties["qualifierEncoded"] = _bytes_to_json(value)

    @property
    def qualifier_string(self):
        """str: A valid UTF-8 string qualifier

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#BigtableColumn.FIELDS.qualifier_string
        """
        return self._properties.get("qualifierString")

    @qualifier_string.setter
    def qualifier_string(self, value):
        self._properties["qualifierString"] = value

    @property
    def type_(self):
        """str: The type to convert the value in cells of this column.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#BigtableColumn.FIELDS.type
        """
        return self._properties.get("type")

    @type_.setter
    def type_(self, value):
        self._properties["type"] = value

    def to_api_repr(self) -> dict:
        """Build an API representation of this object.

        Returns:
            Dict[str, Any]:
                A dictionary in the format used by the BigQuery API.
        """
        return copy.deepcopy(self._properties)

    @classmethod
    def from_api_repr(cls, resource: dict) -> "BigtableColumn":
        """Factory: construct a :class:`~.external_config.BigtableColumn`
        instance given its API representation.

        Args:
            resource (Dict[str, Any]):
                Definition of a :class:`~.external_config.BigtableColumn`
                instance in the same representation as is returned from the
                API.

        Returns:
            external_config.BigtableColumn: Configuration parsed from ``resource``.
        """
        config = cls()
        config._properties = copy.deepcopy(resource)
        return config


class BigtableColumnFamily(object):
    """Options for a Bigtable column family."""

    def __init__(self):
        self._properties = {}

    @property
    def encoding(self):
        """str: The encoding of the values when the type is not `STRING`

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#BigtableColumnFamily.FIELDS.encoding
        """
        return self._properties.get("encoding")

    @encoding.setter
    def encoding(self, value):
        self._properties["encoding"] = value

    @property
    def family_id(self):
        """str: Identifier of the column family.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#BigtableColumnFamily.FIELDS.family_id
        """
        return self._properties.get("familyId")

    @family_id.setter
    def family_id(self, value):
        self._properties["familyId"] = value

    @property
    def only_read_latest(self):
        """bool: If this is set only the latest version of value are exposed
        for all columns in this column family.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#BigtableColumnFamily.FIELDS.only_read_latest
        """
        return self._properties.get("onlyReadLatest")

    @only_read_latest.setter
    def only_read_latest(self, value):
        self._properties["onlyReadLatest"] = value

    @property
    def type_(self):
        """str: The type to convert the value in cells of this column family.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#BigtableColumnFamily.FIELDS.type
        """
        return self._properties.get("type")

    @type_.setter
    def type_(self, value):
        self._properties["type"] = value

    @property
    def columns(self):
        """List[BigtableColumn]: Lists of columns
        that should be exposed as individual fields.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#BigtableColumnFamily.FIELDS.columns
        """
        prop = self._properties.get("columns", [])
        return [BigtableColumn.from_api_repr(col) for col in prop]

    @columns.setter
    def columns(self, value):
        self._properties["columns"] = [col.to_api_repr() for col in value]

    def to_api_repr(self) -> dict:
        """Build an API representation of this object.

        Returns:
            Dict[str, Any]:
                A dictionary in the format used by the BigQuery API.
        """
        return copy.deepcopy(self._properties)

    @classmethod
    def from_api_repr(cls, resource: dict) -> "BigtableColumnFamily":
        """Factory: construct a :class:`~.external_config.BigtableColumnFamily`
        instance given its API representation.

        Args:
            resource (Dict[str, Any]):
                Definition of a :class:`~.external_config.BigtableColumnFamily`
                instance in the same representation as is returned from the
                API.

        Returns:
            :class:`~.external_config.BigtableColumnFamily`:
                Configuration parsed from ``resource``.
        """
        config = cls()
        config._properties = copy.deepcopy(resource)
        return config


class BigtableOptions(object):
    """Options that describe how to treat Bigtable tables as BigQuery tables."""

    _SOURCE_FORMAT = "BIGTABLE"
    _RESOURCE_NAME = "bigtableOptions"

    def __init__(self):
        self._properties = {}

    @property
    def ignore_unspecified_column_families(self):
        """bool: If :data:`True`, ignore columns not specified in
        :attr:`column_families` list. Defaults to :data:`False`.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#BigtableOptions.FIELDS.ignore_unspecified_column_families
        """
        return self._properties.get("ignoreUnspecifiedColumnFamilies")

    @ignore_unspecified_column_families.setter
    def ignore_unspecified_column_families(self, value):
        self._properties["ignoreUnspecifiedColumnFamilies"] = value

    @property
    def read_rowkey_as_string(self):
        """bool: If :data:`True`, rowkey column families will be read and
        converted to string. Defaults to :data:`False`.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#BigtableOptions.FIELDS.read_rowkey_as_string
        """
        return self._properties.get("readRowkeyAsString")

    @read_rowkey_as_string.setter
    def read_rowkey_as_string(self, value):
        self._properties["readRowkeyAsString"] = value

    @property
    def column_families(self):
        """List[:class:`~.external_config.BigtableColumnFamily`]: List of
        column families to expose in the table schema along with their types.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#BigtableOptions.FIELDS.column_families
        """
        prop = self._properties.get("columnFamilies", [])
        return [BigtableColumnFamily.from_api_repr(cf) for cf in prop]

    @column_families.setter
    def column_families(self, value):
        self._properties["columnFamilies"] = [cf.to_api_repr() for cf in value]

    def to_api_repr(self) -> dict:
        """Build an API representation of this object.

        Returns:
            Dict[str, Any]:
                A dictionary in the format used by the BigQuery API.
        """
        return copy.deepcopy(self._properties)

    @classmethod
    def from_api_repr(cls, resource: dict) -> "BigtableOptions":
        """Factory: construct a :class:`~.external_config.BigtableOptions`
        instance given its API representation.

        Args:
            resource (Dict[str, Any]):
                Definition of a :class:`~.external_config.BigtableOptions`
                instance in the same representation as is returned from the
                API.

        Returns:
            BigtableOptions: Configuration parsed from ``resource``.
        """
        config = cls()
        config._properties = copy.deepcopy(resource)
        return config


class CSVOptions(object):
    """Options that describe how to treat CSV files as BigQuery tables."""

    _SOURCE_FORMAT = "CSV"
    _RESOURCE_NAME = "csvOptions"

    def __init__(self):
        self._properties = {}

    @property
    def allow_jagged_rows(self):
        """bool: If :data:`True`, BigQuery treats missing trailing columns as
        null values. Defaults to :data:`False`.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#CsvOptions.FIELDS.allow_jagged_rows
        """
        return self._properties.get("allowJaggedRows")

    @allow_jagged_rows.setter
    def allow_jagged_rows(self, value):
        self._properties["allowJaggedRows"] = value

    @property
    def allow_quoted_newlines(self):
        """bool: If :data:`True`, quoted data sections that contain newline
        characters in a CSV file are allowed. Defaults to :data:`False`.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#CsvOptions.FIELDS.allow_quoted_newlines
        """
        return self._properties.get("allowQuotedNewlines")

    @allow_quoted_newlines.setter
    def allow_quoted_newlines(self, value):
        self._properties["allowQuotedNewlines"] = value

    @property
    def encoding(self):
        """str: The character encoding of the data.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#CsvOptions.FIELDS.encoding
        """
        return self._properties.get("encoding")

    @encoding.setter
    def encoding(self, value):
        self._properties["encoding"] = value

    @property
    def preserve_ascii_control_characters(self):
        """bool: Indicates if the embedded ASCII control characters
        (the first 32 characters in the ASCII-table, from '\x00' to '\x1F') are preserved.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#CsvOptions.FIELDS.preserve_ascii_control_characters
        """
        return self._properties.get("preserveAsciiControlCharacters")

    @preserve_ascii_control_characters.setter
    def preserve_ascii_control_characters(self, value):
        self._properties["preserveAsciiControlCharacters"] = value

    @property
    def field_delimiter(self):
        """str: The separator for fields in a CSV file. Defaults to comma (',').

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#CsvOptions.FIELDS.field_delimiter
        """
        return self._properties.get("fieldDelimiter")

    @field_delimiter.setter
    def field_delimiter(self, value):
        self._properties["fieldDelimiter"] = value

    @property
    def quote_character(self):
        """str: The value that is used to quote data sections in a CSV file.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#CsvOptions.FIELDS.quote
        """
        return self._properties.get("quote")

    @quote_character.setter
    def quote_character(self, value):
        self._properties["quote"] = value

    @property
    def skip_leading_rows(self):
        """int: The number of rows at the top of a CSV file.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#CsvOptions.FIELDS.skip_leading_rows
        """
        return _int_or_none(self._properties.get("skipLeadingRows"))

    @skip_leading_rows.setter
    def skip_leading_rows(self, value):
        self._properties["skipLeadingRows"] = str(value)

    @property
    def source_column_match(self) -> Optional[SourceColumnMatch]:
        """Optional[google.cloud.bigquery.enums.SourceColumnMatch]: Controls the
        strategy used to match loaded columns to the schema. If not set, a sensible
        default is chosen based on how the schema is provided. If autodetect is
        used, then columns are matched by name. Otherwise, columns are matched by
        position. This is done to keep the behavior backward-compatible.

        Acceptable values are:

            SOURCE_COLUMN_MATCH_UNSPECIFIED: Unspecified column name match option.
            POSITION: matches by position. This assumes that the columns are ordered
            the same way as the schema.
            NAME: matches by name. This reads the header row as column names and
            reorders columns to match the field names in the schema.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#CsvOptions.FIELDS.source_column_match
        """

        value = self._properties.get("sourceColumnMatch")
        return SourceColumnMatch(value) if value is not None else None

    @source_column_match.setter
    def source_column_match(self, value: Union[SourceColumnMatch, str, None]):
        if value is not None and not isinstance(value, (SourceColumnMatch, str)):
            raise TypeError(
                "value must be a google.cloud.bigquery.enums.SourceColumnMatch, str, or None"
            )
        if isinstance(value, SourceColumnMatch):
            value = value.value
        self._properties["sourceColumnMatch"] = value if value else None

    @property
    def null_markers(self) -> Optional[Iterable[str]]:
        """Optional[Iterable[str]]: A list of strings represented as SQL NULL values in a CSV file.

        .. note::
            null_marker and null_markers can't be set at the same time.
            If null_marker is set, null_markers has to be not set.
            If null_markers is set, null_marker has to be not set.
            If both null_marker and null_markers are set at the same time, a user error would be thrown.
            Any strings listed in null_markers, including empty string would be interpreted as SQL NULL.
            This applies to all column types.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#CsvOptions.FIELDS.null_markers
        """
        return self._properties.get("nullMarkers")

    @null_markers.setter
    def null_markers(self, value: Optional[Iterable[str]]):
        self._properties["nullMarkers"] = value

    def to_api_repr(self) -> dict:
        """Build an API representation of this object.

        Returns:
            Dict[str, Any]: A dictionary in the format used by the BigQuery API.
        """
        return copy.deepcopy(self._properties)

    @classmethod
    def from_api_repr(cls, resource: dict) -> "CSVOptions":
        """Factory: construct a :class:`~.external_config.CSVOptions` instance
        given its API representation.

        Args:
            resource (Dict[str, Any]):
                Definition of a :class:`~.external_config.CSVOptions`
                instance in the same representation as is returned from the
                API.

        Returns:
            CSVOptions: Configuration parsed from ``resource``.
        """
        config = cls()
        config._properties = copy.deepcopy(resource)
        return config


class GoogleSheetsOptions(object):
    """Options that describe how to treat Google Sheets as BigQuery tables."""

    _SOURCE_FORMAT = "GOOGLE_SHEETS"
    _RESOURCE_NAME = "googleSheetsOptions"

    def __init__(self):
        self._properties = {}

    @property
    def skip_leading_rows(self):
        """int: The number of rows at the top of a sheet that BigQuery will
        skip when reading the data.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#GoogleSheetsOptions.FIELDS.skip_leading_rows
        """
        return _int_or_none(self._properties.get("skipLeadingRows"))

    @skip_leading_rows.setter
    def skip_leading_rows(self, value):
        self._properties["skipLeadingRows"] = str(value)

    @property
    def range(self):
        """str: The range of a sheet that BigQuery will query from.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#GoogleSheetsOptions.FIELDS.range
        """
        return _str_or_none(self._properties.get("range"))

    @range.setter
    def range(self, value):
        self._properties["range"] = value

    def to_api_repr(self) -> dict:
        """Build an API representation of this object.

        Returns:
            Dict[str, Any]: A dictionary in the format used by the BigQuery API.
        """
        return copy.deepcopy(self._properties)

    @classmethod
    def from_api_repr(cls, resource: dict) -> "GoogleSheetsOptions":
        """Factory: construct a :class:`~.external_config.GoogleSheetsOptions`
        instance given its API representation.

        Args:
            resource (Dict[str, Any]):
                Definition of a :class:`~.external_config.GoogleSheetsOptions`
                instance in the same representation as is returned from the
                API.

        Returns:
            GoogleSheetsOptions: Configuration parsed from ``resource``.
        """
        config = cls()
        config._properties = copy.deepcopy(resource)
        return config


_OPTION_CLASSES = (
    AvroOptions,
    BigtableOptions,
    CSVOptions,
    GoogleSheetsOptions,
    ParquetOptions,
)

OptionsType = Union[
    AvroOptions,
    BigtableOptions,
    CSVOptions,
    GoogleSheetsOptions,
    ParquetOptions,
]


class HivePartitioningOptions(object):
    """[Beta] Options that configure hive partitioning.

    .. note::
        **Experimental**. This feature is experimental and might change or
        have limited support.

    See
    https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#HivePartitioningOptions
    """

    def __init__(self) -> None:
        self._properties: Dict[str, Any] = {}

    @property
    def mode(self):
        """Optional[str]: When set, what mode of hive partitioning to use when reading data.

        Two modes are supported: "AUTO" and "STRINGS".

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#HivePartitioningOptions.FIELDS.mode
        """
        return self._properties.get("mode")

    @mode.setter
    def mode(self, value):
        self._properties["mode"] = value

    @property
    def source_uri_prefix(self):
        """Optional[str]: When hive partition detection is requested, a common prefix for
        all source URIs is required.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#HivePartitioningOptions.FIELDS.source_uri_prefix
        """
        return self._properties.get("sourceUriPrefix")

    @source_uri_prefix.setter
    def source_uri_prefix(self, value):
        self._properties["sourceUriPrefix"] = value

    @property
    def require_partition_filter(self):
        """Optional[bool]: If set to true, queries over the partitioned table require a
        partition filter that can be used for partition elimination to be
        specified.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#HivePartitioningOptions.FIELDS.mode
        """
        return self._properties.get("requirePartitionFilter")

    @require_partition_filter.setter
    def require_partition_filter(self, value):
        self._properties["requirePartitionFilter"] = value

    def to_api_repr(self) -> dict:
        """Build an API representation of this object.

        Returns:
            Dict[str, Any]: A dictionary in the format used by the BigQuery API.
        """
        return copy.deepcopy(self._properties)

    @classmethod
    def from_api_repr(cls, resource: dict) -> "HivePartitioningOptions":
        """Factory: construct a :class:`~.external_config.HivePartitioningOptions`
        instance given its API representation.

        Args:
            resource (Dict[str, Any]):
                Definition of a :class:`~.external_config.HivePartitioningOptions`
                instance in the same representation as is returned from the
                API.

        Returns:
            HivePartitioningOptions: Configuration parsed from ``resource``.
        """
        config = cls()
        config._properties = copy.deepcopy(resource)
        return config


class ExternalConfig(object):
    """Description of an external data source.

    Args:
        source_format (ExternalSourceFormat):
            See :attr:`source_format`.
    """

    def __init__(self, source_format) -> None:
        self._properties = {"sourceFormat": source_format}

    @property
    def source_format(self):
        """:class:`~.external_config.ExternalSourceFormat`:
        Format of external source.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ExternalDataConfiguration.FIELDS.source_format
        """
        return self._properties["sourceFormat"]

    @property
    def options(self) -> Optional[OptionsType]:
        """Source-specific options."""
        for optcls in _OPTION_CLASSES:
            # The code below is too much magic for mypy to handle.
            if self.source_format == optcls._SOURCE_FORMAT:  # type: ignore
                options: OptionsType = optcls()  # type: ignore
                options._properties = self._properties.setdefault(
                    optcls._RESOURCE_NAME, {}  # type: ignore
                )
                return options

        # No matching source format found.
        return None

    @property
    def autodetect(self):
        """bool: If :data:`True`, try to detect schema and format options
        automatically.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ExternalDataConfiguration.FIELDS.autodetect
        """
        return self._properties.get("autodetect")

    @autodetect.setter
    def autodetect(self, value):
        self._properties["autodetect"] = value

    @property
    def compression(self):
        """str: The compression type of the data source.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ExternalDataConfiguration.FIELDS.compression
        """
        return self._properties.get("compression")

    @compression.setter
    def compression(self, value):
        self._properties["compression"] = value

    @property
    def decimal_target_types(self) -> Optional[FrozenSet[str]]:
        """Possible SQL data types to which the source decimal values are converted.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ExternalDataConfiguration.FIELDS.decimal_target_types

        .. versionadded:: 2.21.0
        """
        prop = self._properties.get("decimalTargetTypes")
        if prop is not None:
            prop = frozenset(prop)
        return prop

    @decimal_target_types.setter
    def decimal_target_types(self, value: Optional[Iterable[str]]):
        if value is not None:
            self._properties["decimalTargetTypes"] = list(value)
        else:
            if "decimalTargetTypes" in self._properties:
                del self._properties["decimalTargetTypes"]

    @property
    def hive_partitioning(self):
        """Optional[:class:`~.external_config.HivePartitioningOptions`]: [Beta] When set, \
        it configures hive partitioning support.

        .. note::
            **Experimental**. This feature is experimental and might change or
            have limited support.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ExternalDataConfiguration.FIELDS.hive_partitioning_options
        """
        prop = self._properties.get("hivePartitioningOptions")
        if prop is None:
            return None
        return HivePartitioningOptions.from_api_repr(prop)

    @hive_partitioning.setter
    def hive_partitioning(self, value):
        prop = value.to_api_repr() if value is not None else None
        self._properties["hivePartitioningOptions"] = prop

    @property
    def reference_file_schema_uri(self):
        """Optional[str]:
        When creating an external table, the user can provide a reference file with the
        table schema. This is enabled for the following formats:

        AVRO, PARQUET, ORC
        """
        return self._properties.get("referenceFileSchemaUri")

    @reference_file_schema_uri.setter
    def reference_file_schema_uri(self, value):
        self._properties["referenceFileSchemaUri"] = value

    @property
    def ignore_unknown_values(self):
        """bool: If :data:`True`, extra values that are not represented in the
        table schema are ignored. Defaults to :data:`False`.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ExternalDataConfiguration.FIELDS.ignore_unknown_values
        """
        return self._properties.get("ignoreUnknownValues")

    @ignore_unknown_values.setter
    def ignore_unknown_values(self, value):
        self._properties["ignoreUnknownValues"] = value

    @property
    def max_bad_records(self):
        """int: The maximum number of bad records that BigQuery can ignore when
        reading data.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ExternalDataConfiguration.FIELDS.max_bad_records
        """
        return self._properties.get("maxBadRecords")

    @max_bad_records.setter
    def max_bad_records(self, value):
        self._properties["maxBadRecords"] = value

    @property
    def source_uris(self):
        """List[str]: URIs that point to your data in Google Cloud.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ExternalDataConfiguration.FIELDS.source_uris
        """
        return self._properties.get("sourceUris", [])

    @source_uris.setter
    def source_uris(self, value):
        self._properties["sourceUris"] = value

    @property
    def schema(self):
        """List[:class:`~google.cloud.bigquery.schema.SchemaField`]: The schema
        for the data.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ExternalDataConfiguration.FIELDS.schema
        """
        prop: Dict[str, Any] = typing.cast(
            Dict[str, Any], self._properties.get("schema", {})
        )
        return [SchemaField.from_api_repr(field) for field in prop.get("fields", [])]

    @schema.setter
    def schema(self, value):
        prop = value
        if value is not None:
            prop = {"fields": [field.to_api_repr() for field in value]}
        self._properties["schema"] = prop

    @property
    def date_format(self) -> Optional[str]:
        """Optional[str]: Format used to parse DATE values. Supports C-style and SQL-style values.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ExternalDataConfiguration.FIELDS.date_format
        """
        result = self._properties.get("dateFormat")
        return typing.cast(str, result)

    @date_format.setter
    def date_format(self, value: Optional[str]):
        self._properties["dateFormat"] = value

    @property
    def datetime_format(self) -> Optional[str]:
        """Optional[str]: Format used to parse DATETIME values. Supports C-style
        and SQL-style values.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ExternalDataConfiguration.FIELDS.datetime_format
        """
        result = self._properties.get("datetimeFormat")
        return typing.cast(str, result)

    @datetime_format.setter
    def datetime_format(self, value: Optional[str]):
        self._properties["datetimeFormat"] = value

    @property
    def time_zone(self) -> Optional[str]:
        """Optional[str]: Time zone used when parsing timestamp values that do not
        have specific time zone information (e.g. 2024-04-20 12:34:56). The expected
        format is an IANA timezone string (e.g. America/Los_Angeles).

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ExternalDataConfiguration.FIELDS.time_zone
        """

        result = self._properties.get("timeZone")
        return typing.cast(str, result)

    @time_zone.setter
    def time_zone(self, value: Optional[str]):
        self._properties["timeZone"] = value

    @property
    def time_format(self) -> Optional[str]:
        """Optional[str]: Format used to parse TIME values. Supports C-style and SQL-style values.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ExternalDataConfiguration.FIELDS.time_format
        """
        result = self._properties.get("timeFormat")
        return typing.cast(str, result)

    @time_format.setter
    def time_format(self, value: Optional[str]):
        self._properties["timeFormat"] = value

    @property
    def timestamp_format(self) -> Optional[str]:
        """Optional[str]: Format used to parse TIMESTAMP values. Supports C-style and SQL-style values.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ExternalDataConfiguration.FIELDS.timestamp_format
        """
        result = self._properties.get("timestampFormat")
        return typing.cast(str, result)

    @timestamp_format.setter
    def timestamp_format(self, value: Optional[str]):
        self._properties["timestampFormat"] = value

    @property
    def connection_id(self):
        """Optional[str]: [Experimental] ID of a BigQuery Connection API
        resource.

        .. WARNING::

           This feature is experimental. Pre-GA features may have limited
           support, and changes to pre-GA features may not be compatible with
           other pre-GA versions.
        """
        return self._properties.get("connectionId")

    @connection_id.setter
    def connection_id(self, value):
        self._properties["connectionId"] = value

    @property
    def avro_options(self) -> Optional[AvroOptions]:
        """Additional properties to set if ``sourceFormat`` is set to AVRO.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ExternalDataConfiguration.FIELDS.avro_options
        """
        if self.source_format == ExternalSourceFormat.AVRO:
            self._properties.setdefault(AvroOptions._RESOURCE_NAME, {})
        resource = self._properties.get(AvroOptions._RESOURCE_NAME)
        if resource is None:
            return None
        options = AvroOptions()
        options._properties = resource
        return options

    @avro_options.setter
    def avro_options(self, value):
        if self.source_format != ExternalSourceFormat.AVRO:
            msg = f"Cannot set Avro options, source format is {self.source_format}"
            raise TypeError(msg)
        self._properties[AvroOptions._RESOURCE_NAME] = value._properties

    @property
    def bigtable_options(self) -> Optional[BigtableOptions]:
        """Additional properties to set if ``sourceFormat`` is set to BIGTABLE.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ExternalDataConfiguration.FIELDS.bigtable_options
        """
        if self.source_format == ExternalSourceFormat.BIGTABLE:
            self._properties.setdefault(BigtableOptions._RESOURCE_NAME, {})
        resource = self._properties.get(BigtableOptions._RESOURCE_NAME)
        if resource is None:
            return None
        options = BigtableOptions()
        options._properties = resource
        return options

    @bigtable_options.setter
    def bigtable_options(self, value):
        if self.source_format != ExternalSourceFormat.BIGTABLE:
            msg = f"Cannot set Bigtable options, source format is {self.source_format}"
            raise TypeError(msg)
        self._properties[BigtableOptions._RESOURCE_NAME] = value._properties

    @property
    def csv_options(self) -> Optional[CSVOptions]:
        """Additional properties to set if ``sourceFormat`` is set to CSV.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ExternalDataConfiguration.FIELDS.csv_options
        """
        if self.source_format == ExternalSourceFormat.CSV:
            self._properties.setdefault(CSVOptions._RESOURCE_NAME, {})
        resource = self._properties.get(CSVOptions._RESOURCE_NAME)
        if resource is None:
            return None
        options = CSVOptions()
        options._properties = resource
        return options

    @csv_options.setter
    def csv_options(self, value):
        if self.source_format != ExternalSourceFormat.CSV:
            msg = f"Cannot set CSV options, source format is {self.source_format}"
            raise TypeError(msg)
        self._properties[CSVOptions._RESOURCE_NAME] = value._properties

    @property
    def google_sheets_options(self) -> Optional[GoogleSheetsOptions]:
        """Additional properties to set if ``sourceFormat`` is set to
        GOOGLE_SHEETS.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ExternalDataConfiguration.FIELDS.google_sheets_options
        """
        if self.source_format == ExternalSourceFormat.GOOGLE_SHEETS:
            self._properties.setdefault(GoogleSheetsOptions._RESOURCE_NAME, {})
        resource = self._properties.get(GoogleSheetsOptions._RESOURCE_NAME)
        if resource is None:
            return None
        options = GoogleSheetsOptions()
        options._properties = resource
        return options

    @google_sheets_options.setter
    def google_sheets_options(self, value):
        if self.source_format != ExternalSourceFormat.GOOGLE_SHEETS:
            msg = f"Cannot set Google Sheets options, source format is {self.source_format}"
            raise TypeError(msg)
        self._properties[GoogleSheetsOptions._RESOURCE_NAME] = value._properties

    @property
    def parquet_options(self) -> Optional[ParquetOptions]:
        """Additional properties to set if ``sourceFormat`` is set to PARQUET.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#ExternalDataConfiguration.FIELDS.parquet_options
        """
        if self.source_format == ExternalSourceFormat.PARQUET:
            self._properties.setdefault(ParquetOptions._RESOURCE_NAME, {})
        resource = self._properties.get(ParquetOptions._RESOURCE_NAME)
        if resource is None:
            return None
        options = ParquetOptions()
        options._properties = resource
        return options

    @parquet_options.setter
    def parquet_options(self, value):
        if self.source_format != ExternalSourceFormat.PARQUET:
            msg = f"Cannot set Parquet options, source format is {self.source_format}"
            raise TypeError(msg)
        self._properties[ParquetOptions._RESOURCE_NAME] = value._properties

    def to_api_repr(self) -> dict:
        """Build an API representation of this object.

        Returns:
            Dict[str, Any]:
                A dictionary in the format used by the BigQuery API.
        """
        config = copy.deepcopy(self._properties)
        return config

    @classmethod
    def from_api_repr(cls, resource: dict) -> "ExternalConfig":
        """Factory: construct an :class:`~.external_config.ExternalConfig`
        instance given its API representation.

        Args:
            resource (Dict[str, Any]):
                Definition of an :class:`~.external_config.ExternalConfig`
                instance in the same representation as is returned from the
                API.

        Returns:
            ExternalConfig: Configuration parsed from ``resource``.
        """
        config = cls(resource["sourceFormat"])
        config._properties = copy.deepcopy(resource)
        return config


class ExternalCatalogDatasetOptions:
    """Options defining open source compatible datasets living in the BigQuery catalog.
    Contains metadata of open source database, schema or namespace represented
    by the current dataset.

    Args:
        default_storage_location_uri (Optional[str]): The storage location URI for all
            tables in the dataset. Equivalent to hive metastore's database
            locationUri. Maximum length of 1024 characters. (str)
        parameters (Optional[dict[str, Any]]): A map of key value pairs defining the parameters
            and properties of the open source schema. Maximum size of 2Mib.
    """

    def __init__(
        self,
        default_storage_location_uri: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self._properties: Dict[str, Any] = {}
        self.default_storage_location_uri = default_storage_location_uri
        self.parameters = parameters

    @property
    def default_storage_location_uri(self) -> Optional[str]:
        """Optional. The storage location URI for all tables in the dataset.
        Equivalent to hive metastore's database locationUri. Maximum length of
        1024 characters."""

        return self._properties.get("defaultStorageLocationUri")

    @default_storage_location_uri.setter
    def default_storage_location_uri(self, value: Optional[str]):
        value = _helpers._isinstance_or_raise(value, str, none_allowed=True)
        self._properties["defaultStorageLocationUri"] = value

    @property
    def parameters(self) -> Optional[Dict[str, Any]]:
        """Optional. A map of key value pairs defining the parameters and
        properties of the open source schema. Maximum size of 2Mib."""

        return self._properties.get("parameters")

    @parameters.setter
    def parameters(self, value: Optional[Dict[str, Any]]):
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
    def from_api_repr(cls, api_repr: dict) -> ExternalCatalogDatasetOptions:
        """Factory: constructs an instance of the class (cls)
        given its API representation.

        Args:
            api_repr (Dict[str, Any]):
                API representation of the object to be instantiated.

        Returns:
            An instance of the class initialized with data from 'resource'.
        """
        config = cls()
        config._properties = api_repr
        return config


class ExternalCatalogTableOptions:
    """Metadata about open source compatible table. The fields contained in these
    options correspond to hive metastore's table level properties.

    Args:
        connection_id  (Optional[str]): The connection specifying the credentials to be
            used to read external storage, such as Azure Blob, Cloud Storage, or
            S3. The connection is needed to read the open source table from
            BigQuery Engine. The connection_id can have the form `..` or
            `projects//locations//connections/`.
        parameters (Union[Dict[str, Any], None]): A map of key value pairs defining the parameters
            and properties of the open source table. Corresponds with hive meta
            store table parameters. Maximum size of 4Mib.
        storage_descriptor (Optional[StorageDescriptor]): A storage descriptor containing information
            about the physical storage of this table.
    """

    def __init__(
        self,
        connection_id: Optional[str] = None,
        parameters: Union[Dict[str, Any], None] = None,
        storage_descriptor: Optional[schema.StorageDescriptor] = None,
    ):
        self._properties: Dict[str, Any] = {}
        self.connection_id = connection_id
        self.parameters = parameters
        self.storage_descriptor = storage_descriptor

    @property
    def connection_id(self) -> Optional[str]:
        """Optional. The connection specifying the credentials to be
        used to read external storage, such as Azure Blob, Cloud Storage, or
        S3. The connection is needed to read the open source table from
        BigQuery Engine. The connection_id can have the form `..` or
        `projects//locations//connections/`.
        """

        return self._properties.get("connectionId")

    @connection_id.setter
    def connection_id(self, value: Optional[str]):
        value = _helpers._isinstance_or_raise(value, str, none_allowed=True)
        self._properties["connectionId"] = value

    @property
    def parameters(self) -> Union[Dict[str, Any], None]:
        """Optional. A map of key value pairs defining the parameters and
        properties of the open source table. Corresponds with hive meta
        store table parameters. Maximum size of 4Mib.
        """

        return self._properties.get("parameters")

    @parameters.setter
    def parameters(self, value: Union[Dict[str, Any], None]):
        value = _helpers._isinstance_or_raise(value, dict, none_allowed=True)
        self._properties["parameters"] = value

    @property
    def storage_descriptor(self) -> Any:
        """Optional. A storage descriptor containing information about the
        physical storage of this table."""

        prop = _helpers._get_sub_prop(self._properties, ["storageDescriptor"])

        if prop is not None:
            return schema.StorageDescriptor.from_api_repr(prop)
        return None

    @storage_descriptor.setter
    def storage_descriptor(self, value: Union[schema.StorageDescriptor, dict, None]):
        value = _helpers._isinstance_or_raise(
            value, (schema.StorageDescriptor, dict), none_allowed=True
        )
        if isinstance(value, schema.StorageDescriptor):
            self._properties["storageDescriptor"] = value.to_api_repr()
        else:
            self._properties["storageDescriptor"] = value

    def to_api_repr(self) -> dict:
        """Build an API representation of this object.

        Returns:
            Dict[str, Any]:
                A dictionary in the format used by the BigQuery API.
        """

        return self._properties

    @classmethod
    def from_api_repr(cls, api_repr: dict) -> ExternalCatalogTableOptions:
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
