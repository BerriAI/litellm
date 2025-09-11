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

"""Classes for load jobs."""

import typing
from typing import FrozenSet, List, Iterable, Optional, Union

from google.cloud.bigquery.encryption_configuration import EncryptionConfiguration
from google.cloud.bigquery.enums import SourceColumnMatch
from google.cloud.bigquery.external_config import HivePartitioningOptions
from google.cloud.bigquery.format_options import ParquetOptions
from google.cloud.bigquery import _helpers
from google.cloud.bigquery.schema import SchemaField
from google.cloud.bigquery.schema import _to_schema_fields
from google.cloud.bigquery.table import RangePartitioning
from google.cloud.bigquery.table import TableReference
from google.cloud.bigquery.table import TimePartitioning
from google.cloud.bigquery.job.base import _AsyncJob
from google.cloud.bigquery.job.base import _JobConfig
from google.cloud.bigquery.job.base import _JobReference
from google.cloud.bigquery.query import ConnectionProperty


class ColumnNameCharacterMap:
    """Indicates the character map used for column names.

    https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#columnnamecharactermap
    """

    COLUMN_NAME_CHARACTER_MAP_UNSPECIFIED = "COLUMN_NAME_CHARACTER_MAP_UNSPECIFIED"
    """Unspecified column name character map."""

    STRICT = "STRICT"
    """Support flexible column name and reject invalid column names."""

    V1 = "V1"
    """	Support alphanumeric + underscore characters and names must start with
    a letter or underscore. Invalid column names will be normalized."""

    V2 = "V2"
    """Support flexible column name. Invalid column names will be normalized."""


class LoadJobConfig(_JobConfig):
    """Configuration options for load jobs.

    Set properties on the constructed configuration by using the property name
    as the name of a keyword argument. Values which are unset or :data:`None`
    use the BigQuery REST API default values. See the `BigQuery REST API
    reference documentation
    <https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad>`_
    for a list of default values.

    Required options differ based on the
    :attr:`~google.cloud.bigquery.job.LoadJobConfig.source_format` value.
    For example, the BigQuery API's default value for
    :attr:`~google.cloud.bigquery.job.LoadJobConfig.source_format` is ``"CSV"``.
    When loading a CSV file, either
    :attr:`~google.cloud.bigquery.job.LoadJobConfig.schema` must be set or
    :attr:`~google.cloud.bigquery.job.LoadJobConfig.autodetect` must be set to
    :data:`True`.
    """

    def __init__(self, **kwargs) -> None:
        super(LoadJobConfig, self).__init__("load", **kwargs)

    @property
    def allow_jagged_rows(self):
        """Optional[bool]: Allow missing trailing optional columns (CSV only).

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.allow_jagged_rows
        """
        return self._get_sub_prop("allowJaggedRows")

    @allow_jagged_rows.setter
    def allow_jagged_rows(self, value):
        self._set_sub_prop("allowJaggedRows", value)

    @property
    def allow_quoted_newlines(self):
        """Optional[bool]: Allow quoted data containing newline characters (CSV only).

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.allow_quoted_newlines
        """
        return self._get_sub_prop("allowQuotedNewlines")

    @allow_quoted_newlines.setter
    def allow_quoted_newlines(self, value):
        self._set_sub_prop("allowQuotedNewlines", value)

    @property
    def autodetect(self):
        """Optional[bool]: Automatically infer the schema from a sample of the data.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.autodetect
        """
        return self._get_sub_prop("autodetect")

    @autodetect.setter
    def autodetect(self, value):
        self._set_sub_prop("autodetect", value)

    @property
    def clustering_fields(self):
        """Optional[List[str]]: Fields defining clustering for the table

        (Defaults to :data:`None`).

        Clustering fields are immutable after table creation.

        .. note::

           BigQuery supports clustering for both partitioned and
           non-partitioned tables.
        """
        prop = self._get_sub_prop("clustering")
        if prop is not None:
            return list(prop.get("fields", ()))

    @clustering_fields.setter
    def clustering_fields(self, value):
        """Optional[List[str]]: Fields defining clustering for the table

        (Defaults to :data:`None`).
        """
        if value is not None:
            self._set_sub_prop("clustering", {"fields": value})
        else:
            self._del_sub_prop("clustering")

    @property
    def connection_properties(self) -> List[ConnectionProperty]:
        """Connection properties.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.connection_properties

        .. versionadded:: 3.7.0
        """
        resource = self._get_sub_prop("connectionProperties", [])
        return [ConnectionProperty.from_api_repr(prop) for prop in resource]

    @connection_properties.setter
    def connection_properties(self, value: Iterable[ConnectionProperty]):
        self._set_sub_prop(
            "connectionProperties",
            [prop.to_api_repr() for prop in value],
        )

    @property
    def create_disposition(self):
        """Optional[google.cloud.bigquery.job.CreateDisposition]: Specifies behavior
        for creating tables.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.create_disposition
        """
        return self._get_sub_prop("createDisposition")

    @create_disposition.setter
    def create_disposition(self, value):
        self._set_sub_prop("createDisposition", value)

    @property
    def create_session(self) -> Optional[bool]:
        """[Preview] If :data:`True`, creates a new session, where
        :attr:`~google.cloud.bigquery.job.LoadJob.session_info` will contain a
        random server generated session id.

        If :data:`False`, runs load job with an existing ``session_id`` passed in
        :attr:`~google.cloud.bigquery.job.LoadJobConfig.connection_properties`,
        otherwise runs load job in non-session mode.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.create_session

        .. versionadded:: 3.7.0
        """
        return self._get_sub_prop("createSession")

    @create_session.setter
    def create_session(self, value: Optional[bool]):
        self._set_sub_prop("createSession", value)

    @property
    def decimal_target_types(self) -> Optional[FrozenSet[str]]:
        """Possible SQL data types to which the source decimal values are converted.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.decimal_target_types

        .. versionadded:: 2.21.0
        """
        prop = self._get_sub_prop("decimalTargetTypes")
        if prop is not None:
            prop = frozenset(prop)
        return prop

    @decimal_target_types.setter
    def decimal_target_types(self, value: Optional[Iterable[str]]):
        if value is not None:
            self._set_sub_prop("decimalTargetTypes", list(value))
        else:
            self._del_sub_prop("decimalTargetTypes")

    @property
    def destination_encryption_configuration(self):
        """Optional[google.cloud.bigquery.encryption_configuration.EncryptionConfiguration]: Custom
        encryption configuration for the destination table.

        Custom encryption configuration (e.g., Cloud KMS keys) or :data:`None`
        if using default encryption.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.destination_encryption_configuration
        """
        prop = self._get_sub_prop("destinationEncryptionConfiguration")
        if prop is not None:
            prop = EncryptionConfiguration.from_api_repr(prop)
        return prop

    @destination_encryption_configuration.setter
    def destination_encryption_configuration(self, value):
        api_repr = value
        if value is not None:
            api_repr = value.to_api_repr()
            self._set_sub_prop("destinationEncryptionConfiguration", api_repr)
        else:
            self._del_sub_prop("destinationEncryptionConfiguration")

    @property
    def destination_table_description(self):
        """Optional[str]: Description of the destination table.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#DestinationTableProperties.FIELDS.description
        """
        prop = self._get_sub_prop("destinationTableProperties")
        if prop is not None:
            return prop["description"]

    @destination_table_description.setter
    def destination_table_description(self, value):
        keys = [self._job_type, "destinationTableProperties", "description"]
        if value is not None:
            _helpers._set_sub_prop(self._properties, keys, value)
        else:
            _helpers._del_sub_prop(self._properties, keys)

    @property
    def destination_table_friendly_name(self):
        """Optional[str]: Name given to destination table.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#DestinationTableProperties.FIELDS.friendly_name
        """
        prop = self._get_sub_prop("destinationTableProperties")
        if prop is not None:
            return prop["friendlyName"]

    @destination_table_friendly_name.setter
    def destination_table_friendly_name(self, value):
        keys = [self._job_type, "destinationTableProperties", "friendlyName"]
        if value is not None:
            _helpers._set_sub_prop(self._properties, keys, value)
        else:
            _helpers._del_sub_prop(self._properties, keys)

    @property
    def encoding(self):
        """Optional[google.cloud.bigquery.job.Encoding]: The character encoding of the
        data.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.encoding
        """
        return self._get_sub_prop("encoding")

    @encoding.setter
    def encoding(self, value):
        self._set_sub_prop("encoding", value)

    @property
    def field_delimiter(self):
        """Optional[str]: The separator for fields in a CSV file.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.field_delimiter
        """
        return self._get_sub_prop("fieldDelimiter")

    @field_delimiter.setter
    def field_delimiter(self, value):
        self._set_sub_prop("fieldDelimiter", value)

    @property
    def hive_partitioning(self):
        """Optional[:class:`~.external_config.HivePartitioningOptions`]: [Beta] When set, \
        it configures hive partitioning support.

        .. note::
            **Experimental**. This feature is experimental and might change or
            have limited support.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.hive_partitioning_options
        """
        prop = self._get_sub_prop("hivePartitioningOptions")
        if prop is None:
            return None
        return HivePartitioningOptions.from_api_repr(prop)

    @hive_partitioning.setter
    def hive_partitioning(self, value):
        if value is not None:
            if isinstance(value, HivePartitioningOptions):
                value = value.to_api_repr()
            else:
                raise TypeError("Expected a HivePartitioningOptions instance or None.")

        self._set_sub_prop("hivePartitioningOptions", value)

    @property
    def ignore_unknown_values(self):
        """Optional[bool]: Ignore extra values not represented in the table schema.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.ignore_unknown_values
        """
        return self._get_sub_prop("ignoreUnknownValues")

    @ignore_unknown_values.setter
    def ignore_unknown_values(self, value):
        self._set_sub_prop("ignoreUnknownValues", value)

    @property
    def json_extension(self):
        """Optional[str]: The extension to use for writing JSON data to BigQuery. Only supports GeoJSON currently.

        See: https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.json_extension

        """
        return self._get_sub_prop("jsonExtension")

    @json_extension.setter
    def json_extension(self, value):
        self._set_sub_prop("jsonExtension", value)

    @property
    def max_bad_records(self):
        """Optional[int]: Number of invalid rows to ignore.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.max_bad_records
        """
        return _helpers._int_or_none(self._get_sub_prop("maxBadRecords"))

    @max_bad_records.setter
    def max_bad_records(self, value):
        self._set_sub_prop("maxBadRecords", value)

    @property
    def null_marker(self):
        """Optional[str]: Represents a null value (CSV only).

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.null_marker
        """
        return self._get_sub_prop("nullMarker")

    @null_marker.setter
    def null_marker(self, value):
        self._set_sub_prop("nullMarker", value)

    @property
    def null_markers(self) -> Optional[List[str]]:
        """Optional[List[str]]: A list of strings represented as SQL NULL values in a CSV file.

        .. note::
            null_marker and null_markers can't be set at the same time.
            If null_marker is set, null_markers has to be not set.
            If null_markers is set, null_marker has to be not set.
            If both null_marker and null_markers are set at the same time, a user error would be thrown.
            Any strings listed in null_markers, including empty string would be interpreted as SQL NULL.
            This applies to all column types.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.null_markers
        """
        return self._get_sub_prop("nullMarkers")

    @null_markers.setter
    def null_markers(self, value: Optional[List[str]]):
        self._set_sub_prop("nullMarkers", value)

    @property
    def preserve_ascii_control_characters(self):
        """Optional[bool]: Preserves the embedded ASCII control characters when sourceFormat is set to CSV.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.preserve_ascii_control_characters
        """
        return self._get_sub_prop("preserveAsciiControlCharacters")

    @preserve_ascii_control_characters.setter
    def preserve_ascii_control_characters(self, value):
        self._set_sub_prop("preserveAsciiControlCharacters", bool(value))

    @property
    def projection_fields(self) -> Optional[List[str]]:
        """Optional[List[str]]: If
        :attr:`google.cloud.bigquery.job.LoadJobConfig.source_format` is set to
        "DATASTORE_BACKUP", indicates which entity properties to load into
        BigQuery from a Cloud Datastore backup.

        Property names are case sensitive and must be top-level properties. If
        no properties are specified, BigQuery loads all properties. If any
        named property isn't found in the Cloud Datastore backup, an invalid
        error is returned in the job result.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.projection_fields
        """
        return self._get_sub_prop("projectionFields")

    @projection_fields.setter
    def projection_fields(self, value: Optional[List[str]]):
        self._set_sub_prop("projectionFields", value)

    @property
    def quote_character(self):
        """Optional[str]: Character used to quote data sections (CSV only).

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.quote
        """
        return self._get_sub_prop("quote")

    @quote_character.setter
    def quote_character(self, value):
        self._set_sub_prop("quote", value)

    @property
    def range_partitioning(self):
        """Optional[google.cloud.bigquery.table.RangePartitioning]:
        Configures range-based partitioning for destination table.

        .. note::
            **Beta**. The integer range partitioning feature is in a
            pre-release state and might change or have limited support.

        Only specify at most one of
        :attr:`~google.cloud.bigquery.job.LoadJobConfig.time_partitioning` or
        :attr:`~google.cloud.bigquery.job.LoadJobConfig.range_partitioning`.

        Raises:
            ValueError:
                If the value is not
                :class:`~google.cloud.bigquery.table.RangePartitioning` or
                :data:`None`.
        """
        resource = self._get_sub_prop("rangePartitioning")
        if resource is not None:
            return RangePartitioning(_properties=resource)

    @range_partitioning.setter
    def range_partitioning(self, value):
        resource = value
        if isinstance(value, RangePartitioning):
            resource = value._properties
        elif value is not None:
            raise ValueError(
                "Expected value to be RangePartitioning or None, got {}.".format(value)
            )
        self._set_sub_prop("rangePartitioning", resource)

    @property
    def reference_file_schema_uri(self):
        """Optional[str]:
        When creating an external table, the user can provide a reference file with the
        table schema. This is enabled for the following formats:

        AVRO, PARQUET, ORC
        """
        return self._get_sub_prop("referenceFileSchemaUri")

    @reference_file_schema_uri.setter
    def reference_file_schema_uri(self, value):
        return self._set_sub_prop("referenceFileSchemaUri", value)

    @property
    def schema(self):
        """Optional[Sequence[Union[ \
            :class:`~google.cloud.bigquery.schema.SchemaField`, \
            Mapping[str, Any] \
        ]]]: Schema of the destination table.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.schema
        """
        schema = _helpers._get_sub_prop(self._properties, ["load", "schema", "fields"])
        if schema is None:
            return
        return [SchemaField.from_api_repr(field) for field in schema]

    @schema.setter
    def schema(self, value):
        if value is None:
            self._del_sub_prop("schema")
            return

        value = _to_schema_fields(value)

        _helpers._set_sub_prop(
            self._properties,
            ["load", "schema", "fields"],
            [field.to_api_repr() for field in value],
        )

    @property
    def schema_update_options(self):
        """Optional[List[google.cloud.bigquery.job.SchemaUpdateOption]]: Specifies
        updates to the destination table schema to allow as a side effect of
        the load job.
        """
        return self._get_sub_prop("schemaUpdateOptions")

    @schema_update_options.setter
    def schema_update_options(self, values):
        self._set_sub_prop("schemaUpdateOptions", values)

    @property
    def skip_leading_rows(self):
        """Optional[int]: Number of rows to skip when reading data (CSV only).

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.skip_leading_rows
        """
        return _helpers._int_or_none(self._get_sub_prop("skipLeadingRows"))

    @skip_leading_rows.setter
    def skip_leading_rows(self, value):
        self._set_sub_prop("skipLeadingRows", str(value))

    @property
    def source_format(self):
        """Optional[google.cloud.bigquery.job.SourceFormat]: File format of the data.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.source_format
        """
        return self._get_sub_prop("sourceFormat")

    @source_format.setter
    def source_format(self, value):
        self._set_sub_prop("sourceFormat", value)

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

        See:

        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.source_column_match
        """
        value = self._get_sub_prop("sourceColumnMatch")
        return SourceColumnMatch(value) if value is not None else None

    @source_column_match.setter
    def source_column_match(self, value: Union[SourceColumnMatch, str, None]):
        if value is not None and not isinstance(value, (SourceColumnMatch, str)):
            raise TypeError(
                "value must be a google.cloud.bigquery.enums.SourceColumnMatch, str, or None"
            )
        if isinstance(value, SourceColumnMatch):
            value = value.value
        self._set_sub_prop("sourceColumnMatch", value if value else None)

    @property
    def date_format(self) -> Optional[str]:
        """Optional[str]: Date format used for parsing DATE values.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.date_format
        """
        return self._get_sub_prop("dateFormat")

    @date_format.setter
    def date_format(self, value: Optional[str]):
        self._set_sub_prop("dateFormat", value)

    @property
    def datetime_format(self) -> Optional[str]:
        """Optional[str]: Date format used for parsing DATETIME values.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.datetime_format
        """
        return self._get_sub_prop("datetimeFormat")

    @datetime_format.setter
    def datetime_format(self, value: Optional[str]):
        self._set_sub_prop("datetimeFormat", value)

    @property
    def time_zone(self) -> Optional[str]:
        """Optional[str]: Default time zone that will apply when parsing timestamp
        values that have no specific time zone.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.time_zone
        """
        return self._get_sub_prop("timeZone")

    @time_zone.setter
    def time_zone(self, value: Optional[str]):
        self._set_sub_prop("timeZone", value)

    @property
    def time_format(self) -> Optional[str]:
        """Optional[str]: Date format used for parsing TIME values.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.time_format
        """
        return self._get_sub_prop("timeFormat")

    @time_format.setter
    def time_format(self, value: Optional[str]):
        self._set_sub_prop("timeFormat", value)

    @property
    def timestamp_format(self) -> Optional[str]:
        """Optional[str]: Date format used for parsing TIMESTAMP values.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.timestamp_format
        """
        return self._get_sub_prop("timestampFormat")

    @timestamp_format.setter
    def timestamp_format(self, value: Optional[str]):
        self._set_sub_prop("timestampFormat", value)

    @property
    def time_partitioning(self):
        """Optional[google.cloud.bigquery.table.TimePartitioning]: Specifies time-based
        partitioning for the destination table.

        Only specify at most one of
        :attr:`~google.cloud.bigquery.job.LoadJobConfig.time_partitioning` or
        :attr:`~google.cloud.bigquery.job.LoadJobConfig.range_partitioning`.
        """
        prop = self._get_sub_prop("timePartitioning")
        if prop is not None:
            prop = TimePartitioning.from_api_repr(prop)
        return prop

    @time_partitioning.setter
    def time_partitioning(self, value):
        api_repr = value
        if value is not None:
            api_repr = value.to_api_repr()
            self._set_sub_prop("timePartitioning", api_repr)
        else:
            self._del_sub_prop("timePartitioning")

    @property
    def use_avro_logical_types(self):
        """Optional[bool]: For loads of Avro data, governs whether Avro logical types are
        converted to their corresponding BigQuery types (e.g. TIMESTAMP) rather than
        raw types (e.g. INTEGER).
        """
        return self._get_sub_prop("useAvroLogicalTypes")

    @use_avro_logical_types.setter
    def use_avro_logical_types(self, value):
        self._set_sub_prop("useAvroLogicalTypes", bool(value))

    @property
    def write_disposition(self):
        """Optional[google.cloud.bigquery.job.WriteDisposition]: Action that occurs if
        the destination table already exists.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.write_disposition
        """
        return self._get_sub_prop("writeDisposition")

    @write_disposition.setter
    def write_disposition(self, value):
        self._set_sub_prop("writeDisposition", value)

    @property
    def parquet_options(self):
        """Optional[google.cloud.bigquery.format_options.ParquetOptions]: Additional
            properties to set if ``sourceFormat`` is set to PARQUET.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.parquet_options
        """
        prop = self._get_sub_prop("parquetOptions")
        if prop is not None:
            prop = ParquetOptions.from_api_repr(prop)
        return prop

    @parquet_options.setter
    def parquet_options(self, value):
        if value is not None:
            self._set_sub_prop("parquetOptions", value.to_api_repr())
        else:
            self._del_sub_prop("parquetOptions")

    @property
    def column_name_character_map(self) -> str:
        """Optional[google.cloud.bigquery.job.ColumnNameCharacterMap]:
        Character map supported for column names in CSV/Parquet loads. Defaults
        to STRICT and can be overridden by Project Config Service. Using this
        option with unsupported load formats will result in an error.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.column_name_character_map
        """
        return self._get_sub_prop(
            "columnNameCharacterMap",
            ColumnNameCharacterMap.COLUMN_NAME_CHARACTER_MAP_UNSPECIFIED,
        )

    @column_name_character_map.setter
    def column_name_character_map(self, value: Optional[str]):
        if value is None:
            value = ColumnNameCharacterMap.COLUMN_NAME_CHARACTER_MAP_UNSPECIFIED
        self._set_sub_prop("columnNameCharacterMap", value)


class LoadJob(_AsyncJob):
    """Asynchronous job for loading data into a table.

    Can load from Google Cloud Storage URIs or from a file.

    Args:
        job_id (str): the job's ID

        source_uris (Optional[Sequence[str]]):
            URIs of one or more data files to be loaded.  See
            https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.source_uris
            for supported URI formats. Pass None for jobs that load from a file.

        destination (google.cloud.bigquery.table.TableReference): reference to table into which data is to be loaded.

        client (google.cloud.bigquery.client.Client):
            A client which holds credentials and project configuration
            for the dataset (which requires a project).
    """

    _JOB_TYPE = "load"
    _CONFIG_CLASS = LoadJobConfig

    def __init__(self, job_id, source_uris, destination, client, job_config=None):
        super(LoadJob, self).__init__(job_id, client)

        if job_config is not None:
            self._properties["configuration"] = job_config._properties

        if source_uris is not None:
            _helpers._set_sub_prop(
                self._properties, ["configuration", "load", "sourceUris"], source_uris
            )

        if destination is not None:
            _helpers._set_sub_prop(
                self._properties,
                ["configuration", "load", "destinationTable"],
                destination.to_api_repr(),
            )

    @property
    def configuration(self) -> LoadJobConfig:
        """The configuration for this load job."""
        return typing.cast(LoadJobConfig, super().configuration)

    @property
    def destination(self):
        """google.cloud.bigquery.table.TableReference: table where loaded rows are written

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.destination_table
        """
        dest_config = _helpers._get_sub_prop(
            self._properties, ["configuration", "load", "destinationTable"]
        )
        return TableReference.from_api_repr(dest_config)

    @property
    def source_uris(self):
        """Optional[Sequence[str]]: URIs of data files to be loaded. See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.source_uris
        for supported URI formats. None for jobs that load from a file.
        """
        return _helpers._get_sub_prop(
            self._properties, ["configuration", "load", "sourceUris"]
        )

    @property
    def allow_jagged_rows(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.allow_jagged_rows`.
        """
        return self.configuration.allow_jagged_rows

    @property
    def allow_quoted_newlines(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.allow_quoted_newlines`.
        """
        return self.configuration.allow_quoted_newlines

    @property
    def autodetect(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.autodetect`.
        """
        return self.configuration.autodetect

    @property
    def connection_properties(self) -> List[ConnectionProperty]:
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.connection_properties`.

        .. versionadded:: 3.7.0
        """
        return self.configuration.connection_properties

    @property
    def create_disposition(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.create_disposition`.
        """
        return self.configuration.create_disposition

    @property
    def create_session(self) -> Optional[bool]:
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.create_session`.

        .. versionadded:: 3.7.0
        """
        return self.configuration.create_session

    @property
    def encoding(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.encoding`.
        """
        return self.configuration.encoding

    @property
    def field_delimiter(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.field_delimiter`.
        """
        return self.configuration.field_delimiter

    @property
    def ignore_unknown_values(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.ignore_unknown_values`.
        """
        return self.configuration.ignore_unknown_values

    @property
    def max_bad_records(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.max_bad_records`.
        """
        return self.configuration.max_bad_records

    @property
    def null_marker(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.null_marker`.
        """
        return self.configuration.null_marker

    @property
    def null_markers(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.null_markers`.
        """
        return self.configuration.null_markers

    @property
    def quote_character(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.quote_character`.
        """
        return self.configuration.quote_character

    @property
    def reference_file_schema_uri(self):
        """See:
        attr:`google.cloud.bigquery.job.LoadJobConfig.reference_file_schema_uri`.
        """
        return self.configuration.reference_file_schema_uri

    @property
    def skip_leading_rows(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.skip_leading_rows`.
        """
        return self.configuration.skip_leading_rows

    @property
    def source_format(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.source_format`.
        """
        return self.configuration.source_format

    @property
    def write_disposition(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.write_disposition`.
        """
        return self.configuration.write_disposition

    @property
    def schema(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.schema`.
        """
        return self.configuration.schema

    @property
    def destination_encryption_configuration(self):
        """google.cloud.bigquery.encryption_configuration.EncryptionConfiguration: Custom
        encryption configuration for the destination table.

        Custom encryption configuration (e.g., Cloud KMS keys)
        or :data:`None` if using default encryption.

        See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.destination_encryption_configuration`.
        """
        return self.configuration.destination_encryption_configuration

    @property
    def destination_table_description(self):
        """Optional[str] name given to destination table.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#DestinationTableProperties.FIELDS.description
        """
        return self.configuration.destination_table_description

    @property
    def destination_table_friendly_name(self):
        """Optional[str] name given to destination table.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#DestinationTableProperties.FIELDS.friendly_name
        """
        return self.configuration.destination_table_friendly_name

    @property
    def range_partitioning(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.range_partitioning`.
        """
        return self.configuration.range_partitioning

    @property
    def time_partitioning(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.time_partitioning`.
        """
        return self.configuration.time_partitioning

    @property
    def use_avro_logical_types(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.use_avro_logical_types`.
        """
        return self.configuration.use_avro_logical_types

    @property
    def clustering_fields(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.clustering_fields`.
        """
        return self.configuration.clustering_fields

    @property
    def source_column_match(self) -> Optional[SourceColumnMatch]:
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.source_column_match`.
        """
        return self.configuration.source_column_match

    @property
    def date_format(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.date_format`.
        """
        return self.configuration.date_format

    @property
    def datetime_format(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.datetime_format`.
        """
        return self.configuration.datetime_format

    @property
    def time_zone(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.time_zone`.
        """
        return self.configuration.time_zone

    @property
    def time_format(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.time_format`.
        """
        return self.configuration.time_format

    @property
    def timestamp_format(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.timestamp_format`.
        """
        return self.configuration.timestamp_format

    @property
    def schema_update_options(self):
        """See
        :attr:`google.cloud.bigquery.job.LoadJobConfig.schema_update_options`.
        """
        return self.configuration.schema_update_options

    @property
    def input_file_bytes(self):
        """Count of bytes loaded from source files.

        Returns:
            Optional[int]: the count (None until set from the server).

        Raises:
            ValueError: for invalid value types.
        """
        return _helpers._int_or_none(
            _helpers._get_sub_prop(
                self._properties, ["statistics", "load", "inputFileBytes"]
            )
        )

    @property
    def input_files(self):
        """Count of source files.

        Returns:
            Optional[int]: the count (None until set from the server).
        """
        return _helpers._int_or_none(
            _helpers._get_sub_prop(
                self._properties, ["statistics", "load", "inputFiles"]
            )
        )

    @property
    def output_bytes(self):
        """Count of bytes saved to destination table.

        Returns:
            Optional[int]: the count (None until set from the server).
        """
        return _helpers._int_or_none(
            _helpers._get_sub_prop(
                self._properties, ["statistics", "load", "outputBytes"]
            )
        )

    @property
    def output_rows(self):
        """Count of rows saved to destination table.

        Returns:
            Optional[int]: the count (None until set from the server).
        """
        return _helpers._int_or_none(
            _helpers._get_sub_prop(
                self._properties, ["statistics", "load", "outputRows"]
            )
        )

    def to_api_repr(self):
        """Generate a resource for :meth:`_begin`."""
        # Exclude statistics, if set.
        return {
            "jobReference": self._properties["jobReference"],
            "configuration": self._properties["configuration"],
        }

    @classmethod
    def from_api_repr(cls, resource: dict, client) -> "LoadJob":
        """Factory:  construct a job given its API representation

        .. note::

           This method assumes that the project found in the resource matches
           the client's project.

        Args:
            resource (Dict): dataset job representation returned from the API

            client (google.cloud.bigquery.client.Client):
                Client which holds credentials and project
                configuration for the dataset.

        Returns:
            google.cloud.bigquery.job.LoadJob: Job parsed from ``resource``.
        """
        cls._check_resource_config(resource)
        job_ref = _JobReference._from_api_repr(resource["jobReference"])
        job = cls(job_ref, None, None, client)
        job._set_properties(resource)
        return job
