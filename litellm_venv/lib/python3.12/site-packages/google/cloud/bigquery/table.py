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

"""Define API Tables."""

from __future__ import absolute_import

import copy
import datetime
import functools
import operator
import typing
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union, Sequence

import warnings

try:
    import pandas  # type: ignore
except ImportError:
    pandas = None

try:
    import pyarrow  # type: ignore
except ImportError:
    pyarrow = None

try:
    import db_dtypes  # type: ignore
except ImportError:
    db_dtypes = None

try:
    import geopandas  # type: ignore
except ImportError:
    geopandas = None
finally:
    _COORDINATE_REFERENCE_SYSTEM = "EPSG:4326"

try:
    import shapely  # type: ignore
    from shapely import wkt  # type: ignore
except ImportError:
    shapely = None
else:
    _read_wkt = wkt.loads

import google.api_core.exceptions
from google.api_core.page_iterator import HTTPIterator

import google.cloud._helpers  # type: ignore
from google.cloud.bigquery import _helpers
from google.cloud.bigquery import _pandas_helpers
from google.cloud.bigquery import _versions_helpers
from google.cloud.bigquery import exceptions as bq_exceptions
from google.cloud.bigquery._tqdm_helpers import get_progress_bar
from google.cloud.bigquery.encryption_configuration import EncryptionConfiguration
from google.cloud.bigquery.enums import DefaultPandasDTypes
from google.cloud.bigquery.external_config import ExternalConfig
from google.cloud.bigquery import schema as _schema
from google.cloud.bigquery.schema import _build_schema_resource
from google.cloud.bigquery.schema import _parse_schema_resource
from google.cloud.bigquery.schema import _to_schema_fields
from google.cloud.bigquery import external_config

if typing.TYPE_CHECKING:  # pragma: NO COVER
    # Unconditionally import optional dependencies again to tell pytype that
    # they are not None, avoiding false "no attribute" errors.
    import pandas
    import pyarrow
    import geopandas  # type: ignore
    from google.cloud import bigquery_storage  # type: ignore
    from google.cloud.bigquery.dataset import DatasetReference


_NO_GEOPANDAS_ERROR = (
    "The geopandas library is not installed, please install "
    "geopandas to use the to_geodataframe() function."
)
_NO_PYARROW_ERROR = (
    "The pyarrow library is not installed, please install "
    "pyarrow to use the to_arrow() function."
)
_NO_SHAPELY_ERROR = (
    "The shapely library is not installed, please install "
    "shapely to use the geography_as_object option."
)

_TABLE_HAS_NO_SCHEMA = 'Table has no schema:  call "client.get_table()"'

_NO_SUPPORTED_DTYPE = (
    "The dtype cannot to be converted to a pandas ExtensionArray "
    "because the necessary `__from_arrow__` attribute is missing."
)

_RANGE_PYARROW_WARNING = (
    "Unable to represent RANGE schema as struct using pandas ArrowDtype. Using "
    "`object` instead. To use ArrowDtype, use pandas >= 1.5 and "
    "pyarrow >= 10.0.1."
)

# How many of the total rows need to be downloaded already for us to skip
# calling the BQ Storage API?
#
# In microbenchmarks on 2024-05-21, I (tswast@) measure that at about 2 MB of
# remaining results, it's faster to use the BQ Storage Read API to download
# the results than use jobs.getQueryResults. Since we don't have a good way to
# know the remaining bytes, we estimate by remaining number of rows.
#
# Except when rows themselves are larger, I observe that the a single page of
# results will be around 10 MB. Therefore, the proportion of rows already
# downloaded should be 10 (first page) / 12 (all results) or less for it to be
# worth it to make a call to jobs.getQueryResults.
ALMOST_COMPLETELY_CACHED_RATIO = 0.833333


def _reference_getter(table):
    """A :class:`~google.cloud.bigquery.table.TableReference` pointing to
    this table.

    Returns:
        google.cloud.bigquery.table.TableReference: pointer to this table.
    """
    from google.cloud.bigquery import dataset

    dataset_ref = dataset.DatasetReference(table.project, table.dataset_id)
    return TableReference(dataset_ref, table.table_id)


def _view_use_legacy_sql_getter(
    table: Union["Table", "TableListItem"]
) -> Optional[bool]:
    """bool: Specifies whether to execute the view with Legacy or Standard SQL.

    This boolean specifies whether to execute the view with Legacy SQL
    (:data:`True`) or Standard SQL (:data:`False`). The client side default is
    :data:`False`. The server-side default is :data:`True`. If this table is
    not a view, :data:`None` is returned.

    Raises:
        ValueError: For invalid value types.
    """

    view: Optional[Dict[str, Any]] = table._properties.get("view")
    if view is not None:
        # The server-side default for useLegacySql is True.
        return view.get("useLegacySql", True) if view is not None else True
    # In some cases, such as in a table list no view object is present, but the
    # resource still represents a view. Use the type as a fallback.
    if table.table_type == "VIEW":
        # The server-side default for useLegacySql is True.
        return True
    return None  # explicit return statement to appease mypy


class _TableBase:
    """Base class for Table-related classes with common functionality."""

    _PROPERTY_TO_API_FIELD: Dict[str, Union[str, List[str]]] = {
        "dataset_id": ["tableReference", "datasetId"],
        "project": ["tableReference", "projectId"],
        "table_id": ["tableReference", "tableId"],
    }

    def __init__(self):
        self._properties = {}

    @property
    def project(self) -> str:
        """Project bound to the table."""
        return _helpers._get_sub_prop(
            self._properties, self._PROPERTY_TO_API_FIELD["project"]
        )

    @property
    def dataset_id(self) -> str:
        """ID of dataset containing the table."""
        return _helpers._get_sub_prop(
            self._properties, self._PROPERTY_TO_API_FIELD["dataset_id"]
        )

    @property
    def table_id(self) -> str:
        """The table ID."""
        return _helpers._get_sub_prop(
            self._properties, self._PROPERTY_TO_API_FIELD["table_id"]
        )

    @property
    def path(self) -> str:
        """URL path for the table's APIs."""
        return (
            f"/projects/{self.project}/datasets/{self.dataset_id}"
            f"/tables/{self.table_id}"
        )

    def __eq__(self, other):
        if isinstance(other, _TableBase):
            return (
                self.project == other.project
                and self.dataset_id == other.dataset_id
                and self.table_id == other.table_id
            )
        else:
            return NotImplemented

    def __hash__(self):
        return hash((self.project, self.dataset_id, self.table_id))


class TableReference(_TableBase):
    """TableReferences are pointers to tables.

    See
    https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#tablereference

    Args:
        dataset_ref: A pointer to the dataset
        table_id: The ID of the table
    """

    _PROPERTY_TO_API_FIELD = {
        "dataset_id": "datasetId",
        "project": "projectId",
        "table_id": "tableId",
    }

    def __init__(self, dataset_ref: "DatasetReference", table_id: str):
        self._properties = {}

        _helpers._set_sub_prop(
            self._properties,
            self._PROPERTY_TO_API_FIELD["project"],
            dataset_ref.project,
        )
        _helpers._set_sub_prop(
            self._properties,
            self._PROPERTY_TO_API_FIELD["dataset_id"],
            dataset_ref.dataset_id,
        )
        _helpers._set_sub_prop(
            self._properties,
            self._PROPERTY_TO_API_FIELD["table_id"],
            table_id,
        )

    @classmethod
    def from_string(
        cls, table_id: str, default_project: Optional[str] = None
    ) -> "TableReference":
        """Construct a table reference from table ID string.

        Args:
            table_id (str):
                A table ID in standard SQL format. If ``default_project``
                is not specified, this must included a project ID, dataset
                ID, and table ID, each separated by ``.``.
            default_project (Optional[str]):
                The project ID to use when ``table_id`` does not
                include a project ID.

        Returns:
            TableReference: Table reference parsed from ``table_id``.

        Examples:
            >>> TableReference.from_string('my-project.mydataset.mytable')
            TableRef...(DatasetRef...('my-project', 'mydataset'), 'mytable')

        Raises:
            ValueError:
                If ``table_id`` is not a fully-qualified table ID in
                standard SQL format.
        """
        from google.cloud.bigquery.dataset import DatasetReference

        (
            output_project_id,
            output_dataset_id,
            output_table_id,
        ) = _helpers._parse_3_part_id(
            table_id, default_project=default_project, property_name="table_id"
        )

        return cls(
            DatasetReference(output_project_id, output_dataset_id), output_table_id
        )

    @classmethod
    def from_api_repr(cls, resource: dict) -> "TableReference":
        """Factory:  construct a table reference given its API representation

        Args:
            resource (Dict[str, object]):
                Table reference representation returned from the API

        Returns:
            google.cloud.bigquery.table.TableReference:
                Table reference parsed from ``resource``.
        """
        from google.cloud.bigquery.dataset import DatasetReference

        project = resource["projectId"]
        dataset_id = resource["datasetId"]
        table_id = resource["tableId"]

        return cls(DatasetReference(project, dataset_id), table_id)

    def to_api_repr(self) -> dict:
        """Construct the API resource representation of this table reference.

        Returns:
            Dict[str, object]: Table reference represented as an API resource
        """
        return copy.deepcopy(self._properties)

    def to_bqstorage(self) -> str:
        """Construct a BigQuery Storage API representation of this table.

        Install the ``google-cloud-bigquery-storage`` package to use this
        feature.

        If the ``table_id`` contains a partition identifier (e.g.
        ``my_table$201812``) or a snapshot identifier (e.g.
        ``mytable@1234567890``), it is ignored. Use
        :class:`google.cloud.bigquery_storage.types.ReadSession.TableReadOptions`
        to filter rows by partition. Use
        :class:`google.cloud.bigquery_storage.types.ReadSession.TableModifiers`
        to select a specific snapshot to read from.

        Returns:
            str: A reference to this table in the BigQuery Storage API.
        """

        table_id, _, _ = self.table_id.partition("@")
        table_id, _, _ = table_id.partition("$")

        table_ref = (
            f"projects/{self.project}/datasets/{self.dataset_id}/tables/{table_id}"
        )
        return table_ref

    def __str__(self):
        return f"{self.project}.{self.dataset_id}.{self.table_id}"

    def __repr__(self):
        from google.cloud.bigquery.dataset import DatasetReference

        dataset_ref = DatasetReference(self.project, self.dataset_id)
        return f"TableReference({dataset_ref!r}, '{self.table_id}')"


class Table(_TableBase):
    """Tables represent a set of rows whose values correspond to a schema.

    See
    https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#resource-table

    Args:
        table_ref (Union[google.cloud.bigquery.table.TableReference, str]):
            A pointer to a table. If ``table_ref`` is a string, it must
            included a project ID, dataset ID, and table ID, each separated
            by ``.``.
        schema (Optional[Sequence[Union[ \
                :class:`~google.cloud.bigquery.schema.SchemaField`, \
                Mapping[str, Any] \
        ]]]):
            The table's schema. If any item is a mapping, its content must be
            compatible with
            :meth:`~google.cloud.bigquery.schema.SchemaField.from_api_repr`.
    """

    _PROPERTY_TO_API_FIELD: Dict[str, Any] = {
        **_TableBase._PROPERTY_TO_API_FIELD,
        "biglake_configuration": "biglakeConfiguration",
        "clustering_fields": "clustering",
        "created": "creationTime",
        "description": "description",
        "encryption_configuration": "encryptionConfiguration",
        "etag": "etag",
        "expires": "expirationTime",
        "external_data_configuration": "externalDataConfiguration",
        "friendly_name": "friendlyName",
        "full_table_id": "id",
        "labels": "labels",
        "location": "location",
        "modified": "lastModifiedTime",
        "mview_enable_refresh": "materializedView",
        "mview_last_refresh_time": ["materializedView", "lastRefreshTime"],
        "mview_query": "materializedView",
        "mview_refresh_interval": "materializedView",
        "mview_allow_non_incremental_definition": "materializedView",
        "num_bytes": "numBytes",
        "num_rows": "numRows",
        "partition_expiration": "timePartitioning",
        "partitioning_type": "timePartitioning",
        "range_partitioning": "rangePartitioning",
        "time_partitioning": "timePartitioning",
        "schema": ["schema", "fields"],
        "snapshot_definition": "snapshotDefinition",
        "clone_definition": "cloneDefinition",
        "streaming_buffer": "streamingBuffer",
        "self_link": "selfLink",
        "type": "type",
        "view_use_legacy_sql": "view",
        "view_query": "view",
        "require_partition_filter": "requirePartitionFilter",
        "table_constraints": "tableConstraints",
        "max_staleness": "maxStaleness",
        "resource_tags": "resourceTags",
        "external_catalog_table_options": "externalCatalogTableOptions",
        "foreign_type_info": ["schema", "foreignTypeInfo"],
    }

    def __init__(self, table_ref, schema=None) -> None:
        table_ref = _table_arg_to_table_ref(table_ref)
        self._properties: Dict[str, Any] = {
            "tableReference": table_ref.to_api_repr(),
            "labels": {},
        }
        # Let the @property do validation.
        if schema is not None:
            self.schema = schema

    reference = property(_reference_getter)

    @property
    def biglake_configuration(self):
        """google.cloud.bigquery.table.BigLakeConfiguration: Configuration
        for managed tables for Apache Iceberg.

        See https://cloud.google.com/bigquery/docs/iceberg-tables for more information.
        """
        prop = self._properties.get(
            self._PROPERTY_TO_API_FIELD["biglake_configuration"]
        )
        if prop is not None:
            prop = BigLakeConfiguration.from_api_repr(prop)
        return prop

    @biglake_configuration.setter
    def biglake_configuration(self, value):
        api_repr = value
        if value is not None:
            api_repr = value.to_api_repr()
        self._properties[
            self._PROPERTY_TO_API_FIELD["biglake_configuration"]
        ] = api_repr

    @property
    def require_partition_filter(self):
        """bool: If set to true, queries over the partitioned table require a
        partition filter that can be used for partition elimination to be
        specified.
        """
        return self._properties.get(
            self._PROPERTY_TO_API_FIELD["require_partition_filter"]
        )

    @require_partition_filter.setter
    def require_partition_filter(self, value):
        self._properties[
            self._PROPERTY_TO_API_FIELD["require_partition_filter"]
        ] = value

    @property
    def schema(self):
        """Sequence[Union[ \
                :class:`~google.cloud.bigquery.schema.SchemaField`, \
                Mapping[str, Any] \
        ]]:
            Table's schema.

        Raises:
            Exception:
                If ``schema`` is not a sequence, or if any item in the sequence
                is not a :class:`~google.cloud.bigquery.schema.SchemaField`
                instance or a compatible mapping representation of the field.

        .. Note::
            If you are referencing a schema for an external catalog table such
            as a Hive table, it will also be necessary to populate the foreign_type_info
            attribute. This is not necessary if defining the schema for a BigQuery table.

            For details, see:
            https://cloud.google.com/bigquery/docs/external-tables
            https://cloud.google.com/bigquery/docs/datasets-intro#external_datasets

        """
        prop = _helpers._get_sub_prop(
            self._properties, self._PROPERTY_TO_API_FIELD["schema"]
        )
        if not prop:
            return []
        else:
            return _parse_schema_resource(prop)

    @schema.setter
    def schema(self, value):
        api_field = self._PROPERTY_TO_API_FIELD["schema"]

        if value is None:
            _helpers._set_sub_prop(
                self._properties,
                api_field,
                None,
            )
        elif isinstance(value, Sequence):
            value = _to_schema_fields(value)
            value = _build_schema_resource(value)
            _helpers._set_sub_prop(
                self._properties,
                api_field,
                value,
            )
        else:
            raise TypeError("Schema must be a Sequence (e.g. a list) or None.")

    @property
    def labels(self):
        """Dict[str, str]: Labels for the table.

        This method always returns a dict. To change a table's labels,
        modify the dict, then call ``Client.update_table``. To delete a
        label, set its value to :data:`None` before updating.

        Raises:
            ValueError: If ``value`` type is invalid.
        """
        return self._properties.setdefault(self._PROPERTY_TO_API_FIELD["labels"], {})

    @labels.setter
    def labels(self, value):
        if not isinstance(value, dict):
            raise ValueError("Pass a dict")
        self._properties[self._PROPERTY_TO_API_FIELD["labels"]] = value

    @property
    def encryption_configuration(self):
        """google.cloud.bigquery.encryption_configuration.EncryptionConfiguration: Custom
        encryption configuration for the table.

        Custom encryption configuration (e.g., Cloud KMS keys) or :data:`None`
        if using default encryption.

        See `protecting data with Cloud KMS keys
        <https://cloud.google.com/bigquery/docs/customer-managed-encryption>`_
        in the BigQuery documentation.
        """
        prop = self._properties.get(
            self._PROPERTY_TO_API_FIELD["encryption_configuration"]
        )
        if prop is not None:
            prop = EncryptionConfiguration.from_api_repr(prop)
        return prop

    @encryption_configuration.setter
    def encryption_configuration(self, value):
        api_repr = value
        if value is not None:
            api_repr = value.to_api_repr()
        self._properties[
            self._PROPERTY_TO_API_FIELD["encryption_configuration"]
        ] = api_repr

    @property
    def created(self):
        """Union[datetime.datetime, None]: Datetime at which the table was
        created (:data:`None` until set from the server).
        """
        creation_time = self._properties.get(self._PROPERTY_TO_API_FIELD["created"])
        if creation_time is not None:
            # creation_time will be in milliseconds.
            return google.cloud._helpers._datetime_from_microseconds(
                1000.0 * float(creation_time)
            )

    @property
    def etag(self):
        """Union[str, None]: ETag for the table resource (:data:`None` until
        set from the server).
        """
        return self._properties.get(self._PROPERTY_TO_API_FIELD["etag"])

    @property
    def modified(self):
        """Union[datetime.datetime, None]: Datetime at which the table was last
        modified (:data:`None` until set from the server).
        """
        modified_time = self._properties.get(self._PROPERTY_TO_API_FIELD["modified"])
        if modified_time is not None:
            # modified_time will be in milliseconds.
            return google.cloud._helpers._datetime_from_microseconds(
                1000.0 * float(modified_time)
            )

    @property
    def num_bytes(self):
        """Union[int, None]: The size of the table in bytes (:data:`None` until
        set from the server).
        """
        return _helpers._int_or_none(
            self._properties.get(self._PROPERTY_TO_API_FIELD["num_bytes"])
        )

    @property
    def num_rows(self):
        """Union[int, None]: The number of rows in the table (:data:`None`
        until set from the server).
        """
        return _helpers._int_or_none(
            self._properties.get(self._PROPERTY_TO_API_FIELD["num_rows"])
        )

    @property
    def self_link(self):
        """Union[str, None]: URL for the table resource (:data:`None` until set
        from the server).
        """
        return self._properties.get(self._PROPERTY_TO_API_FIELD["self_link"])

    @property
    def full_table_id(self):
        """Union[str, None]: ID for the table (:data:`None` until set from the
        server).

        In the format ``project-id:dataset_id.table_id``.
        """
        return self._properties.get(self._PROPERTY_TO_API_FIELD["full_table_id"])

    @property
    def table_type(self):
        """Union[str, None]: The type of the table (:data:`None` until set from
        the server).

        Possible values are ``'TABLE'``, ``'VIEW'``, ``'MATERIALIZED_VIEW'`` or
        ``'EXTERNAL'``.
        """
        return self._properties.get(self._PROPERTY_TO_API_FIELD["type"])

    @property
    def range_partitioning(self):
        """Optional[google.cloud.bigquery.table.RangePartitioning]:
        Configures range-based partitioning for a table.

        .. note::
            **Beta**. The integer range partitioning feature is in a
            pre-release state and might change or have limited support.

        Only specify at most one of
        :attr:`~google.cloud.bigquery.table.Table.time_partitioning` or
        :attr:`~google.cloud.bigquery.table.Table.range_partitioning`.

        Raises:
            ValueError:
                If the value is not
                :class:`~google.cloud.bigquery.table.RangePartitioning` or
                :data:`None`.
        """
        resource = self._properties.get(
            self._PROPERTY_TO_API_FIELD["range_partitioning"]
        )
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
        self._properties[self._PROPERTY_TO_API_FIELD["range_partitioning"]] = resource

    @property
    def time_partitioning(self):
        """Optional[google.cloud.bigquery.table.TimePartitioning]: Configures time-based
        partitioning for a table.

        Only specify at most one of
        :attr:`~google.cloud.bigquery.table.Table.time_partitioning` or
        :attr:`~google.cloud.bigquery.table.Table.range_partitioning`.

        Raises:
            ValueError:
                If the value is not
                :class:`~google.cloud.bigquery.table.TimePartitioning` or
                :data:`None`.
        """
        prop = self._properties.get(self._PROPERTY_TO_API_FIELD["time_partitioning"])
        if prop is not None:
            return TimePartitioning.from_api_repr(prop)

    @time_partitioning.setter
    def time_partitioning(self, value):
        api_repr = value
        if isinstance(value, TimePartitioning):
            api_repr = value.to_api_repr()
        elif value is not None:
            raise ValueError(
                "value must be google.cloud.bigquery.table.TimePartitioning " "or None"
            )
        self._properties[self._PROPERTY_TO_API_FIELD["time_partitioning"]] = api_repr

    @property
    def partitioning_type(self):
        """Union[str, None]: Time partitioning of the table if it is
        partitioned (Defaults to :data:`None`).

        """
        warnings.warn(
            "This method will be deprecated in future versions. Please use "
            "Table.time_partitioning.type_ instead.",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        if self.time_partitioning is not None:
            return self.time_partitioning.type_

    @partitioning_type.setter
    def partitioning_type(self, value):
        warnings.warn(
            "This method will be deprecated in future versions. Please use "
            "Table.time_partitioning.type_ instead.",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        api_field = self._PROPERTY_TO_API_FIELD["partitioning_type"]
        if self.time_partitioning is None:
            self._properties[api_field] = {}
        self._properties[api_field]["type"] = value

    @property
    def partition_expiration(self):
        """Union[int, None]: Expiration time in milliseconds for a partition.

        If :attr:`partition_expiration` is set and :attr:`type_` is
        not set, :attr:`type_` will default to
        :attr:`~google.cloud.bigquery.table.TimePartitioningType.DAY`.
        """
        warnings.warn(
            "This method will be deprecated in future versions. Please use "
            "Table.time_partitioning.expiration_ms instead.",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        if self.time_partitioning is not None:
            return self.time_partitioning.expiration_ms

    @partition_expiration.setter
    def partition_expiration(self, value):
        warnings.warn(
            "This method will be deprecated in future versions. Please use "
            "Table.time_partitioning.expiration_ms instead.",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        api_field = self._PROPERTY_TO_API_FIELD["partition_expiration"]

        if self.time_partitioning is None:
            self._properties[api_field] = {"type": TimePartitioningType.DAY}

        if value is None:
            self._properties[api_field]["expirationMs"] = None
        else:
            self._properties[api_field]["expirationMs"] = str(value)

    @property
    def clustering_fields(self):
        """Union[List[str], None]: Fields defining clustering for the table

        (Defaults to :data:`None`).

        Clustering fields are immutable after table creation.

        .. note::

           BigQuery supports clustering for both partitioned and
           non-partitioned tables.
        """
        prop = self._properties.get(self._PROPERTY_TO_API_FIELD["clustering_fields"])
        if prop is not None:
            return list(prop.get("fields", ()))

    @clustering_fields.setter
    def clustering_fields(self, value):
        """Union[List[str], None]: Fields defining clustering for the table

        (Defaults to :data:`None`).
        """
        api_field = self._PROPERTY_TO_API_FIELD["clustering_fields"]

        if value is not None:
            prop = self._properties.setdefault(api_field, {})
            prop["fields"] = value
        else:
            # In order to allow unsetting clustering fields completely, we explicitly
            # set this property to None (as oposed to merely removing the key).
            self._properties[api_field] = None

    @property
    def description(self):
        """Union[str, None]: Description of the table (defaults to
        :data:`None`).

        Raises:
            ValueError: For invalid value types.
        """
        return self._properties.get(self._PROPERTY_TO_API_FIELD["description"])

    @description.setter
    def description(self, value):
        if not isinstance(value, str) and value is not None:
            raise ValueError("Pass a string, or None")
        self._properties[self._PROPERTY_TO_API_FIELD["description"]] = value

    @property
    def expires(self):
        """Union[datetime.datetime, None]: Datetime at which the table will be
        deleted.

        Raises:
            ValueError: For invalid value types.
        """
        expiration_time = self._properties.get(self._PROPERTY_TO_API_FIELD["expires"])
        if expiration_time is not None:
            # expiration_time will be in milliseconds.
            return google.cloud._helpers._datetime_from_microseconds(
                1000.0 * float(expiration_time)
            )

    @expires.setter
    def expires(self, value):
        if not isinstance(value, datetime.datetime) and value is not None:
            raise ValueError("Pass a datetime, or None")
        value_ms = google.cloud._helpers._millis_from_datetime(value)
        self._properties[
            self._PROPERTY_TO_API_FIELD["expires"]
        ] = _helpers._str_or_none(value_ms)

    @property
    def friendly_name(self):
        """Union[str, None]: Title of the table (defaults to :data:`None`).

        Raises:
            ValueError: For invalid value types.
        """
        return self._properties.get(self._PROPERTY_TO_API_FIELD["friendly_name"])

    @friendly_name.setter
    def friendly_name(self, value):
        if not isinstance(value, str) and value is not None:
            raise ValueError("Pass a string, or None")
        self._properties[self._PROPERTY_TO_API_FIELD["friendly_name"]] = value

    @property
    def location(self):
        """Union[str, None]: Location in which the table is hosted

        Defaults to :data:`None`.
        """
        return self._properties.get(self._PROPERTY_TO_API_FIELD["location"])

    @property
    def view_query(self):
        """Union[str, None]: SQL query defining the table as a view (defaults
        to :data:`None`).

        By default, the query is treated as Standard SQL. To use Legacy
        SQL, set :attr:`view_use_legacy_sql` to :data:`True`.

        Raises:
            ValueError: For invalid value types.
        """
        api_field = self._PROPERTY_TO_API_FIELD["view_query"]
        return _helpers._get_sub_prop(self._properties, [api_field, "query"])

    @view_query.setter
    def view_query(self, value):
        if not isinstance(value, str):
            raise ValueError("Pass a string")

        api_field = self._PROPERTY_TO_API_FIELD["view_query"]
        _helpers._set_sub_prop(self._properties, [api_field, "query"], value)
        view = self._properties[api_field]
        # The service defaults useLegacySql to True, but this
        # client uses Standard SQL by default.
        if view.get("useLegacySql") is None:
            view["useLegacySql"] = False

    @view_query.deleter
    def view_query(self):
        """Delete SQL query defining the table as a view."""
        self._properties.pop(self._PROPERTY_TO_API_FIELD["view_query"], None)

    view_use_legacy_sql = property(_view_use_legacy_sql_getter)

    @view_use_legacy_sql.setter  # type: ignore  # (redefinition from above)
    def view_use_legacy_sql(self, value):
        if not isinstance(value, bool):
            raise ValueError("Pass a boolean")

        api_field = self._PROPERTY_TO_API_FIELD["view_query"]
        if self._properties.get(api_field) is None:
            self._properties[api_field] = {}
        self._properties[api_field]["useLegacySql"] = value

    @property
    def mview_query(self):
        """Optional[str]: SQL query defining the table as a materialized
        view (defaults to :data:`None`).
        """
        api_field = self._PROPERTY_TO_API_FIELD["mview_query"]
        return _helpers._get_sub_prop(self._properties, [api_field, "query"])

    @mview_query.setter
    def mview_query(self, value):
        api_field = self._PROPERTY_TO_API_FIELD["mview_query"]
        _helpers._set_sub_prop(self._properties, [api_field, "query"], str(value))

    @mview_query.deleter
    def mview_query(self):
        """Delete SQL query defining the table as a materialized view."""
        self._properties.pop(self._PROPERTY_TO_API_FIELD["mview_query"], None)

    @property
    def mview_last_refresh_time(self):
        """Optional[datetime.datetime]: Datetime at which the materialized view was last
        refreshed (:data:`None` until set from the server).
        """
        refresh_time = _helpers._get_sub_prop(
            self._properties, self._PROPERTY_TO_API_FIELD["mview_last_refresh_time"]
        )
        if refresh_time is not None:
            # refresh_time will be in milliseconds.
            return google.cloud._helpers._datetime_from_microseconds(
                1000 * int(refresh_time)
            )

    @property
    def mview_enable_refresh(self):
        """Optional[bool]: Enable automatic refresh of the materialized view
        when the base table is updated. The default value is :data:`True`.
        """
        api_field = self._PROPERTY_TO_API_FIELD["mview_enable_refresh"]
        return _helpers._get_sub_prop(self._properties, [api_field, "enableRefresh"])

    @mview_enable_refresh.setter
    def mview_enable_refresh(self, value):
        api_field = self._PROPERTY_TO_API_FIELD["mview_enable_refresh"]
        return _helpers._set_sub_prop(
            self._properties, [api_field, "enableRefresh"], value
        )

    @property
    def mview_refresh_interval(self):
        """Optional[datetime.timedelta]: The maximum frequency at which this
        materialized view will be refreshed. The default value is 1800000
        milliseconds (30 minutes).
        """
        api_field = self._PROPERTY_TO_API_FIELD["mview_refresh_interval"]
        refresh_interval = _helpers._get_sub_prop(
            self._properties, [api_field, "refreshIntervalMs"]
        )
        if refresh_interval is not None:
            return datetime.timedelta(milliseconds=int(refresh_interval))

    @mview_refresh_interval.setter
    def mview_refresh_interval(self, value):
        if value is None:
            refresh_interval_ms = None
        else:
            refresh_interval_ms = str(value // datetime.timedelta(milliseconds=1))

        api_field = self._PROPERTY_TO_API_FIELD["mview_refresh_interval"]
        _helpers._set_sub_prop(
            self._properties,
            [api_field, "refreshIntervalMs"],
            refresh_interval_ms,
        )

    @property
    def mview_allow_non_incremental_definition(self):
        """Optional[bool]: This option declares the intention to construct a
        materialized view that isn't refreshed incrementally.
        The default value is :data:`False`.
        """
        api_field = self._PROPERTY_TO_API_FIELD[
            "mview_allow_non_incremental_definition"
        ]
        return _helpers._get_sub_prop(
            self._properties, [api_field, "allowNonIncrementalDefinition"]
        )

    @mview_allow_non_incremental_definition.setter
    def mview_allow_non_incremental_definition(self, value):
        api_field = self._PROPERTY_TO_API_FIELD[
            "mview_allow_non_incremental_definition"
        ]
        _helpers._set_sub_prop(
            self._properties, [api_field, "allowNonIncrementalDefinition"], value
        )

    @property
    def streaming_buffer(self):
        """google.cloud.bigquery.StreamingBuffer: Information about a table's
        streaming buffer.
        """
        sb = self._properties.get(self._PROPERTY_TO_API_FIELD["streaming_buffer"])
        if sb is not None:
            return StreamingBuffer(sb)

    @property
    def external_data_configuration(self):
        """Union[google.cloud.bigquery.ExternalConfig, None]: Configuration for
        an external data source (defaults to :data:`None`).

        Raises:
            ValueError: For invalid value types.
        """
        prop = self._properties.get(
            self._PROPERTY_TO_API_FIELD["external_data_configuration"]
        )
        if prop is not None:
            prop = ExternalConfig.from_api_repr(prop)
        return prop

    @external_data_configuration.setter
    def external_data_configuration(self, value):
        if not (value is None or isinstance(value, ExternalConfig)):
            raise ValueError("Pass an ExternalConfig or None")
        api_repr = value
        if value is not None:
            api_repr = value.to_api_repr()
        self._properties[
            self._PROPERTY_TO_API_FIELD["external_data_configuration"]
        ] = api_repr

    @property
    def snapshot_definition(self) -> Optional["SnapshotDefinition"]:
        """Information about the snapshot. This value is set via snapshot creation.

        See: https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#Table.FIELDS.snapshot_definition
        """
        snapshot_info = self._properties.get(
            self._PROPERTY_TO_API_FIELD["snapshot_definition"]
        )
        if snapshot_info is not None:
            snapshot_info = SnapshotDefinition(snapshot_info)
        return snapshot_info

    @property
    def clone_definition(self) -> Optional["CloneDefinition"]:
        """Information about the clone. This value is set via clone creation.

        See: https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#Table.FIELDS.clone_definition
        """
        clone_info = self._properties.get(
            self._PROPERTY_TO_API_FIELD["clone_definition"]
        )
        if clone_info is not None:
            clone_info = CloneDefinition(clone_info)
        return clone_info

    @property
    def table_constraints(self) -> Optional["TableConstraints"]:
        """Tables Primary Key and Foreign Key information."""
        table_constraints = self._properties.get(
            self._PROPERTY_TO_API_FIELD["table_constraints"]
        )
        if table_constraints is not None:
            table_constraints = TableConstraints.from_api_repr(table_constraints)
        return table_constraints

    @table_constraints.setter
    def table_constraints(self, value):
        """Tables Primary Key and Foreign Key information."""
        api_repr = value
        if not isinstance(value, TableConstraints) and value is not None:
            raise ValueError(
                "value must be google.cloud.bigquery.table.TableConstraints or None"
            )
        api_repr = value.to_api_repr() if value else None
        self._properties[self._PROPERTY_TO_API_FIELD["table_constraints"]] = api_repr

    @property
    def resource_tags(self):
        """Dict[str, str]: Resource tags for the table.

        See: https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#Table.FIELDS.resource_tags
        """
        return self._properties.setdefault(
            self._PROPERTY_TO_API_FIELD["resource_tags"], {}
        )

    @resource_tags.setter
    def resource_tags(self, value):
        if not isinstance(value, dict) and value is not None:
            raise ValueError("resource_tags must be a dict or None")
        self._properties[self._PROPERTY_TO_API_FIELD["resource_tags"]] = value

    @property
    def external_catalog_table_options(
        self,
    ) -> Optional[external_config.ExternalCatalogTableOptions]:
        """Options defining open source compatible datasets living in the
        BigQuery catalog. Contains metadata of open source database, schema
        or namespace represented by the current dataset."""

        prop = self._properties.get(
            self._PROPERTY_TO_API_FIELD["external_catalog_table_options"]
        )
        if prop is not None:
            return external_config.ExternalCatalogTableOptions.from_api_repr(prop)
        return None

    @external_catalog_table_options.setter
    def external_catalog_table_options(
        self, value: Union[external_config.ExternalCatalogTableOptions, dict, None]
    ):
        value = _helpers._isinstance_or_raise(
            value,
            (external_config.ExternalCatalogTableOptions, dict),
            none_allowed=True,
        )
        if isinstance(value, external_config.ExternalCatalogTableOptions):
            self._properties[
                self._PROPERTY_TO_API_FIELD["external_catalog_table_options"]
            ] = value.to_api_repr()
        else:
            self._properties[
                self._PROPERTY_TO_API_FIELD["external_catalog_table_options"]
            ] = value

    @property
    def foreign_type_info(self) -> Optional[_schema.ForeignTypeInfo]:
        """Optional. Specifies metadata of the foreign data type definition in
        field schema (TableFieldSchema.foreign_type_definition).
        Returns:
            Optional[schema.ForeignTypeInfo]:
                Foreign type information, or :data:`None` if not set.
        .. Note::
            foreign_type_info is only required if you are referencing an
            external catalog such as a Hive table.
            For details, see:
            https://cloud.google.com/bigquery/docs/external-tables
            https://cloud.google.com/bigquery/docs/datasets-intro#external_datasets
        """

        prop = _helpers._get_sub_prop(
            self._properties, self._PROPERTY_TO_API_FIELD["foreign_type_info"]
        )
        if prop is not None:
            return _schema.ForeignTypeInfo.from_api_repr(prop)
        return None

    @foreign_type_info.setter
    def foreign_type_info(self, value: Union[_schema.ForeignTypeInfo, dict, None]):
        value = _helpers._isinstance_or_raise(
            value,
            (_schema.ForeignTypeInfo, dict),
            none_allowed=True,
        )
        if isinstance(value, _schema.ForeignTypeInfo):
            value = value.to_api_repr()
        _helpers._set_sub_prop(
            self._properties, self._PROPERTY_TO_API_FIELD["foreign_type_info"], value
        )

    @classmethod
    def from_string(cls, full_table_id: str) -> "Table":
        """Construct a table from fully-qualified table ID.

        Args:
            full_table_id (str):
                A fully-qualified table ID in standard SQL format. Must
                included a project ID, dataset ID, and table ID, each
                separated by ``.``.

        Returns:
            Table: Table parsed from ``full_table_id``.

        Examples:
            >>> Table.from_string('my-project.mydataset.mytable')
            Table(TableRef...(D...('my-project', 'mydataset'), 'mytable'))

        Raises:
            ValueError:
                If ``full_table_id`` is not a fully-qualified table ID in
                standard SQL format.
        """
        return cls(TableReference.from_string(full_table_id))

    @classmethod
    def from_api_repr(cls, resource: dict) -> "Table":
        """Factory: construct a table given its API representation

        Args:
            resource (Dict[str, object]):
                Table resource representation from the API

        Returns:
            google.cloud.bigquery.table.Table: Table parsed from ``resource``.

        Raises:
            KeyError:
                If the ``resource`` lacks the key ``'tableReference'``, or if
                the ``dict`` stored within the key ``'tableReference'`` lacks
                the keys ``'tableId'``, ``'projectId'``, or ``'datasetId'``.
        """
        from google.cloud.bigquery import dataset

        if (
            "tableReference" not in resource
            or "tableId" not in resource["tableReference"]
        ):
            raise KeyError(
                "Resource lacks required identity information:"
                '["tableReference"]["tableId"]'
            )
        project_id = _helpers._get_sub_prop(
            resource, cls._PROPERTY_TO_API_FIELD["project"]
        )
        table_id = _helpers._get_sub_prop(
            resource, cls._PROPERTY_TO_API_FIELD["table_id"]
        )
        dataset_id = _helpers._get_sub_prop(
            resource, cls._PROPERTY_TO_API_FIELD["dataset_id"]
        )
        dataset_ref = dataset.DatasetReference(project_id, dataset_id)

        table = cls(dataset_ref.table(table_id))
        table._properties = resource

        return table

    def to_api_repr(self) -> dict:
        """Constructs the API resource of this table

        Returns:
            Dict[str, object]: Table represented as an API resource
        """
        return copy.deepcopy(self._properties)

    def to_bqstorage(self) -> str:
        """Construct a BigQuery Storage API representation of this table.

        Returns:
            str: A reference to this table in the BigQuery Storage API.
        """
        return self.reference.to_bqstorage()

    def _build_resource(self, filter_fields):
        """Generate a resource for ``update``."""
        return _helpers._build_resource_from_properties(self, filter_fields)

    def __repr__(self):
        return "Table({})".format(repr(self.reference))

    def __str__(self):
        return f"{self.project}.{self.dataset_id}.{self.table_id}"

    @property
    def max_staleness(self):
        """Union[str, None]: The maximum staleness of data that could be returned when the table is queried.

        Staleness encoded as a string encoding of sql IntervalValue type.
        This property is optional and defaults to None.

        According to the BigQuery API documentation, maxStaleness specifies the maximum time
        interval for which stale data can be returned when querying the table.
        It helps control data freshness in scenarios like metadata-cached external tables.

        Returns:
            Optional[str]: A string representing the maximum staleness interval
            (e.g., '1h', '30m', '15s' for hours, minutes, seconds respectively).
        """
        return self._properties.get(self._PROPERTY_TO_API_FIELD["max_staleness"])

    @max_staleness.setter
    def max_staleness(self, value):
        """Set the maximum staleness for the table.

        Args:
            value (Optional[str]): A string representing the maximum staleness interval.
                Must be a valid time interval string.
                Examples include '1h' (1 hour), '30m' (30 minutes), '15s' (15 seconds).

        Raises:
            ValueError: If the value is not None and not a string.
        """
        if value is not None and not isinstance(value, str):
            raise ValueError("max_staleness must be a string or None")

        self._properties[self._PROPERTY_TO_API_FIELD["max_staleness"]] = value


class TableListItem(_TableBase):
    """A read-only table resource from a list operation.

    For performance reasons, the BigQuery API only includes some of the table
    properties when listing tables. Notably,
    :attr:`~google.cloud.bigquery.table.Table.schema` and
    :attr:`~google.cloud.bigquery.table.Table.num_rows` are missing.

    For a full list of the properties that the BigQuery API returns, see the
    `REST documentation for tables.list
    <https://cloud.google.com/bigquery/docs/reference/rest/v2/tables/list>`_.


    Args:
        resource (Dict[str, object]):
            A table-like resource object from a table list response. A
            ``tableReference`` property is required.

    Raises:
        ValueError:
            If ``tableReference`` or one of its required members is missing
            from ``resource``.
    """

    def __init__(self, resource):
        if "tableReference" not in resource:
            raise ValueError("resource must contain a tableReference value")
        if "projectId" not in resource["tableReference"]:
            raise ValueError(
                "resource['tableReference'] must contain a projectId value"
            )
        if "datasetId" not in resource["tableReference"]:
            raise ValueError(
                "resource['tableReference'] must contain a datasetId value"
            )
        if "tableId" not in resource["tableReference"]:
            raise ValueError("resource['tableReference'] must contain a tableId value")

        self._properties = resource

    @property
    def created(self):
        """Union[datetime.datetime, None]: Datetime at which the table was
        created (:data:`None` until set from the server).
        """
        creation_time = self._properties.get("creationTime")
        if creation_time is not None:
            # creation_time will be in milliseconds.
            return google.cloud._helpers._datetime_from_microseconds(
                1000.0 * float(creation_time)
            )

    @property
    def expires(self):
        """Union[datetime.datetime, None]: Datetime at which the table will be
        deleted.
        """
        expiration_time = self._properties.get("expirationTime")
        if expiration_time is not None:
            # expiration_time will be in milliseconds.
            return google.cloud._helpers._datetime_from_microseconds(
                1000.0 * float(expiration_time)
            )

    reference = property(_reference_getter)

    @property
    def labels(self):
        """Dict[str, str]: Labels for the table.

        This method always returns a dict. To change a table's labels,
        modify the dict, then call ``Client.update_table``. To delete a
        label, set its value to :data:`None` before updating.
        """
        return self._properties.setdefault("labels", {})

    @property
    def full_table_id(self):
        """Union[str, None]: ID for the table (:data:`None` until set from the
        server).

        In the format ``project_id:dataset_id.table_id``.
        """
        return self._properties.get("id")

    @property
    def table_type(self):
        """Union[str, None]: The type of the table (:data:`None` until set from
        the server).

        Possible values are ``'TABLE'``, ``'VIEW'``, or ``'EXTERNAL'``.
        """
        return self._properties.get("type")

    @property
    def time_partitioning(self):
        """google.cloud.bigquery.table.TimePartitioning: Configures time-based
        partitioning for a table.
        """
        prop = self._properties.get("timePartitioning")
        if prop is not None:
            return TimePartitioning.from_api_repr(prop)

    @property
    def partitioning_type(self):
        """Union[str, None]: Time partitioning of the table if it is
        partitioned (Defaults to :data:`None`).
        """
        warnings.warn(
            "This method will be deprecated in future versions. Please use "
            "TableListItem.time_partitioning.type_ instead.",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        if self.time_partitioning is not None:
            return self.time_partitioning.type_

    @property
    def partition_expiration(self):
        """Union[int, None]: Expiration time in milliseconds for a partition.

        If this property is set and :attr:`type_` is not set, :attr:`type_`
        will default to :attr:`TimePartitioningType.DAY`.
        """
        warnings.warn(
            "This method will be deprecated in future versions. Please use "
            "TableListItem.time_partitioning.expiration_ms instead.",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        if self.time_partitioning is not None:
            return self.time_partitioning.expiration_ms

    @property
    def friendly_name(self):
        """Union[str, None]: Title of the table (defaults to :data:`None`)."""
        return self._properties.get("friendlyName")

    view_use_legacy_sql = property(_view_use_legacy_sql_getter)

    @property
    def clustering_fields(self):
        """Union[List[str], None]: Fields defining clustering for the table

        (Defaults to :data:`None`).

        Clustering fields are immutable after table creation.

        .. note::

           BigQuery supports clustering for both partitioned and
           non-partitioned tables.
        """
        prop = self._properties.get("clustering")
        if prop is not None:
            return list(prop.get("fields", ()))

    @classmethod
    def from_string(cls, full_table_id: str) -> "TableListItem":
        """Construct a table from fully-qualified table ID.

        Args:
            full_table_id (str):
                A fully-qualified table ID in standard SQL format. Must
                included a project ID, dataset ID, and table ID, each
                separated by ``.``.

        Returns:
            Table: Table parsed from ``full_table_id``.

        Examples:
            >>> Table.from_string('my-project.mydataset.mytable')
            Table(TableRef...(D...('my-project', 'mydataset'), 'mytable'))

        Raises:
            ValueError:
                If ``full_table_id`` is not a fully-qualified table ID in
                standard SQL format.
        """
        return cls(
            {"tableReference": TableReference.from_string(full_table_id).to_api_repr()}
        )

    def to_bqstorage(self) -> str:
        """Construct a BigQuery Storage API representation of this table.

        Returns:
            str: A reference to this table in the BigQuery Storage API.
        """
        return self.reference.to_bqstorage()

    def to_api_repr(self) -> dict:
        """Constructs the API resource of this table

        Returns:
            Dict[str, object]: Table represented as an API resource
        """
        return copy.deepcopy(self._properties)


def _row_from_mapping(mapping, schema):
    """Convert a mapping to a row tuple using the schema.

    Args:
        mapping (Dict[str, object])
            Mapping of row data: must contain keys for all required fields in
            the schema. Keys which do not correspond to a field in the schema
            are ignored.
        schema (List[google.cloud.bigquery.schema.SchemaField]):
            The schema of the table destination for the rows

    Returns:
        Tuple[object]:
            Tuple whose elements are ordered according to the schema.

    Raises:
        ValueError: If schema is empty.
    """
    if len(schema) == 0:
        raise ValueError(_TABLE_HAS_NO_SCHEMA)

    row = []
    for field in schema:
        if field.mode == "REQUIRED":
            row.append(mapping[field.name])
        elif field.mode == "REPEATED":
            row.append(mapping.get(field.name, ()))
        elif field.mode == "NULLABLE":
            row.append(mapping.get(field.name))
        else:
            raise ValueError("Unknown field mode: {}".format(field.mode))
    return tuple(row)


class StreamingBuffer(object):
    """Information about a table's streaming buffer.

    See https://cloud.google.com/bigquery/streaming-data-into-bigquery.

    Args:
        resource (Dict[str, object]):
            streaming buffer representation returned from the API
    """

    def __init__(self, resource):
        self.estimated_bytes = None
        if "estimatedBytes" in resource:
            self.estimated_bytes = int(resource["estimatedBytes"])
        self.estimated_rows = None
        if "estimatedRows" in resource:
            self.estimated_rows = int(resource["estimatedRows"])
        self.oldest_entry_time = None
        if "oldestEntryTime" in resource:
            self.oldest_entry_time = google.cloud._helpers._datetime_from_microseconds(
                1000.0 * int(resource["oldestEntryTime"])
            )


class SnapshotDefinition:
    """Information about base table and snapshot time of the snapshot.

    See https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#snapshotdefinition

    Args:
        resource: Snapshot definition representation returned from the API.
    """

    def __init__(self, resource: Dict[str, Any]):
        self.base_table_reference = None
        if "baseTableReference" in resource:
            self.base_table_reference = TableReference.from_api_repr(
                resource["baseTableReference"]
            )

        self.snapshot_time = None
        if "snapshotTime" in resource:
            self.snapshot_time = google.cloud._helpers._rfc3339_to_datetime(
                resource["snapshotTime"]
            )


class CloneDefinition:
    """Information about base table and clone time of the clone.

    See https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#clonedefinition

    Args:
        resource: Clone definition representation returned from the API.
    """

    def __init__(self, resource: Dict[str, Any]):
        self.base_table_reference = None
        if "baseTableReference" in resource:
            self.base_table_reference = TableReference.from_api_repr(
                resource["baseTableReference"]
            )

        self.clone_time = None
        if "cloneTime" in resource:
            self.clone_time = google.cloud._helpers._rfc3339_to_datetime(
                resource["cloneTime"]
            )


class Row(object):
    """A BigQuery row.

    Values can be accessed by position (index), by key like a dict,
    or as properties.

    Args:
        values (Sequence[object]): The row values
        field_to_index (Dict[str, int]):
            A mapping from schema field names to indexes
    """

    # Choose unusual field names to try to avoid conflict with schema fields.
    __slots__ = ("_xxx_values", "_xxx_field_to_index")

    def __init__(self, values, field_to_index) -> None:
        self._xxx_values = values
        self._xxx_field_to_index = field_to_index

    def values(self):
        """Return the values included in this row.

        Returns:
            Sequence[object]: A sequence of length ``len(row)``.
        """
        return copy.deepcopy(self._xxx_values)

    def keys(self) -> Iterable[str]:
        """Return the keys for using a row as a dict.

        Returns:
            Iterable[str]: The keys corresponding to the columns of a row

        Examples:

            >>> list(Row(('a', 'b'), {'x': 0, 'y': 1}).keys())
            ['x', 'y']
        """
        return self._xxx_field_to_index.keys()

    def items(self) -> Iterable[Tuple[str, Any]]:
        """Return items as ``(key, value)`` pairs.

        Returns:
            Iterable[Tuple[str, object]]:
                The ``(key, value)`` pairs representing this row.

        Examples:

            >>> list(Row(('a', 'b'), {'x': 0, 'y': 1}).items())
            [('x', 'a'), ('y', 'b')]
        """
        for key, index in self._xxx_field_to_index.items():
            yield (key, copy.deepcopy(self._xxx_values[index]))

    def get(self, key: str, default: Any = None) -> Any:
        """Return a value for key, with a default value if it does not exist.

        Args:
            key (str): The key of the column to access
            default (object):
                The default value to use if the key does not exist. (Defaults
                to :data:`None`.)

        Returns:
            object:
                The value associated with the provided key, or a default value.

        Examples:
            When the key exists, the value associated with it is returned.

            >>> Row(('a', 'b'), {'x': 0, 'y': 1}).get('x')
            'a'

            The default value is :data:`None` when the key does not exist.

            >>> Row(('a', 'b'), {'x': 0, 'y': 1}).get('z')
            None

            The default value can be overridden with the ``default`` parameter.

            >>> Row(('a', 'b'), {'x': 0, 'y': 1}).get('z', '')
            ''

            >>> Row(('a', 'b'), {'x': 0, 'y': 1}).get('z', default = '')
            ''
        """
        index = self._xxx_field_to_index.get(key)
        if index is None:
            return default
        return self._xxx_values[index]

    def __getattr__(self, name):
        value = self._xxx_field_to_index.get(name)
        if value is None:
            raise AttributeError("no row field {!r}".format(name))
        return self._xxx_values[value]

    def __len__(self):
        return len(self._xxx_values)

    def __getitem__(self, key):
        if isinstance(key, str):
            value = self._xxx_field_to_index.get(key)
            if value is None:
                raise KeyError("no row field {!r}".format(key))
            key = value
        return self._xxx_values[key]

    def __eq__(self, other):
        if not isinstance(other, Row):
            return NotImplemented
        return (
            self._xxx_values == other._xxx_values
            and self._xxx_field_to_index == other._xxx_field_to_index
        )

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        # sort field dict by value, for determinism
        items = sorted(self._xxx_field_to_index.items(), key=operator.itemgetter(1))
        f2i = "{" + ", ".join("%r: %d" % item for item in items) + "}"
        return "Row({}, {})".format(self._xxx_values, f2i)


class _NoopProgressBarQueue(object):
    """A fake Queue class that does nothing.

    This is used when there is no progress bar to send updates to.
    """

    def put_nowait(self, item):
        """Don't actually do anything with the item."""


class RowIterator(HTTPIterator):
    """A class for iterating through HTTP/JSON API row list responses.

    Args:
        client (Optional[google.cloud.bigquery.Client]):
            The API client instance. This should always be non-`None`, except for
            subclasses that do not use it, namely the ``_EmptyRowIterator``.
        api_request (Callable[google.cloud._http.JSONConnection.api_request]):
            The function to use to make API requests.
        path (str): The method path to query for the list of items.
        schema (Sequence[Union[ \
                :class:`~google.cloud.bigquery.schema.SchemaField`, \
                Mapping[str, Any] \
        ]]):
            The table's schema. If any item is a mapping, its content must be
            compatible with
            :meth:`~google.cloud.bigquery.schema.SchemaField.from_api_repr`.
        page_token (str): A token identifying a page in a result set to start
            fetching results from.
        max_results (Optional[int]): The maximum number of results to fetch.
        page_size (Optional[int]): The maximum number of rows in each page
            of results from this request. Non-positive values are ignored.
            Defaults to a sensible value set by the API.
        extra_params (Optional[Dict[str, object]]):
            Extra query string parameters for the API call.
        table (Optional[Union[ \
            google.cloud.bigquery.table.Table, \
            google.cloud.bigquery.table.TableReference, \
        ]]):
            The table which these rows belong to, or a reference to it. Used to
            call the BigQuery Storage API to fetch rows.
        selected_fields (Optional[Sequence[google.cloud.bigquery.schema.SchemaField]]):
            A subset of columns to select from this table.
        total_rows (Optional[int]):
            Total number of rows in the table.
        first_page_response (Optional[dict]):
            API response for the first page of results. These are returned when
            the first page is requested.
        query (Optional[str]):
            The query text used.
        total_bytes_processed (Optional[int]):
            If representing query results, the total bytes processed by the associated query.
        slot_millis (Optional[int]):
            If representing query results, the number of slot ms billed for the associated query.
        created (Optional[datetime.datetime]):
            If representing query results, the creation time of the associated query.
        started (Optional[datetime.datetime]):
            If representing query results, the start time of the associated query.
        ended (Optional[datetime.datetime]):
            If representing query results, the end time of the associated query.
    """

    def __init__(
        self,
        client,
        api_request,
        path,
        schema,
        page_token=None,
        max_results=None,
        page_size=None,
        extra_params=None,
        table=None,
        selected_fields=None,
        total_rows=None,
        first_page_response=None,
        location: Optional[str] = None,
        job_id: Optional[str] = None,
        query_id: Optional[str] = None,
        project: Optional[str] = None,
        num_dml_affected_rows: Optional[int] = None,
        query: Optional[str] = None,
        total_bytes_processed: Optional[int] = None,
        slot_millis: Optional[int] = None,
        created: Optional[datetime.datetime] = None,
        started: Optional[datetime.datetime] = None,
        ended: Optional[datetime.datetime] = None,
    ):
        super(RowIterator, self).__init__(
            client,
            api_request,
            path,
            item_to_value=_item_to_row,
            items_key="rows",
            page_token=page_token,
            max_results=max_results,
            extra_params=extra_params,
            page_start=_rows_page_start,
            next_token="pageToken",
        )
        schema = _to_schema_fields(schema) if schema else ()
        self._field_to_index = _helpers._field_to_index_mapping(schema)
        self._page_size = page_size
        self._preserve_order = False
        self._schema = schema
        self._selected_fields = selected_fields
        self._table = table
        self._total_rows = total_rows
        self._first_page_response = first_page_response
        self._location = location
        self._job_id = job_id
        self._query_id = query_id
        self._project = project
        self._num_dml_affected_rows = num_dml_affected_rows
        self._query = query
        self._total_bytes_processed = total_bytes_processed
        self._slot_millis = slot_millis
        self._job_created = created
        self._job_started = started
        self._job_ended = ended

    @property
    def _billing_project(self) -> Optional[str]:
        """GCP Project ID where BQ API will bill to (if applicable)."""
        client = self.client
        return client.project if client is not None else None

    @property
    def job_id(self) -> Optional[str]:
        """ID of the query job (if applicable).

        To get the job metadata, call
        ``job = client.get_job(rows.job_id, location=rows.location)``.
        """
        return self._job_id

    @property
    def location(self) -> Optional[str]:
        """Location where the query executed (if applicable).

        See: https://cloud.google.com/bigquery/docs/locations
        """
        return self._location

    @property
    def num_dml_affected_rows(self) -> Optional[int]:
        """If this RowIterator is the result of a DML query, the number of
        rows that were affected.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query#body.QueryResponse.FIELDS.num_dml_affected_rows
        """
        return self._num_dml_affected_rows

    @property
    def project(self) -> Optional[str]:
        """GCP Project ID where these rows are read from."""
        return self._project

    @property
    def query_id(self) -> Optional[str]:
        """[Preview] ID of a completed query.

        This ID is auto-generated and not guaranteed to be populated.
        """
        return self._query_id

    @property
    def query(self) -> Optional[str]:
        """The query text used."""
        return self._query

    @property
    def total_bytes_processed(self) -> Optional[int]:
        """total bytes processed from job statistics, if present."""
        return self._total_bytes_processed

    @property
    def slot_millis(self) -> Optional[int]:
        """Number of slot ms the user is actually billed for."""
        return self._slot_millis

    @property
    def created(self) -> Optional[datetime.datetime]:
        """If representing query results, the creation time of the associated query."""
        return self._job_created

    @property
    def started(self) -> Optional[datetime.datetime]:
        """If representing query results, the start time of the associated query."""
        return self._job_started

    @property
    def ended(self) -> Optional[datetime.datetime]:
        """If representing query results, the end time of the associated query."""
        return self._job_ended

    def _is_almost_completely_cached(self):
        """Check if all results are completely cached.

        This is useful to know, because we can avoid alternative download
        mechanisms.
        """
        if (
            not hasattr(self, "_first_page_response")
            or self._first_page_response is None
        ):
            return False

        total_cached_rows = len(self._first_page_response.get(self._items_key, []))
        if self.max_results is not None and total_cached_rows >= self.max_results:
            return True

        if (
            self.next_page_token is None
            and self._first_page_response.get(self._next_token) is None
        ):
            return True

        if self._total_rows is not None:
            almost_completely = self._total_rows * ALMOST_COMPLETELY_CACHED_RATIO
            if total_cached_rows >= almost_completely:
                return True

        return False

    def _should_use_bqstorage(self, bqstorage_client, create_bqstorage_client):
        """Returns True if the BigQuery Storage API can be used.

        Returns:
            bool
                True if the BigQuery Storage client can be used or created.
        """
        using_bqstorage_api = bqstorage_client or create_bqstorage_client
        if not using_bqstorage_api:
            return False

        if self._table is None:
            return False

        # The developer has already started paging through results if
        # next_page_token is set.
        if hasattr(self, "next_page_token") and self.next_page_token is not None:
            return False

        if self._is_almost_completely_cached():
            return False

        if self.max_results is not None:
            return False

        try:
            _versions_helpers.BQ_STORAGE_VERSIONS.try_import(raise_if_error=True)
        except bq_exceptions.BigQueryStorageNotFoundError:
            warnings.warn(
                "BigQuery Storage module not found, fetch data with the REST "
                "endpoint instead."
            )
            return False
        except bq_exceptions.LegacyBigQueryStorageError as exc:
            warnings.warn(str(exc))
            return False

        return True

    def _get_next_page_response(self):
        """Requests the next page from the path provided.

        Returns:
            Dict[str, object]:
                The parsed JSON response of the next page's contents.
        """
        if self._first_page_response:
            rows = self._first_page_response.get(self._items_key, [])[
                : self.max_results
            ]
            response = {
                self._items_key: rows,
            }
            if self._next_token in self._first_page_response:
                response[self._next_token] = self._first_page_response[self._next_token]

            self._first_page_response = None
            return response

        params = self._get_query_params()

        # If the user has provided page_size and start_index, we need to pass
        # start_index for the first page, but for all subsequent pages, we
        # should not pass start_index. We make a shallow copy of params and do
        # not alter the original, so if the user iterates the results again,
        # start_index is preserved.
        params_copy = copy.copy(params)
        if self._page_size is not None:
            if self.page_number and "startIndex" in params:
                del params_copy["startIndex"]

        return self.api_request(
            method=self._HTTP_METHOD, path=self.path, query_params=params_copy
        )

    @property
    def schema(self):
        """List[google.cloud.bigquery.schema.SchemaField]: The subset of
        columns to be read from the table."""
        return list(self._schema)

    @property
    def total_rows(self):
        """int: The total number of rows in the table or query results."""
        return self._total_rows

    def _maybe_warn_max_results(
        self,
        bqstorage_client: Optional["bigquery_storage.BigQueryReadClient"],
    ):
        """Issue a warning if BQ Storage client is not ``None`` with ``max_results`` set.

        This helper method should be used directly in the relevant top-level public
        methods, so that the warning is issued for the correct line in user code.

        Args:
            bqstorage_client:
                The BigQuery Storage client intended to use for downloading result rows.
        """
        if bqstorage_client is not None and self.max_results is not None:
            warnings.warn(
                "Cannot use bqstorage_client if max_results is set, "
                "reverting to fetching data with the REST endpoint.",
                stacklevel=3,
            )

    def _to_page_iterable(
        self, bqstorage_download, tabledata_list_download, bqstorage_client=None
    ):
        if not self._should_use_bqstorage(bqstorage_client, False):
            bqstorage_client = None

        result_pages = (
            bqstorage_download()
            if bqstorage_client is not None
            else tabledata_list_download()
        )
        yield from result_pages

    def to_arrow_iterable(
        self,
        bqstorage_client: Optional["bigquery_storage.BigQueryReadClient"] = None,
        max_queue_size: int = _pandas_helpers._MAX_QUEUE_SIZE_DEFAULT,  # type: ignore
        max_stream_count: Optional[int] = None,
    ) -> Iterator["pyarrow.RecordBatch"]:
        """[Beta] Create an iterable of class:`pyarrow.RecordBatch`, to process the table as a stream.

        Args:
            bqstorage_client (Optional[google.cloud.bigquery_storage_v1.BigQueryReadClient]):
                A BigQuery Storage API client. If supplied, use the faster
                BigQuery Storage API to fetch rows from BigQuery.

                This method requires the ``pyarrow`` and
                ``google-cloud-bigquery-storage`` libraries.

                This method only exposes a subset of the capabilities of the
                BigQuery Storage API. For full access to all features
                (projections, filters, snapshots) use the Storage API directly.

            max_queue_size (Optional[int]):
                The maximum number of result pages to hold in the internal queue when
                streaming query results over the BigQuery Storage API. Ignored if
                Storage API is not used.

                By default, the max queue size is set to the number of BQ Storage streams
                created by the server. If ``max_queue_size`` is :data:`None`, the queue
                size is infinite.

            max_stream_count (Optional[int]):
                The maximum number of parallel download streams when
                using BigQuery Storage API. Ignored if
                BigQuery Storage API is not used.

                This setting also has no effect if the query result
                is deterministically ordered with ORDER BY,
                in which case, the number of download stream is always 1.

                If set to 0 or None (the default), the number of download
                streams is determined by BigQuery the server. However, this behaviour
                can require a lot of memory to store temporary download result,
                especially with very large queries. In that case,
                setting this parameter value to a value > 0 can help
                reduce system resource consumption.

        Returns:
            pyarrow.RecordBatch:
                A generator of :class:`~pyarrow.RecordBatch`.

        .. versionadded:: 2.31.0
        """
        self._maybe_warn_max_results(bqstorage_client)

        bqstorage_download = functools.partial(
            _pandas_helpers.download_arrow_bqstorage,
            self._billing_project,
            self._table,
            bqstorage_client,
            preserve_order=self._preserve_order,
            selected_fields=self._selected_fields,
            max_queue_size=max_queue_size,
            max_stream_count=max_stream_count,
        )
        tabledata_list_download = functools.partial(
            _pandas_helpers.download_arrow_row_iterator, iter(self.pages), self.schema
        )
        return self._to_page_iterable(
            bqstorage_download,
            tabledata_list_download,
            bqstorage_client=bqstorage_client,
        )

    # If changing the signature of this method, make sure to apply the same
    # changes to job.QueryJob.to_arrow()
    def to_arrow(
        self,
        progress_bar_type: Optional[str] = None,
        bqstorage_client: Optional["bigquery_storage.BigQueryReadClient"] = None,
        create_bqstorage_client: bool = True,
    ) -> "pyarrow.Table":
        """[Beta] Create a class:`pyarrow.Table` by loading all pages of a
        table or query.

        Args:
            progress_bar_type (Optional[str]):
                If set, use the `tqdm <https://tqdm.github.io/>`_ library to
                display a progress bar while the data downloads. Install the
                ``tqdm`` package to use this feature.

                Possible values of ``progress_bar_type`` include:

                ``None``
                  No progress bar.
                ``'tqdm'``
                  Use the :func:`tqdm.tqdm` function to print a progress bar
                  to :data:`sys.stdout`.
                ``'tqdm_notebook'``
                  Use the :func:`tqdm.notebook.tqdm` function to display a
                  progress bar as a Jupyter notebook widget.
                ``'tqdm_gui'``
                  Use the :func:`tqdm.tqdm_gui` function to display a
                  progress bar as a graphical dialog box.
            bqstorage_client (Optional[google.cloud.bigquery_storage_v1.BigQueryReadClient]):
                A BigQuery Storage API client. If supplied, use the faster BigQuery
                Storage API to fetch rows from BigQuery. This API is a billable API.

                This method requires ``google-cloud-bigquery-storage`` library.

                This method only  exposes a subset of the capabilities of the
                BigQuery Storage API.  For full access to all features
                (projections, filters, snapshots) use the Storage API directly.
            create_bqstorage_client (Optional[bool]):
                If ``True`` (default), create a BigQuery Storage API client using
                the default API settings. The BigQuery Storage API is a faster way
                to fetch rows from BigQuery. See the ``bqstorage_client`` parameter
                for more information.

                This argument does nothing if ``bqstorage_client`` is supplied.

                .. versionadded:: 1.24.0

        Returns:
            pyarrow.Table
                A :class:`pyarrow.Table` populated with row data and column
                headers from the query results. The column headers are derived
                from the destination table's schema.

        Raises:
            ValueError: If the :mod:`pyarrow` library cannot be imported.


        .. versionadded:: 1.17.0
        """
        if pyarrow is None:
            raise ValueError(_NO_PYARROW_ERROR)

        self._maybe_warn_max_results(bqstorage_client)

        if not self._should_use_bqstorage(bqstorage_client, create_bqstorage_client):
            create_bqstorage_client = False
            bqstorage_client = None

        owns_bqstorage_client = False
        if not bqstorage_client and create_bqstorage_client:
            bqstorage_client = self.client._ensure_bqstorage_client()
            owns_bqstorage_client = bqstorage_client is not None

        try:
            progress_bar = get_progress_bar(
                progress_bar_type, "Downloading", self.total_rows, "rows"
            )

            record_batches = []
            for record_batch in self.to_arrow_iterable(
                bqstorage_client=bqstorage_client
            ):
                record_batches.append(record_batch)

                if progress_bar is not None:
                    # In some cases, the number of total rows is not populated
                    # until the first page of rows is fetched. Update the
                    # progress bar's total to keep an accurate count.
                    progress_bar.total = progress_bar.total or self.total_rows
                    progress_bar.update(record_batch.num_rows)

            if progress_bar is not None:
                # Indicate that the download has finished.
                progress_bar.close()
        finally:
            if owns_bqstorage_client:
                bqstorage_client._transport.grpc_channel.close()  # type: ignore

        if record_batches and bqstorage_client is not None:
            return pyarrow.Table.from_batches(record_batches)
        else:
            # No records (not record_batches), use schema based on BigQuery schema
            # **or**
            # we used the REST API (bqstorage_client is None),
            # which doesn't add arrow extension metadata, so we let
            # `bq_to_arrow_schema` do it.
            arrow_schema = _pandas_helpers.bq_to_arrow_schema(self._schema)
            return pyarrow.Table.from_batches(record_batches, schema=arrow_schema)

    def to_dataframe_iterable(
        self,
        bqstorage_client: Optional["bigquery_storage.BigQueryReadClient"] = None,
        dtypes: Optional[Dict[str, Any]] = None,
        max_queue_size: int = _pandas_helpers._MAX_QUEUE_SIZE_DEFAULT,  # type: ignore
        max_stream_count: Optional[int] = None,
    ) -> "pandas.DataFrame":
        """Create an iterable of pandas DataFrames, to process the table as a stream.

        Args:
            bqstorage_client (Optional[google.cloud.bigquery_storage_v1.BigQueryReadClient]):
                A BigQuery Storage API client. If supplied, use the faster
                BigQuery Storage API to fetch rows from BigQuery.

                This method requires ``google-cloud-bigquery-storage`` library.

                This method only exposes a subset of the capabilities of the
                BigQuery Storage API. For full access to all features
                (projections, filters, snapshots) use the Storage API directly.

            dtypes (Optional[Map[str, Union[str, pandas.Series.dtype]]]):
                A dictionary of column names pandas ``dtype``s. The provided
                ``dtype`` is used when constructing the series for the column
                specified. Otherwise, the default pandas behavior is used.

            max_queue_size (Optional[int]):
                The maximum number of result pages to hold in the internal queue when
                streaming query results over the BigQuery Storage API. Ignored if
                Storage API is not used.

                By default, the max queue size is set to the number of BQ Storage streams
                created by the server. If ``max_queue_size`` is :data:`None`, the queue
                size is infinite.

                .. versionadded:: 2.14.0

            max_stream_count (Optional[int]):
                The maximum number of parallel download streams when
                using BigQuery Storage API. Ignored if
                BigQuery Storage API is not used.

                This setting also has no effect if the query result
                is deterministically ordered with ORDER BY,
                in which case, the number of download stream is always 1.

                If set to 0 or None (the default), the number of download
                streams is determined by BigQuery the server. However, this behaviour
                can require a lot of memory to store temporary download result,
                especially with very large queries. In that case,
                setting this parameter value to a value > 0 can help
                reduce system resource consumption.

        Returns:
            pandas.DataFrame:
                A generator of :class:`~pandas.DataFrame`.

        Raises:
            ValueError:
                If the :mod:`pandas` library cannot be imported.
        """
        _pandas_helpers.verify_pandas_imports()

        if dtypes is None:
            dtypes = {}

        self._maybe_warn_max_results(bqstorage_client)

        column_names = [field.name for field in self._schema]
        bqstorage_download = functools.partial(
            _pandas_helpers.download_dataframe_bqstorage,
            self._billing_project,
            self._table,
            bqstorage_client,
            column_names,
            dtypes,
            preserve_order=self._preserve_order,
            selected_fields=self._selected_fields,
            max_queue_size=max_queue_size,
            max_stream_count=max_stream_count,
        )
        tabledata_list_download = functools.partial(
            _pandas_helpers.download_dataframe_row_iterator,
            iter(self.pages),
            self.schema,
            dtypes,
        )
        return self._to_page_iterable(
            bqstorage_download,
            tabledata_list_download,
            bqstorage_client=bqstorage_client,
        )

    # If changing the signature of this method, make sure to apply the same
    # changes to job.QueryJob.to_dataframe()
    def to_dataframe(
        self,
        bqstorage_client: Optional["bigquery_storage.BigQueryReadClient"] = None,
        dtypes: Optional[Dict[str, Any]] = None,
        progress_bar_type: Optional[str] = None,
        create_bqstorage_client: bool = True,
        geography_as_object: bool = False,
        bool_dtype: Union[Any, None] = DefaultPandasDTypes.BOOL_DTYPE,
        int_dtype: Union[Any, None] = DefaultPandasDTypes.INT_DTYPE,
        float_dtype: Union[Any, None] = None,
        string_dtype: Union[Any, None] = None,
        date_dtype: Union[Any, None] = DefaultPandasDTypes.DATE_DTYPE,
        datetime_dtype: Union[Any, None] = None,
        time_dtype: Union[Any, None] = DefaultPandasDTypes.TIME_DTYPE,
        timestamp_dtype: Union[Any, None] = None,
        range_date_dtype: Union[Any, None] = DefaultPandasDTypes.RANGE_DATE_DTYPE,
        range_datetime_dtype: Union[
            Any, None
        ] = DefaultPandasDTypes.RANGE_DATETIME_DTYPE,
        range_timestamp_dtype: Union[
            Any, None
        ] = DefaultPandasDTypes.RANGE_TIMESTAMP_DTYPE,
    ) -> "pandas.DataFrame":
        """Create a pandas DataFrame by loading all pages of a query.

        Args:
            bqstorage_client (Optional[google.cloud.bigquery_storage_v1.BigQueryReadClient]):
                A BigQuery Storage API client. If supplied, use the faster
                BigQuery Storage API to fetch rows from BigQuery.

                This method requires ``google-cloud-bigquery-storage`` library.

                This method only exposes a subset of the capabilities of the
                BigQuery Storage API. For full access to all features
                (projections, filters, snapshots) use the Storage API directly.

            dtypes (Optional[Map[str, Union[str, pandas.Series.dtype]]]):
                A dictionary of column names pandas ``dtype``s. The provided
                ``dtype`` is used when constructing the series for the column
                specified. Otherwise, the default pandas behavior is used.
            progress_bar_type (Optional[str]):
                If set, use the `tqdm <https://tqdm.github.io/>`_ library to
                display a progress bar while the data downloads. Install the
                ``tqdm`` package to use this feature.

                Possible values of ``progress_bar_type`` include:

                ``None``
                  No progress bar.
                ``'tqdm'``
                  Use the :func:`tqdm.tqdm` function to print a progress bar
                  to :data:`sys.stdout`.
                ``'tqdm_notebook'``
                  Use the :func:`tqdm.notebook.tqdm` function to display a
                  progress bar as a Jupyter notebook widget.
                ``'tqdm_gui'``
                  Use the :func:`tqdm.tqdm_gui` function to display a
                  progress bar as a graphical dialog box.

                .. versionadded:: 1.11.0

            create_bqstorage_client (Optional[bool]):
                If ``True`` (default), create a BigQuery Storage API client
                using the default API settings. The BigQuery Storage API
                is a faster way to fetch rows from BigQuery. See the
                ``bqstorage_client`` parameter for more information.

                This argument does nothing if ``bqstorage_client`` is supplied.

                .. versionadded:: 1.24.0

            geography_as_object (Optional[bool]):
                If ``True``, convert GEOGRAPHY data to :mod:`shapely`
                geometry objects. If ``False`` (default), don't cast
                geography data to :mod:`shapely` geometry objects.

                .. versionadded:: 2.24.0

            bool_dtype (Optional[pandas.Series.dtype, None]):
                If set, indicate a pandas ExtensionDtype (e.g. ``pandas.BooleanDtype()``)
                to convert BigQuery Boolean type, instead of relying on the default
                ``pandas.BooleanDtype()``. If you explicitly set the value to ``None``,
                then the data type will be ``numpy.dtype("bool")``. BigQuery Boolean
                type can be found at:
                https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#boolean_type

                .. versionadded:: 3.8.0

            int_dtype (Optional[pandas.Series.dtype, None]):
                If set, indicate a pandas ExtensionDtype (e.g. ``pandas.Int64Dtype()``)
                to convert BigQuery Integer types, instead of relying on the default
                ``pandas.Int64Dtype()``. If you explicitly set the value to ``None``,
                then the data type will be ``numpy.dtype("int64")``. A list of BigQuery
                Integer types can be found at:
                https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#integer_types

                .. versionadded:: 3.8.0

            float_dtype (Optional[pandas.Series.dtype, None]):
                If set, indicate a pandas ExtensionDtype (e.g. ``pandas.Float32Dtype()``)
                to convert BigQuery Float type, instead of relying on the default
                ``numpy.dtype("float64")``. If you explicitly set the value to ``None``,
                then the data type will be ``numpy.dtype("float64")``. BigQuery Float
                type can be found at:
                https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#floating_point_types

                .. versionadded:: 3.8.0

            string_dtype (Optional[pandas.Series.dtype, None]):
                If set, indicate a pandas ExtensionDtype (e.g. ``pandas.StringDtype()``) to
                convert BigQuery String type, instead of relying on the default
                ``numpy.dtype("object")``. If you explicitly set the value to ``None``,
                then the data type will be ``numpy.dtype("object")``. BigQuery String
                type can be found at:
                https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#string_type

                .. versionadded:: 3.8.0

            date_dtype (Optional[pandas.Series.dtype, None]):
                If set, indicate a pandas ExtensionDtype (e.g.
                ``pandas.ArrowDtype(pyarrow.date32())``) to convert BigQuery Date
                type, instead of relying on the default ``db_dtypes.DateDtype()``.
                If you explicitly set the value to ``None``, then the data type will be
                ``numpy.dtype("datetime64[ns]")`` or ``object`` if out of bound. BigQuery
                Date type can be found at:
                https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#date_type

                .. versionadded:: 3.10.0

            datetime_dtype (Optional[pandas.Series.dtype, None]):
                If set, indicate a pandas ExtensionDtype (e.g.
                ``pandas.ArrowDtype(pyarrow.timestamp("us"))``) to convert BigQuery Datetime
                type, instead of relying on the default ``numpy.dtype("datetime64[ns]``.
                If you explicitly set the value to ``None``, then the data type will be
                ``numpy.dtype("datetime64[ns]")`` or ``object`` if out of bound. BigQuery
                Datetime type can be found at:
                https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#datetime_type

                .. versionadded:: 3.10.0

            time_dtype (Optional[pandas.Series.dtype, None]):
                If set, indicate a pandas ExtensionDtype (e.g.
                ``pandas.ArrowDtype(pyarrow.time64("us"))``) to convert BigQuery Time
                type, instead of relying on the default ``db_dtypes.TimeDtype()``.
                If you explicitly set the value to ``None``, then the data type will be
                ``numpy.dtype("object")``. BigQuery Time type can be found at:
                https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#time_type

                .. versionadded:: 3.10.0

            timestamp_dtype (Optional[pandas.Series.dtype, None]):
                If set, indicate a pandas ExtensionDtype (e.g.
                ``pandas.ArrowDtype(pyarrow.timestamp("us", tz="UTC"))``) to convert BigQuery Timestamp
                type, instead of relying on the default ``numpy.dtype("datetime64[ns, UTC]")``.
                If you explicitly set the value to ``None``, then the data type will be
                ``numpy.dtype("datetime64[ns, UTC]")`` or ``object`` if out of bound. BigQuery
                Datetime type can be found at:
                https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#timestamp_type

                .. versionadded:: 3.10.0

            range_date_dtype (Optional[pandas.Series.dtype, None]):
                If set, indicate a pandas ExtensionDtype, such as:

                .. code-block:: python

                    pandas.ArrowDtype(pyarrow.struct(
                        [("start", pyarrow.date32()), ("end", pyarrow.date32())]
                    ))

                to convert BigQuery RANGE<DATE> type, instead of relying on
                the default ``object``. If you explicitly set the value to
                ``None``, the data type will be ``object``. BigQuery Range type
                can be found at:
                https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#range_type

                .. versionadded:: 3.21.0

            range_datetime_dtype (Optional[pandas.Series.dtype, None]):
                If set, indicate a pandas ExtensionDtype, such as:

                .. code-block:: python

                    pandas.ArrowDtype(pyarrow.struct(
                        [
                            ("start", pyarrow.timestamp("us")),
                            ("end", pyarrow.timestamp("us")),
                        ]
                    ))

                to convert BigQuery RANGE<DATETIME> type, instead of relying on
                the default ``object``. If you explicitly set the value to
                ``None``, the data type will be ``object``. BigQuery Range type
                can be found at:
                https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#range_type

                .. versionadded:: 3.21.0

            range_timestamp_dtype (Optional[pandas.Series.dtype, None]):
                If set, indicate a pandas ExtensionDtype, such as:

                .. code-block:: python

                    pandas.ArrowDtype(pyarrow.struct(
                        [
                            ("start", pyarrow.timestamp("us", tz="UTC")),
                            ("end", pyarrow.timestamp("us", tz="UTC")),
                        ]
                    ))

                to convert BigQuery RANGE<TIMESTAMP> type, instead of relying
                on the default ``object``. If you explicitly set the value to
                ``None``, the data type will be ``object``. BigQuery Range type
                can be found at:
                https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#range_type

                .. versionadded:: 3.21.0

        Returns:
            pandas.DataFrame:
                A :class:`~pandas.DataFrame` populated with row data and column
                headers from the query results. The column headers are derived
                from the destination table's schema.

        Raises:
            ValueError:
                If the :mod:`pandas` library cannot be imported, or
                the :mod:`google.cloud.bigquery_storage_v1` module is
                required but cannot be imported.  Also if
                `geography_as_object` is `True`, but the
                :mod:`shapely` library cannot be imported. Also if
                `bool_dtype`, `int_dtype` or other dtype parameters
                is not supported dtype.

        """
        _pandas_helpers.verify_pandas_imports()

        if geography_as_object and shapely is None:
            raise ValueError(_NO_SHAPELY_ERROR)

        if bool_dtype is DefaultPandasDTypes.BOOL_DTYPE:
            bool_dtype = pandas.BooleanDtype()

        if int_dtype is DefaultPandasDTypes.INT_DTYPE:
            int_dtype = pandas.Int64Dtype()

        if time_dtype is DefaultPandasDTypes.TIME_DTYPE:
            time_dtype = db_dtypes.TimeDtype()

        if range_date_dtype is DefaultPandasDTypes.RANGE_DATE_DTYPE:
            if _versions_helpers.SUPPORTS_RANGE_PYARROW:
                range_date_dtype = pandas.ArrowDtype(
                    pyarrow.struct(
                        [("start", pyarrow.date32()), ("end", pyarrow.date32())]
                    )
                )
            else:
                warnings.warn(_RANGE_PYARROW_WARNING)
                range_date_dtype = None

        if range_datetime_dtype is DefaultPandasDTypes.RANGE_DATETIME_DTYPE:
            if _versions_helpers.SUPPORTS_RANGE_PYARROW:
                range_datetime_dtype = pandas.ArrowDtype(
                    pyarrow.struct(
                        [
                            ("start", pyarrow.timestamp("us")),
                            ("end", pyarrow.timestamp("us")),
                        ]
                    )
                )
            else:
                warnings.warn(_RANGE_PYARROW_WARNING)
                range_datetime_dtype = None

        if range_timestamp_dtype is DefaultPandasDTypes.RANGE_TIMESTAMP_DTYPE:
            if _versions_helpers.SUPPORTS_RANGE_PYARROW:
                range_timestamp_dtype = pandas.ArrowDtype(
                    pyarrow.struct(
                        [
                            ("start", pyarrow.timestamp("us", tz="UTC")),
                            ("end", pyarrow.timestamp("us", tz="UTC")),
                        ]
                    )
                )
            else:
                warnings.warn(_RANGE_PYARROW_WARNING)
                range_timestamp_dtype = None

        if bool_dtype is not None and not hasattr(bool_dtype, "__from_arrow__"):
            raise ValueError("bool_dtype", _NO_SUPPORTED_DTYPE)

        if int_dtype is not None and not hasattr(int_dtype, "__from_arrow__"):
            raise ValueError("int_dtype", _NO_SUPPORTED_DTYPE)

        if float_dtype is not None and not hasattr(float_dtype, "__from_arrow__"):
            raise ValueError("float_dtype", _NO_SUPPORTED_DTYPE)

        if string_dtype is not None and not hasattr(string_dtype, "__from_arrow__"):
            raise ValueError("string_dtype", _NO_SUPPORTED_DTYPE)

        if (
            date_dtype is not None
            and date_dtype is not DefaultPandasDTypes.DATE_DTYPE
            and not hasattr(date_dtype, "__from_arrow__")
        ):
            raise ValueError("date_dtype", _NO_SUPPORTED_DTYPE)

        if datetime_dtype is not None and not hasattr(datetime_dtype, "__from_arrow__"):
            raise ValueError("datetime_dtype", _NO_SUPPORTED_DTYPE)

        if time_dtype is not None and not hasattr(time_dtype, "__from_arrow__"):
            raise ValueError("time_dtype", _NO_SUPPORTED_DTYPE)

        if timestamp_dtype is not None and not hasattr(
            timestamp_dtype, "__from_arrow__"
        ):
            raise ValueError("timestamp_dtype", _NO_SUPPORTED_DTYPE)

        if dtypes is None:
            dtypes = {}

        self._maybe_warn_max_results(bqstorage_client)

        if not self._should_use_bqstorage(bqstorage_client, create_bqstorage_client):
            create_bqstorage_client = False
            bqstorage_client = None

        record_batch = self.to_arrow(
            progress_bar_type=progress_bar_type,
            bqstorage_client=bqstorage_client,
            create_bqstorage_client=create_bqstorage_client,
        )

        # Default date dtype is `db_dtypes.DateDtype()` that could cause out of bounds error,
        # when pyarrow converts date values to nanosecond precision. To avoid the error, we
        # set the date_as_object parameter to True, if necessary.
        date_as_object = False
        if date_dtype is DefaultPandasDTypes.DATE_DTYPE:
            date_dtype = db_dtypes.DateDtype()
            date_as_object = not all(
                self.__can_cast_timestamp_ns(col)
                for col in record_batch
                # Type can be date32 or date64 (plus units).
                # See: https://arrow.apache.org/docs/python/api/datatypes.html
                if pyarrow.types.is_date(col.type)
            )

        timestamp_as_object = False
        if datetime_dtype is None and timestamp_dtype is None:
            timestamp_as_object = not all(
                self.__can_cast_timestamp_ns(col)
                for col in record_batch
                # Type can be datetime and timestamp (plus units and time zone).
                # See: https://arrow.apache.org/docs/python/api/datatypes.html
                if pyarrow.types.is_timestamp(col.type)
            )

        df = record_batch.to_pandas(
            date_as_object=date_as_object,
            timestamp_as_object=timestamp_as_object,
            integer_object_nulls=True,
            types_mapper=_pandas_helpers.default_types_mapper(
                date_as_object=date_as_object,
                bool_dtype=bool_dtype,
                int_dtype=int_dtype,
                float_dtype=float_dtype,
                string_dtype=string_dtype,
                date_dtype=date_dtype,
                datetime_dtype=datetime_dtype,
                time_dtype=time_dtype,
                timestamp_dtype=timestamp_dtype,
                range_date_dtype=range_date_dtype,
                range_datetime_dtype=range_datetime_dtype,
                range_timestamp_dtype=range_timestamp_dtype,
            ),
        )

        for column in dtypes:
            df[column] = pandas.Series(df[column], dtype=dtypes[column], copy=False)

        if geography_as_object:
            for field in self.schema:
                if field.field_type.upper() == "GEOGRAPHY" and field.mode != "REPEATED":
                    df[field.name] = df[field.name].dropna().apply(_read_wkt)

        return df

    @staticmethod
    def __can_cast_timestamp_ns(column):
        try:
            column.cast("timestamp[ns]")
        except pyarrow.lib.ArrowInvalid:
            return False
        else:
            return True

    # If changing the signature of this method, make sure to apply the same
    # changes to job.QueryJob.to_geodataframe()
    def to_geodataframe(
        self,
        bqstorage_client: Optional["bigquery_storage.BigQueryReadClient"] = None,
        dtypes: Optional[Dict[str, Any]] = None,
        progress_bar_type: Optional[str] = None,
        create_bqstorage_client: bool = True,
        geography_column: Optional[str] = None,
        bool_dtype: Union[Any, None] = DefaultPandasDTypes.BOOL_DTYPE,
        int_dtype: Union[Any, None] = DefaultPandasDTypes.INT_DTYPE,
        float_dtype: Union[Any, None] = None,
        string_dtype: Union[Any, None] = None,
    ) -> "geopandas.GeoDataFrame":
        """Create a GeoPandas GeoDataFrame by loading all pages of a query.

        Args:
            bqstorage_client (Optional[google.cloud.bigquery_storage_v1.BigQueryReadClient]):
                A BigQuery Storage API client. If supplied, use the faster
                BigQuery Storage API to fetch rows from BigQuery.

                This method requires the ``pyarrow`` and
                ``google-cloud-bigquery-storage`` libraries.

                This method only exposes a subset of the capabilities of the
                BigQuery Storage API. For full access to all features
                (projections, filters, snapshots) use the Storage API directly.

            dtypes (Optional[Map[str, Union[str, pandas.Series.dtype]]]):
                A dictionary of column names pandas ``dtype``s. The provided
                ``dtype`` is used when constructing the series for the column
                specified. Otherwise, the default pandas behavior is used.
            progress_bar_type (Optional[str]):
                If set, use the `tqdm <https://tqdm.github.io/>`_ library to
                display a progress bar while the data downloads. Install the
                ``tqdm`` package to use this feature.

                Possible values of ``progress_bar_type`` include:

                ``None``
                  No progress bar.
                ``'tqdm'``
                  Use the :func:`tqdm.tqdm` function to print a progress bar
                  to :data:`sys.stdout`.
                ``'tqdm_notebook'``
                  Use the :func:`tqdm.notebook.tqdm` function to display a
                  progress bar as a Jupyter notebook widget.
                ``'tqdm_gui'``
                  Use the :func:`tqdm.tqdm_gui` function to display a
                  progress bar as a graphical dialog box.

            create_bqstorage_client (Optional[bool]):
                If ``True`` (default), create a BigQuery Storage API client
                using the default API settings. The BigQuery Storage API
                is a faster way to fetch rows from BigQuery. See the
                ``bqstorage_client`` parameter for more information.

                This argument does nothing if ``bqstorage_client`` is supplied.

            geography_column (Optional[str]):
                If there are more than one GEOGRAPHY column,
                identifies which one to use to construct a geopandas
                GeoDataFrame.  This option can be ommitted if there's
                only one GEOGRAPHY column.
            bool_dtype (Optional[pandas.Series.dtype, None]):
                If set, indicate a pandas ExtensionDtype (e.g. ``pandas.BooleanDtype()``)
                to convert BigQuery Boolean type, instead of relying on the default
                ``pandas.BooleanDtype()``. If you explicitly set the value to ``None``,
                then the data type will be ``numpy.dtype("bool")``. BigQuery Boolean
                type can be found at:
                https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#boolean_type
            int_dtype (Optional[pandas.Series.dtype, None]):
                If set, indicate a pandas ExtensionDtype (e.g. ``pandas.Int64Dtype()``)
                to convert BigQuery Integer types, instead of relying on the default
                ``pandas.Int64Dtype()``. If you explicitly set the value to ``None``,
                then the data type will be ``numpy.dtype("int64")``. A list of BigQuery
                Integer types can be found at:
                https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#integer_types
            float_dtype (Optional[pandas.Series.dtype, None]):
                If set, indicate a pandas ExtensionDtype (e.g. ``pandas.Float32Dtype()``)
                to convert BigQuery Float type, instead of relying on the default
                ``numpy.dtype("float64")``. If you explicitly set the value to ``None``,
                then the data type will be ``numpy.dtype("float64")``. BigQuery Float
                type can be found at:
                https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#floating_point_types
            string_dtype (Optional[pandas.Series.dtype, None]):
                If set, indicate a pandas ExtensionDtype (e.g. ``pandas.StringDtype()``) to
                convert BigQuery String type, instead of relying on the default
                ``numpy.dtype("object")``. If you explicitly set the value to ``None``,
                then the data type will be ``numpy.dtype("object")``. BigQuery String
                type can be found at:
                https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#string_type

        Returns:
            geopandas.GeoDataFrame:
                A :class:`geopandas.GeoDataFrame` populated with row
                data and column headers from the query results. The
                column headers are derived from the destination
                table's schema.

        Raises:
            ValueError:
                If the :mod:`geopandas` library cannot be imported, or the
                :mod:`google.cloud.bigquery_storage_v1` module is
                required but cannot be imported.

        .. versionadded:: 2.24.0
        """
        if geopandas is None:
            raise ValueError(_NO_GEOPANDAS_ERROR)

        geography_columns = set(
            field.name
            for field in self.schema
            if field.field_type.upper() == "GEOGRAPHY"
        )
        if not geography_columns:
            raise TypeError(
                "There must be at least one GEOGRAPHY column"
                " to create a GeoDataFrame"
            )

        if geography_column:
            if geography_column not in geography_columns:
                raise ValueError(
                    f"The given geography column, {geography_column}, doesn't name"
                    f" a GEOGRAPHY column in the result."
                )
        elif len(geography_columns) == 1:
            [geography_column] = geography_columns
        else:
            raise ValueError(
                "There is more than one GEOGRAPHY column in the result. "
                "The geography_column argument must be used to specify which "
                "one to use to create a GeoDataFrame"
            )

        df = self.to_dataframe(
            bqstorage_client,
            dtypes,
            progress_bar_type,
            create_bqstorage_client,
            geography_as_object=True,
            bool_dtype=bool_dtype,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            string_dtype=string_dtype,
        )

        return geopandas.GeoDataFrame(
            df, crs=_COORDINATE_REFERENCE_SYSTEM, geometry=geography_column
        )


class _EmptyRowIterator(RowIterator):
    """An empty row iterator.

    This class prevents API requests when there are no rows to fetch or rows
    are impossible to fetch, such as with query results for DDL CREATE VIEW
    statements.
    """

    pages = ()
    total_rows = 0

    def __init__(
        self, client=None, api_request=None, path=None, schema=(), *args, **kwargs
    ):
        super().__init__(
            client=client,
            api_request=api_request,
            path=path,
            schema=schema,
            *args,
            **kwargs,
        )

    def to_arrow(
        self,
        progress_bar_type=None,
        bqstorage_client=None,
        create_bqstorage_client=True,
    ) -> "pyarrow.Table":
        """[Beta] Create an empty class:`pyarrow.Table`.

        Args:
            progress_bar_type (str): Ignored. Added for compatibility with RowIterator.
            bqstorage_client (Any): Ignored. Added for compatibility with RowIterator.
            create_bqstorage_client (bool): Ignored. Added for compatibility with RowIterator.

        Returns:
            pyarrow.Table: An empty :class:`pyarrow.Table`.
        """
        if pyarrow is None:
            raise ValueError(_NO_PYARROW_ERROR)
        return pyarrow.Table.from_arrays(())

    def to_dataframe(
        self,
        bqstorage_client=None,
        dtypes=None,
        progress_bar_type=None,
        create_bqstorage_client=True,
        geography_as_object=False,
        bool_dtype=None,
        int_dtype=None,
        float_dtype=None,
        string_dtype=None,
        date_dtype=None,
        datetime_dtype=None,
        time_dtype=None,
        timestamp_dtype=None,
        range_date_dtype=None,
        range_datetime_dtype=None,
        range_timestamp_dtype=None,
    ) -> "pandas.DataFrame":
        """Create an empty dataframe.

        Args:
            bqstorage_client (Any): Ignored. Added for compatibility with RowIterator.
            dtypes (Any): Ignored. Added for compatibility with RowIterator.
            progress_bar_type (Any): Ignored. Added for compatibility with RowIterator.
            create_bqstorage_client (bool): Ignored. Added for compatibility with RowIterator.
            geography_as_object (bool): Ignored. Added for compatibility with RowIterator.
            bool_dtype (Any): Ignored. Added for compatibility with RowIterator.
            int_dtype (Any): Ignored. Added for compatibility with RowIterator.
            float_dtype (Any): Ignored. Added for compatibility with RowIterator.
            string_dtype (Any): Ignored. Added for compatibility with RowIterator.
            date_dtype (Any): Ignored. Added for compatibility with RowIterator.
            datetime_dtype (Any): Ignored. Added for compatibility with RowIterator.
            time_dtype (Any): Ignored. Added for compatibility with RowIterator.
            timestamp_dtype (Any): Ignored. Added for compatibility with RowIterator.
            range_date_dtype (Any): Ignored. Added for compatibility with RowIterator.
            range_datetime_dtype (Any): Ignored. Added for compatibility with RowIterator.
            range_timestamp_dtype (Any): Ignored. Added for compatibility with RowIterator.

        Returns:
            pandas.DataFrame: An empty :class:`~pandas.DataFrame`.
        """
        _pandas_helpers.verify_pandas_imports()
        return pandas.DataFrame()

    def to_geodataframe(
        self,
        bqstorage_client=None,
        dtypes=None,
        progress_bar_type=None,
        create_bqstorage_client=True,
        geography_column: Optional[str] = None,
        bool_dtype: Union[Any, None] = DefaultPandasDTypes.BOOL_DTYPE,
        int_dtype: Union[Any, None] = DefaultPandasDTypes.INT_DTYPE,
        float_dtype: Union[Any, None] = None,
        string_dtype: Union[Any, None] = None,
    ) -> "pandas.DataFrame":
        """Create an empty dataframe.

        Args:
            bqstorage_client (Any): Ignored. Added for compatibility with RowIterator.
            dtypes (Any): Ignored. Added for compatibility with RowIterator.
            progress_bar_type (Any): Ignored. Added for compatibility with RowIterator.
            create_bqstorage_client (bool): Ignored. Added for compatibility with RowIterator.
            geography_column (str): Ignored. Added for compatibility with RowIterator.
            bool_dtype (Any): Ignored. Added for compatibility with RowIterator.
            int_dtype (Any): Ignored. Added for compatibility with RowIterator.
            float_dtype (Any): Ignored. Added for compatibility with RowIterator.
            string_dtype (Any): Ignored. Added for compatibility with RowIterator.

        Returns:
            pandas.DataFrame: An empty :class:`~pandas.DataFrame`.
        """
        if geopandas is None:
            raise ValueError(_NO_GEOPANDAS_ERROR)

        # Since an empty GeoDataFrame has no geometry column, we do not CRS on it,
        # because that's deprecated.
        return geopandas.GeoDataFrame()

    def to_dataframe_iterable(
        self,
        bqstorage_client: Optional["bigquery_storage.BigQueryReadClient"] = None,
        dtypes: Optional[Dict[str, Any]] = None,
        max_queue_size: Optional[int] = None,
        max_stream_count: Optional[int] = None,
    ) -> Iterator["pandas.DataFrame"]:
        """Create an iterable of pandas DataFrames, to process the table as a stream.

        .. versionadded:: 2.21.0

        Args:
            bqstorage_client:
                Ignored. Added for compatibility with RowIterator.

            dtypes (Optional[Map[str, Union[str, pandas.Series.dtype]]]):
                Ignored. Added for compatibility with RowIterator.

            max_queue_size:
                Ignored. Added for compatibility with RowIterator.

            max_stream_count:
                Ignored. Added for compatibility with RowIterator.

        Returns:
            An iterator yielding a single empty :class:`~pandas.DataFrame`.

        Raises:
            ValueError:
                If the :mod:`pandas` library cannot be imported.
        """
        _pandas_helpers.verify_pandas_imports()
        return iter((pandas.DataFrame(),))

    def to_arrow_iterable(
        self,
        bqstorage_client: Optional["bigquery_storage.BigQueryReadClient"] = None,
        max_queue_size: Optional[int] = None,
        max_stream_count: Optional[int] = None,
    ) -> Iterator["pyarrow.RecordBatch"]:
        """Create an iterable of pandas DataFrames, to process the table as a stream.

        .. versionadded:: 2.31.0

        Args:
            bqstorage_client:
                Ignored. Added for compatibility with RowIterator.

            max_queue_size:
                Ignored. Added for compatibility with RowIterator.

            max_stream_count:
                Ignored. Added for compatibility with RowIterator.

        Returns:
            An iterator yielding a single empty :class:`~pyarrow.RecordBatch`.
        """
        return iter((pyarrow.record_batch([]),))

    def __iter__(self):
        return iter(())


class PartitionRange(object):
    """Definition of the ranges for range partitioning.

    .. note::
        **Beta**. The integer range partitioning feature is in a pre-release
        state and might change or have limited support.

    Args:
        start (Optional[int]):
            Sets the
            :attr:`~google.cloud.bigquery.table.PartitionRange.start`
            property.
        end (Optional[int]):
            Sets the
            :attr:`~google.cloud.bigquery.table.PartitionRange.end`
            property.
        interval (Optional[int]):
            Sets the
            :attr:`~google.cloud.bigquery.table.PartitionRange.interval`
            property.
        _properties (Optional[dict]):
            Private. Used to construct object from API resource.
    """

    def __init__(self, start=None, end=None, interval=None, _properties=None) -> None:
        if _properties is None:
            _properties = {}
        self._properties = _properties

        if start is not None:
            self.start = start
        if end is not None:
            self.end = end
        if interval is not None:
            self.interval = interval

    @property
    def start(self):
        """int: The start of range partitioning, inclusive."""
        return _helpers._int_or_none(self._properties.get("start"))

    @start.setter
    def start(self, value):
        self._properties["start"] = _helpers._str_or_none(value)

    @property
    def end(self):
        """int: The end of range partitioning, exclusive."""
        return _helpers._int_or_none(self._properties.get("end"))

    @end.setter
    def end(self, value):
        self._properties["end"] = _helpers._str_or_none(value)

    @property
    def interval(self):
        """int: The width of each interval."""
        return _helpers._int_or_none(self._properties.get("interval"))

    @interval.setter
    def interval(self, value):
        self._properties["interval"] = _helpers._str_or_none(value)

    def _key(self):
        return tuple(sorted(self._properties.items()))

    def __eq__(self, other):
        if not isinstance(other, PartitionRange):
            return NotImplemented
        return self._key() == other._key()

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        key_vals = ["{}={}".format(key, val) for key, val in self._key()]
        return "PartitionRange({})".format(", ".join(key_vals))


class RangePartitioning(object):
    """Range-based partitioning configuration for a table.

    .. note::
        **Beta**. The integer range partitioning feature is in a pre-release
        state and might change or have limited support.

    Args:
        range_ (Optional[google.cloud.bigquery.table.PartitionRange]):
            Sets the
            :attr:`google.cloud.bigquery.table.RangePartitioning.range_`
            property.
        field (Optional[str]):
            Sets the
            :attr:`google.cloud.bigquery.table.RangePartitioning.field`
            property.
        _properties (Optional[dict]):
            Private. Used to construct object from API resource.
    """

    def __init__(self, range_=None, field=None, _properties=None) -> None:
        if _properties is None:
            _properties = {}
        self._properties: Dict[str, Any] = _properties

        if range_ is not None:
            self.range_ = range_
        if field is not None:
            self.field = field

    # Trailing underscore to prevent conflict with built-in range() function.
    @property
    def range_(self):
        """google.cloud.bigquery.table.PartitionRange: Defines the
        ranges for range partitioning.

        Raises:
            ValueError:
                If the value is not a :class:`PartitionRange`.
        """
        range_properties = self._properties.setdefault("range", {})
        return PartitionRange(_properties=range_properties)

    @range_.setter
    def range_(self, value):
        if not isinstance(value, PartitionRange):
            raise ValueError("Expected a PartitionRange, but got {}.".format(value))
        self._properties["range"] = value._properties

    @property
    def field(self):
        """str: The table is partitioned by this field.

        The field must be a top-level ``NULLABLE`` / ``REQUIRED`` field. The
        only supported type is ``INTEGER`` / ``INT64``.
        """
        return self._properties.get("field")

    @field.setter
    def field(self, value):
        self._properties["field"] = value

    def _key(self):
        return (("field", self.field), ("range_", self.range_))

    def __eq__(self, other):
        if not isinstance(other, RangePartitioning):
            return NotImplemented
        return self._key() == other._key()

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        key_vals = ["{}={}".format(key, repr(val)) for key, val in self._key()]
        return "RangePartitioning({})".format(", ".join(key_vals))


class TimePartitioningType(object):
    """Specifies the type of time partitioning to perform."""

    DAY = "DAY"
    """str: Generates one partition per day."""

    HOUR = "HOUR"
    """str: Generates one partition per hour."""

    MONTH = "MONTH"
    """str: Generates one partition per month."""

    YEAR = "YEAR"
    """str: Generates one partition per year."""


class TimePartitioning(object):
    """Configures time-based partitioning for a table.

    Args:
        type_ (Optional[google.cloud.bigquery.table.TimePartitioningType]):
            Specifies the type of time partitioning to perform. Defaults to
            :attr:`~google.cloud.bigquery.table.TimePartitioningType.DAY`.

            Supported values are:

            * :attr:`~google.cloud.bigquery.table.TimePartitioningType.HOUR`
            * :attr:`~google.cloud.bigquery.table.TimePartitioningType.DAY`
            * :attr:`~google.cloud.bigquery.table.TimePartitioningType.MONTH`
            * :attr:`~google.cloud.bigquery.table.TimePartitioningType.YEAR`

        field (Optional[str]):
            If set, the table is partitioned by this field. If not set, the
            table is partitioned by pseudo column ``_PARTITIONTIME``. The field
            must be a top-level ``TIMESTAMP``, ``DATETIME``, or ``DATE``
            field. Its mode must be ``NULLABLE`` or ``REQUIRED``.

            See the `time-unit column-partitioned tables guide
            <https://cloud.google.com/bigquery/docs/creating-column-partitions>`_
            in the BigQuery documentation.
        expiration_ms(Optional[int]):
            Number of milliseconds for which to keep the storage for a
            partition.
        require_partition_filter (Optional[bool]):
            DEPRECATED: Use
            :attr:`~google.cloud.bigquery.table.Table.require_partition_filter`,
            instead.
    """

    def __init__(
        self, type_=None, field=None, expiration_ms=None, require_partition_filter=None
    ) -> None:
        self._properties: Dict[str, Any] = {}
        if type_ is None:
            self.type_ = TimePartitioningType.DAY
        else:
            self.type_ = type_
        if field is not None:
            self.field = field
        if expiration_ms is not None:
            self.expiration_ms = expiration_ms
        if require_partition_filter is not None:
            self.require_partition_filter = require_partition_filter

    @property
    def type_(self):
        """google.cloud.bigquery.table.TimePartitioningType: The type of time
        partitioning to use.
        """
        return self._properties.get("type")

    @type_.setter
    def type_(self, value):
        self._properties["type"] = value

    @property
    def field(self):
        """str: Field in the table to use for partitioning"""
        return self._properties.get("field")

    @field.setter
    def field(self, value):
        self._properties["field"] = value

    @property
    def expiration_ms(self):
        """int: Number of milliseconds to keep the storage for a partition."""
        return _helpers._int_or_none(self._properties.get("expirationMs"))

    @expiration_ms.setter
    def expiration_ms(self, value):
        if value is not None:
            # Allow explicitly setting the expiration to None.
            value = str(value)
        self._properties["expirationMs"] = value

    @property
    def require_partition_filter(self):
        """bool: Specifies whether partition filters are required for queries

        DEPRECATED: Use
        :attr:`~google.cloud.bigquery.table.Table.require_partition_filter`,
        instead.
        """
        warnings.warn(
            (
                "TimePartitioning.require_partition_filter will be removed in "
                "future versions. Please use Table.require_partition_filter "
                "instead."
            ),
            PendingDeprecationWarning,
            stacklevel=2,
        )
        return self._properties.get("requirePartitionFilter")

    @require_partition_filter.setter
    def require_partition_filter(self, value):
        warnings.warn(
            (
                "TimePartitioning.require_partition_filter will be removed in "
                "future versions. Please use Table.require_partition_filter "
                "instead."
            ),
            PendingDeprecationWarning,
            stacklevel=2,
        )
        self._properties["requirePartitionFilter"] = value

    @classmethod
    def from_api_repr(cls, api_repr: dict) -> "TimePartitioning":
        """Return a :class:`TimePartitioning` object deserialized from a dict.

        This method creates a new ``TimePartitioning`` instance that points to
        the ``api_repr`` parameter as its internal properties dict. This means
        that when a ``TimePartitioning`` instance is stored as a property of
        another object, any changes made at the higher level will also appear
        here::

            >>> time_partitioning = TimePartitioning()
            >>> table.time_partitioning = time_partitioning
            >>> table.time_partitioning.field = 'timecolumn'
            >>> time_partitioning.field
            'timecolumn'

        Args:
            api_repr (Mapping[str, str]):
                The serialized representation of the TimePartitioning, such as
                what is output by :meth:`to_api_repr`.

        Returns:
            google.cloud.bigquery.table.TimePartitioning:
                The ``TimePartitioning`` object.
        """
        instance = cls()
        instance._properties = api_repr
        return instance

    def to_api_repr(self) -> dict:
        """Return a dictionary representing this object.

        This method returns the properties dict of the ``TimePartitioning``
        instance rather than making a copy. This means that when a
        ``TimePartitioning`` instance is stored as a property of another
        object, any changes made at the higher level will also appear here.

        Returns:
            dict:
                A dictionary representing the TimePartitioning object in
                serialized form.
        """
        return self._properties

    def _key(self):
        # because we are only "renaming" top level keys shallow copy is sufficient here.
        properties = self._properties.copy()
        # calling repr for non built-in type objects.
        properties["type_"] = repr(properties.pop("type"))
        if "field" in properties:
            # calling repr for non built-in type objects.
            properties["field"] = repr(properties["field"])
        if "requirePartitionFilter" in properties:
            properties["require_partition_filter"] = properties.pop(
                "requirePartitionFilter"
            )
        if "expirationMs" in properties:
            properties["expiration_ms"] = properties.pop("expirationMs")
        return tuple(sorted(properties.items()))

    def __eq__(self, other):
        if not isinstance(other, TimePartitioning):
            return NotImplemented
        return self._key() == other._key()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        key_vals = ["{}={}".format(key, val) for key, val in self._key()]
        return "TimePartitioning({})".format(",".join(key_vals))


class PrimaryKey:
    """Represents the primary key constraint on a table's columns.

    Args:
        columns: The columns that are composed of the primary key constraint.
    """

    def __init__(self, columns: List[str]):
        self.columns = columns

    def __eq__(self, other):
        if not isinstance(other, PrimaryKey):
            raise TypeError("The value provided is not a BigQuery PrimaryKey.")
        return self.columns == other.columns


class ColumnReference:
    """The pair of the foreign key column and primary key column.

    Args:
        referencing_column: The column that composes the foreign key.
        referenced_column: The column in the primary key that are referenced by the referencingColumn.
    """

    def __init__(self, referencing_column: str, referenced_column: str):
        self.referencing_column = referencing_column
        self.referenced_column = referenced_column

    def __eq__(self, other):
        if not isinstance(other, ColumnReference):
            raise TypeError("The value provided is not a BigQuery ColumnReference.")
        return (
            self.referencing_column == other.referencing_column
            and self.referenced_column == other.referenced_column
        )


class ForeignKey:
    """Represents a foreign key constraint on a table's columns.

    Args:
        name: Set only if the foreign key constraint is named.
        referenced_table: The table that holds the primary key and is referenced by this foreign key.
        column_references: The columns that compose the foreign key.
    """

    def __init__(
        self,
        name: str,
        referenced_table: TableReference,
        column_references: List[ColumnReference],
    ):
        self.name = name
        self.referenced_table = referenced_table
        self.column_references = column_references

    def __eq__(self, other):
        if not isinstance(other, ForeignKey):
            raise TypeError("The value provided is not a BigQuery ForeignKey.")
        return (
            self.name == other.name
            and self.referenced_table == other.referenced_table
            and self.column_references == other.column_references
        )

    @classmethod
    def from_api_repr(cls, api_repr: Dict[str, Any]) -> "ForeignKey":
        """Create an instance from API representation."""
        return cls(
            name=api_repr["name"],
            referenced_table=TableReference.from_api_repr(api_repr["referencedTable"]),
            column_references=[
                ColumnReference(
                    column_reference_resource["referencingColumn"],
                    column_reference_resource["referencedColumn"],
                )
                for column_reference_resource in api_repr["columnReferences"]
            ],
        )

    def to_api_repr(self) -> Dict[str, Any]:
        """Return a dictionary representing this object."""
        return {
            "name": self.name,
            "referencedTable": self.referenced_table.to_api_repr(),
            "columnReferences": [
                {
                    "referencingColumn": column_reference.referencing_column,
                    "referencedColumn": column_reference.referenced_column,
                }
                for column_reference in self.column_references
            ],
        }


class TableConstraints:
    """The TableConstraints defines the primary key and foreign key.

    Args:
        primary_key:
            Represents a primary key constraint on a table's columns. Present only if the table
            has a primary key. The primary key is not enforced.
        foreign_keys:
            Present only if the table has a foreign key. The foreign key is not enforced.

    """

    def __init__(
        self,
        primary_key: Optional[PrimaryKey],
        foreign_keys: Optional[List[ForeignKey]],
    ):
        self.primary_key = primary_key
        self.foreign_keys = foreign_keys

    def __eq__(self, other):
        if not isinstance(other, TableConstraints) and other is not None:
            raise TypeError("The value provided is not a BigQuery TableConstraints.")
        return self.primary_key == (
            other.primary_key if other.primary_key else None
        ) and self.foreign_keys == (other.foreign_keys if other.foreign_keys else None)

    @classmethod
    def from_api_repr(cls, resource: Dict[str, Any]) -> "TableConstraints":
        """Create an instance from API representation."""
        primary_key = None
        if "primaryKey" in resource:
            primary_key = PrimaryKey(resource["primaryKey"]["columns"])

        foreign_keys = None
        if "foreignKeys" in resource:
            foreign_keys = [
                ForeignKey.from_api_repr(foreign_key_resource)
                for foreign_key_resource in resource["foreignKeys"]
            ]
        return cls(primary_key, foreign_keys)

    def to_api_repr(self) -> Dict[str, Any]:
        """Return a dictionary representing this object."""
        resource: Dict[str, Any] = {}
        if self.primary_key:
            resource["primaryKey"] = {"columns": self.primary_key.columns}
        if self.foreign_keys:
            resource["foreignKeys"] = [
                foreign_key.to_api_repr() for foreign_key in self.foreign_keys
            ]
        return resource


class BigLakeConfiguration(object):
    """Configuration for managed tables for Apache Iceberg, formerly
       known as BigLake.

    Args:
        connection_id (Optional[str]):
            The connection specifying the credentials to be used to read and write to external
            storage, such as Cloud Storage. The connection_id can have the form
            ``{project}.{location}.{connection_id}`` or
            ``projects/{project}/locations/{location}/connections/{connection_id}``.
        storage_uri (Optional[str]):
            The fully qualified location prefix of the external folder where table data is
            stored. The '*' wildcard character is not allowed. The URI should be in the
            format ``gs://bucket/path_to_table/``.
        file_format (Optional[str]):
            The file format the table data is stored in. See BigLakeFileFormat for available
            values.
        table_format (Optional[str]):
            The table format the metadata only snapshots are stored in. See BigLakeTableFormat
            for available values.
        _properties (Optional[dict]):
            Private. Used to construct object from API resource.
    """

    def __init__(
        self,
        connection_id: Optional[str] = None,
        storage_uri: Optional[str] = None,
        file_format: Optional[str] = None,
        table_format: Optional[str] = None,
        _properties: Optional[dict] = None,
    ) -> None:
        if _properties is None:
            _properties = {}
        self._properties = _properties
        if connection_id is not None:
            self.connection_id = connection_id
        if storage_uri is not None:
            self.storage_uri = storage_uri
        if file_format is not None:
            self.file_format = file_format
        if table_format is not None:
            self.table_format = table_format

    @property
    def connection_id(self) -> Optional[str]:
        """str: The connection specifying the credentials to be used to read and write to external
        storage, such as Cloud Storage."""
        return self._properties.get("connectionId")

    @connection_id.setter
    def connection_id(self, value: Optional[str]):
        self._properties["connectionId"] = value

    @property
    def storage_uri(self) -> Optional[str]:
        """str: The fully qualified location prefix of the external folder where table data is
        stored."""
        return self._properties.get("storageUri")

    @storage_uri.setter
    def storage_uri(self, value: Optional[str]):
        self._properties["storageUri"] = value

    @property
    def file_format(self) -> Optional[str]:
        """str: The file format the table data is stored in. See BigLakeFileFormat for available
        values."""
        return self._properties.get("fileFormat")

    @file_format.setter
    def file_format(self, value: Optional[str]):
        self._properties["fileFormat"] = value

    @property
    def table_format(self) -> Optional[str]:
        """str: The table format the metadata only snapshots are stored in. See BigLakeTableFormat
        for available values."""
        return self._properties.get("tableFormat")

    @table_format.setter
    def table_format(self, value: Optional[str]):
        self._properties["tableFormat"] = value

    def _key(self):
        return tuple(sorted(self._properties.items()))

    def __eq__(self, other):
        if not isinstance(other, BigLakeConfiguration):
            return NotImplemented
        return self._key() == other._key()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        key_vals = ["{}={}".format(key, val) for key, val in self._key()]
        return "BigLakeConfiguration({})".format(",".join(key_vals))

    @classmethod
    def from_api_repr(cls, resource: Dict[str, Any]) -> "BigLakeConfiguration":
        """Factory: construct a BigLakeConfiguration given its API representation.

        Args:
            resource:
                BigLakeConfiguration representation returned from the API

        Returns:
           BigLakeConfiguration parsed from ``resource``.
        """
        ref = cls()
        ref._properties = resource
        return ref

    def to_api_repr(self) -> Dict[str, Any]:
        """Construct the API resource representation of this BigLakeConfiguration.

        Returns:
            BigLakeConfiguration represented as an API resource.
        """
        return copy.deepcopy(self._properties)


def _item_to_row(iterator, resource):
    """Convert a JSON row to the native object.

    .. note::

        This assumes that the ``schema`` attribute has been
        added to the iterator after being created, which
        should be done by the caller.

    Args:
        iterator (google.api_core.page_iterator.Iterator): The iterator that is currently in use.
        resource (Dict): An item to be converted to a row.

    Returns:
        google.cloud.bigquery.table.Row: The next row in the page.
    """
    return Row(
        _helpers._row_tuple_from_json(resource, iterator.schema),
        iterator._field_to_index,
    )


def _row_iterator_page_columns(schema, response):
    """Make a generator of all the columns in a page from tabledata.list.

    This enables creating a :class:`pandas.DataFrame` and other
    column-oriented data structures such as :class:`pyarrow.RecordBatch`
    """
    columns = []
    rows = response.get("rows", [])

    def get_column_data(field_index, field):
        for row in rows:
            yield _helpers.DATA_FRAME_CELL_DATA_PARSER.to_py(
                row["f"][field_index]["v"], field
            )

    for field_index, field in enumerate(schema):
        columns.append(get_column_data(field_index, field))

    return columns


# pylint: disable=unused-argument
def _rows_page_start(iterator, page, response):
    """Grab total rows when :class:`~google.cloud.iterator.Page` starts.

    Args:
        iterator (google.api_core.page_iterator.Iterator): The iterator that is currently in use.
        page (google.api_core.page_iterator.Page): The page that was just created.
        response (Dict): The JSON API response for a page of rows in a table.
    """
    # Make a (lazy) copy of the page in column-oriented format for use in data
    # science packages.
    page._columns = _row_iterator_page_columns(iterator._schema, response)

    total_rows = response.get("totalRows")
    # Don't reset total_rows if it's not present in the next API response.
    if total_rows is not None:
        iterator._total_rows = int(total_rows)


# pylint: enable=unused-argument


def _table_arg_to_table_ref(value, default_project=None) -> TableReference:
    """Helper to convert a string or Table to TableReference.

    This function keeps TableReference and other kinds of objects unchanged.
    """
    if isinstance(value, str):
        value = TableReference.from_string(value, default_project=default_project)
    if isinstance(value, (Table, TableListItem)):
        value = value.reference
    return value


def _table_arg_to_table(value, default_project=None) -> Table:
    """Helper to convert a string or TableReference to a Table.

    This function keeps Table and other kinds of objects unchanged.
    """
    if isinstance(value, str):
        value = TableReference.from_string(value, default_project=default_project)
    if isinstance(value, TableReference):
        value = Table(value)
    if isinstance(value, TableListItem):
        newvalue = Table(value.reference)
        newvalue._properties = value._properties
        value = newvalue

    return value
