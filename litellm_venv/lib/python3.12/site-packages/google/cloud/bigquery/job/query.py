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

"""Classes for query jobs."""

import concurrent.futures
import copy
import re
import time
import typing
from typing import Any, Dict, Iterable, List, Optional, Union

from google.api_core import exceptions
from google.api_core import retry as retries
import requests

from google.cloud.bigquery.dataset import Dataset
from google.cloud.bigquery.dataset import DatasetListItem
from google.cloud.bigquery.dataset import DatasetReference
from google.cloud.bigquery.encryption_configuration import EncryptionConfiguration
from google.cloud.bigquery.enums import KeyResultStatementKind, DefaultPandasDTypes
from google.cloud.bigquery.external_config import ExternalConfig
from google.cloud.bigquery import _helpers
from google.cloud.bigquery.query import (
    _query_param_from_api_repr,
    ArrayQueryParameter,
    ConnectionProperty,
    ScalarQueryParameter,
    StructQueryParameter,
    UDFResource,
)
from google.cloud.bigquery.retry import (
    DEFAULT_RETRY,
    DEFAULT_JOB_RETRY,
    POLLING_DEFAULT_VALUE,
)
from google.cloud.bigquery.routine import RoutineReference
from google.cloud.bigquery.schema import SchemaField
from google.cloud.bigquery.table import _EmptyRowIterator
from google.cloud.bigquery.table import RangePartitioning
from google.cloud.bigquery.table import _table_arg_to_table_ref
from google.cloud.bigquery.table import TableReference
from google.cloud.bigquery.table import TimePartitioning
from google.cloud.bigquery._tqdm_helpers import wait_for_query

from google.cloud.bigquery.job.base import _AsyncJob
from google.cloud.bigquery.job.base import _JobConfig
from google.cloud.bigquery.job.base import _JobReference

try:
    import pandas  # type: ignore
except ImportError:
    pandas = None

if typing.TYPE_CHECKING:  # pragma: NO COVER
    # Assumption: type checks are only used by library developers and CI environments
    # that have all optional dependencies installed, thus no conditional imports.
    import pandas  # type: ignore
    import geopandas  # type: ignore
    import pyarrow  # type: ignore
    from google.cloud import bigquery_storage
    from google.cloud.bigquery.client import Client
    from google.cloud.bigquery.table import RowIterator


_CONTAINS_ORDER_BY = re.compile(r"ORDER\s+BY", re.IGNORECASE)
_EXCEPTION_FOOTER_TEMPLATE = "{message}\n\nLocation: {location}\nJob ID: {job_id}\n"
_TIMEOUT_BUFFER_SECS = 0.1


def _contains_order_by(query):
    """Do we need to preserve the order of the query results?

    This function has known false positives, such as with ordered window
    functions:

    .. code-block:: sql

       SELECT SUM(x) OVER (
           window_name
           PARTITION BY...
           ORDER BY...
           window_frame_clause)
       FROM ...

    This false positive failure case means the behavior will be correct, but
    downloading results with the BigQuery Storage API may be slower than it
    otherwise would. This is preferable to the false negative case, where
    results are expected to be in order but are not (due to parallel reads).
    """
    return query and _CONTAINS_ORDER_BY.search(query)


def _from_api_repr_query_parameters(resource):
    return [_query_param_from_api_repr(mapping) for mapping in resource]


def _to_api_repr_query_parameters(value):
    return [query_parameter.to_api_repr() for query_parameter in value]


def _from_api_repr_udf_resources(resource):
    udf_resources = []
    for udf_mapping in resource:
        for udf_type, udf_value in udf_mapping.items():
            udf_resources.append(UDFResource(udf_type, udf_value))
    return udf_resources


def _to_api_repr_udf_resources(value):
    return [{udf_resource.udf_type: udf_resource.value} for udf_resource in value]


def _from_api_repr_table_defs(resource):
    return {k: ExternalConfig.from_api_repr(v) for k, v in resource.items()}


def _to_api_repr_table_defs(value):
    return {k: ExternalConfig.to_api_repr(v) for k, v in value.items()}


class BiEngineReason(typing.NamedTuple):
    """Reason for BI Engine acceleration failure

    https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#bienginereason
    """

    code: str = "CODE_UNSPECIFIED"

    reason: str = ""

    @classmethod
    def from_api_repr(cls, reason: Dict[str, str]) -> "BiEngineReason":
        return cls(reason.get("code", "CODE_UNSPECIFIED"), reason.get("message", ""))


class BiEngineStats(typing.NamedTuple):
    """Statistics for a BI Engine query

    https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#bienginestatistics
    """

    mode: str = "ACCELERATION_MODE_UNSPECIFIED"
    """ Specifies which mode of BI Engine acceleration was performed (if any)
    """

    reasons: List[BiEngineReason] = []
    """ Contains explanatory messages in case of DISABLED / PARTIAL acceleration
    """

    @classmethod
    def from_api_repr(cls, stats: Dict[str, Any]) -> "BiEngineStats":
        mode = stats.get("biEngineMode", "ACCELERATION_MODE_UNSPECIFIED")
        reasons = [
            BiEngineReason.from_api_repr(r) for r in stats.get("biEngineReasons", [])
        ]
        return cls(mode, reasons)


class DmlStats(typing.NamedTuple):
    """Detailed statistics for DML statements.

    https://cloud.google.com/bigquery/docs/reference/rest/v2/DmlStats
    """

    inserted_row_count: int = 0
    """Number of inserted rows. Populated by DML INSERT and MERGE statements."""

    deleted_row_count: int = 0
    """Number of deleted rows. populated by DML DELETE, MERGE and TRUNCATE statements.
    """

    updated_row_count: int = 0
    """Number of updated rows. Populated by DML UPDATE and MERGE statements."""

    @classmethod
    def from_api_repr(cls, stats: Dict[str, str]) -> "DmlStats":
        # NOTE: The field order here must match the order of fields set at the
        # class level.
        api_fields = ("insertedRowCount", "deletedRowCount", "updatedRowCount")

        args = (
            int(stats.get(api_field, default_val))
            for api_field, default_val in zip(api_fields, cls.__new__.__defaults__)  # type: ignore
        )
        return cls(*args)


class IndexUnusedReason(typing.NamedTuple):
    """Reason about why no search index was used in the search query (or sub-query).

    https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#indexunusedreason
    """

    code: Optional[str] = None
    """Specifies the high-level reason for the scenario when no search index was used.
    """

    message: Optional[str] = None
    """Free form human-readable reason for the scenario when no search index was used.
    """

    baseTable: Optional[TableReference] = None
    """Specifies the base table involved in the reason that no search index was used.
    """

    indexName: Optional[str] = None
    """Specifies the name of the unused search index, if available."""

    @classmethod
    def from_api_repr(cls, reason):
        code = reason.get("code")
        message = reason.get("message")
        baseTable = reason.get("baseTable")
        indexName = reason.get("indexName")

        return cls(code, message, baseTable, indexName)


class SearchStats(typing.NamedTuple):
    """Statistics related to Search Queries. Populated as part of JobStatistics2.

    https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#searchstatistics
    """

    mode: Optional[str] = None
    """Indicates the type of search index usage in the entire search query."""

    reason: List[IndexUnusedReason] = []
    """Reason about why no search index was used in the search query (or sub-query)"""

    @classmethod
    def from_api_repr(cls, stats: Dict[str, Any]):
        mode = stats.get("indexUsageMode", None)
        reason = [
            IndexUnusedReason.from_api_repr(r)
            for r in stats.get("indexUnusedReasons", [])
        ]
        return cls(mode, reason)


class ScriptOptions:
    """Options controlling the execution of scripts.

    https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#ScriptOptions
    """

    def __init__(
        self,
        statement_timeout_ms: Optional[int] = None,
        statement_byte_budget: Optional[int] = None,
        key_result_statement: Optional[KeyResultStatementKind] = None,
    ):
        self._properties: Dict[str, Any] = {}
        self.statement_timeout_ms = statement_timeout_ms
        self.statement_byte_budget = statement_byte_budget
        self.key_result_statement = key_result_statement

    @classmethod
    def from_api_repr(cls, resource: Dict[str, Any]) -> "ScriptOptions":
        """Factory: construct instance from the JSON repr.

        Args:
            resource(Dict[str: Any]):
                ScriptOptions representation returned from API.

        Returns:
            google.cloud.bigquery.ScriptOptions:
                ScriptOptions sample parsed from ``resource``.
        """
        entry = cls()
        entry._properties = copy.deepcopy(resource)
        return entry

    def to_api_repr(self) -> Dict[str, Any]:
        """Construct the API resource representation."""
        return copy.deepcopy(self._properties)

    @property
    def statement_timeout_ms(self) -> Union[int, None]:
        """Timeout period for each statement in a script."""
        return _helpers._int_or_none(self._properties.get("statementTimeoutMs"))

    @statement_timeout_ms.setter
    def statement_timeout_ms(self, value: Union[int, None]):
        new_value = None if value is None else str(value)
        self._properties["statementTimeoutMs"] = new_value

    @property
    def statement_byte_budget(self) -> Union[int, None]:
        """Limit on the number of bytes billed per statement.

        Exceeding this budget results in an error.
        """
        return _helpers._int_or_none(self._properties.get("statementByteBudget"))

    @statement_byte_budget.setter
    def statement_byte_budget(self, value: Union[int, None]):
        new_value = None if value is None else str(value)
        self._properties["statementByteBudget"] = new_value

    @property
    def key_result_statement(self) -> Union[KeyResultStatementKind, None]:
        """Determines which statement in the script represents the "key result".

        This is used to populate the schema and query results of the script job.
        Default is ``KeyResultStatementKind.LAST``.
        """
        return self._properties.get("keyResultStatement")

    @key_result_statement.setter
    def key_result_statement(self, value: Union[KeyResultStatementKind, None]):
        self._properties["keyResultStatement"] = value


class QueryJobConfig(_JobConfig):
    """Configuration options for query jobs.

    All properties in this class are optional. Values which are :data:`None` ->
    server defaults. Set properties on the constructed configuration by using
    the property name as the name of a keyword argument.
    """

    def __init__(self, **kwargs) -> None:
        super(QueryJobConfig, self).__init__("query", **kwargs)

    @property
    def destination_encryption_configuration(self):
        """google.cloud.bigquery.encryption_configuration.EncryptionConfiguration: Custom
        encryption configuration for the destination table.

        Custom encryption configuration (e.g., Cloud KMS keys) or :data:`None`
        if using default encryption.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.destination_encryption_configuration
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

    @property
    def allow_large_results(self):
        """bool: Allow large query results tables (legacy SQL, only)

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.allow_large_results
        """
        return self._get_sub_prop("allowLargeResults")

    @allow_large_results.setter
    def allow_large_results(self, value):
        self._set_sub_prop("allowLargeResults", value)

    @property
    def connection_properties(self) -> List[ConnectionProperty]:
        """Connection properties.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.connection_properties

        .. versionadded:: 2.29.0
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
        """google.cloud.bigquery.job.CreateDisposition: Specifies behavior
        for creating tables.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.create_disposition
        """
        return self._get_sub_prop("createDisposition")

    @create_disposition.setter
    def create_disposition(self, value):
        self._set_sub_prop("createDisposition", value)

    @property
    def create_session(self) -> Optional[bool]:
        """[Preview] If :data:`True`, creates a new session, where
        :attr:`~google.cloud.bigquery.job.QueryJob.session_info` will contain a
        random server generated session id.

        If :data:`False`, runs query with an existing ``session_id`` passed in
        :attr:`~google.cloud.bigquery.job.QueryJobConfig.connection_properties`,
        otherwise runs query in non-session mode.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.create_session

        .. versionadded:: 2.29.0
        """
        return self._get_sub_prop("createSession")

    @create_session.setter
    def create_session(self, value: Optional[bool]):
        self._set_sub_prop("createSession", value)

    @property
    def default_dataset(self):
        """google.cloud.bigquery.dataset.DatasetReference: the default dataset
        to use for unqualified table names in the query or :data:`None` if not
        set.

        The ``default_dataset`` setter accepts:

        - a :class:`~google.cloud.bigquery.dataset.Dataset`, or
        - a :class:`~google.cloud.bigquery.dataset.DatasetReference`, or
        - a :class:`str` of the fully-qualified dataset ID in standard SQL
          format. The value must included a project ID and dataset ID
          separated by ``.``. For example: ``your-project.your_dataset``.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.default_dataset
        """
        prop = self._get_sub_prop("defaultDataset")
        if prop is not None:
            prop = DatasetReference.from_api_repr(prop)
        return prop

    @default_dataset.setter
    def default_dataset(self, value):
        if value is None:
            self._set_sub_prop("defaultDataset", None)
            return

        if isinstance(value, str):
            value = DatasetReference.from_string(value)

        if isinstance(value, (Dataset, DatasetListItem)):
            value = value.reference

        resource = value.to_api_repr()
        self._set_sub_prop("defaultDataset", resource)

    @property
    def destination(self):
        """google.cloud.bigquery.table.TableReference: table where results are
        written or :data:`None` if not set.

        The ``destination`` setter accepts:

        - a :class:`~google.cloud.bigquery.table.Table`, or
        - a :class:`~google.cloud.bigquery.table.TableReference`, or
        - a :class:`str` of the fully-qualified table ID in standard SQL
          format. The value must included a project ID, dataset ID, and table
          ID, each separated by ``.``. For example:
          ``your-project.your_dataset.your_table``.

        .. note::

            Only table ID is passed to the backend, so any configuration
            in `~google.cloud.bigquery.table.Table` is discarded.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.destination_table
        """
        prop = self._get_sub_prop("destinationTable")
        if prop is not None:
            prop = TableReference.from_api_repr(prop)
        return prop

    @destination.setter
    def destination(self, value):
        if value is None:
            self._set_sub_prop("destinationTable", None)
            return

        value = _table_arg_to_table_ref(value)
        resource = value.to_api_repr()
        self._set_sub_prop("destinationTable", resource)

    @property
    def dry_run(self):
        """bool: :data:`True` if this query should be a dry run to estimate
        costs.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfiguration.FIELDS.dry_run
        """
        return self._properties.get("dryRun")

    @dry_run.setter
    def dry_run(self, value):
        self._properties["dryRun"] = value

    @property
    def flatten_results(self):
        """bool: Flatten nested/repeated fields in results. (Legacy SQL only)

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.flatten_results
        """
        return self._get_sub_prop("flattenResults")

    @flatten_results.setter
    def flatten_results(self, value):
        self._set_sub_prop("flattenResults", value)

    @property
    def maximum_billing_tier(self):
        """int: Deprecated. Changes the billing tier to allow high-compute
        queries.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.maximum_billing_tier
        """
        return self._get_sub_prop("maximumBillingTier")

    @maximum_billing_tier.setter
    def maximum_billing_tier(self, value):
        self._set_sub_prop("maximumBillingTier", value)

    @property
    def maximum_bytes_billed(self):
        """int: Maximum bytes to be billed for this job or :data:`None` if not set.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.maximum_bytes_billed
        """
        return _helpers._int_or_none(self._get_sub_prop("maximumBytesBilled"))

    @maximum_bytes_billed.setter
    def maximum_bytes_billed(self, value):
        self._set_sub_prop("maximumBytesBilled", str(value))

    @property
    def priority(self):
        """google.cloud.bigquery.job.QueryPriority: Priority of the query.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.priority
        """
        return self._get_sub_prop("priority")

    @priority.setter
    def priority(self, value):
        self._set_sub_prop("priority", value)

    @property
    def query_parameters(self):
        """List[Union[google.cloud.bigquery.query.ArrayQueryParameter, \
        google.cloud.bigquery.query.ScalarQueryParameter, \
        google.cloud.bigquery.query.StructQueryParameter]]: list of parameters
        for parameterized query (empty by default)

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.query_parameters
        """
        prop = self._get_sub_prop("queryParameters", default=[])
        return _from_api_repr_query_parameters(prop)

    @query_parameters.setter
    def query_parameters(self, values):
        self._set_sub_prop("queryParameters", _to_api_repr_query_parameters(values))

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
    def udf_resources(self):
        """List[google.cloud.bigquery.query.UDFResource]: user
        defined function resources (empty by default)

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.user_defined_function_resources
        """
        prop = self._get_sub_prop("userDefinedFunctionResources", default=[])
        return _from_api_repr_udf_resources(prop)

    @udf_resources.setter
    def udf_resources(self, values):
        self._set_sub_prop(
            "userDefinedFunctionResources", _to_api_repr_udf_resources(values)
        )

    @property
    def use_legacy_sql(self):
        """bool: Use legacy SQL syntax.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.use_legacy_sql
        """
        return self._get_sub_prop("useLegacySql")

    @use_legacy_sql.setter
    def use_legacy_sql(self, value):
        self._set_sub_prop("useLegacySql", value)

    @property
    def use_query_cache(self):
        """bool: Look for the query result in the cache.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.use_query_cache
        """
        return self._get_sub_prop("useQueryCache")

    @use_query_cache.setter
    def use_query_cache(self, value):
        self._set_sub_prop("useQueryCache", value)

    @property
    def write_disposition(self):
        """google.cloud.bigquery.job.WriteDisposition: Action that occurs if
        the destination table already exists.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.write_disposition
        """
        return self._get_sub_prop("writeDisposition")

    @write_disposition.setter
    def write_disposition(self, value):
        self._set_sub_prop("writeDisposition", value)

    @property
    def write_incremental_results(self) -> Optional[bool]:
        """This is only supported for a SELECT query using a temporary table.

        If set, the query is allowed to write results incrementally to the temporary result
        table. This may incur a performance penalty. This option cannot be used with Legacy SQL.

        This feature is not generally available.
        """
        return self._get_sub_prop("writeIncrementalResults")

    @write_incremental_results.setter
    def write_incremental_results(self, value):
        self._set_sub_prop("writeIncrementalResults", value)

    @property
    def table_definitions(self):
        """Dict[str, google.cloud.bigquery.external_config.ExternalConfig]:
        Definitions for external tables or :data:`None` if not set.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.external_table_definitions
        """
        prop = self._get_sub_prop("tableDefinitions")
        if prop is not None:
            prop = _from_api_repr_table_defs(prop)
        return prop

    @table_definitions.setter
    def table_definitions(self, values):
        self._set_sub_prop("tableDefinitions", _to_api_repr_table_defs(values))

    @property
    def time_partitioning(self):
        """Optional[google.cloud.bigquery.table.TimePartitioning]: Specifies
        time-based partitioning for the destination table.

        Only specify at most one of
        :attr:`~google.cloud.bigquery.job.LoadJobConfig.time_partitioning` or
        :attr:`~google.cloud.bigquery.job.LoadJobConfig.range_partitioning`.

        Raises:
            ValueError:
                If the value is not
                :class:`~google.cloud.bigquery.table.TimePartitioning` or
                :data:`None`.
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
    def schema_update_options(self):
        """List[google.cloud.bigquery.job.SchemaUpdateOption]: Specifies
        updates to the destination table schema to allow as a side effect of
        the query job.
        """
        return self._get_sub_prop("schemaUpdateOptions")

    @schema_update_options.setter
    def schema_update_options(self, values):
        self._set_sub_prop("schemaUpdateOptions", values)

    @property
    def script_options(self) -> ScriptOptions:
        """Options controlling the execution of scripts.

        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#scriptoptions
        """
        prop = self._get_sub_prop("scriptOptions")
        if prop is not None:
            prop = ScriptOptions.from_api_repr(prop)
        return prop

    @script_options.setter
    def script_options(self, value: Union[ScriptOptions, None]):
        new_value = None if value is None else value.to_api_repr()
        self._set_sub_prop("scriptOptions", new_value)

    def to_api_repr(self) -> dict:
        """Build an API representation of the query job config.

        Returns:
            Dict: A dictionary in the format used by the BigQuery API.
        """
        resource = copy.deepcopy(self._properties)
        # Query parameters have an addition property associated with them
        # to indicate if the query is using named or positional parameters.
        query_parameters = resource.get("query", {}).get("queryParameters")
        if query_parameters:
            if query_parameters[0].get("name") is None:
                resource["query"]["parameterMode"] = "POSITIONAL"
            else:
                resource["query"]["parameterMode"] = "NAMED"

        return resource


class QueryJob(_AsyncJob):
    """Asynchronous job: query tables.

    Args:
        job_id (str): the job's ID, within the project belonging to ``client``.

        query (str): SQL query string.

        client (google.cloud.bigquery.client.Client):
            A client which holds credentials and project configuration
            for the dataset (which requires a project).

        job_config (Optional[google.cloud.bigquery.job.QueryJobConfig]):
            Extra configuration options for the query job.
    """

    _JOB_TYPE = "query"
    _UDF_KEY = "userDefinedFunctionResources"
    _CONFIG_CLASS = QueryJobConfig

    def __init__(self, job_id, query, client, job_config=None):
        super(QueryJob, self).__init__(job_id, client)

        if job_config is not None:
            self._properties["configuration"] = job_config._properties
        if self.configuration.use_legacy_sql is None:
            self.configuration.use_legacy_sql = False

        if query:
            _helpers._set_sub_prop(
                self._properties, ["configuration", "query", "query"], query
            )
        self._query_results = None
        self._done_timeout = None
        self._transport_timeout = None

    @property
    def allow_large_results(self):
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.allow_large_results`.
        """
        return self.configuration.allow_large_results

    @property
    def configuration(self) -> QueryJobConfig:
        """The configuration for this query job."""
        return typing.cast(QueryJobConfig, super().configuration)

    @property
    def connection_properties(self) -> List[ConnectionProperty]:
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.connection_properties`.

        .. versionadded:: 2.29.0
        """
        return self.configuration.connection_properties

    @property
    def create_disposition(self):
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.create_disposition`.
        """
        return self.configuration.create_disposition

    @property
    def create_session(self) -> Optional[bool]:
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.create_session`.

        .. versionadded:: 2.29.0
        """
        return self.configuration.create_session

    @property
    def default_dataset(self):
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.default_dataset`.
        """
        return self.configuration.default_dataset

    @property
    def destination(self):
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.destination`.
        """
        return self.configuration.destination

    @property
    def destination_encryption_configuration(self):
        """google.cloud.bigquery.encryption_configuration.EncryptionConfiguration: Custom
        encryption configuration for the destination table.

        Custom encryption configuration (e.g., Cloud KMS keys) or :data:`None`
        if using default encryption.

        See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.destination_encryption_configuration`.
        """
        return self.configuration.destination_encryption_configuration

    @property
    def dry_run(self):
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.dry_run`.
        """
        return self.configuration.dry_run

    @property
    def flatten_results(self):
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.flatten_results`.
        """
        return self.configuration.flatten_results

    @property
    def priority(self):
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.priority`.
        """
        return self.configuration.priority

    @property
    def search_stats(self) -> Optional[SearchStats]:
        """Returns a SearchStats object."""

        stats = self._job_statistics().get("searchStatistics")
        if stats is not None:
            return SearchStats.from_api_repr(stats)
        return None

    @property
    def query(self):
        """str: The query text used in this query job.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery.FIELDS.query
        """
        return _helpers._get_sub_prop(
            self._properties, ["configuration", "query", "query"]
        )

    @property
    def query_id(self) -> Optional[str]:
        """[Preview] ID of a completed query.

        This ID is auto-generated and not guaranteed to be populated.
        """
        query_results = self._query_results
        return query_results.query_id if query_results is not None else None

    @property
    def query_parameters(self):
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.query_parameters`.
        """
        return self.configuration.query_parameters

    @property
    def udf_resources(self):
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.udf_resources`.
        """
        return self.configuration.udf_resources

    @property
    def use_legacy_sql(self):
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.use_legacy_sql`.
        """
        return self.configuration.use_legacy_sql

    @property
    def use_query_cache(self):
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.use_query_cache`.
        """
        return self.configuration.use_query_cache

    @property
    def write_disposition(self):
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.write_disposition`.
        """
        return self.configuration.write_disposition

    @property
    def maximum_billing_tier(self):
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.maximum_billing_tier`.
        """
        return self.configuration.maximum_billing_tier

    @property
    def maximum_bytes_billed(self):
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.maximum_bytes_billed`.
        """
        return self.configuration.maximum_bytes_billed

    @property
    def range_partitioning(self):
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.range_partitioning`.
        """
        return self.configuration.range_partitioning

    @property
    def table_definitions(self):
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.table_definitions`.
        """
        return self.configuration.table_definitions

    @property
    def time_partitioning(self):
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.time_partitioning`.
        """
        return self.configuration.time_partitioning

    @property
    def clustering_fields(self):
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.clustering_fields`.
        """
        return self.configuration.clustering_fields

    @property
    def schema_update_options(self):
        """See
        :attr:`google.cloud.bigquery.job.QueryJobConfig.schema_update_options`.
        """
        return self.configuration.schema_update_options

    def to_api_repr(self):
        """Generate a resource for :meth:`_begin`."""
        # Use to_api_repr to allow for some configuration properties to be set
        # automatically.
        configuration = self.configuration.to_api_repr()
        return {
            "jobReference": self._properties["jobReference"],
            "configuration": configuration,
        }

    @classmethod
    def from_api_repr(cls, resource: dict, client: "Client") -> "QueryJob":
        """Factory:  construct a job given its API representation

        Args:
            resource (Dict): dataset job representation returned from the API

            client (google.cloud.bigquery.client.Client):
                Client which holds credentials and project
                configuration for the dataset.

        Returns:
            google.cloud.bigquery.job.QueryJob: Job parsed from ``resource``.
        """
        job_ref_properties = resource.setdefault(
            "jobReference", {"projectId": client.project, "jobId": None}
        )
        job_ref = _JobReference._from_api_repr(job_ref_properties)
        job = cls(job_ref, None, client=client)
        job._set_properties(resource)
        return job

    @property
    def query_plan(self):
        """Return query plan from job statistics, if present.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatistics2.FIELDS.query_plan

        Returns:
            List[google.cloud.bigquery.job.QueryPlanEntry]:
                mappings describing the query plan, or an empty list
                if the query has not yet completed.
        """
        plan_entries = self._job_statistics().get("queryPlan", ())
        return [QueryPlanEntry.from_api_repr(entry) for entry in plan_entries]

    @property
    def schema(self) -> Optional[List[SchemaField]]:
        """The schema of the results.

        Present only for successful dry run of non-legacy SQL queries.
        """
        resource = self._job_statistics().get("schema")
        if resource is None:
            return None
        fields = resource.get("fields", [])
        return [SchemaField.from_api_repr(field) for field in fields]

    @property
    def timeline(self):
        """List(TimelineEntry): Return the query execution timeline
        from job statistics.
        """
        raw = self._job_statistics().get("timeline", ())
        return [TimelineEntry.from_api_repr(entry) for entry in raw]

    @property
    def total_bytes_processed(self):
        """Return total bytes processed from job statistics, if present.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatistics2.FIELDS.total_bytes_processed

        Returns:
            Optional[int]:
                Total bytes processed by the job, or None if job is not
                yet complete.
        """
        result = self._job_statistics().get("totalBytesProcessed")
        if result is not None:
            result = int(result)
        return result

    @property
    def total_bytes_billed(self):
        """Return total bytes billed from job statistics, if present.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatistics2.FIELDS.total_bytes_billed

        Returns:
            Optional[int]:
                Total bytes processed by the job, or None if job is not
                yet complete.
        """
        result = self._job_statistics().get("totalBytesBilled")
        if result is not None:
            result = int(result)
        return result

    @property
    def billing_tier(self):
        """Return billing tier from job statistics, if present.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatistics2.FIELDS.billing_tier

        Returns:
            Optional[int]:
                Billing tier used by the job, or None if job is not
                yet complete.
        """
        return self._job_statistics().get("billingTier")

    @property
    def cache_hit(self):
        """Return whether or not query results were served from cache.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatistics2.FIELDS.cache_hit

        Returns:
            Optional[bool]:
                whether the query results were returned from cache, or None
                if job is not yet complete.
        """
        return self._job_statistics().get("cacheHit")

    @property
    def ddl_operation_performed(self):
        """Optional[str]: Return the DDL operation performed.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatistics2.FIELDS.ddl_operation_performed

        """
        return self._job_statistics().get("ddlOperationPerformed")

    @property
    def ddl_target_routine(self):
        """Optional[google.cloud.bigquery.routine.RoutineReference]: Return the DDL target routine, present
            for CREATE/DROP FUNCTION/PROCEDURE  queries.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatistics2.FIELDS.ddl_target_routine
        """
        prop = self._job_statistics().get("ddlTargetRoutine")
        if prop is not None:
            prop = RoutineReference.from_api_repr(prop)
        return prop

    @property
    def ddl_target_table(self):
        """Optional[google.cloud.bigquery.table.TableReference]: Return the DDL target table, present
            for CREATE/DROP TABLE/VIEW queries.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatistics2.FIELDS.ddl_target_table
        """
        prop = self._job_statistics().get("ddlTargetTable")
        if prop is not None:
            prop = TableReference.from_api_repr(prop)
        return prop

    @property
    def num_dml_affected_rows(self) -> Optional[int]:
        """Return the number of DML rows affected by the job.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatistics2.FIELDS.num_dml_affected_rows

        Returns:
            Optional[int]:
                number of DML rows affected by the job, or None if job is not
                yet complete.
        """
        result = self._job_statistics().get("numDmlAffectedRows")
        if result is not None:
            result = int(result)
        return result

    @property
    def slot_millis(self):
        """Union[int, None]: Slot-milliseconds used by this query job."""
        return _helpers._int_or_none(self._job_statistics().get("totalSlotMs"))

    @property
    def statement_type(self):
        """Return statement type from job statistics, if present.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatistics2.FIELDS.statement_type

        Returns:
            Optional[str]:
                type of statement used by the job, or None if job is not
                yet complete.
        """
        return self._job_statistics().get("statementType")

    @property
    def referenced_tables(self):
        """Return referenced tables from job statistics, if present.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatistics2.FIELDS.referenced_tables

        Returns:
            List[Dict]:
                mappings describing the query plan, or an empty list
                if the query has not yet completed.
        """
        tables = []
        datasets_by_project_name = {}

        for table in self._job_statistics().get("referencedTables", ()):
            t_project = table["projectId"]

            ds_id = table["datasetId"]
            t_dataset = datasets_by_project_name.get((t_project, ds_id))
            if t_dataset is None:
                t_dataset = DatasetReference(t_project, ds_id)
                datasets_by_project_name[(t_project, ds_id)] = t_dataset

            t_name = table["tableId"]
            tables.append(t_dataset.table(t_name))

        return tables

    @property
    def undeclared_query_parameters(self):
        """Return undeclared query parameters from job statistics, if present.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatistics2.FIELDS.undeclared_query_parameters

        Returns:
            List[Union[ \
                google.cloud.bigquery.query.ArrayQueryParameter, \
                google.cloud.bigquery.query.ScalarQueryParameter, \
                google.cloud.bigquery.query.StructQueryParameter \
            ]]:
                Undeclared parameters, or an empty list if the query has
                not yet completed.
        """
        parameters = []
        undeclared = self._job_statistics().get("undeclaredQueryParameters", ())

        for parameter in undeclared:
            p_type = parameter["parameterType"]

            if "arrayType" in p_type:
                klass = ArrayQueryParameter
            elif "structTypes" in p_type:
                klass = StructQueryParameter
            else:
                klass = ScalarQueryParameter

            parameters.append(klass.from_api_repr(parameter))

        return parameters

    @property
    def estimated_bytes_processed(self):
        """Return the estimated number of bytes processed by the query.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatistics2.FIELDS.estimated_bytes_processed

        Returns:
            Optional[int]:
                number of DML rows affected by the job, or None if job is not
                yet complete.
        """
        result = self._job_statistics().get("estimatedBytesProcessed")
        if result is not None:
            result = int(result)
        return result

    @property
    def dml_stats(self) -> Optional[DmlStats]:
        stats = self._job_statistics().get("dmlStats")
        if stats is None:
            return None
        else:
            return DmlStats.from_api_repr(stats)

    @property
    def bi_engine_stats(self) -> Optional[BiEngineStats]:
        stats = self._job_statistics().get("biEngineStatistics")

        if stats is None:
            return None
        else:
            return BiEngineStats.from_api_repr(stats)

    def _blocking_poll(self, timeout=None, **kwargs):
        self._done_timeout = timeout
        self._transport_timeout = timeout
        super(QueryJob, self)._blocking_poll(timeout=timeout, **kwargs)

    @staticmethod
    def _format_for_exception(message: str, query: str):
        """Format a query for the output in exception message.

        Args:
            message (str): The original exception message.
            query (str): The SQL query to format.

        Returns:
            str: A formatted query text.
        """
        template = "{message}\n\n{header}\n\n{ruler}\n{body}\n{ruler}"

        lines = query.splitlines() if query is not None else [""]
        max_line_len = max(len(line) for line in lines)

        header = "-----Query Job SQL Follows-----"
        header = "{:^{total_width}}".format(header, total_width=max_line_len + 5)

        # Print out a "ruler" above and below the SQL so we can judge columns.
        # Left pad for the line numbers (4 digits plus ":").
        ruler = "    |" + "    .    |" * (max_line_len // 10)

        # Put line numbers next to the SQL.
        body = "\n".join(
            "{:4}:{}".format(n, line) for n, line in enumerate(lines, start=1)
        )

        return template.format(message=message, header=header, ruler=ruler, body=body)

    def _begin(self, client=None, retry=DEFAULT_RETRY, timeout=None):
        """API call:  begin the job via a POST request

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/insert

        Args:
            client (Optional[google.cloud.bigquery.client.Client]):
                The client to use. If not passed, falls back to the ``client``
                associated with the job object or``NoneType``.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

        Raises:
            ValueError: If the job has already begun.
        """

        try:
            super(QueryJob, self)._begin(client=client, retry=retry, timeout=timeout)
        except exceptions.GoogleAPICallError as exc:
            exc.message = _EXCEPTION_FOOTER_TEMPLATE.format(
                message=exc.message, location=self.location, job_id=self.job_id
            )
            exc.debug_message = self._format_for_exception(exc.message, self.query)
            exc.query_job = self
            raise

    def _reload_query_results(
        self,
        retry: "retries.Retry" = DEFAULT_RETRY,
        timeout: Optional[float] = None,
        page_size: int = 0,
        start_index: Optional[int] = None,
    ):
        """Refresh the cached query results unless already cached and complete.

        Args:
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the call that retrieves query results.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
            page_size (int):
                Maximum number of rows in a single response. See maxResults in
                the jobs.getQueryResults REST API.
            start_index (Optional[int]):
                Zero-based index of the starting row. See startIndex in the
                jobs.getQueryResults REST API.
        """
        # Optimization: avoid a call to jobs.getQueryResults if it's already
        # been fetched, e.g. from jobs.query first page of results.
        if self._query_results and self._query_results.complete:
            return

        # Since the API to getQueryResults can hang up to the timeout value
        # (default of 10 seconds), set the timeout parameter to ensure that
        # the timeout from the futures API is respected. See:
        # https://github.com/GoogleCloudPlatform/google-cloud-python/issues/4135
        timeout_ms = None

        # Python_API_core, as part of a major rewrite of the deadline, timeout,
        # retry process sets the timeout value as a Python object().
        # Our system does not natively handle that and instead expects
        # either None or a numeric value. If passed a Python object, convert to
        # None.
        if type(self._done_timeout) is object:  # pragma: NO COVER
            self._done_timeout = None

        if self._done_timeout is not None:  # pragma: NO COVER
            # Subtract a buffer for context switching, network latency, etc.
            api_timeout = self._done_timeout - _TIMEOUT_BUFFER_SECS
            api_timeout = max(min(api_timeout, 10), 0)
            self._done_timeout -= api_timeout
            self._done_timeout = max(0, self._done_timeout)
            timeout_ms = int(api_timeout * 1000)

        # If an explicit timeout is not given, fall back to the transport timeout
        # stored in _blocking_poll() in the process of polling for job completion.
        if timeout is not None:
            transport_timeout = timeout
        else:
            transport_timeout = self._transport_timeout

            # Handle PollingJob._DEFAULT_VALUE.
            if not isinstance(transport_timeout, (float, int)):
                transport_timeout = None

        self._query_results = self._client._get_query_results(
            self.job_id,
            retry,
            project=self.project,
            timeout_ms=timeout_ms,
            location=self.location,
            timeout=transport_timeout,
            page_size=page_size,
            start_index=start_index,
        )

    def result(  # type: ignore  # (incompatible with supertype)
        self,
        page_size: Optional[int] = None,
        max_results: Optional[int] = None,
        retry: Optional[retries.Retry] = DEFAULT_RETRY,
        timeout: Optional[Union[float, object]] = POLLING_DEFAULT_VALUE,
        start_index: Optional[int] = None,
        job_retry: Optional[retries.Retry] = DEFAULT_JOB_RETRY,
    ) -> Union["RowIterator", _EmptyRowIterator]:
        """Start the job and wait for it to complete and get the result.

        Args:
            page_size (Optional[int]):
                The maximum number of rows in each page of results from this
                request. Non-positive values are ignored.
            max_results (Optional[int]):
                The maximum total number of rows from this request.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the call that retrieves rows.  This only
                applies to making RPC calls.  It isn't used to retry
                failed jobs.  This has a reasonable default that
                should only be overridden with care. If the job state
                is ``DONE``, retrying is aborted early even if the
                results are not available, as this will not change
                anymore.
            timeout (Optional[Union[float, \
                google.api_core.future.polling.PollingFuture._DEFAULT_VALUE, \
            ]]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``. If ``None``, wait indefinitely
                unless an error is returned. If unset, only the
                underlying API calls have their default timeouts, but we still
                wait indefinitely for the job to finish.
            start_index (Optional[int]):
                The zero-based index of the starting row to read.
            job_retry (Optional[google.api_core.retry.Retry]):
                How to retry failed jobs.  The default retries
                rate-limit-exceeded errors. Passing ``None`` disables
                job retry.

                Not all jobs can be retried.  If ``job_id`` was
                provided to the query that created this job, then the
                job returned by the query will not be retryable, and
                an exception will be raised if non-``None``
                non-default ``job_retry`` is also provided.

        Returns:
            google.cloud.bigquery.table.RowIterator:
                Iterator of row data
                :class:`~google.cloud.bigquery.table.Row`-s. During each
                page, the iterator will have the ``total_rows`` attribute
                set, which counts the total number of rows **in the result
                set** (this is distinct from the total number of rows in the
                current page: ``iterator.page.num_items``).

                If the query is a special query that produces no results, e.g.
                a DDL query, an ``_EmptyRowIterator`` instance is returned.

        Raises:
            google.api_core.exceptions.GoogleAPICallError:
                If the job failed and retries aren't successful.
            concurrent.futures.TimeoutError:
                If the job did not complete in the given timeout.
            TypeError:
                If Non-``None`` and non-default ``job_retry`` is
                provided and the job is not retryable.
        """
        # Note: Since waiting for a query job to finish is more complex than
        # refreshing the job state in a loop, we avoid calling the superclass
        # in this method.

        if self.dry_run:
            return _EmptyRowIterator(
                project=self.project,
                location=self.location,
                schema=self.schema,
                total_bytes_processed=self.total_bytes_processed,
                # Intentionally omit job_id and query_id since this doesn't
                # actually correspond to a finished query job.
            )

        # Setting max_results should be equivalent to setting page_size with
        # regards to allowing the user to tune how many results to download
        # while we wait for the query to finish. See internal issue:
        # 344008814. But if start_index is set, user is trying to access a
        # specific page, so we don't need to set page_size. See issue #1950.
        if page_size is None and max_results is not None and start_index is None:
            page_size = max_results

        # When timeout has default sentinel value ``object()``, do not pass
        # anything to invoke default timeouts in subsequent calls.
        done_kwargs: Dict[str, Union[_helpers.TimeoutType, object]] = {}
        reload_query_results_kwargs: Dict[str, Union[_helpers.TimeoutType, object]] = {}
        list_rows_kwargs: Dict[str, Union[_helpers.TimeoutType, object]] = {}
        if type(timeout) is not object:
            done_kwargs["timeout"] = timeout
            list_rows_kwargs["timeout"] = timeout
            reload_query_results_kwargs["timeout"] = timeout

        if page_size is not None:
            reload_query_results_kwargs["page_size"] = page_size

        if start_index is not None:
            reload_query_results_kwargs["start_index"] = start_index

        try:
            retry_do_query = getattr(self, "_retry_do_query", None)
            if retry_do_query is not None:
                if job_retry is DEFAULT_JOB_RETRY:
                    job_retry = self._job_retry  # type: ignore
            else:
                if job_retry is not None and job_retry is not DEFAULT_JOB_RETRY:
                    raise TypeError(
                        "`job_retry` was provided, but this job is"
                        " not retryable, because a custom `job_id` was"
                        " provided to the query that created this job."
                    )

            restart_query_job = False

            def is_job_done():
                nonlocal restart_query_job

                if restart_query_job:
                    restart_query_job = False

                    # The original job has failed. Create a new one.
                    #
                    # Note that we won't get here if retry_do_query is
                    # None, because we won't use a retry.
                    job = retry_do_query()

                    # Become the new job:
                    self.__dict__.clear()
                    self.__dict__.update(job.__dict__)

                    # It's possible the job fails again and we'll have to
                    # retry that too.
                    self._retry_do_query = retry_do_query
                    self._job_retry = job_retry

                # If the job hasn't been created, create it now. Related:
                # https://github.com/googleapis/python-bigquery/issues/1940
                if self.state is None:
                    self._begin(retry=retry, **done_kwargs)

                # Refresh the job status with jobs.get because some of the
                # exceptions thrown by jobs.getQueryResults like timeout and
                # rateLimitExceeded errors are ambiguous. We want to know if
                # the query job failed and not just the call to
                # jobs.getQueryResults.
                if self.done(retry=retry, **done_kwargs):
                    # If it's already failed, we might as well stop.
                    job_failed_exception = self.exception()
                    if job_failed_exception is not None:
                        # Only try to restart the query job if the job failed for
                        # a retriable reason. For example, don't restart the query
                        # if the call to reload the job metadata within self.done()
                        # timed out.
                        #
                        # The `restart_query_job` must only be called after a
                        # successful call to the `jobs.get` REST API and we
                        # determine that the job has failed.
                        #
                        # The `jobs.get` REST API
                        # (https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/get)
                        #  is called via `self.done()` which calls
                        # `self.reload()`.
                        #
                        # To determine if the job failed, the `self.exception()`
                        # is set from `self.reload()` via
                        # `self._set_properties()`, which translates the
                        # `Job.status.errorResult` field
                        # (https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatus.FIELDS.error_result)
                        # into an exception that can be processed by the
                        # `job_retry` predicate.
                        restart_query_job = True
                        raise job_failed_exception
                    else:
                        # Make sure that the _query_results are cached so we
                        # can return a complete RowIterator.
                        #
                        # Note: As an optimization, _reload_query_results
                        # doesn't make any API calls if the query results are
                        # already cached and have jobComplete=True in the
                        # response from the REST API. This ensures we aren't
                        # making any extra API calls if the previous loop
                        # iteration fetched the finished job.
                        self._reload_query_results(
                            retry=retry, **reload_query_results_kwargs
                        )
                        return True

                # Call jobs.getQueryResults with max results set to 0 just to
                # wait for the query to finish. Unlike most methods,
                # jobs.getQueryResults hangs as long as it can to ensure we
                # know when the query has finished as soon as possible.
                self._reload_query_results(retry=retry, **reload_query_results_kwargs)

                # Even if the query is finished now according to
                # jobs.getQueryResults, we'll want to reload the job status if
                # it's not already DONE.
                return False

            if retry_do_query is not None and job_retry is not None:
                is_job_done = job_retry(is_job_done)

            # timeout can be a number of seconds, `None`, or a
            # `google.api_core.future.polling.PollingFuture._DEFAULT_VALUE`
            # sentinel object indicating a default timeout if we choose to add
            # one some day. This value can come from our PollingFuture
            # superclass and was introduced in
            # https://github.com/googleapis/python-api-core/pull/462.
            if isinstance(timeout, (float, int)):
                remaining_timeout = timeout
            else:
                # Note: we may need to handle _DEFAULT_VALUE as a separate
                # case someday, but even then the best we can do for queries
                # is 72+ hours for hyperparameter tuning jobs:
                # https://cloud.google.com/bigquery/quotas#query_jobs
                #
                # The timeout for a multi-statement query is 24+ hours. See:
                # https://cloud.google.com/bigquery/quotas#multi_statement_query_limits
                remaining_timeout = None

            if remaining_timeout is None:
                # Since is_job_done() calls jobs.getQueryResults, which is a
                # long-running API, don't delay the next request at all.
                while not is_job_done():
                    pass
            else:
                # Use a monotonic clock since we don't actually care about
                # daylight savings or similar, just the elapsed time.
                previous_time = time.monotonic()

                while not is_job_done():
                    current_time = time.monotonic()
                    elapsed_time = current_time - previous_time
                    remaining_timeout = remaining_timeout - elapsed_time
                    previous_time = current_time

                    if remaining_timeout < 0:
                        raise concurrent.futures.TimeoutError()

        except exceptions.GoogleAPICallError as exc:
            exc.message = _EXCEPTION_FOOTER_TEMPLATE.format(
                message=exc.message, location=self.location, job_id=self.job_id
            )
            exc.debug_message = self._format_for_exception(exc.message, self.query)  # type: ignore
            exc.query_job = self  # type: ignore
            raise
        except requests.exceptions.Timeout as exc:
            raise concurrent.futures.TimeoutError from exc

        # If the query job is complete but there are no query results, this was
        # special job, such as a DDL query. Return an empty result set to
        # indicate success and avoid calling tabledata.list on a table which
        # can't be read (such as a view table).
        if self._query_results.total_rows is None:
            return _EmptyRowIterator(
                location=self.location,
                project=self.project,
                job_id=self.job_id,
                query_id=self.query_id,
                schema=self.schema,
                num_dml_affected_rows=self._query_results.num_dml_affected_rows,
                query=self.query,
                total_bytes_processed=self.total_bytes_processed,
                slot_millis=self.slot_millis,
            )

        # We know that there's at least 1 row, so only treat the response from
        # jobs.getQueryResults / jobs.query as the first page of the
        # RowIterator response if there are any rows in it. This prevents us
        # from stopping the iteration early in the cases where we set
        # maxResults=0. In that case, we're missing rows and there's no next
        # page token.
        first_page_response = self._query_results._properties
        if "rows" not in first_page_response:
            first_page_response = None

        rows = self._client._list_rows_from_query_results(
            self.job_id,
            self.location,
            self.project,
            self._query_results.schema,
            total_rows=self._query_results.total_rows,
            destination=self.destination,
            page_size=page_size,
            max_results=max_results,
            start_index=start_index,
            retry=retry,
            query_id=self.query_id,
            first_page_response=first_page_response,
            num_dml_affected_rows=self._query_results.num_dml_affected_rows,
            query=self.query,
            total_bytes_processed=self.total_bytes_processed,
            slot_millis=self.slot_millis,
            created=self.created,
            started=self.started,
            ended=self.ended,
            **list_rows_kwargs,
        )
        rows._preserve_order = _contains_order_by(self.query)
        return rows

    # If changing the signature of this method, make sure to apply the same
    # changes to table.RowIterator.to_arrow(), except for the max_results parameter
    # that should only exist here in the QueryJob method.
    def to_arrow(
        self,
        progress_bar_type: Optional[str] = None,
        bqstorage_client: Optional["bigquery_storage.BigQueryReadClient"] = None,
        create_bqstorage_client: bool = True,
        max_results: Optional[int] = None,
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
                A BigQuery Storage API client. If supplied, use the faster
                BigQuery Storage API to fetch rows from BigQuery. This API
                is a billable API.

                This method requires ``google-cloud-bigquery-storage`` library.

                Reading from a specific partition or snapshot is not
                currently supported by this method.
            create_bqstorage_client (Optional[bool]):
                If ``True`` (default), create a BigQuery Storage API client
                using the default API settings. The BigQuery Storage API
                is a faster way to fetch rows from BigQuery. See the
                ``bqstorage_client`` parameter for more information.

                This argument does nothing if ``bqstorage_client`` is supplied.

                .. versionadded:: 1.24.0

            max_results (Optional[int]):
                Maximum number of rows to include in the result. No limit by default.

                .. versionadded:: 2.21.0

        Returns:
            pyarrow.Table
                A :class:`pyarrow.Table` populated with row data and column
                headers from the query results. The column headers are derived
                from the destination table's schema.

        Raises:
            ValueError:
                If the :mod:`pyarrow` library cannot be imported.

        .. versionadded:: 1.17.0
        """
        query_result = wait_for_query(self, progress_bar_type, max_results=max_results)
        return query_result.to_arrow(
            progress_bar_type=progress_bar_type,
            bqstorage_client=bqstorage_client,
            create_bqstorage_client=create_bqstorage_client,
        )

    # If changing the signature of this method, make sure to apply the same
    # changes to table.RowIterator.to_dataframe(), except for the max_results parameter
    # that should only exist here in the QueryJob method.
    def to_dataframe(
        self,
        bqstorage_client: Optional["bigquery_storage.BigQueryReadClient"] = None,
        dtypes: Optional[Dict[str, Any]] = None,
        progress_bar_type: Optional[str] = None,
        create_bqstorage_client: bool = True,
        max_results: Optional[int] = None,
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
        """Return a pandas DataFrame from a QueryJob

        Args:
            bqstorage_client (Optional[google.cloud.bigquery_storage_v1.BigQueryReadClient]):
                A BigQuery Storage API client. If supplied, use the faster
                BigQuery Storage API to fetch rows from BigQuery. This
                API is a billable API.

                This method requires the ``fastavro`` and
                ``google-cloud-bigquery-storage`` libraries.

                Reading from a specific partition or snapshot is not
                currently supported by this method.

            dtypes (Optional[Map[str, Union[str, pandas.Series.dtype]]]):
                A dictionary of column names pandas ``dtype``s. The provided
                ``dtype`` is used when constructing the series for the column
                specified. Otherwise, the default pandas behavior is used.

            progress_bar_type (Optional[str]):
                If set, use the `tqdm <https://tqdm.github.io/>`_ library to
                display a progress bar while the data downloads. Install the
                ``tqdm`` package to use this feature.

                See
                :func:`~google.cloud.bigquery.table.RowIterator.to_dataframe`
                for details.

                .. versionadded:: 1.11.0
            create_bqstorage_client (Optional[bool]):
                If ``True`` (default), create a BigQuery Storage API client
                using the default API settings. The BigQuery Storage API
                is a faster way to fetch rows from BigQuery. See the
                ``bqstorage_client`` parameter for more information.

                This argument does nothing if ``bqstorage_client`` is supplied.

                .. versionadded:: 1.24.0

            max_results (Optional[int]):
                Maximum number of rows to include in the result. No limit by default.

                .. versionadded:: 2.21.0

            geography_as_object (Optional[bool]):
                If ``True``, convert GEOGRAPHY data to :mod:`shapely`
                geometry objects.  If ``False`` (default), don't cast
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
                A :class:`~pandas.DataFrame` populated with row data
                and column headers from the query results. The column
                headers are derived from the destination table's
                schema.

        Raises:
            ValueError:
                If the :mod:`pandas` library cannot be imported, or
                the :mod:`google.cloud.bigquery_storage_v1` module is
                required but cannot be imported.  Also if
                `geography_as_object` is `True`, but the
                :mod:`shapely` library cannot be imported.
        """
        query_result = wait_for_query(self, progress_bar_type, max_results=max_results)
        return query_result.to_dataframe(
            bqstorage_client=bqstorage_client,
            dtypes=dtypes,
            progress_bar_type=progress_bar_type,
            create_bqstorage_client=create_bqstorage_client,
            geography_as_object=geography_as_object,
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
        )

    # If changing the signature of this method, make sure to apply the same
    # changes to table.RowIterator.to_dataframe(), except for the max_results parameter
    # that should only exist here in the QueryJob method.
    def to_geodataframe(
        self,
        bqstorage_client: Optional["bigquery_storage.BigQueryReadClient"] = None,
        dtypes: Optional[Dict[str, Any]] = None,
        progress_bar_type: Optional[str] = None,
        create_bqstorage_client: bool = True,
        max_results: Optional[int] = None,
        geography_column: Optional[str] = None,
        bool_dtype: Union[Any, None] = DefaultPandasDTypes.BOOL_DTYPE,
        int_dtype: Union[Any, None] = DefaultPandasDTypes.INT_DTYPE,
        float_dtype: Union[Any, None] = None,
        string_dtype: Union[Any, None] = None,
    ) -> "geopandas.GeoDataFrame":
        """Return a GeoPandas GeoDataFrame from a QueryJob

        Args:
            bqstorage_client (Optional[google.cloud.bigquery_storage_v1.BigQueryReadClient]):
                A BigQuery Storage API client. If supplied, use the faster
                BigQuery Storage API to fetch rows from BigQuery. This
                API is a billable API.

                This method requires the ``fastavro`` and
                ``google-cloud-bigquery-storage`` libraries.

                Reading from a specific partition or snapshot is not
                currently supported by this method.

            dtypes (Optional[Map[str, Union[str, pandas.Series.dtype]]]):
                A dictionary of column names pandas ``dtype``s. The provided
                ``dtype`` is used when constructing the series for the column
                specified. Otherwise, the default pandas behavior is used.

            progress_bar_type (Optional[str]):
                If set, use the `tqdm <https://tqdm.github.io/>`_ library to
                display a progress bar while the data downloads. Install the
                ``tqdm`` package to use this feature.

                See
                :func:`~google.cloud.bigquery.table.RowIterator.to_dataframe`
                for details.

                .. versionadded:: 1.11.0
            create_bqstorage_client (Optional[bool]):
                If ``True`` (default), create a BigQuery Storage API client
                using the default API settings. The BigQuery Storage API
                is a faster way to fetch rows from BigQuery. See the
                ``bqstorage_client`` parameter for more information.

                This argument does nothing if ``bqstorage_client`` is supplied.

                .. versionadded:: 1.24.0

            max_results (Optional[int]):
                Maximum number of rows to include in the result. No limit by default.

                .. versionadded:: 2.21.0

            geography_column (Optional[str]):
                If there are more than one GEOGRAPHY column,
                identifies which one to use to construct a GeoPandas
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
        query_result = wait_for_query(self, progress_bar_type, max_results=max_results)
        return query_result.to_geodataframe(
            bqstorage_client=bqstorage_client,
            dtypes=dtypes,
            progress_bar_type=progress_bar_type,
            create_bqstorage_client=create_bqstorage_client,
            geography_column=geography_column,
            bool_dtype=bool_dtype,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            string_dtype=string_dtype,
        )

    def __iter__(self):
        return iter(self.result())


class QueryPlanEntryStep(object):
    """Map a single step in a query plan entry.

    Args:
        kind (str): step type.
        substeps (List): names of substeps.
    """

    def __init__(self, kind, substeps):
        self.kind = kind
        self.substeps = list(substeps)

    @classmethod
    def from_api_repr(cls, resource: dict) -> "QueryPlanEntryStep":
        """Factory: construct instance from the JSON repr.

        Args:
            resource (Dict): JSON representation of the entry.

        Returns:
            google.cloud.bigquery.job.QueryPlanEntryStep:
                New instance built from the resource.
        """
        return cls(kind=resource.get("kind"), substeps=resource.get("substeps", ()))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.kind == other.kind and self.substeps == other.substeps


class QueryPlanEntry(object):
    """QueryPlanEntry represents a single stage of a query execution plan.

    See
    https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#ExplainQueryStage
    for the underlying API representation within query statistics.
    """

    def __init__(self):
        self._properties = {}

    @classmethod
    def from_api_repr(cls, resource: dict) -> "QueryPlanEntry":
        """Factory: construct instance from the JSON repr.

        Args:
            resource(Dict[str: object]):
                ExplainQueryStage representation returned from API.

        Returns:
            google.cloud.bigquery.job.QueryPlanEntry:
                Query plan entry parsed from ``resource``.
        """
        entry = cls()
        entry._properties = resource
        return entry

    @property
    def name(self):
        """Optional[str]: Human-readable name of the stage."""
        return self._properties.get("name")

    @property
    def entry_id(self):
        """Optional[str]: Unique ID for the stage within the plan."""
        return self._properties.get("id")

    @property
    def start(self):
        """Optional[Datetime]: Datetime when the stage started."""
        if self._properties.get("startMs") is None:
            return None
        return _helpers._datetime_from_microseconds(
            int(self._properties.get("startMs")) * 1000.0
        )

    @property
    def end(self):
        """Optional[Datetime]: Datetime when the stage ended."""
        if self._properties.get("endMs") is None:
            return None
        return _helpers._datetime_from_microseconds(
            int(self._properties.get("endMs")) * 1000.0
        )

    @property
    def input_stages(self):
        """List(int): Entry IDs for stages that were inputs for this stage."""
        if self._properties.get("inputStages") is None:
            return []
        return [
            _helpers._int_or_none(entry)
            for entry in self._properties.get("inputStages")
        ]

    @property
    def parallel_inputs(self):
        """Optional[int]: Number of parallel input segments within
        the stage.
        """
        return _helpers._int_or_none(self._properties.get("parallelInputs"))

    @property
    def completed_parallel_inputs(self):
        """Optional[int]: Number of parallel input segments completed."""
        return _helpers._int_or_none(self._properties.get("completedParallelInputs"))

    @property
    def wait_ms_avg(self):
        """Optional[int]: Milliseconds the average worker spent waiting to
        be scheduled.
        """
        return _helpers._int_or_none(self._properties.get("waitMsAvg"))

    @property
    def wait_ms_max(self):
        """Optional[int]: Milliseconds the slowest worker spent waiting to
        be scheduled.
        """
        return _helpers._int_or_none(self._properties.get("waitMsMax"))

    @property
    def wait_ratio_avg(self):
        """Optional[float]: Ratio of time the average worker spent waiting
        to be scheduled, relative to the longest time spent by any worker in
        any stage of the overall plan.
        """
        return self._properties.get("waitRatioAvg")

    @property
    def wait_ratio_max(self):
        """Optional[float]: Ratio of time the slowest worker spent waiting
        to be scheduled, relative to the longest time spent by any worker in
        any stage of the overall plan.
        """
        return self._properties.get("waitRatioMax")

    @property
    def read_ms_avg(self):
        """Optional[int]: Milliseconds the average worker spent reading
        input.
        """
        return _helpers._int_or_none(self._properties.get("readMsAvg"))

    @property
    def read_ms_max(self):
        """Optional[int]: Milliseconds the slowest worker spent reading
        input.
        """
        return _helpers._int_or_none(self._properties.get("readMsMax"))

    @property
    def read_ratio_avg(self):
        """Optional[float]: Ratio of time the average worker spent reading
        input, relative to the longest time spent by any worker in any stage
        of the overall plan.
        """
        return self._properties.get("readRatioAvg")

    @property
    def read_ratio_max(self):
        """Optional[float]: Ratio of time the slowest worker spent reading
        to be scheduled, relative to the longest time spent by any worker in
        any stage of the overall plan.
        """
        return self._properties.get("readRatioMax")

    @property
    def compute_ms_avg(self):
        """Optional[int]: Milliseconds the average worker spent on CPU-bound
        processing.
        """
        return _helpers._int_or_none(self._properties.get("computeMsAvg"))

    @property
    def compute_ms_max(self):
        """Optional[int]: Milliseconds the slowest worker spent on CPU-bound
        processing.
        """
        return _helpers._int_or_none(self._properties.get("computeMsMax"))

    @property
    def compute_ratio_avg(self):
        """Optional[float]: Ratio of time the average worker spent on
        CPU-bound processing, relative to the longest time spent by any
        worker in any stage of the overall plan.
        """
        return self._properties.get("computeRatioAvg")

    @property
    def compute_ratio_max(self):
        """Optional[float]: Ratio of time the slowest worker spent on
        CPU-bound processing, relative to the longest time spent by any
        worker in any stage of the overall plan.
        """
        return self._properties.get("computeRatioMax")

    @property
    def write_ms_avg(self):
        """Optional[int]: Milliseconds the average worker spent writing
        output data.
        """
        return _helpers._int_or_none(self._properties.get("writeMsAvg"))

    @property
    def write_ms_max(self):
        """Optional[int]: Milliseconds the slowest worker spent writing
        output data.
        """
        return _helpers._int_or_none(self._properties.get("writeMsMax"))

    @property
    def write_ratio_avg(self):
        """Optional[float]: Ratio of time the average worker spent writing
        output data, relative to the longest time spent by any worker in any
        stage of the overall plan.
        """
        return self._properties.get("writeRatioAvg")

    @property
    def write_ratio_max(self):
        """Optional[float]: Ratio of time the slowest worker spent writing
        output data, relative to the longest time spent by any worker in any
        stage of the overall plan.
        """
        return self._properties.get("writeRatioMax")

    @property
    def records_read(self):
        """Optional[int]: Number of records read by this stage."""
        return _helpers._int_or_none(self._properties.get("recordsRead"))

    @property
    def records_written(self):
        """Optional[int]: Number of records written by this stage."""
        return _helpers._int_or_none(self._properties.get("recordsWritten"))

    @property
    def status(self):
        """Optional[str]: status of this stage."""
        return self._properties.get("status")

    @property
    def shuffle_output_bytes(self):
        """Optional[int]: Number of bytes written by this stage to
        intermediate shuffle.
        """
        return _helpers._int_or_none(self._properties.get("shuffleOutputBytes"))

    @property
    def shuffle_output_bytes_spilled(self):
        """Optional[int]: Number of bytes written by this stage to
        intermediate shuffle and spilled to disk.
        """
        return _helpers._int_or_none(self._properties.get("shuffleOutputBytesSpilled"))

    @property
    def steps(self):
        """List(QueryPlanEntryStep): List of step operations performed by
        each worker in the stage.
        """
        return [
            QueryPlanEntryStep.from_api_repr(step)
            for step in self._properties.get("steps", [])
        ]

    @property
    def slot_ms(self):
        """Optional[int]: Slot-milliseconds used by the stage."""
        return _helpers._int_or_none(self._properties.get("slotMs"))


class TimelineEntry(object):
    """TimelineEntry represents progress of a query job at a particular
    point in time.

    See
    https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#querytimelinesample
    for the underlying API representation within query statistics.
    """

    def __init__(self):
        self._properties = {}

    @classmethod
    def from_api_repr(cls, resource):
        """Factory: construct instance from the JSON repr.

        Args:
            resource(Dict[str: object]):
                QueryTimelineSample representation returned from API.

        Returns:
            google.cloud.bigquery.TimelineEntry:
                Timeline sample parsed from ``resource``.
        """
        entry = cls()
        entry._properties = resource
        return entry

    @property
    def elapsed_ms(self):
        """Optional[int]: Milliseconds elapsed since start of query
        execution."""
        return _helpers._int_or_none(self._properties.get("elapsedMs"))

    @property
    def active_units(self):
        """Optional[int]: Current number of input units being processed
        by workers, reported as largest value since the last sample."""
        return _helpers._int_or_none(self._properties.get("activeUnits"))

    @property
    def pending_units(self):
        """Optional[int]: Current number of input units remaining for
        query stages active at this sample time."""
        return _helpers._int_or_none(self._properties.get("pendingUnits"))

    @property
    def completed_units(self):
        """Optional[int]: Current number of input units completed by
        this query."""
        return _helpers._int_or_none(self._properties.get("completedUnits"))

    @property
    def slot_millis(self):
        """Optional[int]: Cumulative slot-milliseconds consumed by
        this query."""
        return _helpers._int_or_none(self._properties.get("totalSlotMs"))
