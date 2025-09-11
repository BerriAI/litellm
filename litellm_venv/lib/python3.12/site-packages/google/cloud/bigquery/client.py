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

"""Client for interacting with the Google BigQuery API."""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division

from collections import abc as collections_abc
import copy
import datetime
import functools
import gzip
import io
import itertools
import json
import math
import os
import tempfile
import typing
from typing import (
    Any,
    Callable,
    Dict,
    IO,
    Iterable,
    Mapping,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import uuid
import warnings

import requests

from google import resumable_media  # type: ignore
from google.resumable_media.requests import MultipartUpload  # type: ignore
from google.resumable_media.requests import ResumableUpload

import google.api_core.client_options
import google.api_core.exceptions as core_exceptions
from google.api_core.iam import Policy
from google.api_core import page_iterator
from google.api_core import retry as retries
import google.cloud._helpers  # type: ignore
from google.cloud import exceptions  # pytype: disable=import-error
from google.cloud.client import ClientWithProject  # type: ignore  # pytype: disable=import-error

try:
    from google.cloud.bigquery_storage_v1.services.big_query_read.client import (
        DEFAULT_CLIENT_INFO as DEFAULT_BQSTORAGE_CLIENT_INFO,
    )
except ImportError:
    DEFAULT_BQSTORAGE_CLIENT_INFO = None  # type: ignore


from google.auth.credentials import Credentials
from google.cloud.bigquery._http import Connection
from google.cloud.bigquery import _job_helpers
from google.cloud.bigquery import _pandas_helpers
from google.cloud.bigquery import _versions_helpers
from google.cloud.bigquery import enums
from google.cloud.bigquery import exceptions as bq_exceptions
from google.cloud.bigquery import job
from google.cloud.bigquery._helpers import _get_sub_prop
from google.cloud.bigquery._helpers import _record_field_to_json
from google.cloud.bigquery._helpers import _str_or_none
from google.cloud.bigquery._helpers import _verify_job_config_type
from google.cloud.bigquery._helpers import _get_bigquery_host
from google.cloud.bigquery._helpers import _DEFAULT_HOST
from google.cloud.bigquery._helpers import _DEFAULT_HOST_TEMPLATE
from google.cloud.bigquery._helpers import _DEFAULT_UNIVERSE
from google.cloud.bigquery._helpers import _validate_universe
from google.cloud.bigquery._helpers import _get_client_universe
from google.cloud.bigquery._helpers import TimeoutType
from google.cloud.bigquery._job_helpers import make_job_id as _make_job_id
from google.cloud.bigquery.dataset import Dataset
from google.cloud.bigquery.dataset import DatasetListItem
from google.cloud.bigquery.dataset import DatasetReference

from google.cloud.bigquery.enums import AutoRowIDs, DatasetView, UpdateMode
from google.cloud.bigquery.format_options import ParquetOptions
from google.cloud.bigquery.job import (
    CopyJob,
    CopyJobConfig,
    ExtractJob,
    ExtractJobConfig,
    LoadJob,
    LoadJobConfig,
    QueryJob,
    QueryJobConfig,
)
from google.cloud.bigquery.model import Model
from google.cloud.bigquery.model import ModelReference
from google.cloud.bigquery.model import _model_arg_to_model_ref
from google.cloud.bigquery.opentelemetry_tracing import create_span
from google.cloud.bigquery.query import _QueryResults
from google.cloud.bigquery.retry import (
    DEFAULT_JOB_RETRY,
    DEFAULT_RETRY,
    DEFAULT_TIMEOUT,
    DEFAULT_GET_JOB_TIMEOUT,
    POLLING_DEFAULT_VALUE,
)
from google.cloud.bigquery.routine import Routine
from google.cloud.bigquery.routine import RoutineReference
from google.cloud.bigquery.schema import SchemaField
from google.cloud.bigquery.table import _table_arg_to_table
from google.cloud.bigquery.table import _table_arg_to_table_ref
from google.cloud.bigquery.table import Table
from google.cloud.bigquery.table import TableListItem
from google.cloud.bigquery.table import TableReference
from google.cloud.bigquery.table import RowIterator

pyarrow = _versions_helpers.PYARROW_VERSIONS.try_import()
pandas = (
    _versions_helpers.PANDAS_VERSIONS.try_import()
)  # mypy check fails because pandas import is outside module, there are type: ignore comments related to this


ResumableTimeoutType = Union[
    None, float, Tuple[float, float]
]  # for resumable media methods

if typing.TYPE_CHECKING:  # pragma: NO COVER
    # os.PathLike is only subscriptable in Python 3.9+, thus shielding with a condition.
    PathType = Union[str, bytes, os.PathLike[str], os.PathLike[bytes]]
_DEFAULT_CHUNKSIZE = 100 * 1024 * 1024  # 100 MB
_MAX_MULTIPART_SIZE = 5 * 1024 * 1024
_DEFAULT_NUM_RETRIES = 6
_BASE_UPLOAD_TEMPLATE = "{host}/upload/bigquery/v2/projects/{project}/jobs?uploadType="
_MULTIPART_URL_TEMPLATE = _BASE_UPLOAD_TEMPLATE + "multipart"
_RESUMABLE_URL_TEMPLATE = _BASE_UPLOAD_TEMPLATE + "resumable"
_GENERIC_CONTENT_TYPE = "*/*"
_READ_LESS_THAN_SIZE = (
    "Size {:d} was specified but the file-like object only had " "{:d} bytes remaining."
)
_NEED_TABLE_ARGUMENT = (
    "The table argument should be a table ID string, Table, or TableReference"
)
_LIST_ROWS_FROM_QUERY_RESULTS_FIELDS = "jobReference,totalRows,pageToken,rows"

# In microbenchmarks, it's been shown that even in ideal conditions (query
# finished, local data), requests to getQueryResults can take 10+ seconds.
# In less-than-ideal situations, the response can take even longer, as it must
# be able to download a full 100+ MB row in that time. Don't let the
# connection timeout before data can be downloaded.
# https://github.com/googleapis/python-bigquery/issues/438
_MIN_GET_QUERY_RESULTS_TIMEOUT = 120

TIMEOUT_HEADER = "X-Server-Timeout"


class Project(object):
    """Wrapper for resource describing a BigQuery project.

    Args:
        project_id (str): Opaque ID of the project

        numeric_id (int): Numeric ID of the project

        friendly_name (str): Display name of the project
    """

    def __init__(self, project_id, numeric_id, friendly_name):
        self.project_id = project_id
        self.numeric_id = numeric_id
        self.friendly_name = friendly_name

    @classmethod
    def from_api_repr(cls, resource):
        """Factory: construct an instance from a resource dict."""
        return cls(resource["id"], resource["numericId"], resource["friendlyName"])


class Client(ClientWithProject):
    """Client to bundle configuration needed for API requests.

    Args:
        project (Optional[str]):
            Project ID for the project which the client acts on behalf of.
            Will be passed when creating a dataset / job. If not passed,
            falls back to the default inferred from the environment.
        credentials (Optional[google.auth.credentials.Credentials]):
            The OAuth2 Credentials to use for this client. If not passed
            (and if no ``_http`` object is passed), falls back to the
            default inferred from the environment.
        _http (Optional[requests.Session]):
            HTTP object to make requests. Can be any object that
            defines ``request()`` with the same interface as
            :meth:`requests.Session.request`. If not passed, an ``_http``
            object is created that is bound to the ``credentials`` for the
            current object.
            This parameter should be considered private, and could change in
            the future.
        location (Optional[str]):
            Default location for jobs / datasets / tables.
        default_query_job_config (Optional[google.cloud.bigquery.job.QueryJobConfig]):
            Default ``QueryJobConfig``.
            Will be merged into job configs passed into the ``query`` method.
        default_load_job_config (Optional[google.cloud.bigquery.job.LoadJobConfig]):
            Default ``LoadJobConfig``.
            Will be merged into job configs passed into the ``load_table_*`` methods.
        client_info (Optional[google.api_core.client_info.ClientInfo]):
            The client info used to send a user-agent string along with API
            requests. If ``None``, then default info will be used. Generally,
            you only need to set this if you're developing your own library
            or partner tool.
        client_options (Optional[Union[google.api_core.client_options.ClientOptions, Dict]]):
            Client options used to set user options on the client. API Endpoint
            should be set through client_options.
        default_job_creation_mode (Optional[str]):
            Sets the default job creation mode used by query methods such as
            query_and_wait().  For lightweight queries, JOB_CREATION_OPTIONAL is
            generally recommended.

    Raises:
        google.auth.exceptions.DefaultCredentialsError:
            Raised if ``credentials`` is not specified and the library fails
            to acquire default credentials.
    """

    SCOPE = ("https://www.googleapis.com/auth/cloud-platform",)  # type: ignore
    """The scopes required for authenticating as a BigQuery consumer."""

    def __init__(
        self,
        project: Optional[str] = None,
        credentials: Optional[Credentials] = None,
        _http: Optional[requests.Session] = None,
        location: Optional[str] = None,
        default_query_job_config: Optional[QueryJobConfig] = None,
        default_load_job_config: Optional[LoadJobConfig] = None,
        client_info: Optional[google.api_core.client_info.ClientInfo] = None,
        client_options: Optional[
            Union[google.api_core.client_options.ClientOptions, Dict[str, Any]]
        ] = None,
        default_job_creation_mode: Optional[str] = None,
    ) -> None:
        if client_options is None:
            client_options = {}
        if isinstance(client_options, dict):
            client_options = google.api_core.client_options.from_dict(client_options)
        # assert isinstance(client_options, google.api_core.client_options.ClientOptions)

        super(Client, self).__init__(
            project=project,
            credentials=credentials,
            client_options=client_options,
            _http=_http,
        )

        kw_args: Dict[str, Any] = {"client_info": client_info}
        bq_host = _get_bigquery_host()
        kw_args["api_endpoint"] = bq_host if bq_host != _DEFAULT_HOST else None
        client_universe = None
        if client_options.api_endpoint:
            api_endpoint = client_options.api_endpoint
            kw_args["api_endpoint"] = api_endpoint
        else:
            client_universe = _get_client_universe(client_options)
            if client_universe != _DEFAULT_UNIVERSE:
                kw_args["api_endpoint"] = _DEFAULT_HOST_TEMPLATE.replace(
                    "{UNIVERSE_DOMAIN}", client_universe
                )
        # Ensure credentials and universe are not in conflict.
        if hasattr(self, "_credentials") and client_universe is not None:
            _validate_universe(client_universe, self._credentials)

        self._connection = Connection(self, **kw_args)
        self._location = location
        self._default_load_job_config = copy.deepcopy(default_load_job_config)
        self.default_job_creation_mode = default_job_creation_mode

        # Use property setter so validation can run.
        self.default_query_job_config = default_query_job_config

    @property
    def location(self):
        """Default location for jobs / datasets / tables."""
        return self._location

    @property
    def default_job_creation_mode(self):
        """Default job creation mode used for query execution."""
        return self._default_job_creation_mode

    @default_job_creation_mode.setter
    def default_job_creation_mode(self, value: Optional[str]):
        self._default_job_creation_mode = value

    @property
    def default_query_job_config(self) -> Optional[QueryJobConfig]:
        """Default ``QueryJobConfig`` or ``None``.

        Will be merged into job configs passed into the ``query`` or
        ``query_and_wait`` methods.
        """
        return self._default_query_job_config

    @default_query_job_config.setter
    def default_query_job_config(self, value: Optional[QueryJobConfig]):
        if value is not None:
            _verify_job_config_type(
                value, QueryJobConfig, param_name="default_query_job_config"
            )
        self._default_query_job_config = copy.deepcopy(value)

    @property
    def default_load_job_config(self):
        """Default ``LoadJobConfig``.
        Will be merged into job configs passed into the ``load_table_*`` methods.
        """
        return self._default_load_job_config

    @default_load_job_config.setter
    def default_load_job_config(self, value: LoadJobConfig):
        self._default_load_job_config = copy.deepcopy(value)

    def close(self):
        """Close the underlying transport objects, releasing system resources.

        .. note::

            The client instance can be used for making additional requests even
            after closing, in which case the underlying connections are
            automatically re-created.
        """
        self._http._auth_request.session.close()
        self._http.close()

    def get_service_account_email(
        self,
        project: Optional[str] = None,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
    ) -> str:
        """Get the email address of the project's BigQuery service account

        Example:

        .. code-block:: python

            from google.cloud import bigquery
            client = bigquery.Client()
            client.get_service_account_email()
            # returns an email similar to: my_service_account@my-project.iam.gserviceaccount.com

        Note:
            This is the service account that BigQuery uses to manage tables
            encrypted by a key in KMS.

        Args:
            project (Optional[str]):
                Project ID to use for retreiving service account email.
                Defaults to the client's project.
            retry (Optional[google.api_core.retry.Retry]): How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

        Returns:
            str:
                service account email address

        """
        if project is None:
            project = self.project
        path = "/projects/%s/serviceAccount" % (project,)
        span_attributes = {"path": path}
        api_response = self._call_api(
            retry,
            span_name="BigQuery.getServiceAccountEmail",
            span_attributes=span_attributes,
            method="GET",
            path=path,
            timeout=timeout,
        )
        return api_response["email"]

    def list_projects(
        self,
        max_results: Optional[int] = None,
        page_token: Optional[str] = None,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
        page_size: Optional[int] = None,
    ) -> page_iterator.Iterator:
        """List projects for the project associated with this client.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/projects/list

        Args:
            max_results (Optional[int]):
                Maximum number of projects to return.
                Defaults to a value set by the API.

            page_token (Optional[str]):
                Token representing a cursor into the projects. If not passed,
                the API will return the first page of projects. The token marks
                the beginning of the iterator to be returned and the value of
                the ``page_token`` can be accessed at ``next_page_token`` of the
                :class:`~google.api_core.page_iterator.HTTPIterator`.

            retry (Optional[google.api_core.retry.Retry]): How to retry the RPC.

            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

            page_size (Optional[int]):
                Maximum number of projects to return in each page.
                Defaults to a value set by the API.

        Returns:
            google.api_core.page_iterator.Iterator:
                Iterator of :class:`~google.cloud.bigquery.client.Project`
                accessible to the current client.
        """
        span_attributes = {"path": "/projects"}

        def api_request(*args, **kwargs):
            return self._call_api(
                retry,
                span_name="BigQuery.listProjects",
                span_attributes=span_attributes,
                *args,
                timeout=timeout,
                **kwargs,
            )

        return page_iterator.HTTPIterator(
            client=self,
            api_request=api_request,
            path="/projects",
            item_to_value=_item_to_project,
            items_key="projects",
            page_token=page_token,
            max_results=max_results,
            page_size=page_size,
        )

    def list_datasets(
        self,
        project: Optional[str] = None,
        include_all: bool = False,
        filter: Optional[str] = None,
        max_results: Optional[int] = None,
        page_token: Optional[str] = None,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
        page_size: Optional[int] = None,
    ) -> page_iterator.Iterator:
        """List datasets for the project associated with this client.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/datasets/list

        Args:
            project (Optional[str]):
                Project ID to use for retreiving datasets. Defaults to the
                client's project.
            include_all (Optional[bool]):
                True if results include hidden datasets. Defaults to False.
            filter (Optional[str]):
                An expression for filtering the results by label.
                For syntax, see
                https://cloud.google.com/bigquery/docs/reference/rest/v2/datasets/list#body.QUERY_PARAMETERS.filter
            max_results (Optional[int]):
                Maximum number of datasets to return.
            page_token (Optional[str]):
                Token representing a cursor into the datasets. If not passed,
                the API will return the first page of datasets. The token marks
                the beginning of the iterator to be returned and the value of
                the ``page_token`` can be accessed at ``next_page_token`` of the
                :class:`~google.api_core.page_iterator.HTTPIterator`.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
            page_size (Optional[int]):
                Maximum number of datasets to return per page.

        Returns:
            google.api_core.page_iterator.Iterator:
                Iterator of :class:`~google.cloud.bigquery.dataset.DatasetListItem`.
                associated with the project.
        """
        extra_params: Dict[str, Any] = {}
        if project is None:
            project = self.project
        if include_all:
            extra_params["all"] = True
        if filter:
            # TODO: consider supporting a dict of label -> value for filter,
            # and converting it into a string here.
            extra_params["filter"] = filter
        path = "/projects/%s/datasets" % (project,)

        span_attributes = {"path": path}

        def api_request(*args, **kwargs):
            return self._call_api(
                retry,
                span_name="BigQuery.listDatasets",
                span_attributes=span_attributes,
                *args,
                timeout=timeout,
                **kwargs,
            )

        return page_iterator.HTTPIterator(
            client=self,
            api_request=api_request,
            path=path,
            item_to_value=_item_to_dataset,
            items_key="datasets",
            page_token=page_token,
            max_results=max_results,
            extra_params=extra_params,
            page_size=page_size,
        )

    def dataset(
        self, dataset_id: str, project: Optional[str] = None
    ) -> DatasetReference:
        """Deprecated: Construct a reference to a dataset.

        .. deprecated:: 1.24.0
           Construct a
           :class:`~google.cloud.bigquery.dataset.DatasetReference` using its
           constructor or use a string where previously a reference object
           was used.

           As of ``google-cloud-bigquery`` version 1.7.0, all client methods
           that take a
           :class:`~google.cloud.bigquery.dataset.DatasetReference` or
           :class:`~google.cloud.bigquery.table.TableReference` also take a
           string in standard SQL format, e.g. ``project.dataset_id`` or
           ``project.dataset_id.table_id``.

        Args:
            dataset_id (str): ID of the dataset.

            project (Optional[str]):
                Project ID for the dataset (defaults to the project of the client).

        Returns:
            google.cloud.bigquery.dataset.DatasetReference:
                a new ``DatasetReference`` instance.
        """
        if project is None:
            project = self.project

        warnings.warn(
            "Client.dataset is deprecated and will be removed in a future version. "
            "Use a string like 'my_project.my_dataset' or a "
            "cloud.google.bigquery.DatasetReference object, instead.",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        return DatasetReference(project, dataset_id)

    def _ensure_bqstorage_client(
        self,
        bqstorage_client: Optional[
            "google.cloud.bigquery_storage.BigQueryReadClient"
        ] = None,
        client_options: Optional[google.api_core.client_options.ClientOptions] = None,
        client_info: Optional[
            "google.api_core.gapic_v1.client_info.ClientInfo"
        ] = DEFAULT_BQSTORAGE_CLIENT_INFO,
    ) -> Optional["google.cloud.bigquery_storage.BigQueryReadClient"]:
        """Create a BigQuery Storage API client using this client's credentials.

        Args:
            bqstorage_client:
                An existing BigQuery Storage client instance. If ``None``, a new
                instance is created and returned.
            client_options:
                Custom options used with a new BigQuery Storage client instance
                if one is created.
            client_info:
                The client info used with a new BigQuery Storage client
                instance if one is created.

        Returns:
            A BigQuery Storage API client.
        """

        try:
            bigquery_storage = _versions_helpers.BQ_STORAGE_VERSIONS.try_import(
                raise_if_error=True
            )
        except bq_exceptions.BigQueryStorageNotFoundError:
            warnings.warn(
                "Cannot create BigQuery Storage client, the dependency "
                "google-cloud-bigquery-storage is not installed."
            )
            return None
        except bq_exceptions.LegacyBigQueryStorageError as exc:
            warnings.warn(
                "Dependency google-cloud-bigquery-storage is outdated: " + str(exc)
            )
            return None

        if bqstorage_client is None:  # pragma: NO COVER
            bqstorage_client = bigquery_storage.BigQueryReadClient(
                credentials=self._credentials,
                client_options=client_options,
                client_info=client_info,  # type: ignore  # (None is also accepted)
            )

        return bqstorage_client

    def _dataset_from_arg(self, dataset) -> Union[Dataset, DatasetReference]:
        if isinstance(dataset, str):
            dataset = DatasetReference.from_string(
                dataset, default_project=self.project
            )

        if not isinstance(dataset, (Dataset, DatasetReference)):
            if isinstance(dataset, DatasetListItem):
                dataset = dataset.reference
            else:
                raise TypeError(
                    "dataset must be a Dataset, DatasetReference, DatasetListItem,"
                    " or string"
                )
        return dataset

    def create_dataset(
        self,
        dataset: Union[str, Dataset, DatasetReference, DatasetListItem],
        exists_ok: bool = False,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
    ) -> Dataset:
        """API call: create the dataset via a POST request.


        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/datasets/insert

        Example:

        .. code-block:: python

            from google.cloud import bigquery
            client = bigquery.Client()
            dataset = bigquery.Dataset('my_project.my_dataset')
            dataset = client.create_dataset(dataset)

        Args:
            dataset (Union[ \
                google.cloud.bigquery.dataset.Dataset, \
                google.cloud.bigquery.dataset.DatasetReference, \
                google.cloud.bigquery.dataset.DatasetListItem, \
                str, \
            ]):
                A :class:`~google.cloud.bigquery.dataset.Dataset` to create.
                If ``dataset`` is a reference, an empty dataset is created
                with the specified ID and client's default location.
            exists_ok (Optional[bool]):
                Defaults to ``False``. If ``True``, ignore "already exists"
                errors when creating the dataset.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

        Returns:
            google.cloud.bigquery.dataset.Dataset:
                A new ``Dataset`` returned from the API.

        Raises:
            google.cloud.exceptions.Conflict:
                If the dataset already exists.
        """
        dataset = self._dataset_from_arg(dataset)
        if isinstance(dataset, DatasetReference):
            dataset = Dataset(dataset)

        path = "/projects/%s/datasets" % (dataset.project,)

        data = dataset.to_api_repr()
        if data.get("location") is None and self.location is not None:
            data["location"] = self.location

        try:
            span_attributes = {"path": path}

            api_response = self._call_api(
                retry,
                span_name="BigQuery.createDataset",
                span_attributes=span_attributes,
                method="POST",
                path=path,
                data=data,
                timeout=timeout,
            )
            return Dataset.from_api_repr(api_response)
        except core_exceptions.Conflict:
            if not exists_ok:
                raise
            return self.get_dataset(dataset.reference, retry=retry)

    def create_routine(
        self,
        routine: Routine,
        exists_ok: bool = False,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
    ) -> Routine:
        """[Beta] Create a routine via a POST request.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/routines/insert

        Args:
            routine (google.cloud.bigquery.routine.Routine):
                A :class:`~google.cloud.bigquery.routine.Routine` to create.
                The dataset that the routine belongs to must already exist.
            exists_ok (Optional[bool]):
                Defaults to ``False``. If ``True``, ignore "already exists"
                errors when creating the routine.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

        Returns:
            google.cloud.bigquery.routine.Routine:
                A new ``Routine`` returned from the service.

        Raises:
            google.cloud.exceptions.Conflict:
                If the routine already exists.
        """
        reference = routine.reference
        path = "/projects/{}/datasets/{}/routines".format(
            reference.project, reference.dataset_id
        )
        resource = routine.to_api_repr()
        try:
            span_attributes = {"path": path}
            api_response = self._call_api(
                retry,
                span_name="BigQuery.createRoutine",
                span_attributes=span_attributes,
                method="POST",
                path=path,
                data=resource,
                timeout=timeout,
            )
            return Routine.from_api_repr(api_response)
        except core_exceptions.Conflict:
            if not exists_ok:
                raise
            return self.get_routine(routine.reference, retry=retry)

    def create_table(
        self,
        table: Union[str, Table, TableReference, TableListItem],
        exists_ok: bool = False,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
    ) -> Table:
        """API call:  create a table via a PUT request

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables/insert

        Args:
            table (Union[ \
                google.cloud.bigquery.table.Table, \
                google.cloud.bigquery.table.TableReference, \
                google.cloud.bigquery.table.TableListItem, \
                str, \
            ]):
                A :class:`~google.cloud.bigquery.table.Table` to create.
                If ``table`` is a reference, an empty table is created
                with the specified ID. The dataset that the table belongs to
                must already exist.
            exists_ok (Optional[bool]):
                Defaults to ``False``. If ``True``, ignore "already exists"
                errors when creating the table.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

        Returns:
            google.cloud.bigquery.table.Table:
                A new ``Table`` returned from the service.

        Raises:
            google.cloud.exceptions.Conflict:
                If the table already exists.
        """
        table = _table_arg_to_table(table, default_project=self.project)
        dataset_id = table.dataset_id
        path = "/projects/%s/datasets/%s/tables" % (table.project, dataset_id)
        data = table.to_api_repr()
        try:
            span_attributes = {"path": path, "dataset_id": dataset_id}
            api_response = self._call_api(
                retry,
                span_name="BigQuery.createTable",
                span_attributes=span_attributes,
                method="POST",
                path=path,
                data=data,
                timeout=timeout,
            )
            return Table.from_api_repr(api_response)
        except core_exceptions.Conflict:
            if not exists_ok:
                raise
            return self.get_table(table.reference, retry=retry)

    def _call_api(
        self,
        retry,
        span_name=None,
        span_attributes=None,
        job_ref=None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        kwargs = _add_server_timeout_header(headers, kwargs)
        call = functools.partial(self._connection.api_request, **kwargs)

        if retry:
            call = retry(call)

        if span_name is not None:
            with create_span(
                name=span_name, attributes=span_attributes, client=self, job_ref=job_ref
            ):
                return call()

        return call()

    def get_dataset(
        self,
        dataset_ref: Union[DatasetReference, str],
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
        dataset_view: Optional[DatasetView] = None,
    ) -> Dataset:
        """Fetch the dataset referenced by ``dataset_ref``

        Args:
            dataset_ref (Union[ \
                google.cloud.bigquery.dataset.DatasetReference, \
                str, \
            ]):
                A reference to the dataset to fetch from the BigQuery API.
                If a string is passed in, this method attempts to create a
                dataset reference from a string using
                :func:`~google.cloud.bigquery.dataset.DatasetReference.from_string`.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
            dataset_view (Optional[google.cloud.bigquery.enums.DatasetView]):
                Specifies the view that determines which dataset information is
                returned. By default, dataset metadata (e.g. friendlyName, description,
                labels, etc) and ACL information are returned. This argument can
                take on the following possible enum values.

                * :attr:`~google.cloud.bigquery.enums.DatasetView.ACL`:
                    Includes dataset metadata and the ACL.
                * :attr:`~google.cloud.bigquery.enums.DatasetView.FULL`:
                    Includes all dataset metadata, including the ACL and table metadata.
                    This view is not supported by the `datasets.list` API method.
                * :attr:`~google.cloud.bigquery.enums.DatasetView.METADATA`:
                    Includes basic dataset metadata, but not the ACL.
                * :attr:`~google.cloud.bigquery.enums.DatasetView.DATASET_VIEW_UNSPECIFIED`:
                    The server will decide which view to use. Currently defaults to FULL.
        Returns:
            google.cloud.bigquery.dataset.Dataset:
                A ``Dataset`` instance.
        """
        if isinstance(dataset_ref, str):
            dataset_ref = DatasetReference.from_string(
                dataset_ref, default_project=self.project
            )
        path = dataset_ref.path

        if dataset_view:
            query_params = {"datasetView": dataset_view.value}
        else:
            query_params = {}

        span_attributes = {"path": path}
        api_response = self._call_api(
            retry,
            span_name="BigQuery.getDataset",
            span_attributes=span_attributes,
            method="GET",
            path=path,
            timeout=timeout,
            query_params=query_params,
        )
        return Dataset.from_api_repr(api_response)

    def get_iam_policy(
        self,
        table: Union[Table, TableReference, TableListItem, str],
        requested_policy_version: int = 1,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
    ) -> Policy:
        """Return the access control policy for a table resource.

        Args:
            table (Union[ \
                google.cloud.bigquery.table.Table, \
                google.cloud.bigquery.table.TableReference, \
                google.cloud.bigquery.table.TableListItem, \
                str, \
            ]):
                The table to get the access control policy for.
                If a string is passed in, this method attempts to create a
                table reference from a string using
                :func:`~google.cloud.bigquery.table.TableReference.from_string`.
            requested_policy_version (int):
                Optional. The maximum policy version that will be used to format the policy.

                Only version ``1`` is currently supported.

                See: https://cloud.google.com/bigquery/docs/reference/rest/v2/GetPolicyOptions
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

        Returns:
            google.api_core.iam.Policy:
                The access control policy.
        """
        table = _table_arg_to_table_ref(table, default_project=self.project)

        if requested_policy_version != 1:
            raise ValueError("only IAM policy version 1 is supported")

        body = {"options": {"requestedPolicyVersion": 1}}

        path = "{}:getIamPolicy".format(table.path)
        span_attributes = {"path": path}
        response = self._call_api(
            retry,
            span_name="BigQuery.getIamPolicy",
            span_attributes=span_attributes,
            method="POST",
            path=path,
            data=body,
            timeout=timeout,
        )

        return Policy.from_api_repr(response)

    def set_iam_policy(
        self,
        table: Union[Table, TableReference, TableListItem, str],
        policy: Policy,
        updateMask: Optional[str] = None,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
        *,
        fields: Sequence[str] = (),
    ) -> Policy:
        """Return the access control policy for a table resource.

        Args:
            table (Union[ \
                google.cloud.bigquery.table.Table, \
                google.cloud.bigquery.table.TableReference, \
                google.cloud.bigquery.table.TableListItem, \
                str, \
            ]):
                The table to get the access control policy for.
                If a string is passed in, this method attempts to create a
                table reference from a string using
                :func:`~google.cloud.bigquery.table.TableReference.from_string`.
            policy (google.api_core.iam.Policy):
                The access control policy to set.
            updateMask (Optional[str]):
                Mask as defined by
                https://cloud.google.com/bigquery/docs/reference/rest/v2/tables/setIamPolicy#body.request_body.FIELDS.update_mask

                Incompatible with ``fields``.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
            fields (Sequence[str]):
                Which properties to set on the policy. See:
                https://cloud.google.com/bigquery/docs/reference/rest/v2/tables/setIamPolicy#body.request_body.FIELDS.update_mask

                Incompatible with ``updateMask``.

        Returns:
            google.api_core.iam.Policy:
                The updated access control policy.
        """
        if updateMask is not None and not fields:
            update_mask = updateMask
        elif updateMask is not None and fields:
            raise ValueError("Cannot set both fields and updateMask")
        elif fields:
            update_mask = ",".join(fields)
        else:
            update_mask = None

        table = _table_arg_to_table_ref(table, default_project=self.project)

        if not isinstance(policy, (Policy)):
            raise TypeError("policy must be a Policy")

        body = {"policy": policy.to_api_repr()}

        if update_mask is not None:
            body["updateMask"] = update_mask

        path = "{}:setIamPolicy".format(table.path)
        span_attributes = {"path": path}

        response = self._call_api(
            retry,
            span_name="BigQuery.setIamPolicy",
            span_attributes=span_attributes,
            method="POST",
            path=path,
            data=body,
            timeout=timeout,
        )

        return Policy.from_api_repr(response)

    def test_iam_permissions(
        self,
        table: Union[Table, TableReference, TableListItem, str],
        permissions: Sequence[str],
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
    ) -> Dict[str, Any]:
        table = _table_arg_to_table_ref(table, default_project=self.project)

        body = {"permissions": permissions}

        path = "{}:testIamPermissions".format(table.path)
        span_attributes = {"path": path}
        response = self._call_api(
            retry,
            span_name="BigQuery.testIamPermissions",
            span_attributes=span_attributes,
            method="POST",
            path=path,
            data=body,
            timeout=timeout,
        )

        return response

    def get_model(
        self,
        model_ref: Union[ModelReference, str],
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
    ) -> Model:
        """[Beta] Fetch the model referenced by ``model_ref``.

         Args:
            model_ref (Union[ \
                google.cloud.bigquery.model.ModelReference, \
                str, \
            ]):
                A reference to the model to fetch from the BigQuery API.
                If a string is passed in, this method attempts to create a
                model reference from a string using
                :func:`google.cloud.bigquery.model.ModelReference.from_string`.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

         Returns:
            google.cloud.bigquery.model.Model: A ``Model`` instance.
        """
        if isinstance(model_ref, str):
            model_ref = ModelReference.from_string(
                model_ref, default_project=self.project
            )
        path = model_ref.path
        span_attributes = {"path": path}

        api_response = self._call_api(
            retry,
            span_name="BigQuery.getModel",
            span_attributes=span_attributes,
            method="GET",
            path=path,
            timeout=timeout,
        )
        return Model.from_api_repr(api_response)

    def get_routine(
        self,
        routine_ref: Union[Routine, RoutineReference, str],
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
    ) -> Routine:
        """[Beta] Get the routine referenced by ``routine_ref``.

         Args:
            routine_ref (Union[ \
                google.cloud.bigquery.routine.Routine, \
                google.cloud.bigquery.routine.RoutineReference, \
                str, \
            ]):
                A reference to the routine to fetch from the BigQuery API. If
                a string is passed in, this method attempts to create a
                reference from a string using
                :func:`google.cloud.bigquery.routine.RoutineReference.from_string`.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the API call.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

         Returns:
            google.cloud.bigquery.routine.Routine:
                A ``Routine`` instance.
        """
        if isinstance(routine_ref, str):
            routine_ref = RoutineReference.from_string(
                routine_ref, default_project=self.project
            )
        path = routine_ref.path
        span_attributes = {"path": path}
        api_response = self._call_api(
            retry,
            span_name="BigQuery.getRoutine",
            span_attributes=span_attributes,
            method="GET",
            path=path,
            timeout=timeout,
        )
        return Routine.from_api_repr(api_response)

    def get_table(
        self,
        table: Union[Table, TableReference, TableListItem, str],
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
    ) -> Table:
        """Fetch the table referenced by ``table``.

        Args:
            table (Union[ \
                google.cloud.bigquery.table.Table, \
                google.cloud.bigquery.table.TableReference, \
                google.cloud.bigquery.table.TableListItem, \
                str, \
            ]):
                A reference to the table to fetch from the BigQuery API.
                If a string is passed in, this method attempts to create a
                table reference from a string using
                :func:`google.cloud.bigquery.table.TableReference.from_string`.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

        Returns:
            google.cloud.bigquery.table.Table:
                A ``Table`` instance.
        """
        table_ref = _table_arg_to_table_ref(table, default_project=self.project)
        path = table_ref.path
        span_attributes = {"path": path}
        api_response = self._call_api(
            retry,
            span_name="BigQuery.getTable",
            span_attributes=span_attributes,
            method="GET",
            path=path,
            timeout=timeout,
        )
        return Table.from_api_repr(api_response)

    def update_dataset(
        self,
        dataset: Dataset,
        fields: Sequence[str],
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
        update_mode: Optional[UpdateMode] = None,
    ) -> Dataset:
        """Change some fields of a dataset.

        Use ``fields`` to specify which fields to update. At least one field
        must be provided. If a field is listed in ``fields`` and is ``None`` in
        ``dataset``, it will be deleted.

        For example, to update the default expiration times, specify
        both properties in the ``fields`` argument:

        .. code-block:: python

            bigquery_client.update_dataset(
                dataset,
                [
                    "default_partition_expiration_ms",
                    "default_table_expiration_ms",
                ]
            )

        If ``dataset.etag`` is not ``None``, the update will only
        succeed if the dataset on the server has the same ETag. Thus
        reading a dataset with ``get_dataset``, changing its fields,
        and then passing it to ``update_dataset`` will ensure that the changes
        will only be saved if no modifications to the dataset occurred
        since the read.

        Args:
            dataset (google.cloud.bigquery.dataset.Dataset):
                The dataset to update.
            fields (Sequence[str]):
                The properties of ``dataset`` to change. These are strings
                corresponding to the properties of
                :class:`~google.cloud.bigquery.dataset.Dataset`.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
            update_mode (Optional[google.cloud.bigquery.enums.UpdateMode]):
                Specifies the kind of information to update in a dataset.
                By default, dataset metadata (e.g. friendlyName, description,
                labels, etc) and ACL information are updated. This argument can
                take on the following possible enum values.

                * :attr:`~google.cloud.bigquery.enums.UPDATE_MODE_UNSPECIFIED`:
                    The default value. Behavior defaults to UPDATE_FULL.
                * :attr:`~google.cloud.bigquery.enums.UpdateMode.UPDATE_METADATA`:
                    Includes metadata information for the dataset, such as friendlyName, description, labels, etc.
                * :attr:`~google.cloud.bigquery.enums.UpdateMode.UPDATE_ACL`:
                    Includes ACL information for the dataset, which defines dataset access for one or more entities.
                * :attr:`~google.cloud.bigquery.enums.UpdateMode.UPDATE_FULL`:
                    Includes both dataset metadata and ACL information.

        Returns:
            google.cloud.bigquery.dataset.Dataset:
                The modified ``Dataset`` instance.
        """
        partial = dataset._build_resource(fields)
        if dataset.etag is not None:
            headers: Optional[Dict[str, str]] = {"If-Match": dataset.etag}
        else:
            headers = None
        path = dataset.path
        span_attributes = {"path": path, "fields": fields}

        if update_mode:
            query_params = {"updateMode": update_mode.value}
        else:
            query_params = {}

        api_response = self._call_api(
            retry,
            span_name="BigQuery.updateDataset",
            span_attributes=span_attributes,
            method="PATCH",
            path=path,
            data=partial,
            headers=headers,
            timeout=timeout,
            query_params=query_params,
        )
        return Dataset.from_api_repr(api_response)

    def update_model(
        self,
        model: Model,
        fields: Sequence[str],
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
    ) -> Model:
        """[Beta] Change some fields of a model.

        Use ``fields`` to specify which fields to update. At least one field
        must be provided. If a field is listed in ``fields`` and is ``None``
        in ``model``, the field value will be deleted.

        For example, to update the descriptive properties of the model,
        specify them in the ``fields`` argument:

        .. code-block:: python

            bigquery_client.update_model(
                model, ["description", "friendly_name"]
            )

        If ``model.etag`` is not ``None``, the update will only succeed if
        the model on the server has the same ETag. Thus reading a model with
        ``get_model``, changing its fields, and then passing it to
        ``update_model`` will ensure that the changes will only be saved if
        no modifications to the model occurred since the read.

        Args:
            model (google.cloud.bigquery.model.Model): The model to update.
            fields (Sequence[str]):
                The properties of ``model`` to change. These are strings
                corresponding to the properties of
                :class:`~google.cloud.bigquery.model.Model`.
            retry (Optional[google.api_core.retry.Retry]):
                A description of how to retry the API call.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

        Returns:
            google.cloud.bigquery.model.Model:
                The model resource returned from the API call.
        """
        partial = model._build_resource(fields)
        if model.etag:
            headers: Optional[Dict[str, str]] = {"If-Match": model.etag}
        else:
            headers = None
        path = model.path
        span_attributes = {"path": path, "fields": fields}

        api_response = self._call_api(
            retry,
            span_name="BigQuery.updateModel",
            span_attributes=span_attributes,
            method="PATCH",
            path=path,
            data=partial,
            headers=headers,
            timeout=timeout,
        )
        return Model.from_api_repr(api_response)

    def update_routine(
        self,
        routine: Routine,
        fields: Sequence[str],
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
    ) -> Routine:
        """[Beta] Change some fields of a routine.

        Use ``fields`` to specify which fields to update. At least one field
        must be provided. If a field is listed in ``fields`` and is ``None``
        in ``routine``, the field value will be deleted.

        For example, to update the description property of the routine,
        specify it in the ``fields`` argument:

        .. code-block:: python

            bigquery_client.update_routine(
                routine, ["description"]
            )

        .. warning::
           During beta, partial updates are not supported. You must provide
           all fields in the resource.

        If :attr:`~google.cloud.bigquery.routine.Routine.etag` is not
        ``None``, the update will only succeed if the resource on the server
        has the same ETag. Thus reading a routine with
        :func:`~google.cloud.bigquery.client.Client.get_routine`, changing
        its fields, and then passing it to this method will ensure that the
        changes will only be saved if no modifications to the resource
        occurred since the read.

        Args:
            routine (google.cloud.bigquery.routine.Routine):
                The routine to update.
            fields (Sequence[str]):
                The fields of ``routine`` to change, spelled as the
                :class:`~google.cloud.bigquery.routine.Routine` properties.
            retry (Optional[google.api_core.retry.Retry]):
                A description of how to retry the API call.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

        Returns:
            google.cloud.bigquery.routine.Routine:
                The routine resource returned from the API call.
        """
        partial = routine._build_resource(fields)
        if routine.etag:
            headers: Optional[Dict[str, str]] = {"If-Match": routine.etag}
        else:
            headers = None

        # TODO: remove when routines update supports partial requests.
        partial["routineReference"] = routine.reference.to_api_repr()

        path = routine.path
        span_attributes = {"path": path, "fields": fields}

        api_response = self._call_api(
            retry,
            span_name="BigQuery.updateRoutine",
            span_attributes=span_attributes,
            method="PUT",
            path=path,
            data=partial,
            headers=headers,
            timeout=timeout,
        )
        return Routine.from_api_repr(api_response)

    def update_table(
        self,
        table: Table,
        fields: Sequence[str],
        autodetect_schema: bool = False,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
    ) -> Table:
        """Change some fields of a table.

        Use ``fields`` to specify which fields to update. At least one field
        must be provided. If a field is listed in ``fields`` and is ``None``
        in ``table``, the field value will be deleted.

        For example, to update the descriptive properties of the table,
        specify them in the ``fields`` argument:

        .. code-block:: python

            bigquery_client.update_table(
                table,
                ["description", "friendly_name"]
            )

        If ``table.etag`` is not ``None``, the update will only succeed if
        the table on the server has the same ETag. Thus reading a table with
        ``get_table``, changing its fields, and then passing it to
        ``update_table`` will ensure that the changes will only be saved if
        no modifications to the table occurred since the read.

        Args:
            table (google.cloud.bigquery.table.Table): The table to update.
            fields (Sequence[str]):
                The fields of ``table`` to change, spelled as the
                :class:`~google.cloud.bigquery.table.Table` properties.
            autodetect_schema (bool):
                Specifies if the schema of the table should be autodetected when
                updating the table from the underlying source. Only applicable
                for external tables.
            retry (Optional[google.api_core.retry.Retry]):
                A description of how to retry the API call.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

        Returns:
            google.cloud.bigquery.table.Table:
                The table resource returned from the API call.
        """
        partial = table._build_resource(fields)
        if table.etag is not None:
            headers: Optional[Dict[str, str]] = {"If-Match": table.etag}
        else:
            headers = None

        path = table.path
        span_attributes = {"path": path, "fields": fields}

        if autodetect_schema:
            query_params = {"autodetect_schema": True}
        else:
            query_params = {}

        api_response = self._call_api(
            retry,
            span_name="BigQuery.updateTable",
            span_attributes=span_attributes,
            method="PATCH",
            path=path,
            query_params=query_params,
            data=partial,
            headers=headers,
            timeout=timeout,
        )
        return Table.from_api_repr(api_response)

    def list_models(
        self,
        dataset: Union[Dataset, DatasetReference, DatasetListItem, str],
        max_results: Optional[int] = None,
        page_token: Optional[str] = None,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
        page_size: Optional[int] = None,
    ) -> page_iterator.Iterator:
        """[Beta] List models in the dataset.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/models/list

        Args:
            dataset (Union[ \
                google.cloud.bigquery.dataset.Dataset, \
                google.cloud.bigquery.dataset.DatasetReference, \
                google.cloud.bigquery.dataset.DatasetListItem, \
                str, \
            ]):
                A reference to the dataset whose models to list from the
                BigQuery API. If a string is passed in, this method attempts
                to create a dataset reference from a string using
                :func:`google.cloud.bigquery.dataset.DatasetReference.from_string`.
            max_results (Optional[int]):
                Maximum number of models to return. Defaults to a
                value set by the API.
            page_token (Optional[str]):
                Token representing a cursor into the models. If not passed,
                the API will return the first page of models. The token marks
                the beginning of the iterator to be returned and the value of
                the ``page_token`` can be accessed at ``next_page_token`` of the
                :class:`~google.api_core.page_iterator.HTTPIterator`.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
            page_size (Optional[int]):
                Maximum number of models to return per page.
                Defaults to a value set by the API.

         Returns:
            google.api_core.page_iterator.Iterator:
                Iterator of
                :class:`~google.cloud.bigquery.model.Model` contained
                within the requested dataset.
        """
        dataset = self._dataset_from_arg(dataset)

        path = "%s/models" % dataset.path
        span_attributes = {"path": path}

        def api_request(*args, **kwargs):
            return self._call_api(
                retry,
                span_name="BigQuery.listModels",
                span_attributes=span_attributes,
                *args,
                timeout=timeout,
                **kwargs,
            )

        result = page_iterator.HTTPIterator(
            client=self,
            api_request=api_request,
            path=path,
            item_to_value=_item_to_model,
            items_key="models",
            page_token=page_token,
            max_results=max_results,
            page_size=page_size,
        )
        result.dataset = dataset  # type: ignore
        return result

    def list_routines(
        self,
        dataset: Union[Dataset, DatasetReference, DatasetListItem, str],
        max_results: Optional[int] = None,
        page_token: Optional[str] = None,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
        page_size: Optional[int] = None,
    ) -> page_iterator.Iterator:
        """[Beta] List routines in the dataset.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/routines/list

        Args:
            dataset (Union[ \
                google.cloud.bigquery.dataset.Dataset, \
                google.cloud.bigquery.dataset.DatasetReference, \
                google.cloud.bigquery.dataset.DatasetListItem, \
                str, \
            ]):
                A reference to the dataset whose routines to list from the
                BigQuery API. If a string is passed in, this method attempts
                to create a dataset reference from a string using
                :func:`google.cloud.bigquery.dataset.DatasetReference.from_string`.
            max_results (Optional[int]):
                Maximum number of routines to return. Defaults
                to a value set by the API.
            page_token (Optional[str]):
                Token representing a cursor into the routines. If not passed,
                the API will return the first page of routines. The token marks
                the beginning of the iterator to be returned and the value of the
                ``page_token`` can be accessed at ``next_page_token`` of the
                :class:`~google.api_core.page_iterator.HTTPIterator`.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
            page_size (Optional[int]):
                Maximum number of routines to return per page.
                Defaults to a value set by the API.

         Returns:
            google.api_core.page_iterator.Iterator:
                Iterator of all
                :class:`~google.cloud.bigquery.routine.Routine`s contained
                within the requested dataset, limited by ``max_results``.
        """
        dataset = self._dataset_from_arg(dataset)
        path = "{}/routines".format(dataset.path)

        span_attributes = {"path": path}

        def api_request(*args, **kwargs):
            return self._call_api(
                retry,
                span_name="BigQuery.listRoutines",
                span_attributes=span_attributes,
                *args,
                timeout=timeout,
                **kwargs,
            )

        result = page_iterator.HTTPIterator(
            client=self,
            api_request=api_request,
            path=path,
            item_to_value=_item_to_routine,
            items_key="routines",
            page_token=page_token,
            max_results=max_results,
            page_size=page_size,
        )
        result.dataset = dataset  # type: ignore
        return result

    def list_tables(
        self,
        dataset: Union[Dataset, DatasetReference, DatasetListItem, str],
        max_results: Optional[int] = None,
        page_token: Optional[str] = None,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
        page_size: Optional[int] = None,
    ) -> page_iterator.Iterator:
        """List tables in the dataset.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables/list

        Args:
            dataset (Union[ \
                google.cloud.bigquery.dataset.Dataset, \
                google.cloud.bigquery.dataset.DatasetReference, \
                google.cloud.bigquery.dataset.DatasetListItem, \
                str, \
            ]):
                A reference to the dataset whose tables to list from the
                BigQuery API. If a string is passed in, this method attempts
                to create a dataset reference from a string using
                :func:`google.cloud.bigquery.dataset.DatasetReference.from_string`.
            max_results (Optional[int]):
                Maximum number of tables to return. Defaults
                to a value set by the API.
            page_token (Optional[str]):
                Token representing a cursor into the tables. If not passed,
                the API will return the first page of tables. The token marks
                the beginning of the iterator to be returned and the value of
                the ``page_token`` can be accessed at ``next_page_token`` of the
                :class:`~google.api_core.page_iterator.HTTPIterator`.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
            page_size (Optional[int]):
                Maximum number of tables to return per page.
                Defaults to a value set by the API.

        Returns:
            google.api_core.page_iterator.Iterator:
                Iterator of
                :class:`~google.cloud.bigquery.table.TableListItem` contained
                within the requested dataset.
        """
        dataset = self._dataset_from_arg(dataset)
        path = "%s/tables" % dataset.path
        span_attributes = {"path": path}

        def api_request(*args, **kwargs):
            return self._call_api(
                retry,
                span_name="BigQuery.listTables",
                span_attributes=span_attributes,
                *args,
                timeout=timeout,
                **kwargs,
            )

        result = page_iterator.HTTPIterator(
            client=self,
            api_request=api_request,
            path=path,
            item_to_value=_item_to_table,
            items_key="tables",
            page_token=page_token,
            max_results=max_results,
            page_size=page_size,
        )
        result.dataset = dataset  # type: ignore
        return result

    def delete_dataset(
        self,
        dataset: Union[Dataset, DatasetReference, DatasetListItem, str],
        delete_contents: bool = False,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
        not_found_ok: bool = False,
    ) -> None:
        """Delete a dataset.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/datasets/delete

        Args:
            dataset (Union[ \
                google.cloud.bigquery.dataset.Dataset, \
                google.cloud.bigquery.dataset.DatasetReference, \
                google.cloud.bigquery.dataset.DatasetListItem, \
                str, \
            ]):
                A reference to the dataset to delete. If a string is passed
                in, this method attempts to create a dataset reference from a
                string using
                :func:`google.cloud.bigquery.dataset.DatasetReference.from_string`.
            delete_contents (Optional[bool]):
                If True, delete all the tables in the dataset. If False and
                the dataset contains tables, the request will fail.
                Default is False.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
            not_found_ok (Optional[bool]):
                Defaults to ``False``. If ``True``, ignore "not found" errors
                when deleting the dataset.
        """
        dataset = self._dataset_from_arg(dataset)
        params = {}
        path = dataset.path
        if delete_contents:
            params["deleteContents"] = "true"
            span_attributes = {"path": path, "deleteContents": delete_contents}
        else:
            span_attributes = {"path": path}

        try:
            self._call_api(
                retry,
                span_name="BigQuery.deleteDataset",
                span_attributes=span_attributes,
                method="DELETE",
                path=path,
                query_params=params,
                timeout=timeout,
            )
        except core_exceptions.NotFound:
            if not not_found_ok:
                raise

    def delete_model(
        self,
        model: Union[Model, ModelReference, str],
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
        not_found_ok: bool = False,
    ) -> None:
        """[Beta] Delete a model

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/models/delete

        Args:
            model (Union[ \
                google.cloud.bigquery.model.Model, \
                google.cloud.bigquery.model.ModelReference, \
                str, \
            ]):
                A reference to the model to delete. If a string is passed in,
                this method attempts to create a model reference from a
                string using
                :func:`google.cloud.bigquery.model.ModelReference.from_string`.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
            not_found_ok (Optional[bool]):
                Defaults to ``False``. If ``True``, ignore "not found" errors
                when deleting the model.
        """
        if isinstance(model, str):
            model = ModelReference.from_string(model, default_project=self.project)

        if not isinstance(model, (Model, ModelReference)):
            raise TypeError("model must be a Model or a ModelReference")

        path = model.path
        try:
            span_attributes = {"path": path}
            self._call_api(
                retry,
                span_name="BigQuery.deleteModel",
                span_attributes=span_attributes,
                method="DELETE",
                path=path,
                timeout=timeout,
            )
        except core_exceptions.NotFound:
            if not not_found_ok:
                raise

    def delete_job_metadata(
        self,
        job_id: Union[str, LoadJob, CopyJob, ExtractJob, QueryJob],
        project: Optional[str] = None,
        location: Optional[str] = None,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
        not_found_ok: bool = False,
    ):
        """[Beta] Delete job metadata from job history.

        Note: This does not stop a running job. Use
        :func:`~google.cloud.bigquery.client.Client.cancel_job` instead.

        Args:
            job_id (Union[ \
                str, \
                LoadJob, \
                CopyJob, \
                ExtractJob, \
                QueryJob \
            ]): Job or job identifier.
            project (Optional[str]):
                ID of the project which owns the job (defaults to the client's project).
            location (Optional[str]):
                Location where the job was run. Ignored if ``job_id`` is a job
                object.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
            not_found_ok (Optional[bool]):
                Defaults to ``False``. If ``True``, ignore "not found" errors
                when deleting the job.
        """
        extra_params = {}

        project, location, job_id = _extract_job_reference(
            job_id, project=project, location=location
        )

        if project is None:
            project = self.project

        if location is None:
            location = self.location

        # Location is always required for jobs.delete()
        extra_params["location"] = location

        path = f"/projects/{project}/jobs/{job_id}/delete"

        span_attributes = {"path": path, "job_id": job_id, "location": location}

        try:
            self._call_api(
                retry,
                span_name="BigQuery.deleteJob",
                span_attributes=span_attributes,
                method="DELETE",
                path=path,
                query_params=extra_params,
                timeout=timeout,
            )
        except google.api_core.exceptions.NotFound:
            if not not_found_ok:
                raise

    def delete_routine(
        self,
        routine: Union[Routine, RoutineReference, str],
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
        not_found_ok: bool = False,
    ) -> None:
        """[Beta] Delete a routine.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/routines/delete

        Args:
            routine (Union[ \
                google.cloud.bigquery.routine.Routine, \
                google.cloud.bigquery.routine.RoutineReference, \
                str, \
            ]):
                A reference to the routine to delete. If a string is passed
                in, this method attempts to create a routine reference from a
                string using
                :func:`google.cloud.bigquery.routine.RoutineReference.from_string`.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
            not_found_ok (Optional[bool]):
                Defaults to ``False``. If ``True``, ignore "not found" errors
                when deleting the routine.
        """
        if isinstance(routine, str):
            routine = RoutineReference.from_string(
                routine, default_project=self.project
            )
        path = routine.path

        if not isinstance(routine, (Routine, RoutineReference)):
            raise TypeError("routine must be a Routine or a RoutineReference")

        try:
            span_attributes = {"path": path}
            self._call_api(
                retry,
                span_name="BigQuery.deleteRoutine",
                span_attributes=span_attributes,
                method="DELETE",
                path=path,
                timeout=timeout,
            )
        except core_exceptions.NotFound:
            if not not_found_ok:
                raise

    def delete_table(
        self,
        table: Union[Table, TableReference, TableListItem, str],
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
        not_found_ok: bool = False,
    ) -> None:
        """Delete a table

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tables/delete

        Args:
            table (Union[ \
                google.cloud.bigquery.table.Table, \
                google.cloud.bigquery.table.TableReference, \
                google.cloud.bigquery.table.TableListItem, \
                str, \
            ]):
                A reference to the table to delete. If a string is passed in,
                this method attempts to create a table reference from a
                string using
                :func:`google.cloud.bigquery.table.TableReference.from_string`.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
            not_found_ok (Optional[bool]):
                Defaults to ``False``. If ``True``, ignore "not found" errors
                when deleting the table.
        """
        table = _table_arg_to_table_ref(table, default_project=self.project)
        if not isinstance(table, TableReference):
            raise TypeError("Unable to get TableReference for table '{}'".format(table))

        try:
            path = table.path
            span_attributes = {"path": path}
            self._call_api(
                retry,
                span_name="BigQuery.deleteTable",
                span_attributes=span_attributes,
                method="DELETE",
                path=path,
                timeout=timeout,
            )
        except core_exceptions.NotFound:
            if not not_found_ok:
                raise

    def _get_query_results(
        self,
        job_id: str,
        retry: retries.Retry,
        project: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        location: Optional[str] = None,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
        page_size: int = 0,
        start_index: Optional[int] = None,
    ) -> _QueryResults:
        """Get the query results object for a query job.

        Args:
            job_id (str): Name of the query job.
            retry (google.api_core.retry.Retry):
                How to retry the RPC.
            project (Optional[str]):
                Project ID for the query job (defaults to the project of the client).
            timeout_ms (Optional[int]):
                Number of milliseconds the the API call should wait for the query
                to complete before the request times out.
            location (Optional[str]): Location of the query job.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``. If set, this connection timeout may be
                increased to a minimum value. This prevents retries on what
                would otherwise be a successful response.
            page_size (Optional[int]):
                Maximum number of rows in a single response. See maxResults in
                the jobs.getQueryResults REST API.
            start_index (Optional[int]):
                Zero-based index of the starting row. See startIndex in the
                jobs.getQueryResults REST API.

        Returns:
            google.cloud.bigquery.query._QueryResults:
                A new ``_QueryResults`` instance.
        """

        extra_params: Dict[str, Any] = {"maxResults": page_size}

        if timeout is not None:
            if not isinstance(timeout, (int, float)):
                timeout = _MIN_GET_QUERY_RESULTS_TIMEOUT
            else:
                timeout = max(timeout, _MIN_GET_QUERY_RESULTS_TIMEOUT)

        if page_size > 0:
            extra_params["formatOptions.useInt64Timestamp"] = True

        if project is None:
            project = self.project

        if timeout_ms is not None:
            extra_params["timeoutMs"] = timeout_ms

        if location is None:
            location = self.location

        if location is not None:
            extra_params["location"] = location

        if start_index is not None:
            extra_params["startIndex"] = start_index

        path = "/projects/{}/queries/{}".format(project, job_id)

        # This call is typically made in a polling loop that checks whether the
        # job is complete (from QueryJob.done(), called ultimately from
        # QueryJob.result()). So we don't need to poll here.
        span_attributes = {"path": path}
        resource = self._call_api(
            retry,
            span_name="BigQuery.getQueryResults",
            span_attributes=span_attributes,
            method="GET",
            path=path,
            query_params=extra_params,
            timeout=timeout,
        )
        return _QueryResults.from_api_repr(resource)

    def job_from_resource(
        self, resource: dict
    ) -> Union[job.CopyJob, job.ExtractJob, job.LoadJob, job.QueryJob, job.UnknownJob]:
        """Detect correct job type from resource and instantiate.

        Args:
            resource (Dict): one job resource from API response

        Returns:
            Union[job.CopyJob, job.ExtractJob, job.LoadJob, job.QueryJob, job.UnknownJob]:
                The job instance, constructed via the resource.
        """
        config = resource.get("configuration", {})
        if "load" in config:
            return job.LoadJob.from_api_repr(resource, self)
        elif "copy" in config:
            return job.CopyJob.from_api_repr(resource, self)
        elif "extract" in config:
            return job.ExtractJob.from_api_repr(resource, self)
        elif "query" in config:
            return job.QueryJob.from_api_repr(resource, self)
        return job.UnknownJob.from_api_repr(resource, self)

    def create_job(
        self,
        job_config: dict,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
    ) -> Union[job.LoadJob, job.CopyJob, job.ExtractJob, job.QueryJob]:
        """Create a new job.

        Args:
            job_config (dict): configuration job representation returned from the API.
            retry (Optional[google.api_core.retry.Retry]): How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

        Returns:
            Union[ \
                google.cloud.bigquery.job.LoadJob, \
                google.cloud.bigquery.job.CopyJob, \
                google.cloud.bigquery.job.ExtractJob, \
                google.cloud.bigquery.job.QueryJob \
            ]:
                A new job instance.
        """

        if "load" in job_config:
            load_job_config = google.cloud.bigquery.job.LoadJobConfig.from_api_repr(
                job_config
            )
            destination = _get_sub_prop(job_config, ["load", "destinationTable"])
            source_uris = _get_sub_prop(job_config, ["load", "sourceUris"])
            destination = TableReference.from_api_repr(destination)
            return self.load_table_from_uri(
                source_uris,
                destination,
                job_config=typing.cast(LoadJobConfig, load_job_config),
                retry=retry,
                timeout=timeout,
            )
        elif "copy" in job_config:
            copy_job_config = google.cloud.bigquery.job.CopyJobConfig.from_api_repr(
                job_config
            )
            destination = _get_sub_prop(job_config, ["copy", "destinationTable"])
            destination = TableReference.from_api_repr(destination)
            return self.copy_table(
                [],  # Source table(s) already in job_config resource.
                destination,
                job_config=typing.cast(CopyJobConfig, copy_job_config),
                retry=retry,
                timeout=timeout,
            )
        elif "extract" in job_config:
            extract_job_config = (
                google.cloud.bigquery.job.ExtractJobConfig.from_api_repr(job_config)
            )
            source = _get_sub_prop(job_config, ["extract", "sourceTable"])
            if source:
                source_type = "Table"
                source = TableReference.from_api_repr(source)
            else:
                source = _get_sub_prop(job_config, ["extract", "sourceModel"])
                source_type = "Model"
                source = ModelReference.from_api_repr(source)
            destination_uris = _get_sub_prop(job_config, ["extract", "destinationUris"])
            return self.extract_table(
                source,
                destination_uris,
                job_config=typing.cast(ExtractJobConfig, extract_job_config),
                retry=retry,
                timeout=timeout,
                source_type=source_type,
            )
        elif "query" in job_config:
            query_job_config = google.cloud.bigquery.job.QueryJobConfig.from_api_repr(
                job_config
            )
            query = _get_sub_prop(job_config, ["query", "query"])
            return self.query(
                query,
                job_config=typing.cast(QueryJobConfig, query_job_config),
                retry=retry,
                timeout=timeout,
            )
        else:
            raise TypeError("Invalid job configuration received.")

    def get_job(
        self,
        job_id: Union[str, job.LoadJob, job.CopyJob, job.ExtractJob, job.QueryJob],
        project: Optional[str] = None,
        location: Optional[str] = None,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_GET_JOB_TIMEOUT,
    ) -> Union[job.LoadJob, job.CopyJob, job.ExtractJob, job.QueryJob, job.UnknownJob]:
        """Fetch a job for the project associated with this client.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/get

        Args:
            job_id (Union[ \
                str, \
                job.LoadJob, \
                job.CopyJob, \
                job.ExtractJob, \
                job.QueryJob \
            ]):
                Job identifier.
            project (Optional[str]):
                ID of the project which owns the job (defaults to the client's project).
            location (Optional[str]):
                Location where the job was run. Ignored if ``job_id`` is a job
                object.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

        Returns:
            Union[job.LoadJob, job.CopyJob, job.ExtractJob, job.QueryJob, job.UnknownJob]:
                Job instance, based on the resource returned by the API.
        """
        extra_params = {"projection": "full"}

        project, location, job_id = _extract_job_reference(
            job_id, project=project, location=location
        )

        if project is None:
            project = self.project

        if location is None:
            location = self.location

        if location is not None:
            extra_params["location"] = location

        path = "/projects/{}/jobs/{}".format(project, job_id)

        span_attributes = {"path": path, "job_id": job_id, "location": location}

        resource = self._call_api(
            retry,
            span_name="BigQuery.getJob",
            span_attributes=span_attributes,
            method="GET",
            path=path,
            query_params=extra_params,
            timeout=timeout,
        )

        return self.job_from_resource(resource)

    def cancel_job(
        self,
        job_id: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
    ) -> Union[job.LoadJob, job.CopyJob, job.ExtractJob, job.QueryJob]:
        """Attempt to cancel a job from a job ID.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/cancel

        Args:
            job_id (Union[ \
                str, \
                google.cloud.bigquery.job.LoadJob, \
                google.cloud.bigquery.job.CopyJob, \
                google.cloud.bigquery.job.ExtractJob, \
                google.cloud.bigquery.job.QueryJob \
            ]): Job identifier.
            project (Optional[str]):
                ID of the project which owns the job (defaults to the client's project).
            location (Optional[str]):
                Location where the job was run. Ignored if ``job_id`` is a job
                object.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

        Returns:
            Union[ \
                google.cloud.bigquery.job.LoadJob, \
                google.cloud.bigquery.job.CopyJob, \
                google.cloud.bigquery.job.ExtractJob, \
                google.cloud.bigquery.job.QueryJob, \
            ]:
                Job instance, based on the resource returned by the API.
        """
        extra_params = {"projection": "full"}

        project, location, job_id = _extract_job_reference(
            job_id, project=project, location=location
        )

        if project is None:
            project = self.project

        if location is None:
            location = self.location

        if location is not None:
            extra_params["location"] = location

        path = "/projects/{}/jobs/{}/cancel".format(project, job_id)

        span_attributes = {"path": path, "job_id": job_id, "location": location}

        resource = self._call_api(
            retry,
            span_name="BigQuery.cancelJob",
            span_attributes=span_attributes,
            method="POST",
            path=path,
            query_params=extra_params,
            timeout=timeout,
        )

        job_instance = self.job_from_resource(resource["job"])  # never an UnknownJob

        return typing.cast(
            Union[job.LoadJob, job.CopyJob, job.ExtractJob, job.QueryJob],
            job_instance,
        )

    def list_jobs(
        self,
        project: Optional[str] = None,
        parent_job: Optional[Union[QueryJob, str]] = None,
        max_results: Optional[int] = None,
        page_token: Optional[str] = None,
        all_users: Optional[bool] = None,
        state_filter: Optional[str] = None,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
        min_creation_time: Optional[datetime.datetime] = None,
        max_creation_time: Optional[datetime.datetime] = None,
        page_size: Optional[int] = None,
    ) -> page_iterator.Iterator:
        """List jobs for the project associated with this client.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/list

        Args:
            project (Optional[str]):
                Project ID to use for retreiving datasets. Defaults
                to the client's project.
            parent_job (Optional[Union[ \
                google.cloud.bigquery.job._AsyncJob, \
                str, \
            ]]):
                If set, retrieve only child jobs of the specified parent.
            max_results (Optional[int]):
                Maximum number of jobs to return.
            page_token (Optional[str]):
                Opaque marker for the next "page" of jobs. If not
                passed, the API will return the first page of jobs. The token
                marks the beginning of the iterator to be returned and the
                value of the ``page_token`` can be accessed at
                ``next_page_token`` of
                :class:`~google.api_core.page_iterator.HTTPIterator`.
            all_users (Optional[bool]):
                If true, include jobs owned by all users in the project.
                Defaults to :data:`False`.
            state_filter (Optional[str]):
                If set, include only jobs matching the given state. One of:
                    * ``"done"``
                    * ``"pending"``
                    * ``"running"``
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
            min_creation_time (Optional[datetime.datetime]):
                Min value for job creation time. If set, only jobs created
                after or at this timestamp are returned. If the datetime has
                no time zone assumes UTC time.
            max_creation_time (Optional[datetime.datetime]):
                Max value for job creation time. If set, only jobs created
                before or at this timestamp are returned. If the datetime has
                no time zone assumes UTC time.
            page_size (Optional[int]):
                Maximum number of jobs to return per page.

        Returns:
            google.api_core.page_iterator.Iterator:
                Iterable of job instances.
        """
        if isinstance(parent_job, job._AsyncJob):
            parent_job = parent_job.job_id  # pytype: disable=attribute-error

        extra_params = {
            "allUsers": all_users,
            "stateFilter": state_filter,
            "minCreationTime": _str_or_none(
                google.cloud._helpers._millis_from_datetime(min_creation_time)
            ),
            "maxCreationTime": _str_or_none(
                google.cloud._helpers._millis_from_datetime(max_creation_time)
            ),
            "projection": "full",
            "parentJobId": parent_job,
        }

        extra_params = {
            param: value for param, value in extra_params.items() if value is not None
        }

        if project is None:
            project = self.project

        path = "/projects/%s/jobs" % (project,)

        span_attributes = {"path": path}

        def api_request(*args, **kwargs):
            return self._call_api(
                retry,
                span_name="BigQuery.listJobs",
                span_attributes=span_attributes,
                *args,
                timeout=timeout,
                **kwargs,
            )

        return page_iterator.HTTPIterator(
            client=self,
            api_request=api_request,
            path=path,
            item_to_value=_item_to_job,
            items_key="jobs",
            page_token=page_token,
            max_results=max_results,
            extra_params=extra_params,
            page_size=page_size,
        )

    def load_table_from_uri(
        self,
        source_uris: Union[str, Sequence[str]],
        destination: Union[Table, TableReference, TableListItem, str],
        job_id: Optional[str] = None,
        job_id_prefix: Optional[str] = None,
        location: Optional[str] = None,
        project: Optional[str] = None,
        job_config: Optional[LoadJobConfig] = None,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
    ) -> job.LoadJob:
        """Starts a job for loading data into a table from Cloud Storage.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#jobconfigurationload

        Args:
            source_uris (Union[str, Sequence[str]]):
                URIs of data files to be loaded; in format
                ``gs://<bucket_name>/<object_name_or_glob>``.
            destination (Union[ \
                google.cloud.bigquery.table.Table, \
                google.cloud.bigquery.table.TableReference, \
                google.cloud.bigquery.table.TableListItem, \
                str, \
            ]):
                Table into which data is to be loaded. If a string is passed
                in, this method attempts to create a table reference from a
                string using
                :func:`google.cloud.bigquery.table.TableReference.from_string`.
            job_id (Optional[str]): Name of the job.
            job_id_prefix (Optional[str]):
                The user-provided prefix for a randomly generated job ID.
                This parameter will be ignored if a ``job_id`` is also given.
            location (Optional[str]):
                Location where to run the job. Must match the location of the
                destination table.
            project (Optional[str]):
                Project ID of the project of where to run the job. Defaults
                to the client's project.
            job_config (Optional[google.cloud.bigquery.job.LoadJobConfig]):
                Extra configuration options for the job.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

        Returns:
            google.cloud.bigquery.job.LoadJob: A new load job.

        Raises:
            TypeError:
                If ``job_config`` is not an instance of
                :class:`~google.cloud.bigquery.job.LoadJobConfig` class.
        """
        job_id = _make_job_id(job_id, job_id_prefix)

        if project is None:
            project = self.project

        if location is None:
            location = self.location

        job_ref = job._JobReference(job_id, project=project, location=location)

        if isinstance(source_uris, str):
            source_uris = [source_uris]

        destination = _table_arg_to_table_ref(destination, default_project=self.project)

        if job_config is not None:
            _verify_job_config_type(job_config, LoadJobConfig)
        else:
            job_config = job.LoadJobConfig()

        new_job_config = job_config._fill_from_default(self._default_load_job_config)

        load_job = job.LoadJob(job_ref, source_uris, destination, self, new_job_config)
        load_job._begin(retry=retry, timeout=timeout)

        return load_job

    def load_table_from_file(
        self,
        file_obj: IO[bytes],
        destination: Union[Table, TableReference, TableListItem, str],
        rewind: bool = False,
        size: Optional[int] = None,
        num_retries: int = _DEFAULT_NUM_RETRIES,
        job_id: Optional[str] = None,
        job_id_prefix: Optional[str] = None,
        location: Optional[str] = None,
        project: Optional[str] = None,
        job_config: Optional[LoadJobConfig] = None,
        timeout: ResumableTimeoutType = DEFAULT_TIMEOUT,
    ) -> job.LoadJob:
        """Upload the contents of this table from a file-like object.

        Similar to :meth:`load_table_from_uri`, this method creates, starts and
        returns a :class:`~google.cloud.bigquery.job.LoadJob`.

        Args:
            file_obj (IO[bytes]):
                A file handle opened in binary mode for reading.
            destination (Union[Table, \
                TableReference, \
                TableListItem, \
                str \
            ]):
                Table into which data is to be loaded. If a string is passed
                in, this method attempts to create a table reference from a
                string using
                :func:`google.cloud.bigquery.table.TableReference.from_string`.
            rewind (Optional[bool]):
                If True, seek to the beginning of the file handle before
                reading the file. Defaults to False.
            size (Optional[int]):
                The number of bytes to read from the file handle. If size is
                ``None`` or large, resumable upload will be used. Otherwise,
                multipart upload will be used.
            num_retries (Optional[int]): Number of upload retries. Defaults to 6.
            job_id (Optional[str]): Name of the job.
            job_id_prefix (Optional[str]):
                The user-provided prefix for a randomly generated job ID.
                This parameter will be ignored if a ``job_id`` is also given.
            location (Optional[str]):
                Location where to run the job. Must match the location of the
                destination table.
            project (Optional[str]):
                Project ID of the project of where to run the job. Defaults
                to the client's project.
            job_config (Optional[LoadJobConfig]):
                Extra configuration options for the job.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``. Depending on the retry strategy, a request
                may be repeated several times using the same timeout each time.
                Defaults to None.

                Can also be passed as a tuple (connect_timeout, read_timeout).
                See :meth:`requests.Session.request` documentation for details.

        Returns:
            google.cloud.bigquery.job.LoadJob: A new load job.

        Raises:
            ValueError:
                If ``size`` is not passed in and can not be determined, or if
                the ``file_obj`` can be detected to be a file opened in text
                mode.

            TypeError:
                If ``job_config`` is not an instance of
                :class:`~google.cloud.bigquery.job.LoadJobConfig` class.
        """
        job_id = _make_job_id(job_id, job_id_prefix)

        if project is None:
            project = self.project

        if location is None:
            location = self.location

        destination = _table_arg_to_table_ref(destination, default_project=self.project)
        job_ref = job._JobReference(job_id, project=project, location=location)

        if job_config is not None:
            _verify_job_config_type(job_config, LoadJobConfig)
        else:
            job_config = job.LoadJobConfig()

        new_job_config = job_config._fill_from_default(self._default_load_job_config)

        load_job = job.LoadJob(job_ref, None, destination, self, new_job_config)
        job_resource = load_job.to_api_repr()

        if rewind:
            file_obj.seek(0, os.SEEK_SET)

        _check_mode(file_obj)

        try:
            if size is None or size >= _MAX_MULTIPART_SIZE:
                response = self._do_resumable_upload(
                    file_obj, job_resource, num_retries, timeout, project=project
                )
            else:
                response = self._do_multipart_upload(
                    file_obj, job_resource, size, num_retries, timeout, project=project
                )
        except resumable_media.InvalidResponse as exc:
            raise exceptions.from_http_response(exc.response)

        return typing.cast(LoadJob, self.job_from_resource(response.json()))

    def load_table_from_dataframe(
        self,
        dataframe: "pandas.DataFrame",  # type: ignore
        destination: Union[Table, TableReference, str],
        num_retries: int = _DEFAULT_NUM_RETRIES,
        job_id: Optional[str] = None,
        job_id_prefix: Optional[str] = None,
        location: Optional[str] = None,
        project: Optional[str] = None,
        job_config: Optional[LoadJobConfig] = None,
        parquet_compression: str = "snappy",
        timeout: ResumableTimeoutType = DEFAULT_TIMEOUT,
    ) -> job.LoadJob:
        """Upload the contents of a table from a pandas DataFrame.

        Similar to :meth:`load_table_from_uri`, this method creates, starts and
        returns a :class:`~google.cloud.bigquery.job.LoadJob`.

        .. note::

            REPEATED fields are NOT supported when using the CSV source format.
            They are supported when using the PARQUET source format, but
            due to the way they are encoded in the ``parquet`` file,
            a mismatch with the existing table schema can occur, so
            REPEATED fields are not properly supported when using ``pyarrow<4.0.0``
            using the parquet format.

            https://github.com/googleapis/python-bigquery/issues/19

        Args:
            dataframe (pandas.Dataframe):
                A :class:`~pandas.DataFrame` containing the data to load.
            destination (Union[ \
                Table, \
                TableReference, \
                str \
            ]):
                The destination table to use for loading the data. If it is an
                existing table, the schema of the :class:`~pandas.DataFrame`
                must match the schema of the destination table. If the table
                does not yet exist, the schema is inferred from the
                :class:`~pandas.DataFrame`.

                If a string is passed in, this method attempts to create a
                table reference from a string using
                :func:`google.cloud.bigquery.table.TableReference.from_string`.
            num_retries (Optional[int]): Number of upload retries. Defaults to 6.
            job_id (Optional[str]): Name of the job.
            job_id_prefix (Optional[str]):
                The user-provided prefix for a randomly generated
                job ID. This parameter will be ignored if a ``job_id`` is
                also given.
            location (Optional[str]):
                Location where to run the job. Must match the location of the
                destination table.
            project (Optional[str]):
                Project ID of the project of where to run the job. Defaults
                to the client's project.
            job_config (Optional[LoadJobConfig]):
                Extra configuration options for the job.

                To override the default pandas data type conversions, supply
                a value for
                :attr:`~google.cloud.bigquery.job.LoadJobConfig.schema` with
                column names matching those of the dataframe. The BigQuery
                schema is used to determine the correct data type conversion.
                Indexes are not loaded.

                By default, this method uses the parquet source format. To
                override this, supply a value for
                :attr:`~google.cloud.bigquery.job.LoadJobConfig.source_format`
                with the format name. Currently only
                :attr:`~google.cloud.bigquery.job.SourceFormat.CSV` and
                :attr:`~google.cloud.bigquery.job.SourceFormat.PARQUET` are
                supported.
            parquet_compression (Optional[str]):
                [Beta] The compression method to use if intermittently
                serializing ``dataframe`` to a parquet file.
                Defaults to "snappy".

                The argument is directly passed as the ``compression``
                argument to the underlying ``pyarrow.parquet.write_table()``
                method (the default value "snappy" gets converted to uppercase).
                https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html#pyarrow-parquet-write-table

                If the job config schema is missing, the argument is directly
                passed as the ``compression`` argument to the underlying
                ``DataFrame.to_parquet()`` method.
                https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_parquet.html#pandas.DataFrame.to_parquet
            timeout (Optional[flaot]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``. Depending on the retry strategy, a request may
                be repeated several times using the same timeout each time.
                Defaults to None.

                Can also be passed as a tuple (connect_timeout, read_timeout).
                See :meth:`requests.Session.request` documentation for details.

        Returns:
            google.cloud.bigquery.job.LoadJob: A new load job.

        Raises:
            ValueError:
                If a usable parquet engine cannot be found. This method
                requires :mod:`pyarrow` to be installed.
            TypeError:
                If ``job_config`` is not an instance of
                :class:`~google.cloud.bigquery.job.LoadJobConfig` class.
        """
        job_id = _make_job_id(job_id, job_id_prefix)

        if job_config is not None:
            _verify_job_config_type(job_config, LoadJobConfig)
        else:
            job_config = job.LoadJobConfig()

        new_job_config = job_config._fill_from_default(self._default_load_job_config)

        supported_formats = {job.SourceFormat.CSV, job.SourceFormat.PARQUET}
        if new_job_config.source_format is None:
            # default value
            new_job_config.source_format = job.SourceFormat.PARQUET

        if (
            new_job_config.source_format == job.SourceFormat.PARQUET
            and new_job_config.parquet_options is None
        ):
            parquet_options = ParquetOptions()
            # default value
            parquet_options.enable_list_inference = True
            new_job_config.parquet_options = parquet_options

        if new_job_config.source_format not in supported_formats:
            raise ValueError(
                "Got unexpected source_format: '{}'. Currently, only PARQUET and CSV are supported".format(
                    new_job_config.source_format
                )
            )

        if pyarrow is None and new_job_config.source_format == job.SourceFormat.PARQUET:
            # pyarrow is now the only supported parquet engine.
            raise ValueError("This method requires pyarrow to be installed")

        if location is None:
            location = self.location

        # If table schema is not provided, we try to fetch the existing table
        # schema, and check if dataframe schema is compatible with it - except
        # for WRITE_TRUNCATE jobs, the existing schema does not matter then.
        if (
            not new_job_config.schema
            and new_job_config.write_disposition != job.WriteDisposition.WRITE_TRUNCATE
        ):
            try:
                table = self.get_table(destination)
            except core_exceptions.NotFound:
                pass
            else:
                columns_and_indexes = frozenset(
                    name
                    for name, _ in _pandas_helpers.list_columns_and_indexes(dataframe)
                )
                new_job_config.schema = [
                    # Field description and policy tags are not needed to
                    # serialize a data frame.
                    SchemaField(
                        field.name,
                        field.field_type,
                        mode=field.mode,
                        fields=field.fields,
                    )
                    # schema fields not present in the dataframe are not needed
                    for field in table.schema
                    if field.name in columns_and_indexes
                ]

        new_job_config.schema = _pandas_helpers.dataframe_to_bq_schema(
            dataframe, new_job_config.schema
        )

        if not new_job_config.schema:
            # the schema could not be fully detected
            warnings.warn(
                "Schema could not be detected for all columns. Loading from a "
                "dataframe without a schema will be deprecated in the future, "
                "please provide a schema.",
                PendingDeprecationWarning,
                stacklevel=2,
            )

        tmpfd, tmppath = tempfile.mkstemp(
            suffix="_job_{}.{}".format(job_id[:8], new_job_config.source_format.lower())
        )
        os.close(tmpfd)

        try:
            if new_job_config.source_format == job.SourceFormat.PARQUET:
                if new_job_config.schema:
                    if parquet_compression == "snappy":  # adjust the default value
                        parquet_compression = parquet_compression.upper()

                    _pandas_helpers.dataframe_to_parquet(
                        dataframe,
                        new_job_config.schema,
                        tmppath,
                        parquet_compression=parquet_compression,
                        parquet_use_compliant_nested_type=True,
                    )
                else:
                    dataframe.to_parquet(
                        tmppath,
                        engine="pyarrow",
                        compression=parquet_compression,
                        **(
                            {"use_compliant_nested_type": True}
                            if _versions_helpers.PYARROW_VERSIONS.use_compliant_nested_type
                            else {}
                        ),
                    )

            else:
                dataframe.to_csv(
                    tmppath,
                    index=False,
                    header=False,
                    encoding="utf-8",
                    float_format="%.17g",
                    date_format="%Y-%m-%d %H:%M:%S.%f",
                )

            with open(tmppath, "rb") as tmpfile:
                file_size = os.path.getsize(tmppath)
                return self.load_table_from_file(
                    tmpfile,
                    destination,
                    num_retries=num_retries,
                    rewind=True,
                    size=file_size,
                    job_id=job_id,
                    job_id_prefix=job_id_prefix,
                    location=location,
                    project=project,
                    job_config=new_job_config,
                    timeout=timeout,
                )

        finally:
            os.remove(tmppath)

    def load_table_from_json(
        self,
        json_rows: Iterable[Dict[str, Any]],
        destination: Union[Table, TableReference, TableListItem, str],
        num_retries: int = _DEFAULT_NUM_RETRIES,
        job_id: Optional[str] = None,
        job_id_prefix: Optional[str] = None,
        location: Optional[str] = None,
        project: Optional[str] = None,
        job_config: Optional[LoadJobConfig] = None,
        timeout: ResumableTimeoutType = DEFAULT_TIMEOUT,
    ) -> job.LoadJob:
        """Upload the contents of a table from a JSON string or dict.

        Args:
            json_rows (Iterable[Dict[str, Any]]):
                Row data to be inserted. Keys must match the table schema fields
                and values must be JSON-compatible representations.

                .. note::

                    If your data is already a newline-delimited JSON string,
                    it is best to wrap it into a file-like object and pass it
                    to :meth:`~google.cloud.bigquery.client.Client.load_table_from_file`::

                        import io
                        from google.cloud import bigquery

                        data = u'{"foo": "bar"}'
                        data_as_file = io.StringIO(data)

                        client = bigquery.Client()
                        client.load_table_from_file(data_as_file, ...)

            destination (Union[ \
                Table, \
                TableReference, \
                TableListItem, \
                str \
            ]):
                Table into which data is to be loaded. If a string is passed
                in, this method attempts to create a table reference from a
                string using
                :func:`google.cloud.bigquery.table.TableReference.from_string`.
            num_retries (Optional[int]): Number of upload retries. Defaults to 6.
            job_id (Optional[str]): Name of the job.
            job_id_prefix (Optional[str]):
                The user-provided prefix for a randomly generated job ID.
                This parameter will be ignored if a ``job_id`` is also given.
            location (Optional[str]):
                Location where to run the job. Must match the location of the
                destination table.
            project (Optional[str]):
                Project ID of the project of where to run the job. Defaults
                to the client's project.
            job_config (Optional[LoadJobConfig]):
                Extra configuration options for the job. The ``source_format``
                setting is always set to
                :attr:`~google.cloud.bigquery.job.SourceFormat.NEWLINE_DELIMITED_JSON`.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``. Depending on the retry strategy, a request may
                be repeated several times using the same timeout each time.
                Defaults to None.

                Can also be passed as a tuple (connect_timeout, read_timeout).
                See :meth:`requests.Session.request` documentation for details.

        Returns:
            google.cloud.bigquery.job.LoadJob: A new load job.

        Raises:
            TypeError:
                If ``job_config`` is not an instance of
                :class:`~google.cloud.bigquery.job.LoadJobConfig` class.
        """
        job_id = _make_job_id(job_id, job_id_prefix)

        if job_config is not None:
            _verify_job_config_type(job_config, LoadJobConfig)
        else:
            job_config = job.LoadJobConfig()

        new_job_config = job_config._fill_from_default(self._default_load_job_config)

        new_job_config.source_format = job.SourceFormat.NEWLINE_DELIMITED_JSON

        # In specific conditions, we check if the table alread exists, and/or
        # set the autodetect value for the user. For exact conditions, see table
        # https://github.com/googleapis/python-bigquery/issues/1228#issuecomment-1910946297
        if new_job_config.schema is None and new_job_config.autodetect is None:
            if new_job_config.write_disposition in (
                job.WriteDisposition.WRITE_TRUNCATE,
                job.WriteDisposition.WRITE_EMPTY,
            ):
                new_job_config.autodetect = True
            else:
                try:
                    self.get_table(destination)
                except core_exceptions.NotFound:
                    new_job_config.autodetect = True
                else:
                    new_job_config.autodetect = False

        if project is None:
            project = self.project

        if location is None:
            location = self.location

        destination = _table_arg_to_table_ref(destination, default_project=self.project)

        data_str = "\n".join(json.dumps(item, ensure_ascii=False) for item in json_rows)
        encoded_str = data_str.encode()
        data_file = io.BytesIO(encoded_str)
        return self.load_table_from_file(
            data_file,
            destination,
            size=len(encoded_str),
            num_retries=num_retries,
            job_id=job_id,
            job_id_prefix=job_id_prefix,
            location=location,
            project=project,
            job_config=new_job_config,
            timeout=timeout,
        )

    def _do_resumable_upload(
        self,
        stream: IO[bytes],
        metadata: Mapping[str, str],
        num_retries: int,
        timeout: Optional[ResumableTimeoutType],
        project: Optional[str] = None,
    ) -> "requests.Response":
        """Perform a resumable upload.

        Args:
            stream (IO[bytes]): A bytes IO object open for reading.
            metadata (Mapping[str, str]): The metadata associated with the upload.
            num_retries (int):
                Number of upload retries. (Deprecated: This
                argument will be removed in a future release.)
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``. Depending on the retry strategy, a request may
                be repeated several times using the same timeout each time.

                Can also be passed as a tuple (connect_timeout, read_timeout).
                See :meth:`requests.Session.request` documentation for details.
            project (Optional[str]):
                Project ID of the project of where to run the upload. Defaults
                to the client's project.

        Returns:
            The "200 OK" response object returned after the final chunk
            is uploaded.
        """
        upload, transport = self._initiate_resumable_upload(
            stream, metadata, num_retries, timeout, project=project
        )

        while not upload.finished:
            response = upload.transmit_next_chunk(transport, timeout=timeout)

        return response

    def _initiate_resumable_upload(
        self,
        stream: IO[bytes],
        metadata: Mapping[str, str],
        num_retries: int,
        timeout: Optional[ResumableTimeoutType],
        project: Optional[str] = None,
    ):
        """Initiate a resumable upload.

        Args:
            stream (IO[bytes]): A bytes IO object open for reading.
            metadata (Mapping[str, str]): The metadata associated with the upload.
            num_retries (int):
                Number of upload retries. (Deprecated: This
                argument will be removed in a future release.)
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``. Depending on the retry strategy, a request may
                be repeated several times using the same timeout each time.

                Can also be passed as a tuple (connect_timeout, read_timeout).
                See :meth:`requests.Session.request` documentation for details.
            project (Optional[str]):
                Project ID of the project of where to run the upload. Defaults
                to the client's project.

        Returns:
            Tuple:
                Pair of

                * The :class:`~google.resumable_media.requests.ResumableUpload`
                that was created
                * The ``transport`` used to initiate the upload.
        """
        chunk_size = _DEFAULT_CHUNKSIZE
        transport = self._http
        headers = _get_upload_headers(self._connection.user_agent)

        if project is None:
            project = self.project
        # TODO: Increase the minimum version of google-cloud-core to 1.6.0
        # and remove this logic. See:
        # https://github.com/googleapis/python-bigquery/issues/509
        hostname = (
            self._connection.API_BASE_URL
            if not hasattr(self._connection, "get_api_base_url_for_mtls")
            else self._connection.get_api_base_url_for_mtls()
        )
        upload_url = _RESUMABLE_URL_TEMPLATE.format(host=hostname, project=project)

        # TODO: modify ResumableUpload to take a retry.Retry object
        # that it can use for the initial RPC.
        upload = ResumableUpload(upload_url, chunk_size, headers=headers)

        if num_retries is not None:
            upload._retry_strategy = resumable_media.RetryStrategy(
                max_retries=num_retries
            )

        upload.initiate(
            transport,
            stream,
            metadata,
            _GENERIC_CONTENT_TYPE,
            stream_final=False,
            timeout=timeout,
        )

        return upload, transport

    def _do_multipart_upload(
        self,
        stream: IO[bytes],
        metadata: Mapping[str, str],
        size: int,
        num_retries: int,
        timeout: Optional[ResumableTimeoutType],
        project: Optional[str] = None,
    ):
        """Perform a multipart upload.

        Args:
            stream (IO[bytes]): A bytes IO object open for reading.
            metadata (Mapping[str, str]): The metadata associated with the upload.
            size (int):
                The number of bytes to be uploaded (which will be read
                from ``stream``). If not provided, the upload will be
                concluded once ``stream`` is exhausted (or :data:`None`).
            num_retries (int):
                Number of upload retries. (Deprecated: This
                argument will be removed in a future release.)
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``. Depending on the retry strategy, a request may
                be repeated several times using the same timeout each time.

                Can also be passed as a tuple (connect_timeout, read_timeout).
                See :meth:`requests.Session.request` documentation for details.
            project (Optional[str]):
                Project ID of the project of where to run the upload. Defaults
                to the client's project.

        Returns:
            requests.Response:
                The "200 OK" response object returned after the multipart
                upload request.

        Raises:
            ValueError:
                if the ``stream`` has fewer than ``size``
                bytes remaining.
        """
        data = stream.read(size)
        if len(data) < size:
            msg = _READ_LESS_THAN_SIZE.format(size, len(data))
            raise ValueError(msg)

        headers = _get_upload_headers(self._connection.user_agent)

        if project is None:
            project = self.project

        # TODO: Increase the minimum version of google-cloud-core to 1.6.0
        # and remove this logic. See:
        # https://github.com/googleapis/python-bigquery/issues/509
        hostname = (
            self._connection.API_BASE_URL
            if not hasattr(self._connection, "get_api_base_url_for_mtls")
            else self._connection.get_api_base_url_for_mtls()
        )
        upload_url = _MULTIPART_URL_TEMPLATE.format(host=hostname, project=project)
        upload = MultipartUpload(upload_url, headers=headers)

        if num_retries is not None:
            upload._retry_strategy = resumable_media.RetryStrategy(
                max_retries=num_retries
            )

        response = upload.transmit(
            self._http, data, metadata, _GENERIC_CONTENT_TYPE, timeout=timeout
        )

        return response

    def copy_table(
        self,
        sources: Union[
            Table,
            TableReference,
            TableListItem,
            str,
            Sequence[Union[Table, TableReference, TableListItem, str]],
        ],
        destination: Union[Table, TableReference, TableListItem, str],
        job_id: Optional[str] = None,
        job_id_prefix: Optional[str] = None,
        location: Optional[str] = None,
        project: Optional[str] = None,
        job_config: Optional[CopyJobConfig] = None,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
    ) -> job.CopyJob:
        """Copy one or more tables to another table.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#jobconfigurationtablecopy

        Args:
            sources (Union[ \
                google.cloud.bigquery.table.Table, \
                google.cloud.bigquery.table.TableReference, \
                google.cloud.bigquery.table.TableListItem, \
                str, \
                Sequence[ \
                    Union[ \
                        google.cloud.bigquery.table.Table, \
                        google.cloud.bigquery.table.TableReference, \
                        google.cloud.bigquery.table.TableListItem, \
                        str, \
                    ] \
                ], \
            ]):
                Table or tables to be copied.
            destination (Union[ \
                google.cloud.bigquery.table.Table, \
                google.cloud.bigquery.table.TableReference, \
                google.cloud.bigquery.table.TableListItem, \
                str, \
            ]):
                Table into which data is to be copied.
            job_id (Optional[str]): The ID of the job.
            job_id_prefix (Optional[str]):
                The user-provided prefix for a randomly generated job ID.
                This parameter will be ignored if a ``job_id`` is also given.
            location (Optional[str]):
                Location where to run the job. Must match the location of any
                source table as well as the destination table.
            project (Optional[str]):
                Project ID of the project of where to run the job. Defaults
                to the client's project.
            job_config (Optional[google.cloud.bigquery.job.CopyJobConfig]):
                Extra configuration options for the job.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

        Returns:
            google.cloud.bigquery.job.CopyJob: A new copy job instance.

        Raises:
            TypeError:
                If ``job_config`` is not an instance of :class:`~google.cloud.bigquery.job.CopyJobConfig`
                class.
        """
        job_id = _make_job_id(job_id, job_id_prefix)

        if project is None:
            project = self.project

        if location is None:
            location = self.location

        job_ref = job._JobReference(job_id, project=project, location=location)

        # sources can be one of many different input types. (string, Table,
        # TableReference, or a sequence of any of those.) Convert them all to a
        # list of TableReferences.
        #
        # _table_arg_to_table_ref leaves lists unmodified.
        sources = _table_arg_to_table_ref(sources, default_project=self.project)

        if not isinstance(sources, collections_abc.Sequence):
            sources = [sources]

        sources = [
            _table_arg_to_table_ref(source, default_project=self.project)
            for source in sources
        ]

        destination = _table_arg_to_table_ref(destination, default_project=self.project)

        if job_config:
            _verify_job_config_type(job_config, google.cloud.bigquery.job.CopyJobConfig)
            job_config = copy.deepcopy(job_config)

        copy_job = job.CopyJob(
            job_ref, sources, destination, client=self, job_config=job_config
        )
        copy_job._begin(retry=retry, timeout=timeout)

        return copy_job

    def extract_table(
        self,
        source: Union[Table, TableReference, TableListItem, Model, ModelReference, str],
        destination_uris: Union[str, Sequence[str]],
        job_id: Optional[str] = None,
        job_id_prefix: Optional[str] = None,
        location: Optional[str] = None,
        project: Optional[str] = None,
        job_config: Optional[ExtractJobConfig] = None,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
        source_type: str = "Table",
    ) -> job.ExtractJob:
        """Start a job to extract a table into Cloud Storage files.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#jobconfigurationextract

        Args:
            source (Union[ \
                google.cloud.bigquery.table.Table, \
                google.cloud.bigquery.table.TableReference, \
                google.cloud.bigquery.table.TableListItem, \
                google.cloud.bigquery.model.Model, \
                google.cloud.bigquery.model.ModelReference, \
                src, \
            ]):
                Table or Model to be extracted.
            destination_uris (Union[str, Sequence[str]]):
                URIs of Cloud Storage file(s) into which table data is to be
                extracted; in format
                ``gs://<bucket_name>/<object_name_or_glob>``.
            job_id (Optional[str]): The ID of the job.
            job_id_prefix (Optional[str]):
                The user-provided prefix for a randomly generated job ID.
                This parameter will be ignored if a ``job_id`` is also given.
            location (Optional[str]):
                Location where to run the job. Must match the location of the
                source table.
            project (Optional[str]):
                Project ID of the project of where to run the job. Defaults
                to the client's project.
            job_config (Optional[google.cloud.bigquery.job.ExtractJobConfig]):
                Extra configuration options for the job.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
            source_type (Optional[str]):
                Type of source to be extracted.``Table`` or ``Model``. Defaults to ``Table``.
        Returns:
            google.cloud.bigquery.job.ExtractJob: A new extract job instance.

        Raises:
            TypeError:
                If ``job_config`` is not an instance of :class:`~google.cloud.bigquery.job.ExtractJobConfig`
                class.
            ValueError:
                If ``source_type`` is not among ``Table``,``Model``.
            """
        job_id = _make_job_id(job_id, job_id_prefix)

        if project is None:
            project = self.project

        if location is None:
            location = self.location

        job_ref = job._JobReference(job_id, project=project, location=location)
        src = source_type.lower()
        if src == "table":
            source = _table_arg_to_table_ref(source, default_project=self.project)
        elif src == "model":
            source = _model_arg_to_model_ref(source, default_project=self.project)
        else:
            raise ValueError(
                "Cannot pass `{}` as a ``source_type``, pass Table or Model".format(
                    source_type
                )
            )

        if isinstance(destination_uris, str):
            destination_uris = [destination_uris]

        if job_config:
            _verify_job_config_type(
                job_config, google.cloud.bigquery.job.ExtractJobConfig
            )
            job_config = copy.deepcopy(job_config)

        extract_job = job.ExtractJob(
            job_ref, source, destination_uris, client=self, job_config=job_config
        )
        extract_job._begin(retry=retry, timeout=timeout)

        return extract_job

    def query(
        self,
        query: str,
        job_config: Optional[QueryJobConfig] = None,
        job_id: Optional[str] = None,
        job_id_prefix: Optional[str] = None,
        location: Optional[str] = None,
        project: Optional[str] = None,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
        job_retry: Optional[retries.Retry] = DEFAULT_JOB_RETRY,
        api_method: Union[str, enums.QueryApiMethod] = enums.QueryApiMethod.INSERT,
    ) -> job.QueryJob:
        """Run a SQL query.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#jobconfigurationquery

        Args:
            query (str):
                SQL query to be executed. Defaults to the standard SQL
                dialect. Use the ``job_config`` parameter to change dialects.
            job_config (Optional[google.cloud.bigquery.job.QueryJobConfig]):
                Extra configuration options for the job.
                To override any options that were previously set in
                the ``default_query_job_config`` given to the
                ``Client`` constructor, manually set those options to ``None``,
                or whatever value is preferred.
            job_id (Optional[str]): ID to use for the query job.
            job_id_prefix (Optional[str]):
                The prefix to use for a randomly generated job ID. This parameter
                will be ignored if a ``job_id`` is also given.
            location (Optional[str]):
                Location where to run the job. Must match the location of the
                table used in the query as well as the destination table.
            project (Optional[str]):
                Project ID of the project of where to run the job. Defaults
                to the client's project.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.  This only applies to making RPC
                calls.  It isn't used to retry failed jobs.  This has
                a reasonable default that should only be overridden
                with care.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
            job_retry (Optional[google.api_core.retry.Retry]):
                How to retry failed jobs.  The default retries
                rate-limit-exceeded errors.  Passing ``None`` disables
                job retry.

                Not all jobs can be retried.  If ``job_id`` is
                provided, then the job returned by the query will not
                be retryable, and an exception will be raised if a
                non-``None`` (and non-default) value for ``job_retry``
                is also provided.

                Note that errors aren't detected until ``result()`` is
                called on the job returned. The ``job_retry``
                specified here becomes the default ``job_retry`` for
                ``result()``, where it can also be specified.
            api_method (Union[str, enums.QueryApiMethod]):
                Method with which to start the query job.  By default,
                the jobs.insert API is used for starting a query.

                See :class:`google.cloud.bigquery.enums.QueryApiMethod` for
                details on the difference between the query start methods.

        Returns:
            google.cloud.bigquery.job.QueryJob: A new query job instance.

        Raises:
            TypeError:
                If ``job_config`` is not an instance of
                :class:`~google.cloud.bigquery.job.QueryJobConfig`
                class, or if both ``job_id`` and non-``None`` non-default
                ``job_retry`` are provided.
        """
        _job_helpers.validate_job_retry(job_id, job_retry)

        job_id_given = job_id is not None
        if job_id_given and api_method == enums.QueryApiMethod.QUERY:
            raise TypeError(
                "`job_id` was provided, but the 'QUERY' `api_method` was requested."
            )

        if project is None:
            project = self.project

        if location is None:
            location = self.location

        if job_config is not None:
            _verify_job_config_type(job_config, QueryJobConfig)

        job_config = _job_helpers.job_config_with_defaults(
            job_config, self._default_query_job_config
        )

        # Note that we haven't modified the original job_config (or
        # _default_query_job_config) up to this point.
        if api_method == enums.QueryApiMethod.QUERY:
            return _job_helpers.query_jobs_query(
                self,
                query,
                job_config,
                location,
                project,
                retry,
                timeout,
                job_retry,
            )
        elif api_method == enums.QueryApiMethod.INSERT:
            return _job_helpers.query_jobs_insert(
                self,
                query,
                job_config,
                job_id,
                job_id_prefix,
                location,
                project,
                retry,
                timeout,
                job_retry,
            )
        else:
            raise ValueError(f"Got unexpected value for api_method: {repr(api_method)}")

    def query_and_wait(
        self,
        query,
        *,
        job_config: Optional[QueryJobConfig] = None,
        location: Optional[str] = None,
        project: Optional[str] = None,
        api_timeout: TimeoutType = DEFAULT_TIMEOUT,
        wait_timeout: Union[Optional[float], object] = POLLING_DEFAULT_VALUE,
        retry: retries.Retry = DEFAULT_RETRY,
        job_retry: retries.Retry = DEFAULT_JOB_RETRY,
        page_size: Optional[int] = None,
        max_results: Optional[int] = None,
    ) -> RowIterator:
        """Run the query, wait for it to finish, and return the results.

        Args:
            query (str):
                SQL query to be executed. Defaults to the standard SQL
                dialect. Use the ``job_config`` parameter to change dialects.
            job_config (Optional[google.cloud.bigquery.job.QueryJobConfig]):
                Extra configuration options for the job.
                To override any options that were previously set in
                the ``default_query_job_config`` given to the
                ``Client`` constructor, manually set those options to ``None``,
                or whatever value is preferred.
            location (Optional[str]):
                Location where to run the job. Must match the location of the
                table used in the query as well as the destination table.
            project (Optional[str]):
                Project ID of the project of where to run the job. Defaults
                to the client's project.
            api_timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
            wait_timeout (Optional[Union[float, object]]):
                The number of seconds to wait for the query to finish. If the
                query doesn't finish before this timeout, the client attempts
                to cancel the query. If unset, the underlying REST API calls
                have timeouts, but we still wait indefinitely for the job to
                finish.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.  This only applies to making RPC
                calls.  It isn't used to retry failed jobs.  This has
                a reasonable default that should only be overridden
                with care.
            job_retry (Optional[google.api_core.retry.Retry]):
                How to retry failed jobs.  The default retries
                rate-limit-exceeded errors.  Passing ``None`` disables
                job retry. Not all jobs can be retried.
            page_size (Optional[int]):
                The maximum number of rows in each page of results from the
                initial jobs.query request. Non-positive values are ignored.
            max_results (Optional[int]):
                The maximum total number of rows from this request.

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
            TypeError:
                If ``job_config`` is not an instance of
                :class:`~google.cloud.bigquery.job.QueryJobConfig`
                class.
        """
        return self._query_and_wait_bigframes(
            query,
            job_config=job_config,
            location=location,
            project=project,
            api_timeout=api_timeout,
            wait_timeout=wait_timeout,
            retry=retry,
            job_retry=job_retry,
            page_size=page_size,
            max_results=max_results,
        )

    def _query_and_wait_bigframes(
        self,
        query,
        *,
        job_config: Optional[QueryJobConfig] = None,
        location: Optional[str] = None,
        project: Optional[str] = None,
        api_timeout: TimeoutType = DEFAULT_TIMEOUT,
        wait_timeout: Union[Optional[float], object] = POLLING_DEFAULT_VALUE,
        retry: retries.Retry = DEFAULT_RETRY,
        job_retry: retries.Retry = DEFAULT_JOB_RETRY,
        page_size: Optional[int] = None,
        max_results: Optional[int] = None,
        callback: Callable = lambda _: None,
    ) -> RowIterator:
        """See query_and_wait.

        This method has an extra callback parameter, which is used by bigframes
        to create better progress bars.
        """
        if project is None:
            project = self.project

        if location is None:
            location = self.location

        if job_config is not None:
            _verify_job_config_type(job_config, QueryJobConfig)

        job_config = _job_helpers.job_config_with_defaults(
            job_config, self._default_query_job_config
        )

        return _job_helpers.query_and_wait(
            self,
            query,
            job_config=job_config,
            location=location,
            project=project,
            api_timeout=api_timeout,
            wait_timeout=wait_timeout,
            retry=retry,
            job_retry=job_retry,
            page_size=page_size,
            max_results=max_results,
            callback=callback,
        )

    def insert_rows(
        self,
        table: Union[Table, TableReference, str],
        rows: Union[Iterable[Tuple], Iterable[Mapping[str, Any]]],
        selected_fields: Optional[Sequence[SchemaField]] = None,
        **kwargs,
    ) -> Sequence[Dict[str, Any]]:
        """Insert rows into a table via the streaming API.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tabledata/insertAll

        BigQuery will reject insertAll payloads that exceed a defined limit (10MB).
        Additionally, if a payload vastly exceeds this limit, the request is rejected
        by the intermediate architecture, which returns a 413 (Payload Too Large) status code.


        See
        https://cloud.google.com/bigquery/quotas#streaming_inserts

        Args:
            table (Union[ \
                google.cloud.bigquery.table.Table, \
                google.cloud.bigquery.table.TableReference, \
                str, \
            ]):
                The destination table for the row data, or a reference to it.
            rows (Union[Sequence[Tuple], Sequence[Dict]]):
                Row data to be inserted. If a list of tuples is given, each
                tuple should contain data for each schema field on the
                current table and in the same order as the schema fields. If
                a list of dictionaries is given, the keys must include all
                required fields in the schema. Keys which do not correspond
                to a field in the schema are ignored.
            selected_fields (Sequence[google.cloud.bigquery.schema.SchemaField]):
                The fields to return. Required if ``table`` is a
                :class:`~google.cloud.bigquery.table.TableReference`.
            kwargs (dict):
                Keyword arguments to
                :meth:`~google.cloud.bigquery.client.Client.insert_rows_json`.

        Returns:
            Sequence[Mappings]:
                One mapping per row with insert errors: the "index" key
                identifies the row, and the "errors" key contains a list of
                the mappings describing one or more problems with the row.

        Raises:
            ValueError: if table's schema is not set or `rows` is not a `Sequence`.
        """
        if not isinstance(rows, (collections_abc.Sequence, collections_abc.Iterator)):
            raise TypeError("rows argument should be a sequence of dicts or tuples")

        table = _table_arg_to_table(table, default_project=self.project)

        if not isinstance(table, Table):
            raise TypeError(_NEED_TABLE_ARGUMENT)

        schema = table.schema

        # selected_fields can override the table schema.
        if selected_fields is not None:
            schema = selected_fields

        if len(schema) == 0:
            raise ValueError(
                (
                    "Could not determine schema for table '{}'. Call client.get_table() "
                    "or pass in a list of schema fields to the selected_fields argument."
                ).format(table)
            )

        json_rows = [_record_field_to_json(schema, row) for row in rows]

        return self.insert_rows_json(table, json_rows, **kwargs)

    def insert_rows_from_dataframe(
        self,
        table: Union[Table, TableReference, str],
        dataframe,
        selected_fields: Optional[Sequence[SchemaField]] = None,
        chunk_size: int = 500,
        **kwargs: Dict,
    ) -> Sequence[Sequence[dict]]:
        """Insert rows into a table from a dataframe via the streaming API.

        BigQuery will reject insertAll payloads that exceed a defined limit (10MB).
        Additionally, if a payload vastly exceeds this limit, the request is rejected
        by the intermediate architecture, which returns a 413 (Payload Too Large) status code.

        See
        https://cloud.google.com/bigquery/quotas#streaming_inserts

        Args:
            table (Union[ \
                google.cloud.bigquery.table.Table, \
                google.cloud.bigquery.table.TableReference, \
                str, \
            ]):
                The destination table for the row data, or a reference to it.
            dataframe (pandas.DataFrame):
                A :class:`~pandas.DataFrame` containing the data to load. Any
                ``NaN`` values present in the dataframe are omitted from the
                streaming API request(s).
            selected_fields (Sequence[google.cloud.bigquery.schema.SchemaField]):
                The fields to return. Required if ``table`` is a
                :class:`~google.cloud.bigquery.table.TableReference`.
            chunk_size (int):
                The number of rows to stream in a single chunk. Must be positive.
            kwargs (Dict):
                Keyword arguments to
                :meth:`~google.cloud.bigquery.client.Client.insert_rows_json`.

        Returns:
            Sequence[Sequence[Mappings]]:
                A list with insert errors for each insert chunk. Each element
                is a list containing one mapping per row with insert errors:
                the "index" key identifies the row, and the "errors" key
                contains a list of the mappings describing one or more problems
                with the row.

        Raises:
            ValueError: if table's schema is not set
        """
        insert_results = []

        chunk_count = int(math.ceil(len(dataframe) / chunk_size))
        rows_iter = _pandas_helpers.dataframe_to_json_generator(dataframe)

        for _ in range(chunk_count):
            rows_chunk = itertools.islice(rows_iter, chunk_size)
            result = self.insert_rows(table, rows_chunk, selected_fields, **kwargs)
            insert_results.append(result)

        return insert_results

    def insert_rows_json(
        self,
        table: Union[Table, TableReference, TableListItem, str],
        json_rows: Sequence[Mapping[str, Any]],
        row_ids: Union[
            Iterable[Optional[str]], AutoRowIDs, None
        ] = AutoRowIDs.GENERATE_UUID,
        skip_invalid_rows: Optional[bool] = None,
        ignore_unknown_values: Optional[bool] = None,
        template_suffix: Optional[str] = None,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
    ) -> Sequence[dict]:
        """Insert rows into a table without applying local type conversions.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tabledata/insertAll

        BigQuery will reject insertAll payloads that exceed a defined limit (10MB).
        Additionally, if a payload vastly exceeds this limit, the request is rejected
        by the intermediate architecture, which returns a 413 (Payload Too Large) status code.

        See
        https://cloud.google.com/bigquery/quotas#streaming_inserts

        Args:
            table (Union[ \
                google.cloud.bigquery.table.Table \
                google.cloud.bigquery.table.TableReference, \
                google.cloud.bigquery.table.TableListItem, \
                str \
            ]):
                The destination table for the row data, or a reference to it.
            json_rows (Sequence[Dict]):
                Row data to be inserted. Keys must match the table schema fields
                and values must be JSON-compatible representations.
            row_ids (Union[Iterable[str], AutoRowIDs, None]):
                Unique IDs, one per row being inserted. An ID can also be
                ``None``, indicating that an explicit insert ID should **not**
                be used for that row. If the argument is omitted altogether,
                unique IDs are created automatically.

                .. versionchanged:: 2.21.0
                    Can also be an iterable, not just a sequence, or an
                    :class:`AutoRowIDs` enum member.

                .. deprecated:: 2.21.0
                    Passing ``None`` to explicitly request autogenerating insert IDs is
                    deprecated, use :attr:`AutoRowIDs.GENERATE_UUID` instead.

            skip_invalid_rows (Optional[bool]):
                Insert all valid rows of a request, even if invalid rows exist.
                The default value is ``False``, which causes the entire request
                to fail if any invalid rows exist.
            ignore_unknown_values (Optional[bool]):
                Accept rows that contain values that do not match the schema.
                The unknown values are ignored. Default is ``False``, which
                treats unknown values as errors.
            template_suffix (Optional[str]):
                Treat ``name`` as a template table and provide a suffix.
                BigQuery will create the table ``<name> + <template_suffix>``
                based on the schema of the template table. See
                https://cloud.google.com/bigquery/streaming-data-into-bigquery#template-tables
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

        Returns:
            Sequence[Mappings]:
                One mapping per row with insert errors: the "index" key
                identifies the row, and the "errors" key contains a list of
                the mappings describing one or more problems with the row.

        Raises:
            TypeError: if `json_rows` is not a `Sequence`.
        """
        if not isinstance(
            json_rows, (collections_abc.Sequence, collections_abc.Iterator)
        ):
            raise TypeError("json_rows argument should be a sequence of dicts")
        # Convert table to just a reference because unlike insert_rows,
        # insert_rows_json doesn't need the table schema. It's not doing any
        # type conversions.
        table = _table_arg_to_table_ref(table, default_project=self.project)
        rows_info: List[Any] = []
        data: Dict[str, Any] = {"rows": rows_info}

        if row_ids is None:
            warnings.warn(
                "Passing None for row_ids is deprecated. To explicitly request "
                "autogenerated insert IDs, use AutoRowIDs.GENERATE_UUID instead",
                category=DeprecationWarning,
            )
            row_ids = AutoRowIDs.GENERATE_UUID

        if not isinstance(row_ids, AutoRowIDs):
            try:
                row_ids_iter = iter(row_ids)
            except TypeError:
                msg = "row_ids is neither an iterable nor an AutoRowIDs enum member"
                raise TypeError(msg)

        for i, row in enumerate(json_rows):
            info: Dict[str, Any] = {"json": row}

            if row_ids is AutoRowIDs.GENERATE_UUID:
                info["insertId"] = str(uuid.uuid4())
            elif row_ids is AutoRowIDs.DISABLED:
                info["insertId"] = None
            else:
                try:
                    insert_id = next(row_ids_iter)
                except StopIteration:
                    msg = f"row_ids did not generate enough IDs, error at index {i}"
                    raise ValueError(msg)
                else:
                    info["insertId"] = insert_id

            rows_info.append(info)

        if skip_invalid_rows is not None:
            data["skipInvalidRows"] = skip_invalid_rows

        if ignore_unknown_values is not None:
            data["ignoreUnknownValues"] = ignore_unknown_values

        if template_suffix is not None:
            data["templateSuffix"] = template_suffix

        path = "%s/insertAll" % table.path
        # We can always retry, because every row has an insert ID.
        span_attributes = {"path": path}
        response = self._call_api(
            retry,
            span_name="BigQuery.insertRowsJson",
            span_attributes=span_attributes,
            method="POST",
            path=path,
            data=data,
            timeout=timeout,
        )
        errors = []

        for error in response.get("insertErrors", ()):
            errors.append({"index": int(error["index"]), "errors": error["errors"]})

        return errors

    def list_partitions(
        self,
        table: Union[Table, TableReference, TableListItem, str],
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
    ) -> Sequence[str]:
        """List the partitions in a table.

        Args:
            table (Union[ \
                google.cloud.bigquery.table.Table, \
                google.cloud.bigquery.table.TableReference, \
                google.cloud.bigquery.table.TableListItem, \
                str, \
            ]):
                The table or reference from which to get partition info
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
                If multiple requests are made under the hood, ``timeout``
                applies to each individual request.

        Returns:
            List[str]:
                A list of the partition ids present in the partitioned table
        """
        table = _table_arg_to_table_ref(table, default_project=self.project)
        meta_table = self.get_table(
            TableReference(
                DatasetReference(table.project, table.dataset_id),
                "%s$__PARTITIONS_SUMMARY__" % table.table_id,
            ),
            retry=retry,
            timeout=timeout,
        )

        subset = [col for col in meta_table.schema if col.name == "partition_id"]
        return [
            row[0]
            for row in self.list_rows(
                meta_table, selected_fields=subset, retry=retry, timeout=timeout
            )
        ]

    def list_rows(
        self,
        table: Union[Table, TableListItem, TableReference, str],
        selected_fields: Optional[Sequence[SchemaField]] = None,
        max_results: Optional[int] = None,
        page_token: Optional[str] = None,
        start_index: Optional[int] = None,
        page_size: Optional[int] = None,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
    ) -> RowIterator:
        """List the rows of the table.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/tabledata/list

        .. note::

           This method assumes that the provided schema is up-to-date with the
           schema as defined on the back-end: if the two schemas are not
           identical, the values returned may be incomplete. To ensure that the
           local copy of the schema is up-to-date, call ``client.get_table``.

        Args:
            table (Union[ \
                google.cloud.bigquery.table.Table, \
                google.cloud.bigquery.table.TableListItem, \
                google.cloud.bigquery.table.TableReference, \
                str, \
            ]):
                The table to list, or a reference to it. When the table
                object does not contain a schema and ``selected_fields`` is
                not supplied, this method calls ``get_table`` to fetch the
                table schema.
            selected_fields (Sequence[google.cloud.bigquery.schema.SchemaField]):
                The fields to return. If not supplied, data for all columns
                are downloaded.
            max_results (Optional[int]):
                Maximum number of rows to return.
            page_token (Optional[str]):
                Token representing a cursor into the table's rows.
                If not passed, the API will return the first page of the
                rows. The token marks the beginning of the iterator to be
                returned and the value of the ``page_token`` can be accessed
                at ``next_page_token`` of the
                :class:`~google.cloud.bigquery.table.RowIterator`.
            start_index (Optional[int]):
                The zero-based index of the starting row to read.
            page_size (Optional[int]):
                The maximum number of rows in each page of results from this request.
                Non-positive values are ignored. Defaults to a sensible value set by the API.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
                If multiple requests are made under the hood, ``timeout``
                applies to each individual request.

        Returns:
            google.cloud.bigquery.table.RowIterator:
                Iterator of row data
                :class:`~google.cloud.bigquery.table.Row`-s. During each
                page, the iterator will have the ``total_rows`` attribute
                set, which counts the total number of rows **in the table**
                (this is distinct from the total number of rows in the
                current page: ``iterator.page.num_items``).
        """
        table = _table_arg_to_table(table, default_project=self.project)

        if not isinstance(table, Table):
            raise TypeError(_NEED_TABLE_ARGUMENT)

        schema = table.schema

        # selected_fields can override the table schema.
        if selected_fields is not None:
            schema = selected_fields

        # No schema, but no selected_fields. Assume the developer wants all
        # columns, so get the table resource for them rather than failing.
        elif len(schema) == 0:
            table = self.get_table(table.reference, retry=retry, timeout=timeout)
            schema = table.schema

        params: Dict[str, Any] = {}
        if selected_fields is not None:
            params["selectedFields"] = ",".join(field.name for field in selected_fields)
        if start_index is not None:
            params["startIndex"] = start_index

        params["formatOptions.useInt64Timestamp"] = True
        row_iterator = RowIterator(
            client=self,
            api_request=functools.partial(self._call_api, retry, timeout=timeout),
            path="%s/data" % (table.path,),
            schema=schema,
            page_token=page_token,
            max_results=max_results,
            page_size=page_size,
            extra_params=params,
            table=table,
            # Pass in selected_fields separately from schema so that full
            # tables can be fetched without a column filter.
            selected_fields=selected_fields,
            total_rows=getattr(table, "num_rows", None),
            project=table.project,
            location=table.location,
        )
        return row_iterator

    def _list_rows_from_query_results(
        self,
        job_id: str,
        location: str,
        project: str,
        schema: Sequence[SchemaField],
        total_rows: Optional[int] = None,
        destination: Optional[Union[Table, TableReference, TableListItem, str]] = None,
        max_results: Optional[int] = None,
        start_index: Optional[int] = None,
        page_size: Optional[int] = None,
        retry: retries.Retry = DEFAULT_RETRY,
        timeout: TimeoutType = DEFAULT_TIMEOUT,
        query_id: Optional[str] = None,
        first_page_response: Optional[Dict[str, Any]] = None,
        num_dml_affected_rows: Optional[int] = None,
        query: Optional[str] = None,
        total_bytes_processed: Optional[int] = None,
        slot_millis: Optional[int] = None,
        created: Optional[datetime.datetime] = None,
        started: Optional[datetime.datetime] = None,
        ended: Optional[datetime.datetime] = None,
    ) -> RowIterator:
        """List the rows of a completed query.
        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/getQueryResults
        Args:
            job_id (str):
                ID of a query job.
            location (str): Location of the query job.
            project (str):
                ID of the project where the query job was run.
            schema (Sequence[google.cloud.bigquery.schema.SchemaField]):
                The fields expected in these query results. Used to convert
                from JSON to expected Python types.
            total_rows (Optional[int]):
                Total number of rows in the query results.
            destination (Optional[Union[ \
                google.cloud.bigquery.table.Table, \
                google.cloud.bigquery.table.TableListItem, \
                google.cloud.bigquery.table.TableReference, \
                str, \
            ]]):
                Destination table reference. Used to fetch the query results
                with the BigQuery Storage API.
            max_results (Optional[int]):
                Maximum number of rows to return across the whole iterator.
            start_index (Optional[int]):
                The zero-based index of the starting row to read.
            page_size (Optional[int]):
                The maximum number of rows in each page of results from this request.
                Non-positive values are ignored. Defaults to a sensible value set by the API.
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``. If set, this connection timeout may be
                increased to a minimum value. This prevents retries on what
                would otherwise be a successful response.
                If multiple requests are made under the hood, ``timeout``
                applies to each individual request.
            query_id (Optional[str]):
                [Preview] ID of a completed query. This ID is auto-generated
                and not guaranteed to be populated.
            first_page_response (Optional[dict]):
                API response for the first page of results (if available).
            num_dml_affected_rows (Optional[int]):
                If this RowIterator is the result of a DML query, the number of
                rows that were affected.
            query (Optional[str]):
                The query text used.
            total_bytes_processed (Optional[int]):
                total bytes processed from job statistics, if present.
            slot_millis (Optional[int]):
                Number of slot ms the user is actually billed for.
            created (Optional[datetime.datetime]):
                Datetime at which the job was created.
            started (Optional[datetime.datetime]):
                Datetime at which the job was started.
            ended (Optional[datetime.datetime]):
                Datetime at which the job finished.

        Returns:
            google.cloud.bigquery.table.RowIterator:
                Iterator of row data
                :class:`~google.cloud.bigquery.table.Row`-s.
        """
        params: Dict[str, Any] = {
            "fields": _LIST_ROWS_FROM_QUERY_RESULTS_FIELDS,
            "location": location,
        }

        if timeout is not None:
            if not isinstance(timeout, (int, float)):
                timeout = _MIN_GET_QUERY_RESULTS_TIMEOUT
            else:
                timeout = max(timeout, _MIN_GET_QUERY_RESULTS_TIMEOUT)

        if start_index is not None:
            params["startIndex"] = start_index

        params["formatOptions.useInt64Timestamp"] = True
        row_iterator = RowIterator(
            client=self,
            api_request=functools.partial(self._call_api, retry, timeout=timeout),
            path=f"/projects/{project}/queries/{job_id}",
            schema=schema,
            max_results=max_results,
            page_size=page_size,
            table=destination,
            extra_params=params,
            total_rows=total_rows,
            project=project,
            location=location,
            job_id=job_id,
            query_id=query_id,
            first_page_response=first_page_response,
            num_dml_affected_rows=num_dml_affected_rows,
            query=query,
            total_bytes_processed=total_bytes_processed,
            slot_millis=slot_millis,
            created=created,
            started=started,
            ended=ended,
        )
        return row_iterator

    def _schema_from_json_file_object(self, file_obj):
        """Helper function for schema_from_json that takes a
        file object that describes a table schema.

        Returns:
             List of schema field objects.
        """
        json_data = json.load(file_obj)
        return [SchemaField.from_api_repr(field) for field in json_data]

    def _schema_to_json_file_object(self, schema_list, file_obj):
        """Helper function for schema_to_json that takes a schema list and file
        object and writes the schema list to the file object with json.dump
        """
        json.dump(schema_list, file_obj, indent=2, sort_keys=True)

    def schema_from_json(self, file_or_path: "PathType") -> List[SchemaField]:
        """Takes a file object or file path that contains json that describes
        a table schema.

        Returns:
            List[SchemaField]:
                List of :class:`~google.cloud.bigquery.schema.SchemaField` objects.
        """
        if isinstance(file_or_path, io.IOBase):
            return self._schema_from_json_file_object(file_or_path)

        with open(file_or_path) as file_obj:
            return self._schema_from_json_file_object(file_obj)

    def schema_to_json(
        self, schema_list: Sequence[SchemaField], destination: "PathType"
    ):
        """Takes a list of schema field objects.

        Serializes the list of schema field objects as json to a file.

        Destination is a file path or a file object.
        """
        json_schema_list = [f.to_api_repr() for f in schema_list]

        if isinstance(destination, io.IOBase):
            return self._schema_to_json_file_object(json_schema_list, destination)

        with open(destination, mode="w") as file_obj:
            return self._schema_to_json_file_object(json_schema_list, file_obj)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


# pylint: disable=unused-argument
def _item_to_project(iterator, resource):
    """Convert a JSON project to the native object.

    Args:
        iterator (google.api_core.page_iterator.Iterator): The iterator that is currently in use.

        resource (Dict): An item to be converted to a project.

    Returns:
        google.cloud.bigquery.client.Project: The next project in the page.
    """
    return Project.from_api_repr(resource)


# pylint: enable=unused-argument


def _item_to_dataset(iterator, resource):
    """Convert a JSON dataset to the native object.

    Args:
        iterator (google.api_core.page_iterator.Iterator): The iterator that is currently in use.

        resource (Dict): An item to be converted to a dataset.

    Returns:
        google.cloud.bigquery.dataset.DatasetListItem: The next dataset in the page.
    """
    return DatasetListItem(resource)


def _item_to_job(iterator, resource):
    """Convert a JSON job to the native object.

    Args:
        iterator (google.api_core.page_iterator.Iterator): The iterator that is currently in use.

        resource (Dict): An item to be converted to a job.

    Returns:
        job instance: The next job in the page.
    """
    return iterator.client.job_from_resource(resource)


def _item_to_model(iterator, resource):
    """Convert a JSON model to the native object.

    Args:
        iterator (google.api_core.page_iterator.Iterator):
            The iterator that is currently in use.
        resource (Dict): An item to be converted to a model.

    Returns:
        google.cloud.bigquery.model.Model: The next model in the page.
    """
    return Model.from_api_repr(resource)


def _item_to_routine(iterator, resource):
    """Convert a JSON model to the native object.

    Args:
        iterator (google.api_core.page_iterator.Iterator):
            The iterator that is currently in use.
        resource (Dict): An item to be converted to a routine.

    Returns:
        google.cloud.bigquery.routine.Routine: The next routine in the page.
    """
    return Routine.from_api_repr(resource)


def _item_to_table(iterator, resource):
    """Convert a JSON table to the native object.

    Args:
        iterator (google.api_core.page_iterator.Iterator): The iterator that is currently in use.

        resource (Dict): An item to be converted to a table.

    Returns:
        google.cloud.bigquery.table.Table: The next table in the page.
    """
    return TableListItem(resource)


def _extract_job_reference(job, project=None, location=None):
    """Extract fully-qualified job reference from a job-like object.

    Args:
        job_id (Union[ \
            str, \
            google.cloud.bigquery.job.LoadJob, \
            google.cloud.bigquery.job.CopyJob, \
            google.cloud.bigquery.job.ExtractJob, \
            google.cloud.bigquery.job.QueryJob \
        ]): Job identifier.
        project (Optional[str]):
            Project where the job was run. Ignored if ``job_id`` is a job
            object.
        location (Optional[str]):
            Location where the job was run. Ignored if ``job_id`` is a job
            object.

    Returns:
        Tuple[str, str, str]: ``(project, location, job_id)``
    """
    if hasattr(job, "job_id"):
        project = job.project
        job_id = job.job_id
        location = job.location
    else:
        job_id = job

    return (project, location, job_id)


def _check_mode(stream):
    """Check that a stream was opened in read-binary mode.

    Args:
        stream (IO[bytes]): A bytes IO object open for reading.

    Raises:
        ValueError:
            if the ``stream.mode`` is a valid attribute
            and is not among ``rb``, ``r+b`` or ``rb+``.
    """
    mode = getattr(stream, "mode", None)

    if isinstance(stream, gzip.GzipFile):
        if mode != gzip.READ:  # pytype: disable=module-attr
            raise ValueError(
                "Cannot upload gzip files opened in write mode:  use "
                "gzip.GzipFile(filename, mode='rb')"
            )
    else:
        if mode is not None and mode not in ("rb", "r+b", "rb+"):
            raise ValueError(
                "Cannot upload files opened in text mode:  use "
                "open(filename, mode='rb') or open(filename, mode='r+b')"
            )


def _get_upload_headers(user_agent):
    """Get the headers for an upload request.

    Args:
        user_agent (str): The user-agent for requests.

    Returns:
        Dict: The headers to be used for the request.
    """
    return {
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": user_agent,
        "content-type": "application/json; charset=UTF-8",
    }


def _add_server_timeout_header(headers: Optional[Dict[str, str]], kwargs):
    timeout = kwargs.get("timeout")
    if timeout is not None:
        if headers is None:
            headers = {}
        headers[TIMEOUT_HEADER] = str(timeout)

    if headers:
        kwargs["headers"] = headers

    return kwargs
