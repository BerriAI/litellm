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

"""Cursor for the Google BigQuery DB-API."""

from __future__ import annotations

import collections
from collections import abc as collections_abc
import re
from typing import Optional

try:
    from google.cloud.bigquery_storage import ArrowSerializationOptions
except ImportError:
    _ARROW_COMPRESSION_SUPPORT = False
else:
    # Having BQ Storage available implies that pyarrow >=1.0.0 is available, too.
    _ARROW_COMPRESSION_SUPPORT = True

from google.cloud.bigquery import job
from google.cloud.bigquery.dbapi import _helpers
from google.cloud.bigquery.dbapi import exceptions
import google.cloud.exceptions  # type: ignore


# Per PEP 249: A 7-item sequence containing information describing one result
# column. The first two items (name and type_code) are mandatory, the other
# five are optional and are set to None if no meaningful values can be
# provided.
Column = collections.namedtuple(
    "Column",
    [
        "name",
        "type_code",
        "display_size",
        "internal_size",
        "precision",
        "scale",
        "null_ok",
    ],
)


@_helpers.raise_on_closed("Operating on a closed cursor.")
class Cursor(object):
    """DB-API Cursor to Google BigQuery.

    Args:
        connection (google.cloud.bigquery.dbapi.Connection):
            A DB-API connection to Google BigQuery.
    """

    def __init__(self, connection):
        self.connection = connection
        self.description = None
        # Per PEP 249: The attribute is -1 in case no .execute*() has been
        # performed on the cursor or the rowcount of the last operation
        # cannot be determined by the interface.
        self.rowcount = -1
        # Per PEP 249: The arraysize attribute defaults to 1, meaning to fetch
        # a single row at a time. However, we deviate from that, and set the
        # default to None, allowing the backend to automatically determine the
        # most appropriate size.
        self.arraysize = None
        self._query_data = None
        self._query_rows = None
        self._closed = False

    @property
    def query_job(self) -> Optional[job.QueryJob]:
        """google.cloud.bigquery.job.query.QueryJob | None: The query job
        created by the last ``execute*()`` call, if a query job was created.

        .. note::
            If the last ``execute*()`` call was ``executemany()``, this is the
            last job created by ``executemany()``."""
        rows = self._query_rows

        if rows is None:
            return None

        job_id = rows.job_id
        project = rows.project
        location = rows.location
        client = self.connection._client

        if job_id is None:
            return None

        return client.get_job(job_id, location=location, project=project)

    def close(self):
        """Mark the cursor as closed, preventing its further use."""
        self._closed = True

    def _set_description(self, schema):
        """Set description from schema.

        Args:
            schema (Sequence[google.cloud.bigquery.schema.SchemaField]):
                A description of fields in the schema.
        """
        if schema is None:
            self.description = None
            return

        self.description = tuple(
            Column(
                name=field.name,
                type_code=field.field_type,
                display_size=None,
                internal_size=None,
                precision=None,
                scale=None,
                null_ok=field.is_nullable,
            )
            for field in schema
        )

    def _set_rowcount(self, rows):
        """Set the rowcount from a RowIterator.

        Normally, this sets rowcount to the number of rows returned by the
        query, but if it was a DML statement, it sets rowcount to the number
        of modified rows.

        Args:
            query_results (google.cloud.bigquery.query._QueryResults):
                Results of a query.
        """
        total_rows = 0
        num_dml_affected_rows = rows.num_dml_affected_rows

        if rows.total_rows is not None and rows.total_rows > 0:
            total_rows = rows.total_rows
        if num_dml_affected_rows is not None and num_dml_affected_rows > 0:
            total_rows = num_dml_affected_rows
        self.rowcount = total_rows

    def execute(self, operation, parameters=None, job_id=None, job_config=None):
        """Prepare and execute a database operation.

        .. note::
            When setting query parameters, values which are "text"
            (``unicode`` in Python2, ``str`` in Python3) will use
            the 'STRING' BigQuery type. Values which are "bytes" (``str`` in
            Python2, ``bytes`` in Python3), will use using the 'BYTES' type.

            A `~datetime.datetime` parameter without timezone information uses
            the 'DATETIME' BigQuery type (example: Global Pi Day Celebration
            March 14, 2017 at 1:59pm). A `~datetime.datetime` parameter with
            timezone information uses the 'TIMESTAMP' BigQuery type (example:
            a wedding on April 29, 2011 at 11am, British Summer Time).

            For more information about BigQuery data types, see:
            https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types

            ``STRUCT``/``RECORD`` and ``REPEATED`` query parameters are not
            yet supported. See:
            https://github.com/GoogleCloudPlatform/google-cloud-python/issues/3524

        Args:
            operation (str): A Google BigQuery query string.

            parameters (Union[Mapping[str, Any], Sequence[Any]]):
                (Optional) dictionary or sequence of parameter values.

            job_id (str | None):
                (Optional and discouraged) The job ID to use when creating
                the query job. For best performance and reliability, manually
                setting a job ID is discouraged.

            job_config (google.cloud.bigquery.job.QueryJobConfig):
                (Optional) Extra configuration options for the query job.
        """
        formatted_operation, parameter_types = _format_operation(operation, parameters)
        self._execute(
            formatted_operation, parameters, job_id, job_config, parameter_types
        )

    def _execute(
        self, formatted_operation, parameters, job_id, job_config, parameter_types
    ):
        self._query_data = None
        self._query_results = None
        client = self.connection._client

        # The DB-API uses the pyformat formatting, since the way BigQuery does
        # query parameters was not one of the standard options. Convert both
        # the query and the parameters to the format expected by the client
        # libraries.
        query_parameters = _helpers.to_query_parameters(parameters, parameter_types)

        config = job_config or job.QueryJobConfig()
        config.query_parameters = query_parameters

        # Start the query and wait for the query to finish.
        try:
            if job_id is not None:
                rows = client.query(
                    formatted_operation,
                    job_config=job_config,
                    job_id=job_id,
                ).result(
                    page_size=self.arraysize,
                )
            else:
                rows = client.query_and_wait(
                    formatted_operation,
                    job_config=config,
                    page_size=self.arraysize,
                )
        except google.cloud.exceptions.GoogleCloudError as exc:
            raise exceptions.DatabaseError(exc)

        self._query_rows = rows
        self._set_description(rows.schema)

        if config.dry_run:
            self.rowcount = 0
        else:
            self._set_rowcount(rows)

    def executemany(self, operation, seq_of_parameters):
        """Prepare and execute a database operation multiple times.

        Args:
            operation (str): A Google BigQuery query string.

            seq_of_parameters (Union[Sequence[Mapping[str, Any], Sequence[Any]]]):
                Sequence of many sets of parameter values.
        """
        if seq_of_parameters:
            rowcount = 0
            # There's no reason to format the line more than once, as
            # the operation only barely depends on the parameters.  So
            # we just use the first set of parameters. If there are
            # different numbers or types of parameters, we'll error
            # anyway.
            formatted_operation, parameter_types = _format_operation(
                operation, seq_of_parameters[0]
            )
            for parameters in seq_of_parameters:
                self._execute(
                    formatted_operation, parameters, None, None, parameter_types
                )
                rowcount += self.rowcount

            self.rowcount = rowcount

    def _try_fetch(self, size=None):
        """Try to start fetching data, if not yet started.

        Mutates self to indicate that iteration has started.
        """
        if self._query_data is not None:
            # Already started fetching the data.
            return

        rows = self._query_rows
        if rows is None:
            raise exceptions.InterfaceError(
                "No query results: execute() must be called before fetch."
            )

        bqstorage_client = self.connection._bqstorage_client
        if rows._should_use_bqstorage(
            bqstorage_client,
            create_bqstorage_client=False,
        ):
            rows_iterable = self._bqstorage_fetch(bqstorage_client)
            self._query_data = _helpers.to_bq_table_rows(rows_iterable)
            return

        self._query_data = iter(rows)

    def _bqstorage_fetch(self, bqstorage_client):
        """Start fetching data with the BigQuery Storage API.

        The method assumes that the data about the relevant query job already
        exists internally.

        Args:
            bqstorage_client(\
                google.cloud.bigquery_storage_v1.BigQueryReadClient \
            ):
                A client tha know how to talk to the BigQuery Storage API.

        Returns:
            Iterable[Mapping]:
                A sequence of rows, represented as dictionaries.
        """
        # Hitting this code path with a BQ Storage client instance implies that
        # bigquery_storage can indeed be imported here without errors.
        from google.cloud import bigquery_storage

        table_reference = self._query_rows._table

        requested_session = bigquery_storage.types.ReadSession(
            table=table_reference.to_bqstorage(),
            data_format=bigquery_storage.types.DataFormat.ARROW,
        )

        if _ARROW_COMPRESSION_SUPPORT:
            requested_session.read_options.arrow_serialization_options.buffer_compression = (
                ArrowSerializationOptions.CompressionCodec.LZ4_FRAME
            )

        read_session = bqstorage_client.create_read_session(
            parent="projects/{}".format(table_reference.project),
            read_session=requested_session,
            # a single stream only, as DB API is not well-suited for multithreading
            max_stream_count=1,
        )

        if not read_session.streams:
            return iter([])  # empty table, nothing to read

        stream_name = read_session.streams[0].name
        read_rows_stream = bqstorage_client.read_rows(stream_name)

        rows_iterable = read_rows_stream.rows(read_session)
        return rows_iterable

    def fetchone(self):
        """Fetch a single row from the results of the last ``execute*()`` call.

        .. note::
            If a dry run query was executed, no rows are returned.

        Returns:
            Tuple:
                A tuple representing a row or ``None`` if no more data is
                available.

        Raises:
            google.cloud.bigquery.dbapi.InterfaceError: if called before ``execute()``.
        """
        self._try_fetch()
        try:
            return next(self._query_data)
        except StopIteration:
            return None

    def fetchmany(self, size=None):
        """Fetch multiple results from the last ``execute*()`` call.

        .. note::
            If a dry run query was executed, no rows are returned.

        .. note::
            The size parameter is not used for the request/response size.
            Set the ``arraysize`` attribute before calling ``execute()`` to
            set the batch size.

        Args:
            size (int):
                (Optional) Maximum number of rows to return. Defaults to the
                ``arraysize`` property value. If ``arraysize`` is not set, it
                defaults to ``1``.

        Returns:
            List[Tuple]: A list of rows.

        Raises:
            google.cloud.bigquery.dbapi.InterfaceError: if called before ``execute()``.
        """
        if size is None:
            # Since self.arraysize can be None (a deviation from PEP 249),
            # use an actual PEP 249 default of 1 in such case (*some* number
            # is needed here).
            size = self.arraysize if self.arraysize else 1

        self._try_fetch(size=size)
        rows = []

        for row in self._query_data:
            rows.append(row)
            if len(rows) >= size:
                break

        return rows

    def fetchall(self):
        """Fetch all remaining results from the last ``execute*()`` call.

        .. note::
            If a dry run query was executed, no rows are returned.

        Returns:
            List[Tuple]: A list of all the rows in the results.

        Raises:
            google.cloud.bigquery.dbapi.InterfaceError: if called before ``execute()``.
        """
        self._try_fetch()
        return list(self._query_data)

    def setinputsizes(self, sizes):
        """No-op, but for consistency raise an error if cursor is closed."""

    def setoutputsize(self, size, column=None):
        """No-op, but for consistency raise an error if cursor is closed."""

    def __iter__(self):
        self._try_fetch()
        return iter(self._query_data)


def _format_operation_list(operation, parameters):
    """Formats parameters in operation in the way BigQuery expects.

    The input operation will be a query like ``SELECT %s`` and the output
    will be a query like ``SELECT ?``.

    Args:
        operation (str): A Google BigQuery query string.

        parameters (Sequence[Any]): Sequence of parameter values.

    Returns:
        str: A formatted query string.

    Raises:
        google.cloud.bigquery.dbapi.ProgrammingError:
            if a parameter used in the operation is not found in the
            ``parameters`` argument.
    """
    formatted_params = ["?" for _ in parameters]

    try:
        return operation % tuple(formatted_params)
    except (TypeError, ValueError) as exc:
        raise exceptions.ProgrammingError(exc)


def _format_operation_dict(operation, parameters):
    """Formats parameters in operation in the way BigQuery expects.

    The input operation will be a query like ``SELECT %(namedparam)s`` and
    the output will be a query like ``SELECT @namedparam``.

    Args:
        operation (str): A Google BigQuery query string.

        parameters (Mapping[str, Any]): Dictionary of parameter values.

    Returns:
        str: A formatted query string.

    Raises:
        google.cloud.bigquery.dbapi.ProgrammingError:
            if a parameter used in the operation is not found in the
            ``parameters`` argument.
    """
    formatted_params = {}
    for name in parameters:
        escaped_name = name.replace("`", r"\`")
        formatted_params[name] = "@`{}`".format(escaped_name)

    try:
        return operation % formatted_params
    except (KeyError, ValueError, TypeError) as exc:
        raise exceptions.ProgrammingError(exc)


def _format_operation(operation, parameters):
    """Formats parameters in operation in way BigQuery expects.

    Args:
        operation (str): A Google BigQuery query string.

        parameters (Union[Mapping[str, Any], Sequence[Any]]):
            Optional parameter values.

    Returns:
        str: A formatted query string.

    Raises:
        google.cloud.bigquery.dbapi.ProgrammingError:
            if a parameter used in the operation is not found in the
            ``parameters`` argument.
    """
    if parameters is None or len(parameters) == 0:
        return operation.replace("%%", "%"), None  # Still do percent de-escaping.

    operation, parameter_types = _extract_types(operation)
    if parameter_types is None:
        raise exceptions.ProgrammingError(
            f"Parameters were provided, but {repr(operation)} has no placeholders."
        )

    if isinstance(parameters, collections_abc.Mapping):
        return _format_operation_dict(operation, parameters), parameter_types

    return _format_operation_list(operation, parameters), parameter_types


def _extract_types(
    operation,
    extra_type_sub=re.compile(
        r"""
        (%*)          # Extra %s.  We'll deal with these in the replacement code

        %             # Beginning of replacement, %s, %(...)s

        (?:\(         # Begin of optional name and/or type
        ([^:)]*)      # name
        (?::          # ':' introduces type
          (             # start of type group
            [a-zA-Z0-9_<>, ]+ # First part, no parens

            (?:               # start sets of parens + non-paren text
              \([0-9 ,]+\)      # comma-separated groups of digits in parens
                                # (e.g. string(10))
              (?=[, >)])        # Must be followed by ,>) or space
              [a-zA-Z0-9<>, ]*  # Optional non-paren chars
            )*                # Can be zero or more of parens and following text
          )             # end of type group
        )?            # close type clause ":type"
        \))?          # End of optional name and/or type

        s             # End of replacement
        """,
        re.VERBOSE,
    ).sub,
):
    """Remove type information from parameter placeholders.

    For every parameter of the form %(name:type)s, replace with %(name)s and add the
    item name->type to dict that's returned.

    Returns operation without type information and a dictionary of names and types.
    """
    parameter_types = None

    def repl(m):
        nonlocal parameter_types
        prefix, name, type_ = m.groups()
        if len(prefix) % 2:
            # The prefix has an odd number of %s, the last of which
            # escapes the % we're looking for, so we don't want to
            # change anything.
            return m.group(0)

        try:
            if name:
                if not parameter_types:
                    parameter_types = {}
                if type_:
                    if name in parameter_types:
                        if type_ != parameter_types[name]:
                            raise exceptions.ProgrammingError(
                                f"Conflicting types for {name}: "
                                f"{parameter_types[name]} and {type_}."
                            )
                    else:
                        parameter_types[name] = type_
                else:
                    if not isinstance(parameter_types, dict):
                        raise TypeError()

                return f"{prefix}%({name})s"
            else:
                if parameter_types is None:
                    parameter_types = []
                parameter_types.append(type_)
                return f"{prefix}%s"
        except (AttributeError, TypeError):
            raise exceptions.ProgrammingError(
                f"{repr(operation)} mixes named and unamed parameters."
            )

    return extra_type_sub(repl, operation), parameter_types
