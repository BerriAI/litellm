# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for interacting with the job REST APIs from the client.

For queries, there are three cases to consider:

1. jobs.insert: This always returns a job resource.
2. jobs.query, jobCreationMode=JOB_CREATION_REQUIRED:
   This sometimes can return the results inline, but always includes a job ID.
3. jobs.query, jobCreationMode=JOB_CREATION_OPTIONAL:
   This sometimes doesn't create a job at all, instead returning the results.
   For better debugging, an auto-generated query ID is included in the
   response.

Client.query() calls either (1) or (2), depending on what the user provides
for the api_method parameter. query() always returns a QueryJob object, which
can retry the query when the query job fails for a retriable reason.

Client.query_and_wait() calls (3). This returns a RowIterator that may wrap
local results from the response or may wrap a query job containing multiple
pages of results. Even though query_and_wait() waits for the job to complete,
we still need a separate job_retry object because there are different
predicates where it is safe to generate a new query ID.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import functools
import uuid
import textwrap
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING, Union
import warnings

import google.api_core.exceptions as core_exceptions
from google.api_core import retry as retries

from google.cloud.bigquery import job
import google.cloud.bigquery.job.query
import google.cloud.bigquery.query
from google.cloud.bigquery import table
import google.cloud.bigquery.retry
from google.cloud.bigquery.retry import POLLING_DEFAULT_VALUE

# Avoid circular imports
if TYPE_CHECKING:  # pragma: NO COVER
    from google.cloud.bigquery.client import Client


# The purpose of _TIMEOUT_BUFFER_MILLIS is to allow the server-side timeout to
# happen before the client-side timeout. This is not strictly necessary, as the
# client retries client-side timeouts, but the hope by making the server-side
# timeout slightly shorter is that it can save the server from some unncessary
# processing time.
#
# 250 milliseconds is chosen arbitrarily, though should be about the right
# order of magnitude for network latency and switching delays. It is about the
# amount of time for light to circumnavigate the world twice.
_TIMEOUT_BUFFER_MILLIS = 250


def make_job_id(job_id: Optional[str] = None, prefix: Optional[str] = None) -> str:
    """Construct an ID for a new job.

    Args:
        job_id: the user-provided job ID.
        prefix: the user-provided prefix for a job ID.

    Returns:
        str: A job ID
    """
    if job_id is not None:
        return job_id
    elif prefix is not None:
        return str(prefix) + str(uuid.uuid4())
    else:
        return str(uuid.uuid4())


def job_config_with_defaults(
    job_config: Optional[job.QueryJobConfig],
    default_job_config: Optional[job.QueryJobConfig],
) -> Optional[job.QueryJobConfig]:
    """Create a copy of `job_config`, replacing unset values with those from
    `default_job_config`.
    """
    if job_config is None:
        return default_job_config

    if default_job_config is None:
        return job_config

    # Both job_config and default_job_config are not None, so make a copy of
    # job_config merged with default_job_config. Anything already explicitly
    # set on job_config should not be replaced.
    return job_config._fill_from_default(default_job_config)


def query_jobs_insert(
    client: "Client",
    query: str,
    job_config: Optional[job.QueryJobConfig],
    job_id: Optional[str],
    job_id_prefix: Optional[str],
    location: Optional[str],
    project: str,
    retry: Optional[retries.Retry],
    timeout: Optional[float],
    job_retry: Optional[retries.Retry],
    *,
    callback: Callable = lambda _: None,
) -> job.QueryJob:
    """Initiate a query using jobs.insert.

    See: https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/insert

    Args:
        callback (Callable):
            A callback function used by bigframes to report query progress.
    """
    job_id_given = job_id is not None
    job_id_save = job_id
    job_config_save = job_config
    query_sent_factory = QuerySentEventFactory()

    def do_query():
        # Make a copy now, so that original doesn't get changed by the process
        # below and to facilitate retry
        job_config = copy.deepcopy(job_config_save)

        job_id = make_job_id(job_id_save, job_id_prefix)
        job_ref = job._JobReference(job_id, project=project, location=location)
        query_job = job.QueryJob(job_ref, query, client=client, job_config=job_config)

        try:
            query_job._begin(retry=retry, timeout=timeout)
            if job_config is not None and not job_config.dry_run:
                callback(
                    query_sent_factory(
                        query=query,
                        billing_project=query_job.project,
                        location=query_job.location,
                        job_id=query_job.job_id,
                        request_id=None,
                    )
                )
        except core_exceptions.Conflict as create_exc:
            # The thought is if someone is providing their own job IDs and they get
            # their job ID generation wrong, this could end up returning results for
            # the wrong query. We thus only try to recover if job ID was not given.
            if job_id_given:
                raise create_exc

            try:
                # Sometimes we get a 404 after a Conflict. In this case, we
                # have pretty high confidence that by retrying the 404, we'll
                # (hopefully) eventually recover the job.
                # https://github.com/googleapis/python-bigquery/issues/2134
                #
                # Allow users who want to completely disable retries to
                # continue to do so by setting retry to None.
                get_job_retry = retry
                if retry is not None:
                    # TODO(tswast): Amend the user's retry object with allowing
                    # 404 to retry when there's a public way to do so.
                    # https://github.com/googleapis/python-api-core/issues/796
                    get_job_retry = (
                        google.cloud.bigquery.retry._DEFAULT_GET_JOB_CONFLICT_RETRY
                    )

                query_job = client.get_job(
                    job_id,
                    project=project,
                    location=location,
                    retry=get_job_retry,
                    timeout=google.cloud.bigquery.retry.DEFAULT_GET_JOB_TIMEOUT,
                )
            except core_exceptions.GoogleAPIError:  # (includes RetryError)
                raise
            else:
                return query_job
        else:
            return query_job

    # Allow users who want to completely disable retries to
    # continue to do so by setting job_retry to None.
    if job_retry is not None:
        do_query = google.cloud.bigquery.retry._DEFAULT_QUERY_JOB_INSERT_RETRY(do_query)

    future = do_query()

    # The future might be in a failed state now, but if it's
    # unrecoverable, we'll find out when we ask for it's result, at which
    # point, we may retry.
    if not job_id_given:
        future._retry_do_query = do_query  # in case we have to retry later
        future._job_retry = job_retry

    return future


def _validate_job_config(request_body: Dict[str, Any], invalid_key: str):
    """Catch common mistakes, such as passing in a *JobConfig object of the
    wrong type.
    """
    if invalid_key in request_body:
        raise ValueError(f"got unexpected key {repr(invalid_key)} in job_config")


def validate_job_retry(job_id: Optional[str], job_retry: Optional[retries.Retry]):
    """Catch common mistakes, such as setting a job_id and job_retry at the same
    time.
    """
    if job_id is not None and job_retry is not None:
        # TODO(tswast): To avoid breaking changes but still allow a default
        # query job retry, we currently only raise if they explicitly set a
        # job_retry other than the default. In a future version, we may want to
        # avoid this check for DEFAULT_JOB_RETRY and always raise.
        if job_retry is not google.cloud.bigquery.retry.DEFAULT_JOB_RETRY:
            raise TypeError(
                textwrap.dedent(
                    """
                    `job_retry` was provided, but the returned job is
                    not retryable, because a custom `job_id` was
                    provided. To customize the job ID and allow for job
                    retries, set job_id_prefix, instead.
                    """
                ).strip()
            )
        else:
            warnings.warn(
                textwrap.dedent(
                    """
                    job_retry must be explicitly set to None if job_id is set.
                    BigQuery cannot retry a failed job by using the exact
                    same ID. Setting job_id without explicitly disabling
                    job_retry will raise an error in the future. To avoid this
                    warning, either use job_id_prefix instead (preferred) or
                    set job_retry=None.
                    """
                ).strip(),
                category=FutureWarning,
                # user code -> client.query / client.query_and_wait -> validate_job_retry
                stacklevel=3,
            )


def _to_query_request(
    job_config: Optional[job.QueryJobConfig] = None,
    *,
    query: str,
    location: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Transform from Job resource to QueryRequest resource.

    Most of the keys in job.configuration.query are in common with
    QueryRequest. If any configuration property is set that is not available in
    jobs.query, it will result in a server-side error.
    """
    request_body = copy.copy(job_config.to_api_repr()) if job_config else {}

    _validate_job_config(request_body, job.CopyJob._JOB_TYPE)
    _validate_job_config(request_body, job.ExtractJob._JOB_TYPE)
    _validate_job_config(request_body, job.LoadJob._JOB_TYPE)

    # Move query.* properties to top-level.
    query_config_resource = request_body.pop("query", {})
    request_body.update(query_config_resource)

    # Default to standard SQL.
    request_body.setdefault("useLegacySql", False)

    # Since jobs.query can return results, ensure we use the lossless timestamp
    # format. See: https://github.com/googleapis/python-bigquery/issues/395
    request_body.setdefault("formatOptions", {})
    request_body["formatOptions"]["useInt64Timestamp"] = True  # type: ignore

    if timeout is not None:
        # Subtract a buffer for context switching, network latency, etc.
        request_body["timeoutMs"] = max(0, int(1000 * timeout) - _TIMEOUT_BUFFER_MILLIS)

    if location is not None:
        request_body["location"] = location

    request_body["query"] = query

    return request_body


def _to_query_job(
    client: "Client",
    query: str,
    request_config: Optional[job.QueryJobConfig],
    query_response: Dict[str, Any],
) -> job.QueryJob:
    job_ref_resource = query_response["jobReference"]
    job_ref = job._JobReference._from_api_repr(job_ref_resource)
    query_job = job.QueryJob(job_ref, query, client=client)
    query_job._properties.setdefault("configuration", {})

    # Not all relevant properties are in the jobs.query response. Populate some
    # expected properties based on the job configuration.
    if request_config is not None:
        query_job._properties["configuration"].update(request_config.to_api_repr())

    query_job._properties["configuration"].setdefault("query", {})
    query_job._properties["configuration"]["query"]["query"] = query
    query_job._properties["configuration"]["query"].setdefault("useLegacySql", False)

    query_job._properties.setdefault("statistics", {})
    query_job._properties["statistics"].setdefault("query", {})
    query_job._properties["statistics"]["query"]["cacheHit"] = query_response.get(
        "cacheHit"
    )
    query_job._properties["statistics"]["query"]["schema"] = query_response.get(
        "schema"
    )
    query_job._properties["statistics"]["query"][
        "totalBytesProcessed"
    ] = query_response.get("totalBytesProcessed")

    # Set errors if any were encountered.
    query_job._properties.setdefault("status", {})
    if "errors" in query_response:
        # Set errors but not errorResult. If there was an error that failed
        # the job, jobs.query behaves like jobs.getQueryResults and returns a
        # non-success HTTP status code.
        errors = query_response["errors"]
        query_job._properties["status"]["errors"] = errors

    # Avoid an extra call to `getQueryResults` if the query has finished.
    job_complete = query_response.get("jobComplete")
    if job_complete:
        query_job._query_results = google.cloud.bigquery.query._QueryResults(
            query_response
        )

    # We want job.result() to refresh the job state, so the conversion is
    # always "PENDING", even if the job is finished.
    query_job._properties["status"]["state"] = "PENDING"

    return query_job


def _to_query_path(project: str) -> str:
    return f"/projects/{project}/queries"


def query_jobs_query(
    client: "Client",
    query: str,
    job_config: Optional[job.QueryJobConfig],
    location: Optional[str],
    project: str,
    retry: retries.Retry,
    timeout: Optional[float],
    job_retry: Optional[retries.Retry],
) -> job.QueryJob:
    """Initiate a query using jobs.query with jobCreationMode=JOB_CREATION_REQUIRED.

    See: https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query
    """
    path = _to_query_path(project)
    request_body = _to_query_request(
        query=query, job_config=job_config, location=location, timeout=timeout
    )

    def do_query():
        request_body["requestId"] = make_job_id()
        span_attributes = {"path": path}
        api_response = client._call_api(
            retry,
            span_name="BigQuery.query",
            span_attributes=span_attributes,
            method="POST",
            path=path,
            data=request_body,
            timeout=timeout,
        )
        return _to_query_job(client, query, job_config, api_response)

    future = do_query()

    # The future might be in a failed state now, but if it's
    # unrecoverable, we'll find out when we ask for it's result, at which
    # point, we may retry.
    future._retry_do_query = do_query  # in case we have to retry later
    future._job_retry = job_retry

    return future


def query_and_wait(
    client: "Client",
    query: str,
    *,
    job_config: Optional[job.QueryJobConfig],
    location: Optional[str],
    project: str,
    api_timeout: Optional[float] = None,
    wait_timeout: Optional[Union[float, object]] = POLLING_DEFAULT_VALUE,
    retry: Optional[retries.Retry],
    job_retry: Optional[retries.Retry],
    page_size: Optional[int] = None,
    max_results: Optional[int] = None,
    callback: Callable = lambda _: None,
) -> table.RowIterator:
    """Run the query, wait for it to finish, and return the results.


    Args:
        client:
            BigQuery client to make API calls.
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
        project (str):
            Project ID of the project of where to run the job.
        api_timeout (Optional[float]):
            The number of seconds to wait for the underlying HTTP transport
            before using ``retry``.
        wait_timeout (Optional[Union[float, object]]):
            The number of seconds to wait for the query to finish. If the
            query doesn't finish before this timeout, the client attempts
            to cancel the query. If unset, the underlying Client.get_job() API
            call has timeout, but we still wait indefinitely for the job to
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
            The maximum number of rows in each page of results from this
            request. Non-positive values are ignored.
        max_results (Optional[int]):
            The maximum total number of rows from this request.
        callback (Callable):
            A callback function used by bigframes to report query progress.

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
    request_body = _to_query_request(
        query=query, job_config=job_config, location=location, timeout=api_timeout
    )

    # Some API parameters aren't supported by the jobs.query API. In these
    # cases, fallback to a jobs.insert call.
    if not _supported_by_jobs_query(request_body):
        return _wait_or_cancel(
            query_jobs_insert(
                client=client,
                query=query,
                job_id=None,
                job_id_prefix=None,
                job_config=job_config,
                location=location,
                project=project,
                retry=retry,
                timeout=api_timeout,
                job_retry=job_retry,
                callback=callback,
            ),
            api_timeout=api_timeout,
            wait_timeout=wait_timeout,
            retry=retry,
            page_size=page_size,
            max_results=max_results,
            callback=callback,
        )

    path = _to_query_path(project)

    if page_size is not None and max_results is not None:
        request_body["maxResults"] = min(page_size, max_results)
    elif page_size is not None or max_results is not None:
        request_body["maxResults"] = page_size or max_results
    if client.default_job_creation_mode:
        request_body["jobCreationMode"] = client.default_job_creation_mode

    query_sent_factory = QuerySentEventFactory()

    def do_query():
        request_id = make_job_id()
        request_body["requestId"] = request_id
        span_attributes = {"path": path}

        if "dryRun" not in request_body:
            callback(
                query_sent_factory(
                    query=query,
                    billing_project=project,
                    location=location,
                    job_id=None,
                    request_id=request_id,
                )
            )

        # For easier testing, handle the retries ourselves.
        if retry is not None:
            response = retry(client._call_api)(
                retry=None,  # We're calling the retry decorator ourselves.
                span_name="BigQuery.query",
                span_attributes=span_attributes,
                method="POST",
                path=path,
                data=request_body,
                timeout=api_timeout,
            )
        else:
            response = client._call_api(
                retry=None,
                span_name="BigQuery.query",
                span_attributes=span_attributes,
                method="POST",
                path=path,
                data=request_body,
                timeout=api_timeout,
            )

        # Even if we run with JOB_CREATION_OPTIONAL, if there are more pages
        # to fetch, there will be a job ID for jobs.getQueryResults.
        query_results = google.cloud.bigquery.query._QueryResults.from_api_repr(
            response
        )
        page_token = query_results.page_token
        more_pages = page_token is not None

        if more_pages or not query_results.complete:
            # TODO(swast): Avoid a call to jobs.get in some cases (few
            # remaining pages) by waiting for the query to finish and calling
            # client._list_rows_from_query_results directly. Need to update
            # RowIterator to fetch destination table via the job ID if needed.
            return _wait_or_cancel(
                _to_query_job(client, query, job_config, response),
                api_timeout=api_timeout,
                wait_timeout=wait_timeout,
                retry=retry,
                page_size=page_size,
                max_results=max_results,
                callback=callback,
            )

        if "dryRun" not in request_body:
            callback(
                QueryFinishedEvent(
                    billing_project=project,
                    location=query_results.location,
                    query_id=query_results.query_id,
                    job_id=query_results.job_id,
                    total_rows=query_results.total_rows,
                    total_bytes_processed=query_results.total_bytes_processed,
                    slot_millis=query_results.slot_millis,
                    destination=None,
                    created=query_results.created,
                    started=query_results.started,
                    ended=query_results.ended,
                )
            )
        return table.RowIterator(
            client=client,
            api_request=functools.partial(client._call_api, retry, timeout=api_timeout),
            path=None,
            schema=query_results.schema,
            max_results=max_results,
            page_size=page_size,
            total_rows=query_results.total_rows,
            first_page_response=response,
            location=query_results.location,
            job_id=query_results.job_id,
            query_id=query_results.query_id,
            project=query_results.project,
            num_dml_affected_rows=query_results.num_dml_affected_rows,
            query=query,
            total_bytes_processed=query_results.total_bytes_processed,
            slot_millis=query_results.slot_millis,
            created=query_results.created,
            started=query_results.started,
            ended=query_results.ended,
        )

    if job_retry is not None:
        return job_retry(do_query)()
    else:
        return do_query()


def _supported_by_jobs_query(request_body: Dict[str, Any]) -> bool:
    """True if jobs.query can be used. False if jobs.insert is needed."""
    request_keys = frozenset(request_body.keys())

    # Per issue: https://github.com/googleapis/python-bigquery/issues/1867
    # use an allowlist here instead of a denylist because the backend API allows
    # unsupported parameters without any warning or failure. Instead, keep this
    # set in sync with those in QueryRequest:
    # https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query#QueryRequest
    keys_allowlist = {
        "kind",
        "query",
        "maxResults",
        "defaultDataset",
        "timeoutMs",
        "dryRun",
        "preserveNulls",
        "useQueryCache",
        "useLegacySql",
        "parameterMode",
        "queryParameters",
        "location",
        "formatOptions",
        "connectionProperties",
        "labels",
        "maximumBytesBilled",
        "requestId",
        "createSession",
        "writeIncrementalResults",
        "jobTimeoutMs",
        "reservation",
        "maxSlots",
    }

    unsupported_keys = request_keys - keys_allowlist
    return len(unsupported_keys) == 0


def _wait_or_cancel(
    job: job.QueryJob,
    api_timeout: Optional[float],
    wait_timeout: Optional[Union[object, float]],
    retry: Optional[retries.Retry],
    page_size: Optional[int],
    max_results: Optional[int],
    *,
    callback: Callable = lambda _: None,
) -> table.RowIterator:
    """Wait for a job to complete and return the results.

    If we can't return the results within the ``wait_timeout``, try to cancel
    the job.
    """
    try:
        if not job.dry_run:
            callback(
                QueryReceivedEvent(
                    billing_project=job.project,
                    location=job.location,
                    job_id=job.job_id,
                    statement_type=job.statement_type,
                    state=job.state,
                    query_plan=job.query_plan,
                    created=job.created,
                    started=job.started,
                    ended=job.ended,
                )
            )
        query_results = job.result(
            page_size=page_size,
            max_results=max_results,
            retry=retry,
            timeout=wait_timeout,
        )
        if not job.dry_run:
            callback(
                QueryFinishedEvent(
                    billing_project=job.project,
                    location=query_results.location,
                    query_id=query_results.query_id,
                    job_id=query_results.job_id,
                    total_rows=query_results.total_rows,
                    total_bytes_processed=query_results.total_bytes_processed,
                    slot_millis=query_results.slot_millis,
                    destination=job.destination,
                    created=job.created,
                    started=job.started,
                    ended=job.ended,
                )
            )
        return query_results
    except Exception:
        # Attempt to cancel the job since we can't return the results.
        try:
            job.cancel(retry=retry, timeout=api_timeout)
        except Exception:
            # Don't eat the original exception if cancel fails.
            pass
        raise


@dataclasses.dataclass(frozen=True)
class QueryFinishedEvent:
    """Query finished successfully."""

    billing_project: Optional[str]
    location: Optional[str]
    query_id: Optional[str]
    job_id: Optional[str]
    destination: Optional[table.TableReference]
    total_rows: Optional[int]
    total_bytes_processed: Optional[int]
    slot_millis: Optional[int]
    created: Optional[datetime.datetime]
    started: Optional[datetime.datetime]
    ended: Optional[datetime.datetime]


@dataclasses.dataclass(frozen=True)
class QueryReceivedEvent:
    """Query received and acknowledged by the BigQuery API."""

    billing_project: Optional[str]
    location: Optional[str]
    job_id: Optional[str]
    statement_type: Optional[str]
    state: Optional[str]
    query_plan: Optional[list[google.cloud.bigquery.job.query.QueryPlanEntry]]
    created: Optional[datetime.datetime]
    started: Optional[datetime.datetime]
    ended: Optional[datetime.datetime]


@dataclasses.dataclass(frozen=True)
class QuerySentEvent:
    """Query sent to BigQuery."""

    query: str
    billing_project: Optional[str]
    location: Optional[str]
    job_id: Optional[str]
    request_id: Optional[str]


class QueryRetryEvent(QuerySentEvent):
    """Query sent another time because the previous attempt failed."""


class QuerySentEventFactory:
    """Creates a QuerySentEvent first, then QueryRetryEvent after that."""

    def __init__(self):
        self._event_constructor = QuerySentEvent

    def __call__(self, **kwargs):
        result = self._event_constructor(**kwargs)
        self._event_constructor = QueryRetryEvent
        return result
