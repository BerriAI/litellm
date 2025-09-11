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

"""Base classes and helpers for job classes."""

from collections import namedtuple
import copy
import http
import threading
import typing
from typing import ClassVar, Dict, Optional, Sequence

from google.api_core import retry as retries
from google.api_core import exceptions
import google.api_core.future.polling

from google.cloud.bigquery import _helpers
from google.cloud.bigquery._helpers import _int_or_none
from google.cloud.bigquery.retry import (
    DEFAULT_GET_JOB_TIMEOUT,
    DEFAULT_RETRY,
)


_DONE_STATE = "DONE"
_STOPPED_REASON = "stopped"
_ERROR_REASON_TO_EXCEPTION = {
    "accessDenied": http.client.FORBIDDEN,
    "backendError": http.client.INTERNAL_SERVER_ERROR,
    "billingNotEnabled": http.client.FORBIDDEN,
    "billingTierLimitExceeded": http.client.BAD_REQUEST,
    "blocked": http.client.FORBIDDEN,
    "duplicate": http.client.CONFLICT,
    "internalError": http.client.INTERNAL_SERVER_ERROR,
    "invalid": http.client.BAD_REQUEST,
    "invalidQuery": http.client.BAD_REQUEST,
    "notFound": http.client.NOT_FOUND,
    "notImplemented": http.client.NOT_IMPLEMENTED,
    "policyViolation": http.client.FORBIDDEN,
    "quotaExceeded": http.client.FORBIDDEN,
    "rateLimitExceeded": http.client.TOO_MANY_REQUESTS,
    "resourceInUse": http.client.BAD_REQUEST,
    "resourcesExceeded": http.client.BAD_REQUEST,
    "responseTooLarge": http.client.FORBIDDEN,
    "stopped": http.client.OK,
    "tableUnavailable": http.client.BAD_REQUEST,
}


def _error_result_to_exception(error_result, errors=None):
    """Maps BigQuery error reasons to an exception.

    The reasons and their matching HTTP status codes are documented on
    the `troubleshooting errors`_ page.

    .. _troubleshooting errors: https://cloud.google.com/bigquery\
        /troubleshooting-errors

    Args:
        error_result (Mapping[str, str]): The error result from BigQuery.
        errors (Union[Iterable[str], None]): The detailed error messages.

    Returns:
        google.cloud.exceptions.GoogleAPICallError: The mapped exception.
    """
    reason = error_result.get("reason")
    status_code = _ERROR_REASON_TO_EXCEPTION.get(
        reason, http.client.INTERNAL_SERVER_ERROR
    )
    # Manually create error message to preserve both error_result and errors.
    # Can be removed once b/310544564 and b/318889899 are resolved.
    concatenated_errors = ""
    if errors:
        concatenated_errors = "; "
        for err in errors:
            concatenated_errors += ", ".join(
                [f"{key}: {value}" for key, value in err.items()]
            )
            concatenated_errors += "; "

        # strips off the last unneeded semicolon and space
        concatenated_errors = concatenated_errors[:-2]

    error_message = error_result.get("message", "") + concatenated_errors

    return exceptions.from_http_status(
        status_code, error_message, errors=[error_result]
    )


ReservationUsage = namedtuple("ReservationUsage", "name slot_ms")
ReservationUsage.__doc__ = "Job resource usage for a reservation."
ReservationUsage.name.__doc__ = (
    'Reservation name or "unreserved" for on-demand resources usage.'
)
ReservationUsage.slot_ms.__doc__ = (
    "Total slot milliseconds used by the reservation for a particular job."
)


class TransactionInfo(typing.NamedTuple):
    """[Alpha] Information of a multi-statement transaction.

    https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#TransactionInfo

    .. versionadded:: 2.24.0
    """

    transaction_id: str
    """Output only. ID of the transaction."""

    @classmethod
    def from_api_repr(cls, transaction_info: Dict[str, str]) -> "TransactionInfo":
        return cls(transaction_info["transactionId"])


class _JobReference(object):
    """A reference to a job.

    Args:
        job_id (str): ID of the job to run.
        project (str): ID of the project where the job runs.
        location (str): Location of where the job runs.
    """

    def __init__(self, job_id, project, location):
        self._properties = {"jobId": job_id, "projectId": project}
        # The location field must not be populated if it is None.
        if location:
            self._properties["location"] = location

    @property
    def job_id(self):
        """str: ID of the job."""
        return self._properties.get("jobId")

    @property
    def project(self):
        """str: ID of the project where the job runs."""
        return self._properties.get("projectId")

    @property
    def location(self):
        """str: Location where the job runs."""
        return self._properties.get("location")

    def _to_api_repr(self):
        """Returns the API resource representation of the job reference."""
        return copy.deepcopy(self._properties)

    @classmethod
    def _from_api_repr(cls, resource):
        """Returns a job reference for an API resource representation."""
        job_id = resource.get("jobId")
        project = resource.get("projectId")
        location = resource.get("location")
        job_ref = cls(job_id, project, location)
        return job_ref


class _JobConfig(object):
    """Abstract base class for job configuration objects.

    Args:
        job_type (str): The key to use for the job configuration.
    """

    def __init__(self, job_type, **kwargs):
        self._job_type = job_type
        self._properties = {job_type: {}}
        for prop, val in kwargs.items():
            setattr(self, prop, val)

    def __setattr__(self, name, value):
        """Override to be able to raise error if an unknown property is being set"""
        if not name.startswith("_") and not hasattr(type(self), name):
            raise AttributeError(
                "Property {} is unknown for {}.".format(name, type(self))
            )
        super(_JobConfig, self).__setattr__(name, value)

    @property
    def job_timeout_ms(self):
        """Optional parameter. Job timeout in milliseconds. If this time limit is exceeded, BigQuery might attempt to stop the job.
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfiguration.FIELDS.job_timeout_ms
        e.g.

            job_config = bigquery.QueryJobConfig( job_timeout_ms = 5000 )
            or
            job_config.job_timeout_ms = 5000

        Raises:
            ValueError: If ``value`` type is invalid.
        """

        # None as this is an optional parameter.
        if self._properties.get("jobTimeoutMs"):
            return self._properties["jobTimeoutMs"]
        return None

    @job_timeout_ms.setter
    def job_timeout_ms(self, value):
        try:
            value = _int_or_none(value)
        except ValueError as err:
            raise ValueError("Pass an int for jobTimeoutMs, e.g. 5000").with_traceback(
                err.__traceback__
            )

        if value is not None:
            # docs indicate a string is expected by the API
            self._properties["jobTimeoutMs"] = str(value)
        else:
            self._properties.pop("jobTimeoutMs", None)

    @property
    def max_slots(self) -> Optional[int]:
        """The maximum rate of slot consumption to allow for this job.

        If set, the number of slots used to execute the job will be throttled
        to try and keep its slot consumption below the requested rate.
        This feature is not generally available.
        """

        max_slots = self._properties.get("maxSlots")
        if max_slots is not None:
            if isinstance(max_slots, str):
                return int(max_slots)
            if isinstance(max_slots, int):
                return max_slots
        return None

    @max_slots.setter
    def max_slots(self, value):
        try:
            value = _int_or_none(value)
        except ValueError as err:
            raise ValueError("Pass an int for max slots, e.g. 100").with_traceback(
                err.__traceback__
            )

        if value is not None:
            self._properties["maxSlots"] = str(value)
        else:
            self._properties.pop("maxSlots", None)

    @property
    def reservation(self):
        """str: Optional. The reservation that job would use.

        User can specify a reservation to execute the job. If reservation is
        not set, reservation is determined based on the rules defined by the
        reservation assignments. The expected format is
        projects/{project}/locations/{location}/reservations/{reservation}.

        Raises:
            ValueError: If ``value`` type is not None or of string type.
        """
        return self._properties.setdefault("reservation", None)

    @reservation.setter
    def reservation(self, value):
        if value and not isinstance(value, str):
            raise ValueError("Reservation must be None or a string.")
        self._properties["reservation"] = value

    @property
    def labels(self):
        """Dict[str, str]: Labels for the job.

        This method always returns a dict. Once a job has been created on the
        server, its labels cannot be modified anymore.

        Raises:
            ValueError: If ``value`` type is invalid.
        """
        return self._properties.setdefault("labels", {})

    @labels.setter
    def labels(self, value):
        if not isinstance(value, dict):
            raise ValueError("Pass a dict")
        self._properties["labels"] = value

    def _get_sub_prop(self, key, default=None):
        """Get a value in the ``self._properties[self._job_type]`` dictionary.

        Most job properties are inside the dictionary related to the job type
        (e.g. 'copy', 'extract', 'load', 'query'). Use this method to access
        those properties::

            self._get_sub_prop('destinationTable')

        This is equivalent to using the ``_helpers._get_sub_prop`` function::

            _helpers._get_sub_prop(
                self._properties, ['query', 'destinationTable'])

        Args:
            key (str):
                Key for the value to get in the
                ``self._properties[self._job_type]`` dictionary.
            default (Optional[object]):
                Default value to return if the key is not found.
                Defaults to :data:`None`.

        Returns:
            object: The value if present or the default.
        """
        return _helpers._get_sub_prop(
            self._properties, [self._job_type, key], default=default
        )

    def _set_sub_prop(self, key, value):
        """Set a value in the ``self._properties[self._job_type]`` dictionary.

        Most job properties are inside the dictionary related to the job type
        (e.g. 'copy', 'extract', 'load', 'query'). Use this method to set
        those properties::

            self._set_sub_prop('useLegacySql', False)

        This is equivalent to using the ``_helper._set_sub_prop`` function::

            _helper._set_sub_prop(
                self._properties, ['query', 'useLegacySql'], False)

        Args:
            key (str):
                Key to set in the ``self._properties[self._job_type]``
                dictionary.
            value (object): Value to set.
        """
        _helpers._set_sub_prop(self._properties, [self._job_type, key], value)

    def _del_sub_prop(self, key):
        """Remove ``key`` from the ``self._properties[self._job_type]`` dict.

        Most job properties are inside the dictionary related to the job type
        (e.g. 'copy', 'extract', 'load', 'query'). Use this method to clear
        those properties::

            self._del_sub_prop('useLegacySql')

        This is equivalent to using the ``_helper._del_sub_prop`` function::

            _helper._del_sub_prop(
                self._properties, ['query', 'useLegacySql'])

        Args:
            key (str):
                Key to remove in the ``self._properties[self._job_type]``
                dictionary.
        """
        _helpers._del_sub_prop(self._properties, [self._job_type, key])

    def to_api_repr(self) -> dict:
        """Build an API representation of the job config.

        Returns:
            Dict: A dictionary in the format used by the BigQuery API.
        """
        return copy.deepcopy(self._properties)

    def _fill_from_default(self, default_job_config=None):
        """Merge this job config with a default job config.

        The keys in this object take precedence over the keys in the default
        config. The merge is done at the top-level as well as for keys one
        level below the job type.

        Args:
            default_job_config (google.cloud.bigquery.job._JobConfig):
                The default job config that will be used to fill in self.

        Returns:
            google.cloud.bigquery.job._JobConfig: A new (merged) job config.
        """
        if not default_job_config:
            new_job_config = copy.deepcopy(self)
            return new_job_config

        if self._job_type != default_job_config._job_type:
            raise TypeError(
                "attempted to merge two incompatible job types: "
                + repr(self._job_type)
                + ", "
                + repr(default_job_config._job_type)
            )

        # cls is one of the job config subclasses that provides the job_type argument to
        # this base class on instantiation, thus missing-parameter warning is a false
        # positive here.
        new_job_config = self.__class__()  # pytype: disable=missing-parameter

        default_job_properties = copy.deepcopy(default_job_config._properties)
        for key in self._properties:
            if key != self._job_type:
                default_job_properties[key] = self._properties[key]

        default_job_properties[self._job_type].update(self._properties[self._job_type])
        new_job_config._properties = default_job_properties

        return new_job_config

    @classmethod
    def from_api_repr(cls, resource: dict) -> "_JobConfig":
        """Factory: construct a job configuration given its API representation

        Args:
            resource (Dict):
                A job configuration in the same representation as is returned
                from the API.

        Returns:
            google.cloud.bigquery.job._JobConfig: Configuration parsed from ``resource``.
        """
        # cls is one of the job config subclasses that provides the job_type argument to
        # this base class on instantiation, thus missing-parameter warning is a false
        # positive here.
        job_config = cls()  # type: ignore  # pytype: disable=missing-parameter
        job_config._properties = resource
        return job_config


class _AsyncJob(google.api_core.future.polling.PollingFuture):
    """Base class for asynchronous jobs.

    Args:
        job_id (Union[str, _JobReference]):
            Job's ID in the project associated with the client or a
            fully-qualified job reference.
        client (google.cloud.bigquery.client.Client):
            Client which holds credentials and project configuration.
    """

    _JOB_TYPE = "unknown"
    _CONFIG_CLASS: ClassVar

    def __init__(self, job_id, client):
        super(_AsyncJob, self).__init__()

        # The job reference can be either a plain job ID or the full resource.
        # Populate the properties dictionary consistently depending on what has
        # been passed in.
        job_ref = job_id
        if not isinstance(job_id, _JobReference):
            job_ref = _JobReference(job_id, client.project, None)
        self._properties = {"jobReference": job_ref._to_api_repr()}

        self._client = client
        self._result_set = False
        self._completion_lock = threading.Lock()

    @property
    def configuration(self) -> _JobConfig:
        """Job-type specific configurtion."""
        configuration: _JobConfig = self._CONFIG_CLASS()  # pytype: disable=not-callable
        configuration._properties = self._properties.setdefault("configuration", {})
        return configuration

    @property
    def job_id(self):
        """str: ID of the job."""
        return _helpers._get_sub_prop(self._properties, ["jobReference", "jobId"])

    @property
    def parent_job_id(self):
        """Return the ID of the parent job.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatistics.FIELDS.parent_job_id

        Returns:
            Optional[str]: parent job id.
        """
        return _helpers._get_sub_prop(self._properties, ["statistics", "parentJobId"])

    @property
    def script_statistics(self) -> Optional["ScriptStatistics"]:
        """Statistics for a child job of a script."""
        resource = _helpers._get_sub_prop(
            self._properties, ["statistics", "scriptStatistics"]
        )
        if resource is None:
            return None
        return ScriptStatistics(resource)

    @property
    def session_info(self) -> Optional["SessionInfo"]:
        """[Preview] Information of the session if this job is part of one.

        .. versionadded:: 2.29.0
        """
        resource = _helpers._get_sub_prop(
            self._properties, ["statistics", "sessionInfo"]
        )
        if resource is None:
            return None
        return SessionInfo(resource)

    @property
    def num_child_jobs(self):
        """The number of child jobs executed.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatistics.FIELDS.num_child_jobs

        Returns:
            int
        """
        count = _helpers._get_sub_prop(self._properties, ["statistics", "numChildJobs"])
        return int(count) if count is not None else 0

    @property
    def project(self):
        """Project bound to the job.

        Returns:
            str: the project (derived from the client).
        """
        return _helpers._get_sub_prop(self._properties, ["jobReference", "projectId"])

    @property
    def location(self):
        """str: Location where the job runs."""
        return _helpers._get_sub_prop(self._properties, ["jobReference", "location"])

    @property
    def reservation_id(self):
        """str: Name of the primary reservation assigned to this job.

        Note that this could be different than reservations reported in
        the reservation field if parent reservations were used to execute
        this job.
        """
        return _helpers._get_sub_prop(
            self._properties, ["statistics", "reservation_id"]
        )

    def _require_client(self, client):
        """Check client or verify over-ride.

        Args:
            client (Optional[google.cloud.bigquery.client.Client]):
                the client to use.  If not passed, falls back to the
                ``client`` stored on the current dataset.

        Returns:
            google.cloud.bigquery.client.Client:
                The client passed in or the currently bound client.
        """
        if client is None:
            client = self._client
        return client

    @property
    def job_type(self):
        """Type of job.

        Returns:
            str: one of 'load', 'copy', 'extract', 'query'.
        """
        return self._JOB_TYPE

    @property
    def path(self):
        """URL path for the job's APIs.

        Returns:
            str: the path based on project and job ID.
        """
        return "/projects/%s/jobs/%s" % (self.project, self.job_id)

    @property
    def labels(self):
        """Dict[str, str]: Labels for the job."""
        return self._properties.setdefault("configuration", {}).setdefault("labels", {})

    @property
    def etag(self):
        """ETag for the job resource.

        Returns:
            Optional[str]: the ETag (None until set from the server).
        """
        return self._properties.get("etag")

    @property
    def self_link(self):
        """URL for the job resource.

        Returns:
            Optional[str]: the URL (None until set from the server).
        """
        return self._properties.get("selfLink")

    @property
    def user_email(self):
        """E-mail address of user who submitted the job.

        Returns:
            Optional[str]: the URL (None until set from the server).
        """
        return self._properties.get("user_email")

    @property
    def created(self):
        """Datetime at which the job was created.

        Returns:
            Optional[datetime.datetime]:
                the creation time (None until set from the server).
        """
        millis = _helpers._get_sub_prop(
            self._properties, ["statistics", "creationTime"]
        )
        if millis is not None:
            return _helpers._datetime_from_microseconds(millis * 1000.0)

    @property
    def started(self):
        """Datetime at which the job was started.

        Returns:
            Optional[datetime.datetime]:
                the start time (None until set from the server).
        """
        millis = _helpers._get_sub_prop(self._properties, ["statistics", "startTime"])
        if millis is not None:
            return _helpers._datetime_from_microseconds(millis * 1000.0)

    @property
    def ended(self):
        """Datetime at which the job finished.

        Returns:
            Optional[datetime.datetime]:
                the end time (None until set from the server).
        """
        millis = _helpers._get_sub_prop(self._properties, ["statistics", "endTime"])
        if millis is not None:
            return _helpers._datetime_from_microseconds(millis * 1000.0)

    def _job_statistics(self):
        """Helper for job-type specific statistics-based properties."""
        statistics = self._properties.get("statistics", {})
        return statistics.get(self._JOB_TYPE, {})

    @property
    def reservation_usage(self):
        """Job resource usage breakdown by reservation.

        Returns:
            List[google.cloud.bigquery.job.ReservationUsage]:
                Reservation usage stats. Can be empty if not set from the server.
        """
        usage_stats_raw = _helpers._get_sub_prop(
            self._properties, ["statistics", "reservationUsage"], default=()
        )
        return [
            ReservationUsage(name=usage["name"], slot_ms=int(usage["slotMs"]))
            for usage in usage_stats_raw
        ]

    @property
    def transaction_info(self) -> Optional[TransactionInfo]:
        """Information of the multi-statement transaction if this job is part of one.

        Since a scripting query job can execute multiple transactions, this
        property is only expected on child jobs. Use the
        :meth:`google.cloud.bigquery.client.Client.list_jobs` method with the
        ``parent_job`` parameter to iterate over child jobs.

        .. versionadded:: 2.24.0
        """
        info = self._properties.get("statistics", {}).get("transactionInfo")
        if info is None:
            return None
        else:
            return TransactionInfo.from_api_repr(info)

    @property
    def error_result(self):
        """Output only. Final error result of the job.

        If present, indicates that the job has completed and was unsuccessful.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatus.FIELDS.error_result

        Returns:
            Optional[Mapping]: the error information (None until set from the server).
        """
        status = self._properties.get("status")
        if status is not None:
            return status.get("errorResult")

    @property
    def errors(self):
        """Output only. The first errors encountered during the running of the job.

        The final message includes the number of errors that caused the process to stop.
        Errors here do not necessarily mean that the job has not completed or was unsuccessful.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatus.FIELDS.errors

        Returns:
            Optional[List[Mapping]]:
                the error information (None until set from the server).
        """
        status = self._properties.get("status")
        if status is not None:
            return status.get("errors")

    @property
    def state(self):
        """Output only. Running state of the job.

        Valid states include 'PENDING', 'RUNNING', and 'DONE'.

        See:
        https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatus.FIELDS.state

        Returns:
            Optional[str]:
                the state (None until set from the server).
        """
        status = self._properties.get("status", {})
        return status.get("state")

    def _set_properties(self, api_response):
        """Update properties from resource in body of ``api_response``

        Args:
            api_response (Dict): response returned from an API call.
        """
        cleaned = api_response.copy()
        statistics = cleaned.setdefault("statistics", {})
        if "creationTime" in statistics:
            statistics["creationTime"] = float(statistics["creationTime"])
        if "startTime" in statistics:
            statistics["startTime"] = float(statistics["startTime"])
        if "endTime" in statistics:
            statistics["endTime"] = float(statistics["endTime"])

        self._properties = cleaned

        # For Future interface
        self._set_future_result()

    @classmethod
    def _check_resource_config(cls, resource):
        """Helper for :meth:`from_api_repr`

        Args:
            resource (Dict): resource for the job.

        Raises:
            KeyError:
                If the resource has no identifier, or
                is missing the appropriate configuration.
        """
        if "jobReference" not in resource or "jobId" not in resource["jobReference"]:
            raise KeyError(
                "Resource lacks required identity information: "
                '["jobReference"]["jobId"]'
            )
        if (
            "configuration" not in resource
            or cls._JOB_TYPE not in resource["configuration"]
        ):
            raise KeyError(
                "Resource lacks required configuration: "
                '["configuration"]["%s"]' % cls._JOB_TYPE
            )

    def to_api_repr(self):
        """Generate a resource for the job."""
        return copy.deepcopy(self._properties)

    _build_resource = to_api_repr  # backward-compatibility alias

    def _begin(self, client=None, retry=DEFAULT_RETRY, timeout=None):
        """API call:  begin the job via a POST request

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/insert

        Args:
            client (Optional[google.cloud.bigquery.client.Client]):
                The client to use. If not passed, falls back to the ``client``
                associated with the job object or``NoneType``
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

        Raises:
            ValueError:
                If the job has already begun.
        """
        if self.state is not None:
            raise ValueError("Job already begun.")

        client = self._require_client(client)
        path = "/projects/%s/jobs" % (self.project,)

        # jobs.insert is idempotent because we ensure that every new
        # job has an ID.
        span_attributes = {"path": path}
        api_response = client._call_api(
            retry,
            span_name="BigQuery.job.begin",
            span_attributes=span_attributes,
            job_ref=self,
            method="POST",
            path=path,
            data=self.to_api_repr(),
            timeout=timeout,
        )
        self._set_properties(api_response)

    def exists(
        self,
        client=None,
        retry: "retries.Retry" = DEFAULT_RETRY,
        timeout: Optional[float] = None,
    ) -> bool:
        """API call:  test for the existence of the job via a GET request

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/get

        Args:
            client (Optional[google.cloud.bigquery.client.Client]):
                the client to use.  If not passed, falls back to the
                ``client`` stored on the current dataset.

            retry (Optional[google.api_core.retry.Retry]): How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.

        Returns:
            bool: Boolean indicating existence of the job.
        """
        client = self._require_client(client)

        extra_params = {"fields": "id"}
        if self.location:
            extra_params["location"] = self.location

        try:
            span_attributes = {"path": self.path}

            client._call_api(
                retry,
                span_name="BigQuery.job.exists",
                span_attributes=span_attributes,
                job_ref=self,
                method="GET",
                path=self.path,
                query_params=extra_params,
                timeout=timeout,
            )
        except exceptions.NotFound:
            return False
        else:
            return True

    def reload(
        self,
        client=None,
        retry: "retries.Retry" = DEFAULT_RETRY,
        timeout: Optional[float] = DEFAULT_GET_JOB_TIMEOUT,
    ):
        """API call:  refresh job properties via a GET request.

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/get

        Args:
            client (Optional[google.cloud.bigquery.client.Client]):
                the client to use.  If not passed, falls back to the
                ``client`` stored on the current dataset.

            retry (Optional[google.api_core.retry.Retry]): How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
        """
        client = self._require_client(client)

        got_job = client.get_job(
            self,
            project=self.project,
            location=self.location,
            retry=retry,
            timeout=timeout,
        )
        self._set_properties(got_job._properties)

    def cancel(
        self,
        client=None,
        retry: Optional[retries.Retry] = DEFAULT_RETRY,
        timeout: Optional[float] = None,
    ) -> bool:
        """API call:  cancel job via a POST request

        See
        https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/cancel

        Args:
            client (Optional[google.cloud.bigquery.client.Client]):
                the client to use.  If not passed, falls back to the
                ``client`` stored on the current dataset.
            retry (Optional[google.api_core.retry.Retry]): How to retry the RPC.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``

        Returns:
            bool: Boolean indicating that the cancel request was sent.
        """
        client = self._require_client(client)

        extra_params = {}
        if self.location:
            extra_params["location"] = self.location

        path = "{}/cancel".format(self.path)
        span_attributes = {"path": path}

        api_response = client._call_api(
            retry,
            span_name="BigQuery.job.cancel",
            span_attributes=span_attributes,
            job_ref=self,
            method="POST",
            path=path,
            query_params=extra_params,
            timeout=timeout,
        )
        self._set_properties(api_response["job"])
        # The Future interface requires that we return True if the *attempt*
        # to cancel was successful.
        return True

    # The following methods implement the PollingFuture interface. Note that
    # the methods above are from the pre-Future interface and are left for
    # compatibility. The only "overloaded" method is :meth:`cancel`, which
    # satisfies both interfaces.

    def _set_future_result(self):
        """Set the result or exception from the job if it is complete."""
        # This must be done in a lock to prevent the polling thread
        # and main thread from both executing the completion logic
        # at the same time.
        with self._completion_lock:
            # If the operation isn't complete or if the result has already been
            # set, do not call set_result/set_exception again.
            # Note: self._result_set is set to True in set_result and
            # set_exception, in case those methods are invoked directly.
            if not self.done(reload=False) or self._result_set:
                return

            if self.error_result is not None:
                exception = _error_result_to_exception(
                    self.error_result, self.errors or ()
                )
                self.set_exception(exception)
            else:
                self.set_result(self)

    def done(
        self,
        retry: "retries.Retry" = DEFAULT_RETRY,
        timeout: Optional[float] = DEFAULT_GET_JOB_TIMEOUT,
        reload: bool = True,
    ) -> bool:
        """Checks if the job is complete.

        Args:
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC. If the job state is ``DONE``, retrying is aborted
                early, as the job will not change anymore.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
            reload (Optional[bool]):
                If ``True``, make an API call to refresh the job state of
                unfinished jobs before checking. Default ``True``.

        Returns:
            bool: True if the job is complete, False otherwise.
        """
        # Do not refresh is the state is already done, as the job will not
        # change once complete.
        if self.state != _DONE_STATE and reload:
            self.reload(retry=retry, timeout=timeout)
        return self.state == _DONE_STATE

    def result(  # type: ignore  # (incompatible with supertype)
        self,
        retry: Optional[retries.Retry] = DEFAULT_RETRY,
        timeout: Optional[float] = None,
    ) -> "_AsyncJob":
        """Start the job and wait for it to complete and get the result.

        Args:
            retry (Optional[google.api_core.retry.Retry]):
                How to retry the RPC. If the job state is ``DONE``, retrying is aborted
                early, as the job will not change anymore.
            timeout (Optional[float]):
                The number of seconds to wait for the underlying HTTP transport
                before using ``retry``.
                If multiple requests are made under the hood, ``timeout``
                applies to each individual request.

        Returns:
            _AsyncJob: This instance.

        Raises:
            google.cloud.exceptions.GoogleAPICallError:
                if the job failed.
            concurrent.futures.TimeoutError:
                if the job did not complete in the given timeout.
        """
        if self.state is None:
            self._begin(retry=retry, timeout=timeout)

        kwargs = {} if retry is DEFAULT_RETRY else {"retry": retry}
        return super(_AsyncJob, self).result(timeout=timeout, **kwargs)

    def cancelled(self):
        """Check if the job has been cancelled.

        This always returns False. It's not possible to check if a job was
        cancelled in the API. This method is here to satisfy the interface
        for :class:`google.api_core.future.Future`.

        Returns:
            bool: False
        """
        return (
            self.error_result is not None
            and self.error_result.get("reason") == _STOPPED_REASON
        )

    def __repr__(self):
        result = (
            f"{self.__class__.__name__}<"
            f"project={self.project}, location={self.location}, id={self.job_id}"
            ">"
        )
        return result


class ScriptStackFrame(object):
    """Stack frame showing the line/column/procedure name where the current
    evaluation happened.

    Args:
        resource (Map[str, Any]): JSON representation of object.
    """

    def __init__(self, resource):
        self._properties = resource

    @property
    def procedure_id(self):
        """Optional[str]: Name of the active procedure.

        Omitted if in a top-level script.
        """
        return self._properties.get("procedureId")

    @property
    def text(self):
        """str: Text of the current statement/expression."""
        return self._properties.get("text")

    @property
    def start_line(self):
        """int: One-based start line."""
        return _helpers._int_or_none(self._properties.get("startLine"))

    @property
    def start_column(self):
        """int: One-based start column."""
        return _helpers._int_or_none(self._properties.get("startColumn"))

    @property
    def end_line(self):
        """int: One-based end line."""
        return _helpers._int_or_none(self._properties.get("endLine"))

    @property
    def end_column(self):
        """int: One-based end column."""
        return _helpers._int_or_none(self._properties.get("endColumn"))


class ScriptStatistics(object):
    """Statistics for a child job of a script.

    Args:
        resource (Map[str, Any]): JSON representation of object.
    """

    def __init__(self, resource):
        self._properties = resource

    @property
    def stack_frames(self) -> Sequence[ScriptStackFrame]:
        """Stack trace where the current evaluation happened.

        Shows line/column/procedure name of each frame on the stack at the
        point where the current evaluation happened.

        The leaf frame is first, the primary script is last.
        """
        return [
            ScriptStackFrame(frame) for frame in self._properties.get("stackFrames", [])
        ]

    @property
    def evaluation_kind(self) -> Optional[str]:
        """str: Indicates the type of child job.

        Possible values include ``STATEMENT`` and ``EXPRESSION``.
        """
        return self._properties.get("evaluationKind")


class SessionInfo:
    """[Preview] Information of the session if this job is part of one.

    .. versionadded:: 2.29.0

    Args:
        resource (Map[str, Any]): JSON representation of object.
    """

    def __init__(self, resource):
        self._properties = resource

    @property
    def session_id(self) -> Optional[str]:
        """The ID of the session."""
        return self._properties.get("sessionId")


class UnknownJob(_AsyncJob):
    """A job whose type cannot be determined."""

    @classmethod
    def from_api_repr(cls, resource: dict, client) -> "UnknownJob":
        """Construct an UnknownJob from the JSON representation.

        Args:
            resource (Dict): JSON representation of a job.
            client (google.cloud.bigquery.client.Client):
                Client connected to BigQuery API.

        Returns:
            UnknownJob: Job corresponding to the resource.
        """
        job_ref_properties = resource.get(
            "jobReference", {"projectId": client.project, "jobId": None}
        )
        job_ref = _JobReference._from_api_repr(job_ref_properties)
        job = cls(job_ref, client)
        # Populate the job reference with the project, even if it has been
        # redacted, because we know it should equal that of the request.
        resource["jobReference"] = job_ref_properties
        job._properties = resource
        return job
