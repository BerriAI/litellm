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

"""Configure HMAC keys that can be used to authenticate requests to Google Cloud Storage.

See [HMAC keys documentation](https://cloud.google.com/storage/docs/authentication/hmackeys)
"""

from google.cloud.exceptions import NotFound
from google.cloud._helpers import _rfc3339_nanos_to_datetime

from google.cloud.storage._opentelemetry_tracing import create_trace_span
from google.cloud.storage.constants import _DEFAULT_TIMEOUT
from google.cloud.storage.retry import DEFAULT_RETRY
from google.cloud.storage.retry import DEFAULT_RETRY_IF_ETAG_IN_JSON


class HMACKeyMetadata(object):
    """Metadata about an HMAC service account key withn Cloud Storage.

    :type client: :class:`~google.cloud.stoage.client.Client`
    :param client: client associated with the key metadata.

    :type access_id: str
    :param access_id: (Optional) Unique ID of an existing key.

    :type project_id: str
    :param project_id: (Optional) Project ID of an existing key.
        Defaults to client's project.

    :type user_project: str
    :param user_project: (Optional) This parameter is currently ignored.
    """

    ACTIVE_STATE = "ACTIVE"
    """Key is active, and may be used to sign requests."""
    INACTIVE_STATE = "INACTIVE"
    """Key is inactive, and may not be used to sign requests.

    It can be re-activated via :meth:`update`.
    """
    DELETED_STATE = "DELETED"
    """Key is deleted.  It cannot be re-activated."""

    _SETTABLE_STATES = (ACTIVE_STATE, INACTIVE_STATE)

    def __init__(self, client, access_id=None, project_id=None, user_project=None):
        self._client = client
        self._properties = {}

        if access_id is not None:
            self._properties["accessId"] = access_id

        if project_id is not None:
            self._properties["projectId"] = project_id

        self._user_project = user_project

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self._client == other._client and self.access_id == other.access_id

    def __hash__(self):
        return hash(self._client) + hash(self.access_id)

    @property
    def access_id(self):
        """Access ID of the key.

        :rtype: str or None
        :returns: unique identifier of the key within a project.
        """
        return self._properties.get("accessId")

    @property
    def etag(self):
        """ETag identifying the version of the key metadata.

        :rtype: str or None
        :returns: ETag for the version of the key's metadata.
        """
        return self._properties.get("etag")

    @property
    def id(self):
        """ID of the key, including the Project ID and the Access ID.

        :rtype: str or None
        :returns: ID of the key.
        """
        return self._properties.get("id")

    @property
    def project(self):
        """Project ID associated with the key.

        :rtype: str or None
        :returns: project identfier for the key.
        """
        return self._properties.get("projectId")

    @property
    def service_account_email(self):
        """Service account e-mail address associated with the key.

        :rtype: str or None
        :returns: e-mail address for the service account which created the key.
        """
        return self._properties.get("serviceAccountEmail")

    @property
    def state(self):
        """Get / set key's state.

        One of:
            - ``ACTIVE``
            - ``INACTIVE``
            - ``DELETED``

        :rtype: str or None
        :returns: key's current state.
        """
        return self._properties.get("state")

    @state.setter
    def state(self, value):
        self._properties["state"] = value

    @property
    def time_created(self):
        """Retrieve the timestamp at which the HMAC key was created.

        :rtype: :class:`datetime.datetime` or ``NoneType``
        :returns: Datetime object parsed from RFC3339 valid timestamp, or
                  ``None`` if the bucket's resource has not been loaded
                  from the server.
        """
        value = self._properties.get("timeCreated")
        if value is not None:
            return _rfc3339_nanos_to_datetime(value)

    @property
    def updated(self):
        """Retrieve the timestamp at which the HMAC key was created.

        :rtype: :class:`datetime.datetime` or ``NoneType``
        :returns: Datetime object parsed from RFC3339 valid timestamp, or
                  ``None`` if the bucket's resource has not been loaded
                  from the server.
        """
        value = self._properties.get("updated")
        if value is not None:
            return _rfc3339_nanos_to_datetime(value)

    @property
    def path(self):
        """Resource path for the metadata's key."""

        if self.access_id is None:
            raise ValueError("No 'access_id' set.")

        project = self.project
        if project is None:
            project = self._client.project

        return f"/projects/{project}/hmacKeys/{self.access_id}"

    @property
    def user_project(self):
        """Project ID to be billed for API requests made via this bucket.

        This property is currently ignored by the server.

        :rtype: str
        """
        return self._user_project

    @create_trace_span(name="Storage.HmacKey.exists")
    def exists(self, timeout=_DEFAULT_TIMEOUT, retry=DEFAULT_RETRY):
        """Determine whether or not the key for this metadata exists.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC. See: :ref:`configuring_retries`

        :rtype: bool
        :returns: True if the key exists in Cloud Storage.
        """
        try:
            qs_params = {}

            if self.user_project is not None:
                qs_params["userProject"] = self.user_project

            self._client._get_resource(
                self.path,
                query_params=qs_params,
                timeout=timeout,
                retry=retry,
            )
        except NotFound:
            return False
        else:
            return True

    @create_trace_span(name="Storage.HmacKey.reload")
    def reload(self, timeout=_DEFAULT_TIMEOUT, retry=DEFAULT_RETRY):
        """Reload properties from Cloud Storage.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC. See: :ref:`configuring_retries`

        :raises :class:`~google.api_core.exceptions.NotFound`:
            if the key does not exist on the back-end.
        """
        qs_params = {}

        if self.user_project is not None:
            qs_params["userProject"] = self.user_project

        self._properties = self._client._get_resource(
            self.path,
            query_params=qs_params,
            timeout=timeout,
            retry=retry,
        )

    @create_trace_span(name="Storage.HmacKey.update")
    def update(self, timeout=_DEFAULT_TIMEOUT, retry=DEFAULT_RETRY_IF_ETAG_IN_JSON):
        """Save writable properties to Cloud Storage.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC. See: :ref:`configuring_retries`

        :raises :class:`~google.api_core.exceptions.NotFound`:
            if the key does not exist on the back-end.
        """
        qs_params = {}
        if self.user_project is not None:
            qs_params["userProject"] = self.user_project

        payload = {"state": self.state}
        self._properties = self._client._put_resource(
            self.path,
            payload,
            query_params=qs_params,
            timeout=timeout,
            retry=retry,
        )

    @create_trace_span(name="Storage.HmacKey.delete")
    def delete(self, timeout=_DEFAULT_TIMEOUT, retry=DEFAULT_RETRY):
        """Delete the key from Cloud Storage.

        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC. See: :ref:`configuring_retries`

        :raises :class:`~google.api_core.exceptions.NotFound`:
            if the key does not exist on the back-end.
        """
        qs_params = {}
        if self.user_project is not None:
            qs_params["userProject"] = self.user_project

        self._client._delete_resource(
            self.path,
            query_params=qs_params,
            timeout=timeout,
            retry=retry,
        )
