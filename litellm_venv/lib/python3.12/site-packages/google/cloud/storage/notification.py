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

"""Configure bucket notification resources to interact with Google Cloud Pub/Sub.

See [Cloud Pub/Sub Notifications for Google Cloud Storage](https://cloud.google.com/storage/docs/pubsub-notifications)
"""

import re

from google.api_core.exceptions import NotFound

from google.cloud.storage._opentelemetry_tracing import create_trace_span
from google.cloud.storage.constants import _DEFAULT_TIMEOUT
from google.cloud.storage.retry import DEFAULT_RETRY


OBJECT_FINALIZE_EVENT_TYPE = "OBJECT_FINALIZE"
OBJECT_METADATA_UPDATE_EVENT_TYPE = "OBJECT_METADATA_UPDATE"
OBJECT_DELETE_EVENT_TYPE = "OBJECT_DELETE"
OBJECT_ARCHIVE_EVENT_TYPE = "OBJECT_ARCHIVE"

JSON_API_V1_PAYLOAD_FORMAT = "JSON_API_V1"
NONE_PAYLOAD_FORMAT = "NONE"

_TOPIC_REF_FMT = "//pubsub.googleapis.com/projects/{}/topics/{}"
_PROJECT_PATTERN = r"(?P<project>[a-z][a-z0-9-]{4,28}[a-z0-9])"
_TOPIC_NAME_PATTERN = r"(?P<name>[A-Za-z](\w|[-_.~+%])+)"
_TOPIC_REF_PATTERN = _TOPIC_REF_FMT.format(_PROJECT_PATTERN, _TOPIC_NAME_PATTERN)
_TOPIC_REF_RE = re.compile(_TOPIC_REF_PATTERN)
_BAD_TOPIC = (
    "Resource has invalid topic: {}; see "
    "https://cloud.google.com/storage/docs/json_api/v1/"
    "notifications/insert#topic"
)


class BucketNotification(object):
    """Represent a single notification resource for a bucket.

    See: https://cloud.google.com/storage/docs/json_api/v1/notifications

    :type bucket: :class:`google.cloud.storage.bucket.Bucket`
    :param bucket: Bucket to which the notification is bound.

    :type topic_name: str
    :param topic_name:
        (Optional) Topic name to which notifications are published.

    :type topic_project: str
    :param topic_project:
        (Optional) Project ID of topic to which notifications are published.
        If not passed, uses the project ID of the bucket's client.

    :type custom_attributes: dict
    :param custom_attributes:
        (Optional) Additional attributes passed with notification events.

    :type event_types: list(str)
    :param event_types:
        (Optional) Event types for which notification events are published.

    :type blob_name_prefix: str
    :param blob_name_prefix:
        (Optional) Prefix of blob names for which notification events are
        published.

    :type payload_format: str
    :param payload_format:
        (Optional) Format of payload for notification events.

    :type notification_id: str
    :param notification_id:
        (Optional) The ID of the notification.
    """

    def __init__(
        self,
        bucket,
        topic_name=None,
        topic_project=None,
        custom_attributes=None,
        event_types=None,
        blob_name_prefix=None,
        payload_format=NONE_PAYLOAD_FORMAT,
        notification_id=None,
    ):
        self._bucket = bucket
        self._topic_name = topic_name

        if topic_project is None:
            topic_project = bucket.client.project

        if topic_project is None:
            raise ValueError("Client project not set:  pass an explicit topic_project.")

        self._topic_project = topic_project

        self._properties = {}

        if custom_attributes is not None:
            self._properties["custom_attributes"] = custom_attributes

        if event_types is not None:
            self._properties["event_types"] = event_types

        if blob_name_prefix is not None:
            self._properties["object_name_prefix"] = blob_name_prefix

        if notification_id is not None:
            self._properties["id"] = notification_id

        self._properties["payload_format"] = payload_format

    @classmethod
    def from_api_repr(cls, resource, bucket):
        """Construct an instance from the JSON repr returned by the server.

        See: https://cloud.google.com/storage/docs/json_api/v1/notifications

        :type resource: dict
        :param resource: JSON repr of the notification

        :type bucket: :class:`google.cloud.storage.bucket.Bucket`
        :param bucket: Bucket to which the notification is bound.

        :rtype: :class:`BucketNotification`
        :returns: the new notification instance
        """
        topic_path = resource.get("topic")
        if topic_path is None:
            raise ValueError("Resource has no topic")

        name, project = _parse_topic_path(topic_path)
        instance = cls(bucket, name, topic_project=project)
        instance._properties = resource

        return instance

    @property
    def bucket(self):
        """Bucket to which the notification is bound."""
        return self._bucket

    @property
    def topic_name(self):
        """Topic name to which notifications are published."""
        return self._topic_name

    @property
    def topic_project(self):
        """Project ID of topic to which notifications are published."""
        return self._topic_project

    @property
    def custom_attributes(self):
        """Custom attributes passed with notification events."""
        return self._properties.get("custom_attributes")

    @property
    def event_types(self):
        """Event types for which notification events are published."""
        return self._properties.get("event_types")

    @property
    def blob_name_prefix(self):
        """Prefix of blob names for which notification events are published."""
        return self._properties.get("object_name_prefix")

    @property
    def payload_format(self):
        """Format of payload of notification events."""
        return self._properties.get("payload_format")

    @property
    def notification_id(self):
        """Server-set ID of notification resource."""
        return self._properties.get("id")

    @property
    def etag(self):
        """Server-set ETag of notification resource."""
        return self._properties.get("etag")

    @property
    def self_link(self):
        """Server-set ETag of notification resource."""
        return self._properties.get("selfLink")

    @property
    def client(self):
        """The client bound to this notfication."""
        return self.bucket.client

    @property
    def path(self):
        """The URL path for this notification."""
        return f"/b/{self.bucket.name}/notificationConfigs/{self.notification_id}"

    def _require_client(self, client):
        """Check client or verify over-ride.

        :type client: :class:`~google.cloud.storage.client.Client` or
                      ``NoneType``
        :param client: the client to use.

        :rtype: :class:`google.cloud.storage.client.Client`
        :returns: The client passed in or the bucket's client.
        """
        if client is None:
            client = self.client
        return client

    def _set_properties(self, response):
        """Helper for :meth:`reload`.

        :type response: dict
        :param response: resource mapping from server
        """
        self._properties.clear()
        self._properties.update(response)

    @create_trace_span(name="Storage.BucketNotification.create")
    def create(self, client=None, timeout=_DEFAULT_TIMEOUT, retry=None):
        """API wrapper: create the notification.

        See:
        https://cloud.google.com/storage/docs/json_api/v1/notifications/insert

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        :type client: :class:`~google.cloud.storage.client.Client`
        :param client: (Optional) The client to use.  If not passed, falls back
                       to the ``client`` stored on the notification's bucket.
        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC. See: :ref:`configuring_retries`

        :raises ValueError: if the notification already exists.
        """
        if self.notification_id is not None:
            raise ValueError(
                f"notification_id already set to {self.notification_id}; must be None to create a Notification."
            )

        client = self._require_client(client)

        query_params = {}
        if self.bucket.user_project is not None:
            query_params["userProject"] = self.bucket.user_project

        path = f"/b/{self.bucket.name}/notificationConfigs"
        properties = self._properties.copy()

        if self.topic_name is None:
            properties["topic"] = _TOPIC_REF_FMT.format(self.topic_project, "")
        else:
            properties["topic"] = _TOPIC_REF_FMT.format(
                self.topic_project, self.topic_name
            )

        self._properties = client._post_resource(
            path,
            properties,
            query_params=query_params,
            timeout=timeout,
            retry=retry,
        )

    @create_trace_span(name="Storage.BucketNotification.exists")
    def exists(self, client=None, timeout=_DEFAULT_TIMEOUT, retry=DEFAULT_RETRY):
        """Test whether this notification exists.

        See:
        https://cloud.google.com/storage/docs/json_api/v1/notifications/get

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        :type client: :class:`~google.cloud.storage.client.Client` or
                      ``NoneType``
        :param client: (Optional) The client to use.  If not passed, falls back
                       to the ``client`` stored on the current bucket.
        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC. See: :ref:`configuring_retries`

        :rtype: bool
        :returns: True, if the notification exists, else False.
        :raises ValueError: if the notification has no ID.
        """
        if self.notification_id is None:
            raise ValueError("Notification ID not set: set an explicit notification_id")

        client = self._require_client(client)

        query_params = {}
        if self.bucket.user_project is not None:
            query_params["userProject"] = self.bucket.user_project

        try:
            client._get_resource(
                self.path,
                query_params=query_params,
                timeout=timeout,
                retry=retry,
            )
        except NotFound:
            return False
        else:
            return True

    @create_trace_span(name="Storage.BucketNotification.reload")
    def reload(self, client=None, timeout=_DEFAULT_TIMEOUT, retry=DEFAULT_RETRY):
        """Update this notification from the server configuration.

        See:
        https://cloud.google.com/storage/docs/json_api/v1/notifications/get

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        :type client: :class:`~google.cloud.storage.client.Client` or
                      ``NoneType``
        :param client: (Optional) The client to use.  If not passed, falls back
                       to the ``client`` stored on the current bucket.
        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC. See: :ref:`configuring_retries`


        :raises ValueError: if the notification has no ID.
        """
        if self.notification_id is None:
            raise ValueError("Notification ID not set: set an explicit notification_id")

        client = self._require_client(client)

        query_params = {}
        if self.bucket.user_project is not None:
            query_params["userProject"] = self.bucket.user_project

        response = client._get_resource(
            self.path,
            query_params=query_params,
            timeout=timeout,
            retry=retry,
        )
        self._set_properties(response)

    @create_trace_span(name="Storage.BucketNotification.delete")
    def delete(self, client=None, timeout=_DEFAULT_TIMEOUT, retry=DEFAULT_RETRY):
        """Delete this notification.

        See:
        https://cloud.google.com/storage/docs/json_api/v1/notifications/delete

        If :attr:`user_project` is set on the bucket, bills the API request
        to that project.

        :type client: :class:`~google.cloud.storage.client.Client` or
                      ``NoneType``
        :param client: (Optional) The client to use.  If not passed, falls back
                       to the ``client`` stored on the current bucket.
        :type timeout: float or tuple
        :param timeout:
            (Optional) The amount of time, in seconds, to wait
            for the server response.  See: :ref:`configuring_timeouts`

        :type retry: google.api_core.retry.Retry or google.cloud.storage.retry.ConditionalRetryPolicy
        :param retry:
            (Optional) How to retry the RPC. See: :ref:`configuring_retries`

        :raises: :class:`google.api_core.exceptions.NotFound`:
            if the notification does not exist.
        :raises ValueError: if the notification has no ID.
        """
        if self.notification_id is None:
            raise ValueError("Notification ID not set: set an explicit notification_id")

        client = self._require_client(client)

        query_params = {}
        if self.bucket.user_project is not None:
            query_params["userProject"] = self.bucket.user_project

        client._delete_resource(
            self.path,
            query_params=query_params,
            timeout=timeout,
            retry=retry,
        )


def _parse_topic_path(topic_path):
    """Verify that a topic path is in the correct format.

    Expected to be of the form:

        //pubsub.googleapis.com/projects/{project}/topics/{topic}

    where the ``project`` value must be "6 to 30 lowercase letters, digits,
    or hyphens. It must start with a letter. Trailing hyphens are prohibited."
    (see [`resource manager docs`](https://cloud.google.com/resource-manager/reference/rest/v1beta1/projects#Project.FIELDS.project_id))
    and ``topic`` must have length at least two,
    must start with a letter and may only contain alphanumeric characters or
    ``-``, ``_``, ``.``, ``~``, ``+`` or ``%`` (i.e characters used for URL
    encoding, see [`topic spec`](https://cloud.google.com/storage/docs/json_api/v1/notifications/insert#topic)).

    Args:
        topic_path (str): The topic path to be verified.

    Returns:
        Tuple[str, str]: The ``project`` and ``topic`` parsed from the
        ``topic_path``.

    Raises:
        ValueError: If the topic path is invalid.
    """
    match = _TOPIC_REF_RE.match(topic_path)
    if match is None:
        raise ValueError(_BAD_TOPIC.format(topic_path))

    return match.group("name"), match.group("project")
