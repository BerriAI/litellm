# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# pylint: disable=docstring-keyword-should-match-keyword-only

import uuid

from typing import Any, Optional, Union, TYPE_CHECKING

from azure.core.exceptions import HttpResponseError
from azure.core.tracing.decorator import distributed_trace

from ._shared.response_handlers import process_storage_error, return_response_headers
from ._serialize import get_modify_conditions

if TYPE_CHECKING:
    from azure.storage.blob import BlobClient, ContainerClient
    from datetime import datetime


class BlobLeaseClient:  # pylint: disable=client-accepts-api-version-keyword
    """Creates a new BlobLeaseClient.

    This client provides lease operations on a BlobClient or ContainerClient.
    :param client: The client of the blob or container to lease.
    :type client: Union[BlobClient, ContainerClient]
    :param lease_id: A string representing the lease ID of an existing lease. This value does not need to be
    specified in order to acquire a new lease, or break one.
    :type lease_id: Optional[str]
    """

    id: str
    """The ID of the lease currently being maintained. This will be `None` if no
    lease has yet been acquired."""
    etag: Optional[str]
    """The ETag of the lease currently being maintained. This will be `None` if no
    lease has yet been acquired or modified."""
    last_modified: Optional["datetime"]
    """The last modified timestamp of the lease currently being maintained.
    This will be `None` if no lease has yet been acquired or modified."""

    def __init__(  # pylint: disable=missing-client-constructor-parameter-credential, missing-client-constructor-parameter-kwargs
        self,
        client: Union["BlobClient", "ContainerClient"],
        lease_id: Optional[str] = None,
    ) -> None:
        self.id = lease_id or str(uuid.uuid4())
        self.last_modified = None
        self.etag = None
        if hasattr(client, "blob_name"):
            self._client = client._client.blob
        elif hasattr(client, "container_name"):
            self._client = client._client.container
        else:
            raise TypeError("Lease must use either BlobClient or ContainerClient.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    @distributed_trace
    def acquire(self, lease_duration: int = -1, **kwargs: Any) -> None:
        """Requests a new lease.

        If the container does not have an active lease, the Blob service creates a
        lease on the container and returns a new lease ID.

        :param int lease_duration:
            Specifies the duration of the lease, in seconds, or negative one
            (-1) for a lease that never expires. A non-infinite lease can be
            between 15 and 60 seconds. A lease duration cannot be changed
            using renew or change. Default is -1 (infinite lease).
        :keyword ~datetime.datetime if_modified_since:
            A DateTime value. Azure expects the date value passed in to be UTC.
            If timezone is included, any non-UTC datetimes will be converted to UTC.
            If a date is passed in without timezone info, it is assumed to be UTC.
            Specify this header to perform the operation only
            if the resource has been modified since the specified time.
        :keyword ~datetime.datetime if_unmodified_since:
            A DateTime value. Azure expects the date value passed in to be UTC.
            If timezone is included, any non-UTC datetimes will be converted to UTC.
            If a date is passed in without timezone info, it is assumed to be UTC.
            Specify this header to perform the operation only if
            the resource has not been modified since the specified date/time.
        :keyword str etag:
            An ETag value, or the wildcard character (*). Used to check if the resource has changed,
            and act according to the condition specified by the `match_condition` parameter.
        :keyword ~azure.core.MatchConditions match_condition:
            The match condition to use upon the etag.
        :keyword str if_tags_match_condition:
            Specify a SQL where clause on blob tags to operate only on blob with a matching value.
            eg. ``\"\\\"tagname\\\"='my tag'\"``

            .. versionadded:: 12.4.0

        :keyword int timeout:
            Sets the server-side timeout for the operation in seconds. For more details see
            https://learn.microsoft.com/rest/api/storageservices/setting-timeouts-for-blob-service-operations.
            This value is not tracked or validated on the client. To configure client-side network timesouts
            see `here <https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/storage/azure-storage-blob
            #other-client--per-operation-configuration>`__.
        :return: None
        :rtype: None
        """
        mod_conditions = get_modify_conditions(kwargs)
        try:
            response: Any = self._client.acquire_lease(
                timeout=kwargs.pop("timeout", None),
                duration=lease_duration,
                proposed_lease_id=self.id,
                modified_access_conditions=mod_conditions,
                cls=return_response_headers,
                **kwargs
            )
        except HttpResponseError as error:
            process_storage_error(error)
        self.id = response.get("lease_id")
        self.last_modified = response.get("last_modified")
        self.etag = response.get("etag")

    @distributed_trace
    def renew(self, **kwargs: Any) -> None:
        """Renews the lease.

        The lease can be renewed if the lease ID specified in the
        lease client matches that associated with the container or blob. Note that
        the lease may be renewed even if it has expired as long as the container
        or blob has not been leased again since the expiration of that lease. When you
        renew a lease, the lease duration clock resets.

        :keyword ~datetime.datetime if_modified_since:
            A DateTime value. Azure expects the date value passed in to be UTC.
            If timezone is included, any non-UTC datetimes will be converted to UTC.
            If a date is passed in without timezone info, it is assumed to be UTC.
            Specify this header to perform the operation only
            if the resource has been modified since the specified time.
        :keyword ~datetime.datetime if_unmodified_since:
            A DateTime value. Azure expects the date value passed in to be UTC.
            If timezone is included, any non-UTC datetimes will be converted to UTC.
            If a date is passed in without timezone info, it is assumed to be UTC.
            Specify this header to perform the operation only if
            the resource has not been modified since the specified date/time.
        :keyword str etag:
            An ETag value, or the wildcard character (*). Used to check if the resource has changed,
            and act according to the condition specified by the `match_condition` parameter.
        :keyword ~azure.core.MatchConditions match_condition:
            The match condition to use upon the etag.
        :keyword str if_tags_match_condition:
            Specify a SQL where clause on blob tags to operate only on blob with a matching value.
            eg. ``\"\\\"tagname\\\"='my tag'\"``

            .. versionadded:: 12.4.0

        :keyword int timeout:
            Sets the server-side timeout for the operation in seconds. For more details see
            https://learn.microsoft.com/rest/api/storageservices/setting-timeouts-for-blob-service-operations.
            This value is not tracked or validated on the client. To configure client-side network timesouts
            see `here <https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/storage/azure-storage-blob
            #other-client--per-operation-configuration>`__.
        :return: None
        """
        mod_conditions = get_modify_conditions(kwargs)
        try:
            response: Any = self._client.renew_lease(
                lease_id=self.id,
                timeout=kwargs.pop("timeout", None),
                modified_access_conditions=mod_conditions,
                cls=return_response_headers,
                **kwargs
            )
        except HttpResponseError as error:
            process_storage_error(error)
        self.etag = response.get("etag")
        self.id = response.get("lease_id")
        self.last_modified = response.get("last_modified")

    @distributed_trace
    def release(self, **kwargs: Any) -> None:
        """Release the lease.

        The lease may be released if the client lease id specified matches
        that associated with the container or blob. Releasing the lease allows another client
        to immediately acquire the lease for the container or blob as soon as the release is complete.

        :keyword ~datetime.datetime if_modified_since:
            A DateTime value. Azure expects the date value passed in to be UTC.
            If timezone is included, any non-UTC datetimes will be converted to UTC.
            If a date is passed in without timezone info, it is assumed to be UTC.
            Specify this header to perform the operation only
            if the resource has been modified since the specified time.
        :keyword ~datetime.datetime if_unmodified_since:
            A DateTime value. Azure expects the date value passed in to be UTC.
            If timezone is included, any non-UTC datetimes will be converted to UTC.
            If a date is passed in without timezone info, it is assumed to be UTC.
            Specify this header to perform the operation only if
            the resource has not been modified since the specified date/time.
        :keyword str etag:
            An ETag value, or the wildcard character (*). Used to check if the resource has changed,
            and act according to the condition specified by the `match_condition` parameter.
        :keyword ~azure.core.MatchConditions match_condition:
            The match condition to use upon the etag.
        :keyword str if_tags_match_condition:
            Specify a SQL where clause on blob tags to operate only on blob with a matching value.
            eg. ``\"\\\"tagname\\\"='my tag'\"``

            .. versionadded:: 12.4.0

        :keyword int timeout:
            Sets the server-side timeout for the operation in seconds. For more details see
            https://learn.microsoft.com/rest/api/storageservices/setting-timeouts-for-blob-service-operations.
            This value is not tracked or validated on the client. To configure client-side network timesouts
            see `here <https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/storage/azure-storage-blob
            #other-client--per-operation-configuration>`__.
        :return: None
        """
        mod_conditions = get_modify_conditions(kwargs)
        try:
            response: Any = self._client.release_lease(
                lease_id=self.id,
                timeout=kwargs.pop("timeout", None),
                modified_access_conditions=mod_conditions,
                cls=return_response_headers,
                **kwargs
            )
        except HttpResponseError as error:
            process_storage_error(error)
        self.etag = response.get("etag")
        self.id = response.get("lease_id")
        self.last_modified = response.get("last_modified")

    @distributed_trace
    def change(self, proposed_lease_id: str, **kwargs: Any) -> None:
        """Change the lease ID of an active lease.

        :param str proposed_lease_id:
            Proposed lease ID, in a GUID string format. The Blob service returns 400
            (Invalid request) if the proposed lease ID is not in the correct format.
        :keyword ~datetime.datetime if_modified_since:
            A DateTime value. Azure expects the date value passed in to be UTC.
            If timezone is included, any non-UTC datetimes will be converted to UTC.
            If a date is passed in without timezone info, it is assumed to be UTC.
            Specify this header to perform the operation only
            if the resource has been modified since the specified time.
        :keyword ~datetime.datetime if_unmodified_since:
            A DateTime value. Azure expects the date value passed in to be UTC.
            If timezone is included, any non-UTC datetimes will be converted to UTC.
            If a date is passed in without timezone info, it is assumed to be UTC.
            Specify this header to perform the operation only if
            the resource has not been modified since the specified date/time.
        :keyword str etag:
            An ETag value, or the wildcard character (*). Used to check if the resource has changed,
            and act according to the condition specified by the `match_condition` parameter.
        :keyword ~azure.core.MatchConditions match_condition:
            The match condition to use upon the etag.
        :keyword str if_tags_match_condition:
            Specify a SQL where clause on blob tags to operate only on blob with a matching value.
            eg. ``\"\\\"tagname\\\"='my tag'\"``

            .. versionadded:: 12.4.0

        :keyword int timeout:
            Sets the server-side timeout for the operation in seconds. For more details see
            https://learn.microsoft.com/rest/api/storageservices/setting-timeouts-for-blob-service-operations.
            This value is not tracked or validated on the client. To configure client-side network timesouts
            see `here <https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/storage/azure-storage-blob
            #other-client--per-operation-configuration>`__.
        :return: None
        """
        mod_conditions = get_modify_conditions(kwargs)
        try:
            response: Any = self._client.change_lease(
                lease_id=self.id,
                proposed_lease_id=proposed_lease_id,
                timeout=kwargs.pop("timeout", None),
                modified_access_conditions=mod_conditions,
                cls=return_response_headers,
                **kwargs
            )
        except HttpResponseError as error:
            process_storage_error(error)
        self.etag = response.get("etag")
        self.id = response.get("lease_id")
        self.last_modified = response.get("last_modified")

    @distributed_trace
    def break_lease(
        self, lease_break_period: Optional[int] = None, **kwargs: Any
    ) -> int:
        """Break the lease, if the container or blob has an active lease.

        Once a lease is broken, it cannot be renewed. Any authorized request can break the lease;
        the request is not required to specify a matching lease ID. When a lease
        is broken, the lease break period is allowed to elapse, during which time
        no lease operation except break and release can be performed on the container or blob.
        When a lease is successfully broken, the response indicates the interval
        in seconds until a new lease can be acquired.

        :param int lease_break_period:
            This is the proposed duration of seconds that the lease
            should continue before it is broken, between 0 and 60 seconds. This
            break period is only used if it is shorter than the time remaining
            on the lease. If longer, the time remaining on the lease is used.
            A new lease will not be available before the break period has
            expired, but the lease may be held for longer than the break
            period. If this header does not appear with a break
            operation, a fixed-duration lease breaks after the remaining lease
            period elapses, and an infinite lease breaks immediately.
        :keyword ~datetime.datetime if_modified_since:
            A DateTime value. Azure expects the date value passed in to be UTC.
            If timezone is included, any non-UTC datetimes will be converted to UTC.
            If a date is passed in without timezone info, it is assumed to be UTC.
            Specify this header to perform the operation only
            if the resource has been modified since the specified time.
        :keyword ~datetime.datetime if_unmodified_since:
            A DateTime value. Azure expects the date value passed in to be UTC.
            If timezone is included, any non-UTC datetimes will be converted to UTC.
            If a date is passed in without timezone info, it is assumed to be UTC.
            Specify this header to perform the operation only if
            the resource has not been modified since the specified date/time.
        :keyword str if_tags_match_condition:
            Specify a SQL where clause on blob tags to operate only on blob with a matching value.
            eg. ``\"\\\"tagname\\\"='my tag'\"``

            .. versionadded:: 12.4.0

        :keyword int timeout:
            Sets the server-side timeout for the operation in seconds. For more details see
            https://learn.microsoft.com/rest/api/storageservices/setting-timeouts-for-blob-service-operations.
            This value is not tracked or validated on the client. To configure client-side network timesouts
            see `here <https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/storage/azure-storage-blob
            #other-client--per-operation-configuration>`__.
        :return: Approximate time remaining in the lease period, in seconds.
        :rtype: int
        """
        mod_conditions = get_modify_conditions(kwargs)
        try:
            response = self._client.break_lease(
                timeout=kwargs.pop("timeout", None),
                break_period=lease_break_period,
                modified_access_conditions=mod_conditions,
                cls=return_response_headers,
                **kwargs
            )
        except HttpResponseError as error:
            process_storage_error(error)
        return response.get("lease_time")  # type: ignore
