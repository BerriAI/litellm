# -*- coding: utf-8 -*-
# Copyright 2024 Google LLC
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
#
from __future__ import annotations

from typing import MutableMapping, MutableSequence

from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.cloud.resourcemanager.v3",
    manifest={
        "Organization",
        "GetOrganizationRequest",
        "SearchOrganizationsRequest",
        "SearchOrganizationsResponse",
        "DeleteOrganizationMetadata",
        "UndeleteOrganizationMetadata",
    },
)


class Organization(proto.Message):
    r"""The root node in the resource hierarchy to which a particular
    entity's (a company, for example) resources belong.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            Output only. The resource name of the organization. This is
            the organization's relative path in the API. Its format is
            "organizations/[organization_id]". For example,
            "organizations/1234".
        display_name (str):
            Output only. A human-readable string that
            refers to the organization in the Google Cloud
            Console. This string is set by the server and
            cannot be changed. The string will be set to the
            primary domain (for example, "google.com") of
            the Google Workspace customer that owns the
            organization.
        directory_customer_id (str):
            Immutable. The G Suite / Workspace customer
            id used in the Directory API.

            This field is a member of `oneof`_ ``owner``.
        state (google.cloud.resourcemanager_v3.types.Organization.State):
            Output only. The organization's current
            lifecycle state.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when the Organization
            was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when the Organization
            was last modified.
        delete_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when the Organization
            was requested for deletion.
        etag (str):
            Output only. A checksum computed by the
            server based on the current value of the
            Organization resource. This may be sent on
            update and delete requests to ensure the client
            has an up-to-date value before proceeding.
    """

    class State(proto.Enum):
        r"""Organization lifecycle states.

        Values:
            STATE_UNSPECIFIED (0):
                Unspecified state.  This is only useful for
                distinguishing unset values.
            ACTIVE (1):
                The normal and active state.
            DELETE_REQUESTED (2):
                The organization has been marked for deletion
                by the user.
        """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        DELETE_REQUESTED = 2

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    directory_customer_id: str = proto.Field(
        proto.STRING,
        number=3,
        oneof="owner",
    )
    state: State = proto.Field(
        proto.ENUM,
        number=4,
        enum=State,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    delete_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=8,
    )


class GetOrganizationRequest(proto.Message):
    r"""The request sent to the ``GetOrganization`` method. The ``name``
    field is required. ``organization_id`` is no longer accepted.

    Attributes:
        name (str):
            Required. The resource name of the Organization to fetch.
            This is the organization's relative path in the API,
            formatted as "organizations/[organizationId]". For example,
            "organizations/1234".
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class SearchOrganizationsRequest(proto.Message):
    r"""The request sent to the ``SearchOrganizations`` method.

    Attributes:
        page_size (int):
            Optional. The maximum number of organizations
            to return in the response. The server can return
            fewer organizations than requested. If
            unspecified, server picks an appropriate
            default.
        page_token (str):
            Optional. A pagination token returned from a previous call
            to ``SearchOrganizations`` that indicates from where listing
            should continue.
        query (str):
            Optional. An optional query string used to filter the
            Organizations to return in the response. Query rules are
            case-insensitive.

            ::

               | Field            | Description                                |
               |------------------|--------------------------------------------|
               | directoryCustomerId, owner.directoryCustomerId | Filters by directory
               customer id. |
               | domain           | Filters by domain.                         |

            Organizations may be queried by ``directoryCustomerId`` or
            by ``domain``, where the domain is a G Suite domain, for
            example:

            -  Query ``directorycustomerid:123456789`` returns
               Organization resources with
               ``owner.directory_customer_id`` equal to ``123456789``.
            -  Query ``domain:google.com`` returns Organization
               resources corresponding to the domain ``google.com``.
    """

    page_size: int = proto.Field(
        proto.INT32,
        number=1,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )
    query: str = proto.Field(
        proto.STRING,
        number=3,
    )


class SearchOrganizationsResponse(proto.Message):
    r"""The response returned from the ``SearchOrganizations`` method.

    Attributes:
        organizations (MutableSequence[google.cloud.resourcemanager_v3.types.Organization]):
            The list of Organizations that matched the
            search query, possibly paginated.
        next_page_token (str):
            A pagination token to be used to retrieve the
            next page of results. If the result is too large
            to fit within the page size specified in the
            request, this field will be set with a token
            that can be used to fetch the next page of
            results. If this field is empty, it indicates
            that this response contains the last page of
            results.
    """

    @property
    def raw_page(self):
        return self

    organizations: MutableSequence["Organization"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="Organization",
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteOrganizationMetadata(proto.Message):
    r"""A status object which is used as the ``metadata`` field for the
    operation returned by DeleteOrganization.

    """


class UndeleteOrganizationMetadata(proto.Message):
    r"""A status object which is used as the ``metadata`` field for the
    Operation returned by UndeleteOrganization.

    """


__all__ = tuple(sorted(__protobuf__.manifest))
