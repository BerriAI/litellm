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

from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.cloud.resourcemanager.v3",
    manifest={
        "Project",
        "GetProjectRequest",
        "ListProjectsRequest",
        "ListProjectsResponse",
        "SearchProjectsRequest",
        "SearchProjectsResponse",
        "CreateProjectRequest",
        "CreateProjectMetadata",
        "UpdateProjectRequest",
        "UpdateProjectMetadata",
        "MoveProjectRequest",
        "MoveProjectMetadata",
        "DeleteProjectRequest",
        "DeleteProjectMetadata",
        "UndeleteProjectRequest",
        "UndeleteProjectMetadata",
    },
)


class Project(proto.Message):
    r"""A project is a high-level Google Cloud entity. It is a
    container for ACLs, APIs, App Engine Apps, VMs, and other Google
    Cloud Platform resources.

    Attributes:
        name (str):
            Output only. The unique resource name of the project. It is
            an int64 generated number prefixed by "projects/".

            Example: ``projects/415104041262``
        parent (str):
            Optional. A reference to a parent Resource. eg.,
            ``organizations/123`` or ``folders/876``.
        project_id (str):
            Immutable. The unique, user-assigned id of the project. It
            must be 6 to 30 lowercase ASCII letters, digits, or hyphens.
            It must start with a letter. Trailing hyphens are
            prohibited.

            Example: ``tokyo-rain-123``
        state (google.cloud.resourcemanager_v3.types.Project.State):
            Output only. The project lifecycle state.
        display_name (str):
            Optional. A user-assigned display name of the project. When
            present it must be between 4 to 30 characters. Allowed
            characters are: lowercase and uppercase letters, numbers,
            hyphen, single-quote, double-quote, space, and exclamation
            point.

            Example: ``My Project``
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Creation time.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The most recent time this
            resource was modified.
        delete_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time at which this resource
            was requested for deletion.
        etag (str):
            Output only. A checksum computed by the
            server based on the current value of the Project
            resource. This may be sent on update and delete
            requests to ensure the client has an up-to-date
            value before proceeding.
        labels (MutableMapping[str, str]):
            Optional. The labels associated with this project.

            Label keys must be between 1 and 63 characters long and must
            conform to the following regular expression:
            [a-z]([-a-z0-9]*[a-z0-9])?.

            Label values must be between 0 and 63 characters long and
            must conform to the regular expression
            ([a-z]([-a-z0-9]*[a-z0-9])?)?.

            No more than 64 labels can be associated with a given
            resource.

            Clients should store labels in a representation such as JSON
            that does not depend on specific characters being
            disallowed.

            Example: ``"myBusinessDimension" : "businessValue"``
    """

    class State(proto.Enum):
        r"""Project lifecycle states.

        Values:
            STATE_UNSPECIFIED (0):
                Unspecified state.  This is only used/useful
                for distinguishing unset values.
            ACTIVE (1):
                The normal and active state.
            DELETE_REQUESTED (2):
                The project has been marked for deletion by the user (by
                invoking
                [DeleteProject][google.cloud.resourcemanager.v3.Projects.DeleteProject])
                or by the system (Google Cloud Platform). This can generally
                be reversed by invoking [UndeleteProject]
                [google.cloud.resourcemanager.v3.Projects.UndeleteProject].
        """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        DELETE_REQUESTED = 2

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    parent: str = proto.Field(
        proto.STRING,
        number=2,
    )
    project_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=4,
        enum=State,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=5,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    delete_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=8,
        message=timestamp_pb2.Timestamp,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=9,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=10,
    )


class GetProjectRequest(proto.Message):
    r"""The request sent to the
    [GetProject][google.cloud.resourcemanager.v3.Projects.GetProject]
    method.

    Attributes:
        name (str):
            Required. The name of the project (for example,
            ``projects/415104041262``).
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListProjectsRequest(proto.Message):
    r"""The request sent to the
    [ListProjects][google.cloud.resourcemanager.v3.Projects.ListProjects]
    method.

    Attributes:
        parent (str):
            Required. The name of the parent resource whose projects are
            being listed. Only children of this parent resource are
            listed; descendants are not listed.

            If the parent is a folder, use the value
            ``folders/{folder_id}``. If the parent is an organization,
            use the value ``organizations/{org_id}``.
        page_token (str):
            Optional. A pagination token returned from a previous call
            to [ListProjects]
            [google.cloud.resourcemanager.v3.Projects.ListProjects] that
            indicates from where listing should continue.
        page_size (int):
            Optional. The maximum number of projects to
            return in the response. The server can return
            fewer projects than requested. If unspecified,
            server picks an appropriate default.
        show_deleted (bool):
            Optional. Indicate that projects in the ``DELETE_REQUESTED``
            state should also be returned. Normally only ``ACTIVE``
            projects are returned.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    show_deleted: bool = proto.Field(
        proto.BOOL,
        number=4,
    )


class ListProjectsResponse(proto.Message):
    r"""A page of the response received from the
    [ListProjects][google.cloud.resourcemanager.v3.Projects.ListProjects]
    method.

    A paginated response where more pages are available has
    ``next_page_token`` set. This token can be used in a subsequent
    request to retrieve the next request page.

    NOTE: A response may contain fewer elements than the request
    ``page_size`` and still have a ``next_page_token``.

    Attributes:
        projects (MutableSequence[google.cloud.resourcemanager_v3.types.Project]):
            The list of Projects under the parent. This
            list can be paginated.
        next_page_token (str):
            Pagination token.

            If the result set is too large to fit in a single response,
            this token is returned. It encodes the position of the
            current result cursor. Feeding this value into a new list
            request with the ``page_token`` parameter gives the next
            page of the results.

            When ``next_page_token`` is not filled in, there is no next
            page and the list returned is the last page in the result
            set.

            Pagination tokens have a limited lifetime.
    """

    @property
    def raw_page(self):
        return self

    projects: MutableSequence["Project"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="Project",
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class SearchProjectsRequest(proto.Message):
    r"""The request sent to the
    [SearchProjects][google.cloud.resourcemanager.v3.Projects.SearchProjects]
    method.

    Attributes:
        query (str):
            Optional. A query string for searching for projects that the
            caller has ``resourcemanager.projects.get`` permission to.
            If multiple fields are included in the query, then it will
            return results that match any of the fields. Some eligible
            fields are:

            -  **``displayName``, ``name``**: Filters by displayName.
            -  **``parent``**: Project's parent (for example:
               ``folders/123``, ``organizations/*``). Prefer ``parent``
               field over ``parent.type`` and ``parent.id``.
            -  **``parent.type``**: Parent's type: ``folder`` or
               ``organization``.
            -  **``parent.id``**: Parent's id number (for example:
               ``123``).
            -  **``id``, ``projectId``**: Filters by projectId.
            -  **``state``, ``lifecycleState``**: Filters by state.
            -  **``labels``**: Filters by label name or value.
            -  **``labels.<key>`` (where ``<key>`` is the name of a
               label)**: Filters by label name.

            Search expressions are case insensitive.

            Some examples queries:

            -  **``name:how*``**: The project's name starts with "how".
            -  **``name:Howl``**: The project's name is ``Howl`` or
               ``howl``.
            -  **``name:HOWL``**: Equivalent to above.
            -  **``NAME:howl``**: Equivalent to above.
            -  **``labels.color:*``**: The project has the label
               ``color``.
            -  **``labels.color:red``**: The project's label ``color``
               has the value ``red``.
            -  **``labels.color:red labels.size:big``**: The project's
               label ``color`` has the value ``red`` or its label
               ``size`` has the value ``big``.

            If no query is specified, the call will return projects for
            which the user has the ``resourcemanager.projects.get``
            permission.
        page_token (str):
            Optional. A pagination token returned from a previous call
            to [ListProjects]
            [google.cloud.resourcemanager.v3.Projects.ListProjects] that
            indicates from where listing should continue.
        page_size (int):
            Optional. The maximum number of projects to
            return in the response. The server can return
            fewer projects than requested. If unspecified,
            server picks an appropriate default.
    """

    query: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )


class SearchProjectsResponse(proto.Message):
    r"""A page of the response received from the
    [SearchProjects][google.cloud.resourcemanager.v3.Projects.SearchProjects]
    method.

    A paginated response where more pages are available has
    ``next_page_token`` set. This token can be used in a subsequent
    request to retrieve the next request page.

    Attributes:
        projects (MutableSequence[google.cloud.resourcemanager_v3.types.Project]):
            The list of Projects that matched the list
            filter query. This list can be paginated.
        next_page_token (str):
            Pagination token.

            If the result set is too large to fit in a single response,
            this token is returned. It encodes the position of the
            current result cursor. Feeding this value into a new list
            request with the ``page_token`` parameter gives the next
            page of the results.

            When ``next_page_token`` is not filled in, there is no next
            page and the list returned is the last page in the result
            set.

            Pagination tokens have a limited lifetime.
    """

    @property
    def raw_page(self):
        return self

    projects: MutableSequence["Project"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="Project",
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class CreateProjectRequest(proto.Message):
    r"""The request sent to the
    [CreateProject][google.cloud.resourcemanager.v3.Projects.CreateProject]
    method.

    Attributes:
        project (google.cloud.resourcemanager_v3.types.Project):
            Required. The Project to create.

            Project ID is required. If the requested ID is unavailable,
            the request fails.

            If the ``parent`` field is set, the
            ``resourcemanager.projects.create`` permission is checked on
            the parent resource. If no parent is set and the
            authorization credentials belong to an Organization, the
            parent will be set to that Organization.
    """

    project: "Project" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="Project",
    )


class CreateProjectMetadata(proto.Message):
    r"""A status object which is used as the ``metadata`` field for the
    Operation returned by CreateProject. It provides insight for when
    significant phases of Project creation have completed.

    Attributes:
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Creation time of the project creation
            workflow.
        gettable (bool):
            True if the project can be retrieved using ``GetProject``.
            No other operations on the project are guaranteed to work
            until the project creation is complete.
        ready (bool):
            True if the project creation process is
            complete.
    """

    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )
    gettable: bool = proto.Field(
        proto.BOOL,
        number=2,
    )
    ready: bool = proto.Field(
        proto.BOOL,
        number=3,
    )


class UpdateProjectRequest(proto.Message):
    r"""The request sent to the
    [UpdateProject][google.cloud.resourcemanager.v3.Projects.UpdateProject]
    method.

    Only the ``display_name`` and ``labels`` fields can be change. Use
    the
    [MoveProject][google.cloud.resourcemanager.v3.Projects.MoveProject]
    method to change the ``parent`` field.

    Attributes:
        project (google.cloud.resourcemanager_v3.types.Project):
            Required. The new definition of the project.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. An update mask to selectively
            update fields.
    """

    project: "Project" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="Project",
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class UpdateProjectMetadata(proto.Message):
    r"""A status object which is used as the ``metadata`` field for the
    Operation returned by UpdateProject.

    """


class MoveProjectRequest(proto.Message):
    r"""The request sent to
    [MoveProject][google.cloud.resourcemanager.v3.Projects.MoveProject]
    method.

    Attributes:
        name (str):
            Required. The name of the project to move.
        destination_parent (str):
            Required. The new parent to move the Project
            under.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    destination_parent: str = proto.Field(
        proto.STRING,
        number=2,
    )


class MoveProjectMetadata(proto.Message):
    r"""A status object which is used as the ``metadata`` field for the
    Operation returned by MoveProject.

    """


class DeleteProjectRequest(proto.Message):
    r"""[DeleteProject][google.cloud.resourcemanager.v3.Projects.DeleteProject]
    method.

    Attributes:
        name (str):
            Required. The name of the Project (for example,
            ``projects/415104041262``).
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class DeleteProjectMetadata(proto.Message):
    r"""A status object which is used as the ``metadata`` field for the
    Operation returned by ``DeleteProject``.

    """


class UndeleteProjectRequest(proto.Message):
    r"""The request sent to the [UndeleteProject]
    [google.cloud.resourcemanager.v3.Projects.UndeleteProject] method.

    Attributes:
        name (str):
            Required. The name of the project (for example,
            ``projects/415104041262``).

            Required.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UndeleteProjectMetadata(proto.Message):
    r"""A status object which is used as the ``metadata`` field for the
    Operation returned by ``UndeleteProject``.

    """


__all__ = tuple(sorted(__protobuf__.manifest))
