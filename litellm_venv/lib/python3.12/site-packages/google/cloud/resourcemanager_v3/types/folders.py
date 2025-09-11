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
        "Folder",
        "GetFolderRequest",
        "ListFoldersRequest",
        "ListFoldersResponse",
        "SearchFoldersRequest",
        "SearchFoldersResponse",
        "CreateFolderRequest",
        "CreateFolderMetadata",
        "UpdateFolderRequest",
        "UpdateFolderMetadata",
        "MoveFolderRequest",
        "MoveFolderMetadata",
        "DeleteFolderRequest",
        "DeleteFolderMetadata",
        "UndeleteFolderRequest",
        "UndeleteFolderMetadata",
    },
)


class Folder(proto.Message):
    r"""A folder in an organization's resource hierarchy, used to
    organize that organization's resources.

    Attributes:
        name (str):
            Output only. The resource name of the folder. Its format is
            ``folders/{folder_id}``, for example: "folders/1234".
        parent (str):
            Required. The folder's parent's resource name. Updates to
            the folder's parent must be performed using
            [MoveFolder][google.cloud.resourcemanager.v3.Folders.MoveFolder].
        display_name (str):
            The folder's display name. A folder's display name must be
            unique amongst its siblings. For example, no two folders
            with the same parent can share the same display name. The
            display name must start and end with a letter or digit, may
            contain letters, digits, spaces, hyphens and underscores and
            can be no longer than 30 characters. This is captured by the
            regular expression:
            ``[\p{L}\p{N}]([\p{L}\p{N}_- ]{0,28}[\p{L}\p{N}])?``.
        state (google.cloud.resourcemanager_v3.types.Folder.State):
            Output only. The lifecycle state of the folder. Updates to
            the state must be performed using
            [DeleteFolder][google.cloud.resourcemanager.v3.Folders.DeleteFolder]
            and
            [UndeleteFolder][google.cloud.resourcemanager.v3.Folders.UndeleteFolder].
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when the folder was
            created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when the folder was
            last modified.
        delete_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when the folder was
            requested to be deleted.
        etag (str):
            Output only. A checksum computed by the
            server based on the current value of the folder
            resource. This may be sent on update and delete
            requests to ensure the client has an up-to-date
            value before proceeding.
    """

    class State(proto.Enum):
        r"""Folder lifecycle states.

        Values:
            STATE_UNSPECIFIED (0):
                Unspecified state.
            ACTIVE (1):
                The normal and active state.
            DELETE_REQUESTED (2):
                The folder has been marked for deletion by
                the user.
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
    display_name: str = proto.Field(
        proto.STRING,
        number=3,
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


class GetFolderRequest(proto.Message):
    r"""The GetFolder request message.

    Attributes:
        name (str):
            Required. The resource name of the folder to retrieve. Must
            be of the form ``folders/{folder_id}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListFoldersRequest(proto.Message):
    r"""The ListFolders request message.

    Attributes:
        parent (str):
            Required. The name of the parent resource whose folders are
            being listed. Only children of this parent resource are
            listed; descendants are not listed.

            If the parent is a folder, use the value
            ``folders/{folder_id}``. If the parent is an organization,
            use the value ``organizations/{org_id}``.

            Access to this method is controlled by checking the
            ``resourcemanager.folders.list`` permission on the
            ``parent``.
        page_size (int):
            Optional. The maximum number of folders to
            return in the response. The server can return
            fewer folders than requested. If unspecified,
            server picks an appropriate default.
        page_token (str):
            Optional. A pagination token returned from a previous call
            to ``ListFolders`` that indicates where this listing should
            continue from.
        show_deleted (bool):
            Optional. Controls whether folders in the
            [DELETE_REQUESTED][google.cloud.resourcemanager.v3.Folder.State.DELETE_REQUESTED]
            state should be returned. Defaults to false.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )
    show_deleted: bool = proto.Field(
        proto.BOOL,
        number=4,
    )


class ListFoldersResponse(proto.Message):
    r"""The ListFolders response message.

    Attributes:
        folders (MutableSequence[google.cloud.resourcemanager_v3.types.Folder]):
            A possibly paginated list of folders that are
            direct descendants of the specified parent
            resource.
        next_page_token (str):
            A pagination token returned from a previous call to
            ``ListFolders`` that indicates from where listing should
            continue.
    """

    @property
    def raw_page(self):
        return self

    folders: MutableSequence["Folder"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="Folder",
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class SearchFoldersRequest(proto.Message):
    r"""The request message for searching folders.

    Attributes:
        page_size (int):
            Optional. The maximum number of folders to
            return in the response. The server can return
            fewer folders than requested. If unspecified,
            server picks an appropriate default.
        page_token (str):
            Optional. A pagination token returned from a previous call
            to ``SearchFolders`` that indicates from where search should
            continue.
        query (str):
            Optional. Search criteria used to select the folders to
            return. If no search criteria is specified then all
            accessible folders will be returned.

            Query expressions can be used to restrict results based upon
            displayName, state and parent, where the operators ``=``
            (``:``) ``NOT``, ``AND`` and ``OR`` can be used along with
            the suffix wildcard symbol ``*``.

            The ``displayName`` field in a query expression should use
            escaped quotes for values that include whitespace to prevent
            unexpected behavior.

            ::

               | Field                   | Description                            |
               |-------------------------|----------------------------------------|
               | displayName             | Filters by displayName.                |
               | parent                  | Filters by parent (for example: folders/123). |
               | state, lifecycleState   | Filters by state.                      |

            Some example queries are:

            -  Query ``displayName=Test*`` returns Folder resources
               whose display name starts with "Test".
            -  Query ``state=ACTIVE`` returns Folder resources with
               ``state`` set to ``ACTIVE``.
            -  Query ``parent=folders/123`` returns Folder resources
               that have ``folders/123`` as a parent resource.
            -  Query ``parent=folders/123 AND state=ACTIVE`` returns
               active Folder resources that have ``folders/123`` as a
               parent resource.
            -  Query ``displayName=\\"Test String\\"`` returns Folder
               resources with display names that include both "Test" and
               "String".
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


class SearchFoldersResponse(proto.Message):
    r"""The response message for searching folders.

    Attributes:
        folders (MutableSequence[google.cloud.resourcemanager_v3.types.Folder]):
            A possibly paginated folder search results.
            the specified parent resource.
        next_page_token (str):
            A pagination token returned from a previous call to
            ``SearchFolders`` that indicates from where searching should
            continue.
    """

    @property
    def raw_page(self):
        return self

    folders: MutableSequence["Folder"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="Folder",
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class CreateFolderRequest(proto.Message):
    r"""The CreateFolder request message.

    Attributes:
        folder (google.cloud.resourcemanager_v3.types.Folder):
            Required. The folder being created, only the
            display name and parent will be consulted. All
            other fields will be ignored.
    """

    folder: "Folder" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="Folder",
    )


class CreateFolderMetadata(proto.Message):
    r"""Metadata pertaining to the Folder creation process.

    Attributes:
        display_name (str):
            The display name of the folder.
        parent (str):
            The resource name of the folder or
            organization we are creating the folder under.
    """

    display_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    parent: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdateFolderRequest(proto.Message):
    r"""The request sent to the
    [UpdateFolder][google.cloud.resourcemanager.v3.Folder.UpdateFolder]
    method.

    Only the ``display_name`` field can be changed. All other fields
    will be ignored. Use the
    [MoveFolder][google.cloud.resourcemanager.v3.Folders.MoveFolder]
    method to change the ``parent`` field.

    Attributes:
        folder (google.cloud.resourcemanager_v3.types.Folder):
            Required. The new definition of the Folder. It must include
            the ``name`` field, which cannot be changed.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. Fields to be updated. Only the ``display_name``
            can be updated.
    """

    folder: "Folder" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="Folder",
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class UpdateFolderMetadata(proto.Message):
    r"""A status object which is used as the ``metadata`` field for the
    Operation returned by UpdateFolder.

    """


class MoveFolderRequest(proto.Message):
    r"""The MoveFolder request message.

    Attributes:
        name (str):
            Required. The resource name of the Folder to move. Must be
            of the form folders/{folder_id}
        destination_parent (str):
            Required. The resource name of the folder or organization
            which should be the folder's new parent. Must be of the form
            ``folders/{folder_id}`` or ``organizations/{org_id}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    destination_parent: str = proto.Field(
        proto.STRING,
        number=2,
    )


class MoveFolderMetadata(proto.Message):
    r"""Metadata pertaining to the folder move process.

    Attributes:
        display_name (str):
            The display name of the folder.
        source_parent (str):
            The resource name of the folder's parent.
        destination_parent (str):
            The resource name of the folder or
            organization to move the folder to.
    """

    display_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    source_parent: str = proto.Field(
        proto.STRING,
        number=2,
    )
    destination_parent: str = proto.Field(
        proto.STRING,
        number=3,
    )


class DeleteFolderRequest(proto.Message):
    r"""The DeleteFolder request message.

    Attributes:
        name (str):
            Required. The resource name of the folder to be deleted.
            Must be of the form ``folders/{folder_id}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class DeleteFolderMetadata(proto.Message):
    r"""A status object which is used as the ``metadata`` field for the
    ``Operation`` returned by ``DeleteFolder``.

    """


class UndeleteFolderRequest(proto.Message):
    r"""The UndeleteFolder request message.

    Attributes:
        name (str):
            Required. The resource name of the folder to undelete. Must
            be of the form ``folders/{folder_id}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UndeleteFolderMetadata(proto.Message):
    r"""A status object which is used as the ``metadata`` field for the
    ``Operation`` returned by ``UndeleteFolder``.

    """


__all__ = tuple(sorted(__protobuf__.manifest))
