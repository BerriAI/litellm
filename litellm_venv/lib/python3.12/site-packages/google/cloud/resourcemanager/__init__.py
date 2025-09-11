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
from google.cloud.resourcemanager import gapic_version as package_version

__version__ = package_version.__version__


from google.cloud.resourcemanager_v3.services.folders.async_client import (
    FoldersAsyncClient,
)
from google.cloud.resourcemanager_v3.services.folders.client import FoldersClient
from google.cloud.resourcemanager_v3.services.organizations.async_client import (
    OrganizationsAsyncClient,
)
from google.cloud.resourcemanager_v3.services.organizations.client import (
    OrganizationsClient,
)
from google.cloud.resourcemanager_v3.services.projects.async_client import (
    ProjectsAsyncClient,
)
from google.cloud.resourcemanager_v3.services.projects.client import ProjectsClient
from google.cloud.resourcemanager_v3.services.tag_bindings.async_client import (
    TagBindingsAsyncClient,
)
from google.cloud.resourcemanager_v3.services.tag_bindings.client import (
    TagBindingsClient,
)
from google.cloud.resourcemanager_v3.services.tag_holds.async_client import (
    TagHoldsAsyncClient,
)
from google.cloud.resourcemanager_v3.services.tag_holds.client import TagHoldsClient
from google.cloud.resourcemanager_v3.services.tag_keys.async_client import (
    TagKeysAsyncClient,
)
from google.cloud.resourcemanager_v3.services.tag_keys.client import TagKeysClient
from google.cloud.resourcemanager_v3.services.tag_values.async_client import (
    TagValuesAsyncClient,
)
from google.cloud.resourcemanager_v3.services.tag_values.client import TagValuesClient
from google.cloud.resourcemanager_v3.types.folders import (
    CreateFolderMetadata,
    CreateFolderRequest,
    DeleteFolderMetadata,
    DeleteFolderRequest,
    Folder,
    GetFolderRequest,
    ListFoldersRequest,
    ListFoldersResponse,
    MoveFolderMetadata,
    MoveFolderRequest,
    SearchFoldersRequest,
    SearchFoldersResponse,
    UndeleteFolderMetadata,
    UndeleteFolderRequest,
    UpdateFolderMetadata,
    UpdateFolderRequest,
)
from google.cloud.resourcemanager_v3.types.organizations import (
    DeleteOrganizationMetadata,
    GetOrganizationRequest,
    Organization,
    SearchOrganizationsRequest,
    SearchOrganizationsResponse,
    UndeleteOrganizationMetadata,
)
from google.cloud.resourcemanager_v3.types.projects import (
    CreateProjectMetadata,
    CreateProjectRequest,
    DeleteProjectMetadata,
    DeleteProjectRequest,
    GetProjectRequest,
    ListProjectsRequest,
    ListProjectsResponse,
    MoveProjectMetadata,
    MoveProjectRequest,
    Project,
    SearchProjectsRequest,
    SearchProjectsResponse,
    UndeleteProjectMetadata,
    UndeleteProjectRequest,
    UpdateProjectMetadata,
    UpdateProjectRequest,
)
from google.cloud.resourcemanager_v3.types.tag_bindings import (
    CreateTagBindingMetadata,
    CreateTagBindingRequest,
    DeleteTagBindingMetadata,
    DeleteTagBindingRequest,
    EffectiveTag,
    ListEffectiveTagsRequest,
    ListEffectiveTagsResponse,
    ListTagBindingsRequest,
    ListTagBindingsResponse,
    TagBinding,
)
from google.cloud.resourcemanager_v3.types.tag_holds import (
    CreateTagHoldMetadata,
    CreateTagHoldRequest,
    DeleteTagHoldMetadata,
    DeleteTagHoldRequest,
    ListTagHoldsRequest,
    ListTagHoldsResponse,
    TagHold,
)
from google.cloud.resourcemanager_v3.types.tag_keys import (
    CreateTagKeyMetadata,
    CreateTagKeyRequest,
    DeleteTagKeyMetadata,
    DeleteTagKeyRequest,
    GetNamespacedTagKeyRequest,
    GetTagKeyRequest,
    ListTagKeysRequest,
    ListTagKeysResponse,
    Purpose,
    TagKey,
    UpdateTagKeyMetadata,
    UpdateTagKeyRequest,
)
from google.cloud.resourcemanager_v3.types.tag_values import (
    CreateTagValueMetadata,
    CreateTagValueRequest,
    DeleteTagValueMetadata,
    DeleteTagValueRequest,
    GetNamespacedTagValueRequest,
    GetTagValueRequest,
    ListTagValuesRequest,
    ListTagValuesResponse,
    TagValue,
    UpdateTagValueMetadata,
    UpdateTagValueRequest,
)

__all__ = (
    "FoldersClient",
    "FoldersAsyncClient",
    "OrganizationsClient",
    "OrganizationsAsyncClient",
    "ProjectsClient",
    "ProjectsAsyncClient",
    "TagBindingsClient",
    "TagBindingsAsyncClient",
    "TagHoldsClient",
    "TagHoldsAsyncClient",
    "TagKeysClient",
    "TagKeysAsyncClient",
    "TagValuesClient",
    "TagValuesAsyncClient",
    "CreateFolderMetadata",
    "CreateFolderRequest",
    "DeleteFolderMetadata",
    "DeleteFolderRequest",
    "Folder",
    "GetFolderRequest",
    "ListFoldersRequest",
    "ListFoldersResponse",
    "MoveFolderMetadata",
    "MoveFolderRequest",
    "SearchFoldersRequest",
    "SearchFoldersResponse",
    "UndeleteFolderMetadata",
    "UndeleteFolderRequest",
    "UpdateFolderMetadata",
    "UpdateFolderRequest",
    "DeleteOrganizationMetadata",
    "GetOrganizationRequest",
    "Organization",
    "SearchOrganizationsRequest",
    "SearchOrganizationsResponse",
    "UndeleteOrganizationMetadata",
    "CreateProjectMetadata",
    "CreateProjectRequest",
    "DeleteProjectMetadata",
    "DeleteProjectRequest",
    "GetProjectRequest",
    "ListProjectsRequest",
    "ListProjectsResponse",
    "MoveProjectMetadata",
    "MoveProjectRequest",
    "Project",
    "SearchProjectsRequest",
    "SearchProjectsResponse",
    "UndeleteProjectMetadata",
    "UndeleteProjectRequest",
    "UpdateProjectMetadata",
    "UpdateProjectRequest",
    "CreateTagBindingMetadata",
    "CreateTagBindingRequest",
    "DeleteTagBindingMetadata",
    "DeleteTagBindingRequest",
    "EffectiveTag",
    "ListEffectiveTagsRequest",
    "ListEffectiveTagsResponse",
    "ListTagBindingsRequest",
    "ListTagBindingsResponse",
    "TagBinding",
    "CreateTagHoldMetadata",
    "CreateTagHoldRequest",
    "DeleteTagHoldMetadata",
    "DeleteTagHoldRequest",
    "ListTagHoldsRequest",
    "ListTagHoldsResponse",
    "TagHold",
    "CreateTagKeyMetadata",
    "CreateTagKeyRequest",
    "DeleteTagKeyMetadata",
    "DeleteTagKeyRequest",
    "GetNamespacedTagKeyRequest",
    "GetTagKeyRequest",
    "ListTagKeysRequest",
    "ListTagKeysResponse",
    "TagKey",
    "UpdateTagKeyMetadata",
    "UpdateTagKeyRequest",
    "Purpose",
    "CreateTagValueMetadata",
    "CreateTagValueRequest",
    "DeleteTagValueMetadata",
    "DeleteTagValueRequest",
    "GetNamespacedTagValueRequest",
    "GetTagValueRequest",
    "ListTagValuesRequest",
    "ListTagValuesResponse",
    "TagValue",
    "UpdateTagValueMetadata",
    "UpdateTagValueRequest",
)
