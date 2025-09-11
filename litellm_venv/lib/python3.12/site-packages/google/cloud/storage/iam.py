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
"""Storage API IAM policy definitions

For allowed roles / permissions, see:
https://cloud.google.com/storage/docs/access-control/iam
"""

# Storage-specific IAM roles

STORAGE_OBJECT_CREATOR_ROLE = "roles/storage.objectCreator"
"""Role implying rights to create objects, but not delete or overwrite them."""

STORAGE_OBJECT_VIEWER_ROLE = "roles/storage.objectViewer"
"""Role implying rights to view object properties, excluding ACLs."""

STORAGE_OBJECT_ADMIN_ROLE = "roles/storage.objectAdmin"
"""Role implying full control of objects."""

STORAGE_ADMIN_ROLE = "roles/storage.admin"
"""Role implying full control of objects and buckets."""

STORAGE_VIEWER_ROLE = "Viewer"
"""Can list buckets."""

STORAGE_EDITOR_ROLE = "Editor"
"""Can create, list, and delete buckets."""

STORAGE_OWNER_ROLE = "Owners"
"""Can create, list, and delete buckets."""


# Storage-specific permissions

STORAGE_BUCKETS_CREATE = "storage.buckets.create"
"""Permission: create buckets."""

STORAGE_BUCKETS_DELETE = "storage.buckets.delete"
"""Permission: delete buckets."""

STORAGE_BUCKETS_GET = "storage.buckets.get"
"""Permission: read bucket metadata, excluding ACLs."""

STORAGE_BUCKETS_GET_IAM_POLICY = "storage.buckets.getIamPolicy"
"""Permission: read bucket ACLs."""

STORAGE_BUCKETS_LIST = "storage.buckets.list"
"""Permission: list buckets."""

STORAGE_BUCKETS_SET_IAM_POLICY = "storage.buckets.setIamPolicy"
"""Permission: update bucket ACLs."""

STORAGE_BUCKETS_UPDATE = "storage.buckets.list"
"""Permission: update buckets, excluding ACLS."""

STORAGE_OBJECTS_CREATE = "storage.objects.create"
"""Permission: add new objects to a bucket."""

STORAGE_OBJECTS_DELETE = "storage.objects.delete"
"""Permission: delete objects."""

STORAGE_OBJECTS_GET = "storage.objects.get"
"""Permission: read object data / metadata, excluding ACLs."""

STORAGE_OBJECTS_GET_IAM_POLICY = "storage.objects.getIamPolicy"
"""Permission: read object ACLs."""

STORAGE_OBJECTS_LIST = "storage.objects.list"
"""Permission: list objects in a bucket."""

STORAGE_OBJECTS_SET_IAM_POLICY = "storage.objects.setIamPolicy"
"""Permission: update object ACLs."""

STORAGE_OBJECTS_UPDATE = "storage.objects.update"
"""Permission: update object metadat, excluding ACLs."""
