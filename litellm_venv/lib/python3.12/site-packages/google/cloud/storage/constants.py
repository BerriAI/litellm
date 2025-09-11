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

"""Constants used across google.cloud.storage modules.

See [Python Storage Client Constants Page](https://github.com/googleapis/python-storage/blob/main/google/cloud/storage/constants.py)
for constants used across storage classes, location types, public access prevention, etc.

"""

# Storage classes

STANDARD_STORAGE_CLASS = "STANDARD"
"""Storage class for objects accessed more than once per month.

See: https://cloud.google.com/storage/docs/storage-classes
"""

NEARLINE_STORAGE_CLASS = "NEARLINE"
"""Storage class for objects accessed at most once per month.

See: https://cloud.google.com/storage/docs/storage-classes
"""

COLDLINE_STORAGE_CLASS = "COLDLINE"
"""Storage class for objects accessed at most once per year.

See: https://cloud.google.com/storage/docs/storage-classes
"""

ARCHIVE_STORAGE_CLASS = "ARCHIVE"
"""Storage class for objects accessed less frequently than once per year.

See: https://cloud.google.com/storage/docs/storage-classes
"""

MULTI_REGIONAL_LEGACY_STORAGE_CLASS = "MULTI_REGIONAL"
"""Legacy storage class.

Alias for :attr:`STANDARD_STORAGE_CLASS`.

Can only be used for objects in buckets whose
:attr:`~google.cloud.storage.bucket.Bucket.location_type` is
:attr:`~google.cloud.storage.bucket.Bucket.MULTI_REGION_LOCATION_TYPE`.

See: https://cloud.google.com/storage/docs/storage-classes
"""

REGIONAL_LEGACY_STORAGE_CLASS = "REGIONAL"
"""Legacy storage class.

Alias for :attr:`STANDARD_STORAGE_CLASS`.

Can only be used for objects in buckets whose
:attr:`~google.cloud.storage.bucket.Bucket.location_type` is
:attr:`~google.cloud.storage.bucket.Bucket.REGION_LOCATION_TYPE`.

See: https://cloud.google.com/storage/docs/storage-classes
"""

DURABLE_REDUCED_AVAILABILITY_LEGACY_STORAGE_CLASS = "DURABLE_REDUCED_AVAILABILITY"
"""Legacy storage class.

Similar to :attr:`NEARLINE_STORAGE_CLASS`.
"""


# Location types

MULTI_REGION_LOCATION_TYPE = "multi-region"
"""Location type: data will be replicated across regions in a multi-region.

Provides highest availability across largest area.
"""

REGION_LOCATION_TYPE = "region"
"""Location type: data will be stored within a single region.

Provides lowest latency within a single region.
"""

DUAL_REGION_LOCATION_TYPE = "dual-region"
"""Location type: data will be stored within two primary regions.

Provides high availability and low latency across two regions.
"""


# Internal constants

_DEFAULT_TIMEOUT = 60  # in seconds
"""The default request timeout in seconds if a timeout is not explicitly given.
"""

# Public Access Prevention
PUBLIC_ACCESS_PREVENTION_ENFORCED = "enforced"
"""Enforced public access prevention value.

See: https://cloud.google.com/storage/docs/public-access-prevention
"""

PUBLIC_ACCESS_PREVENTION_UNSPECIFIED = "unspecified"
"""Unspecified public access prevention value.

DEPRECATED: Use 'PUBLIC_ACCESS_PREVENTION_INHERITED' instead.

See: https://cloud.google.com/storage/docs/public-access-prevention
"""

PUBLIC_ACCESS_PREVENTION_INHERITED = "inherited"
"""Inherited public access prevention value.

See: https://cloud.google.com/storage/docs/public-access-prevention
"""

RPO_ASYNC_TURBO = "ASYNC_TURBO"
"""The recovery point objective (RPO) indicates how quickly newly written objects are asynchronously replicated to a separate geographic location.
When the RPO value is set to ASYNC_TURBO, the turbo replication feature is enabled.

See: https://cloud.google.com/storage/docs/managing-turbo-replication
"""

RPO_DEFAULT = "DEFAULT"
"""The recovery point objective (RPO) indicates how quickly newly written objects are asynchronously replicated to a separate geographic location.
When the RPO value is set to DEFAULT, the default replication behavior is enabled.

See: https://cloud.google.com/storage/docs/managing-turbo-replication
"""
