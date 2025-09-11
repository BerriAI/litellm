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

"""Comprehensive list of environment variables used in google-cloud.

These enable many types of implicit behavior in both production
and tests.
"""

GCD_DATASET = "DATASTORE_DATASET"
"""Environment variable defining default dataset ID under GCD."""

GCD_HOST = "DATASTORE_EMULATOR_HOST"
"""Environment variable defining host for GCD dataset server."""

PUBSUB_EMULATOR = "PUBSUB_EMULATOR_HOST"
"""Environment variable defining host for Pub/Sub emulator."""

BIGTABLE_EMULATOR = "BIGTABLE_EMULATOR_HOST"
"""Environment variable defining host for Bigtable emulator."""

DISABLE_GRPC = "GOOGLE_CLOUD_DISABLE_GRPC"
"""Environment variable acting as flag to disable gRPC.

To be used for APIs where both an HTTP and gRPC implementation
exist.
"""
