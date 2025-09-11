# Copyright 2020 Google LLC
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
"""BigQuery API IAM policy definitions

For all allowed roles and permissions, see:

https://cloud.google.com/bigquery/docs/access-control
"""

# BigQuery-specific IAM roles available for tables and views

BIGQUERY_DATA_EDITOR_ROLE = "roles/bigquery.dataEditor"
"""When applied to a table or view, this role provides permissions to
read and update data and metadata for the table or view."""

BIGQUERY_DATA_OWNER_ROLE = "roles/bigquery.dataOwner"
"""When applied to a table or view, this role provides permissions to
read and update data and metadata for the table or view, share the
table/view, and delete the table/view."""

BIGQUERY_DATA_VIEWER_ROLE = "roles/bigquery.dataViewer"
"""When applied to a table or view, this role provides permissions to
read data and metadata from the table or view."""

BIGQUERY_METADATA_VIEWER_ROLE = "roles/bigquery.metadataViewer"
"""When applied to a table or view, this role provides persmissions to
read metadata from the table or view."""
