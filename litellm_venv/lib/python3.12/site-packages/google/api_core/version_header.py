# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

API_VERSION_METADATA_KEY = "x-goog-api-version"


def to_api_version_header(version_identifier):
    """Returns data for the API Version header for the given `version_identifier`.

    Args:
        version_identifier (str): The version identifier to be used in the
            tuple returned.

    Returns:
        Tuple(str, str): A tuple containing the API Version metadata key and
            value.
    """
    return (API_VERSION_METADATA_KEY, version_identifier)
