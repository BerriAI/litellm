# Copyright The OpenTelemetry Authors
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


FEATURE_FLAG_KEY = "feature_flag.key"
"""
The unique identifier of the feature flag.
"""

FEATURE_FLAG_PROVIDER_NAME = "feature_flag.provider_name"
"""
The name of the service provider that performs the flag evaluation.
"""

FEATURE_FLAG_VARIANT = "feature_flag.variant"
"""
SHOULD be a semantic identifier for a value. If one is unavailable, a stringified version of the value can be used.
Note: A semantic identifier, commonly referred to as a variant, provides a means
    for referring to a value without including the value itself. This can
    provide additional context for understanding the meaning behind a value.
    For example, the variant `red` maybe be used for the value `#c05543`.

    A stringified version of the value can be used in situations where a
    semantic identifier is unavailable. String representation of the value
    should be determined by the implementer.
"""
