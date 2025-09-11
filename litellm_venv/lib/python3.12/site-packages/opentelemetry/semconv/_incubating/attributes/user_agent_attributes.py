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


USER_AGENT_NAME = "user_agent.name"
"""
Name of the user-agent extracted from original. Usually refers to the browser's name.
Note: [Example](https://www.whatsmyua.info) of extracting browser's name from original string. In the case of using a user-agent for non-browser products, such as microservices with multiple names/versions inside the `user_agent.original`, the most significant name SHOULD be selected. In such a scenario it should align with `user_agent.version`.
"""

USER_AGENT_ORIGINAL = "user_agent.original"
"""
Deprecated in favor of stable :py:const:`opentelemetry.semconv.attributes.user_agent_attributes.USER_AGENT_ORIGINAL`.
"""

USER_AGENT_VERSION = "user_agent.version"
"""
Version of the user-agent extracted from original. Usually refers to the browser's version.
Note: [Example](https://www.whatsmyua.info) of extracting browser's version from original string. In the case of using a user-agent for non-browser products, such as microservices with multiple names/versions inside the `user_agent.original`, the most significant version SHOULD be selected. In such a scenario it should align with `user_agent.name`.
"""
