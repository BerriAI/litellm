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

from enum import Enum


class Schemas(Enum):
    V1_23_1 = "https://opentelemetry.io/schemas/1.23.1"
    """
    The URL of the OpenTelemetry schema version 1.23.1.
    """

    V1_25_0 = "https://opentelemetry.io/schemas/v1.25.0"
    """
    The URL of the OpenTelemetry schema version v1.25.0.
    """

    # when generating new semantic conventions,
    # make sure to add new versions version here.
