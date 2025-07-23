# Copyright 2025 CloudZero
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
# CHANGELOG: 2025-01-19 - Migrated from pandas to polars and requests to httpx (erik.peterson)
# CHANGELOG: 2025-01-19 - Initial creation of LiteLLM to CloudZero ETL tool (erik.peterson)

"""LiteLLM to CloudZero AnyCost ETL Tool."""

from .cli import app

__version__ = "0.1.0"
__all__ = ["app"]
