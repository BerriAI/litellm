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

# FIXME: Use importlib.metadata when support for 3.11 is dropped if the rest of
# the supported versions at that time have the same API.
from importlib_metadata import (  # type: ignore
    EntryPoint,
    EntryPoints,
    entry_points,
    version,
)

# The importlib-metadata library has introduced breaking changes before to its
# API, this module is kept just to act as a layer between the
# importlib-metadata library and our project if in any case it is necessary to
# do so.

__all__ = ["entry_points", "version", "EntryPoint", "EntryPoints"]
