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


CODE_COLUMN = "code.column"
"""
The column number in `code.filepath` best representing the operation. It SHOULD point within the code unit named in `code.function`.
"""

CODE_FILEPATH = "code.filepath"
"""
The source code file name that identifies the code unit as uniquely as possible (preferably an absolute file path).
"""

CODE_FUNCTION = "code.function"
"""
The method or function name, or equivalent (usually rightmost part of the code unit's name).
"""

CODE_LINENO = "code.lineno"
"""
The line number in `code.filepath` best representing the operation. It SHOULD point within the code unit named in `code.function`.
"""

CODE_NAMESPACE = "code.namespace"
"""
The "namespace" within which `code.function` is defined. Usually the qualified class or module name, such that `code.namespace` + some separator + `code.function` form a unique identifier for the code unit.
"""

CODE_STACKTRACE = "code.stacktrace"
"""
A stacktrace as a string in the natural representation for the language runtime. The representation is to be determined and documented by each language SIG.
"""
