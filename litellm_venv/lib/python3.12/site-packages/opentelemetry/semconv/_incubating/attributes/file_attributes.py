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


FILE_DIRECTORY = "file.directory"
"""
Directory where the file is located. It should include the drive letter, when appropriate.
"""

FILE_EXTENSION = "file.extension"
"""
File extension, excluding the leading dot.
Note: When the file name has multiple extensions (example.tar.gz), only the last one should be captured ("gz", not "tar.gz").
"""

FILE_NAME = "file.name"
"""
Name of the file including the extension, without the directory.
"""

FILE_PATH = "file.path"
"""
Full path to the file, including the file name. It should include the drive letter, when appropriate.
"""

FILE_SIZE = "file.size"
"""
File size in bytes.
"""
