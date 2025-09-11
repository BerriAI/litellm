# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
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

from pathlib import Path


def _is_relative_to(path: str, to_path: str) -> bool:
    """Returns whether or not this path is relative to another path.

    This function can be replacted by Path.is_relative_to which is availble in Python 3.9+.

    Args:
        path (str):
            Required. The path to check whether it is relative to the other path.
        to_path (str):
            Required. The path to check whether the other path is relative to it.

    Returns:
        Whether the path is relative to another path.
    """
    try:
        Path(path).expanduser().resolve().relative_to(
            Path(to_path).expanduser().resolve()
        )
        return True
    except ValueError:
        return False
