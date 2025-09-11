# -*- coding: utf-8 -*-

# Copyright 2024 Google LLC
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

import contextlib

_tool_names_to_append = []


@contextlib.contextmanager
def tool_context_manager(tool_name: str) -> None:
    """Context manager for appending tool name to client instantiations.

    Most client instantiations occur at construction time. There are a few
    exceptions such as generate_content that uses lazy instantiation at
    inference time (b/328511605).

    Example Usage:

        aiplatform.init(...)
        with telemetry.tool_context_manager('ClientName'):
            model = GenerativeModel("gemini-pro")
            responses = model.generate_content("Why is the sky blue?", stream=True)

    Args:
        tool_name: The name of the client library to attribute usage to

    Returns:
        None
    """
    _append_tool_name(tool_name)
    try:
        yield
    finally:
        _pop_tool_name(tool_name)


def _append_tool_name(tool_name: str) -> None:
    _tool_names_to_append.append(tool_name)


def _pop_tool_name(tool_name: str) -> None:
    if not _tool_names_to_append or _tool_names_to_append[-1] != tool_name:
        raise RuntimeError(
            "Tool context error detected. This can occur due to parallelization."
        )
    _tool_names_to_append.pop()
