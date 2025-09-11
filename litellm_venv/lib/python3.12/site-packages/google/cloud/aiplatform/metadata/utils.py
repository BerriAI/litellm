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

from typing import List, Optional, Union


def _make_filter_string(
    schema_title: Optional[Union[str, List[str]]] = None,
    in_context: Optional[List[str]] = None,
    parent_contexts: Optional[List[str]] = None,
    uri: Optional[str] = None,
) -> str:
    """Helper method to format filter strings for Metadata querying.

    No enforcement of correctness.

    Args:
        schema_title (Union[str, List[str]]): Optional. schema_titles to filter for.
        in_context (List[str]):
            Optional. Context resource names that the node should be in. Only for Artifacts/Executions.
        parent_contexts (List[str]): Optional. Parent contexts the context should be in. Only for Contexts.
        uri (str): Optional. uri to match for. Only for Artifacts.
    Returns:
        String that can be used for Metadata service filtering.
    """
    parts = []
    if schema_title:
        if isinstance(schema_title, str):
            parts.append(f'schema_title="{schema_title}"')
        else:
            substring = " OR ".join(f'schema_title="{s}"' for s in schema_title)
            parts.append(f"({substring})")
    if in_context:
        for context in in_context:
            parts.append(f'in_context("{context}")')
    if parent_contexts:
        parent_context_str = ",".join([f'"{c}"' for c in parent_contexts])
        parts.append(f"parent_contexts:{parent_context_str}")
    if uri:
        parts.append(f'uri="{uri}"')
    return " AND ".join(parts)
