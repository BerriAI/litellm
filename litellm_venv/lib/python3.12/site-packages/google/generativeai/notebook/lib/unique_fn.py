# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
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
"""Function for de-duping results."""
from __future__ import annotations

from typing import Sequence
from google.generativeai.notebook.lib import llmfn_output_row


def unique_fn(
    rows: Sequence[llmfn_output_row.LLMFnOutputRowView],
) -> Sequence[int]:
    """Returns a list of indices with duplicates removed.

    E.g. if rows has results ["hello", "hello", "world"], the return value would
    be [0, 2], indicating that the results at index 1 is a duplicate and should be
    removed.

    Args:
      rows: The input rows

    Returns:
      A sequence of indices indicating which entries have unique results.
    """
    indices: list[int] = []
    seen_entries = set()
    for idx, row in enumerate(rows):
        value = row.result_value()
        if value in seen_entries:
            continue

        seen_entries.add(value)
        indices.append(idx)

    return indices
