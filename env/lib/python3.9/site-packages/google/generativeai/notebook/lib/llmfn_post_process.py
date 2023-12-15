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
"""Signatures for post-processing functions and other common definitions."""
from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple

from google.generativeai.notebook.lib import llmfn_output_row


class PostProcessExecutionError(RuntimeError):
    """An error while executing a post-processing command."""


# A batch-process function takes a batch of rows, and returns a sequence of
# indices representing which rows to keep.
# This can be used to implement operations such as filtering and sorting.
#
# Requires:
# - Indices must be in the range [0, len(input rows)).
LLMFnPostProcessBatchReorderFn = Callable[
    [Sequence[llmfn_output_row.LLMFnOutputRowView]],
    Sequence[int],
]

# An add function takes a batch of rows and returns a sequence of values to
# be added as new columns.
#
# Requires:
# - Output sequence must be exactly the same length as number of rows.
LLMFnPostProcessBatchAddFn = Callable[
    [Sequence[llmfn_output_row.LLMFnOutputRowView]], Sequence[Any]
]

# A replace function takes a batch of rows and returns a sequence of values
# to replace the existing results.
#
# Requires:
# - Output sequence must be exactly the same length as number of rows.
# - Return type must match the result_type of LLMFnOutputRow.
LLMFnPostProcessBatchReplaceFn = Callable[
    [Sequence[llmfn_output_row.LLMFnOutputRowView]], Sequence[Any]
]

# An add function takes a batch of pairs of rows and returns a sequence of
# values to be added as new columns.
#
# This is used for LLMCompareFunction.
#
# Requires:
# - Output sequence must be exactly the same length as number of rows.
LLMCompareFnPostProcessBatchAddFn = Callable[
    [
        Sequence[
            Tuple[
                llmfn_output_row.LLMFnOutputRowView,
                llmfn_output_row.LLMFnOutputRowView,
            ]
        ]
    ],
    Sequence[Any],
]
