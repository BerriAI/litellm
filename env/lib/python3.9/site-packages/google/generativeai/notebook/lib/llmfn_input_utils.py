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
"""Utilities for handling input variables."""
from __future__ import annotations

from typing import Any, Mapping, Sequence, Union

from google.generativeai.notebook.lib import llmfn_inputs_source


_NormalizedInputsList = llmfn_inputs_source.NormalizedInputsList

_ColumnOrderValuesList = Mapping[str, Sequence[str]]

LLMFunctionInputs = Union[_ColumnOrderValuesList, llmfn_inputs_source.LLMFnInputsSource]


def _is_column_order_values_list(inputs: Any) -> bool:
    """See if inputs is of the form: {"key1": ["val1", "val2", ...]}.

    This is similar to the format produced by:
      pandas.DataFrame.to_dict(orient="list")

    Args:
      inputs: The inputs passed into an LLMFunction.

    Returns:
      Whether `inputs` is a column-ordered list of values.
    """
    if not isinstance(inputs, Mapping):
        return False
    for x in inputs.values():
        if not isinstance(x, Sequence):
            return False
        # Strings and bytes are also considered Sequences but we disallow them
        # here because the values contained in their Sequences are single
        # characters rather than words.
        if isinstance(x, str) or isinstance(x, bytes):
            return False
    return True


# TODO(b/273688393): Perform stricter validation on `inputs`.
def _normalize_column_order_values_list(
    inputs: _ColumnOrderValuesList,
) -> _NormalizedInputsList:
    """Transforms prompt inputs into a list of dictionaries."""
    return_list: list[dict[str, str]] = []
    keys = list(inputs.keys())
    if keys:
        first_key = keys[0]
        for row_num in range(len(inputs[first_key])):
            row_dict = {}
            return_list.append(row_dict)
            for key in keys:
                row_dict[key] = inputs[key][row_num]
    return return_list


def to_normalized_inputs(inputs: LLMFunctionInputs) -> _NormalizedInputsList:
    """Handles the different types of `inputs` and returns a normalized form."""
    normalized_inputs: list[Mapping[str, str]] = []
    if isinstance(inputs, llmfn_inputs_source.LLMFnInputsSource):
        normalized_inputs.extend(inputs.to_normalized_inputs())
    elif _is_column_order_values_list(inputs):
        normalized_inputs.extend(_normalize_column_order_values_list(inputs))
    else:
        raise ValueError("Unsupported input type {!r}".format(inputs))
    return normalized_inputs
