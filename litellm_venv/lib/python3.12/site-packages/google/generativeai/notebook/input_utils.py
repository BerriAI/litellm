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

from typing import Callable, Mapping

from google.generativeai.notebook import parsed_args_lib
from google.generativeai.notebook import py_utils
from google.generativeai.notebook.lib import llmfn_input_utils
from google.generativeai.notebook.lib import llmfn_inputs_source


class _NormalizedInputsSource(llmfn_inputs_source.LLMFnInputsSource):
    """Wrapper around NormalizedInputsList.

    By design LLMFunction does not take NormalizedInputsList as input because
    NormalizedInputsList is an internal representation so we want to minimize
    exposure to the caller.

    When we have inputs already in normalized format (e.g. from
    join_prompt_inputs()) we can wrap it as an LLMFnInputsSource to pass as an
    input to LLMFunction.
    """

    def __init__(self, normalized_inputs: llmfn_inputs_source.NormalizedInputsList):
        super().__init__()
        self._normalized_inputs = normalized_inputs

    def _to_normalized_inputs_impl(
        self,
    ) -> tuple[llmfn_inputs_source.NormalizedInputsList, Callable[[], None]]:
        return self._normalized_inputs, lambda: None


def get_inputs_source_from_py_var(
    var_name: str,
) -> llmfn_inputs_source.LLMFnInputsSource:
    data = py_utils.get_py_var(var_name)
    if isinstance(data, llmfn_inputs_source.LLMFnInputsSource):
        # No conversion needed.
        return data
    normalized_inputs = llmfn_input_utils.to_normalized_inputs(data)
    return _NormalizedInputsSource(normalized_inputs)


def join_inputs_sources(
    parsed_args: parsed_args_lib.ParsedArgs,
    suppress_status_msgs: bool = False,
) -> llmfn_inputs_source.LLMFnInputsSource:
    """Get a single combined input source from `parsed_args."""
    combined_inputs: list[Mapping[str, str]] = []
    for source in parsed_args.inputs:
        combined_inputs.extend(
            source.to_normalized_inputs(suppress_status_msgs=suppress_status_msgs)
        )
    for source in parsed_args.sheets_input_names:
        combined_inputs.extend(
            source.to_normalized_inputs(suppress_status_msgs=suppress_status_msgs)
        )
    return _NormalizedInputsSource(combined_inputs)
