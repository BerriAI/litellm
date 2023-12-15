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
"""LLMFnInputsSource."""
from __future__ import annotations

import abc
from typing import Callable, Mapping, Sequence


NormalizedInputsList = Sequence[Mapping[str, str]]


class LLMFnInputsSource(abc.ABC):
    """Abstract class representing a source of inputs for LLMFunction.

    This class could be extended with concrete implementations that read data
    from external sources, such as Google Sheets.
    """

    def __init__(self):
        self._cached_inputs: NormalizedInputsList | None = None
        self._display_status_fn: Callable[[], None] = lambda: None

    def to_normalized_inputs(self, suppress_status_msgs: bool = False) -> NormalizedInputsList:
        """Returns a sequence of normalized inputs.

        The return value is a sequence of dictionaries of (placeholder, value)
        pairs, e.g. [{"word": "hot"}, {"word: "cold"}, ....]

        These are used for keyword-substitution for prompts in LLMFunctions.

        Args:
          suppress_status_msgs: If True, suppress status messages regarding the
            input being read.

        Returns:
          A sequence of normalized inputs.
        """
        if self._cached_inputs is None:
            (
                self._cached_inputs,
                self._display_status_fn,
            ) = self._to_normalized_inputs_impl()
        if not suppress_status_msgs:
            self._display_status_fn()
        return self._cached_inputs

    @abc.abstractmethod
    def _to_normalized_inputs_impl(
        self,
    ) -> tuple[NormalizedInputsList, Callable[[], None]]:
        """Returns a tuple of NormalizedInputsList and a display function.

        The display function displays some status about the input (e.g. where
        it is read from). This way the status continues to be displayed
        even though the results are cached.
        """
