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
"""Abstract interface for models."""
from __future__ import annotations

import abc
import dataclasses
from typing import Sequence


@dataclasses.dataclass(frozen=True)
class ModelArguments:
    """Common arguments for models.

    Attributes:
      model: The model string to use. If None a default model will be selected.
      temperature: The temperature. Must be greater-than-or-equal-to zero.
      candidate_count: Number of candidates to return.
    """

    model: str | None = None
    temperature: float | None = None
    candidate_count: int | None = None


@dataclasses.dataclass
class ModelResults:
    """Results from calling AbstractModel.call_model()."""

    model_input: str
    text_results: Sequence[str]


class AbstractModel(abc.ABC):
    @abc.abstractmethod
    def call_model(
        self, model_input: str, model_args: ModelArguments | None = None
    ) -> ModelResults:
        """Executes the model."""


class EchoModel(AbstractModel):
    """Model that returns the original input.

    This is primarily used for testing.
    """

    def call_model(
        self, model_input: str, model_args: ModelArguments | None = None
    ) -> ModelResults:
        candidate_count = model_args.candidate_count if model_args else None
        if candidate_count is None:
            candidate_count = 1
        return ModelResults(
            model_input=model_input,
            text_results=[model_input] * candidate_count,
        )
