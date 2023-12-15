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
"""Model that uses the Text service."""
from __future__ import annotations

from google.api_core import retry
from google.generativeai import text
from google.generativeai.types import text_types
from google.generativeai.notebook.lib import model as model_lib


class TextModel(model_lib.AbstractModel):
    """Concrete model that uses the Text service."""

    def _generate_text(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        candidate_count: int | None = None,
        **kwargs,
    ) -> text_types.Completion:
        if model is not None:
            kwargs["model"] = model
        if temperature is not None:
            kwargs["temperature"] = temperature
        if candidate_count is not None:
            kwargs["candidate_count"] = candidate_count
        return text.generate_text(prompt=prompt, **kwargs)

    def call_model(
        self,
        model_input: str,
        model_args: model_lib.ModelArguments | None = None,
    ) -> model_lib.ModelResults:
        if model_args is None:
            model_args = model_lib.ModelArguments()

        # Wrap the generation function here, rather than decorate, so that it
        # applies to any overridden calls too.
        retryable_fn = retry.Retry(retry.if_transient_error)(self._generate_text)
        response = retryable_fn(
            prompt=model_input,
            model=model_args.model,
            temperature=model_args.temperature,
            candidate_count=model_args.candidate_count,
        )

        return model_lib.ModelResults(
            model_input=model_input,
            text_results=[x["output"] for x in response.candidates],
        )
