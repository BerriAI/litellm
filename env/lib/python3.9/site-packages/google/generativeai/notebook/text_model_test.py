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
from __future__ import annotations

from unittest import mock

from absl.testing import absltest

from google.api_core import exceptions
from google.generativeai import text
from google.generativeai.notebook import text_model
from google.generativeai.notebook.lib import model as model_lib


def _fake_generator(
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    candidate_count: int | None = None,
) -> text.Completion:
    return text.Completion(
        prompt=prompt,
        model=model,
        temperature=temperature,
        candidate_count=candidate_count,
        # Smuggle the parameters as text output, so we can make assertions.
        candidates=[
            {"output": f"{prompt}_1"},
            {"output": model},
            {"output": temperature},
            {"output": candidate_count},
        ],
    )


class TestModel(text_model.TextModel):
    """A TextModel, but with _generate_text stubbed out."""

    def _generate_text(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        candidate_count: int | None = None,
        **kwargs,
    ) -> text.Completion:
        return _fake_generator(
            prompt=prompt,
            model=model,
            temperature=temperature,
            candidate_count=candidate_count,
        )


class TextModelTestCase(absltest.TestCase):
    def test_generate_text(self):
        model = TestModel()

        result = model.call_model("prompt goes in")
        self.assertEqual(result.text_results[0], "prompt goes in_1")
        self.assertIsNone(result.text_results[1])
        self.assertIsNone(result.text_results[2])
        self.assertIsNone(result.text_results[3])

        args = model_lib.ModelArguments(model="model_name", temperature=0.42, candidate_count=5)
        result = model.call_model("prompt goes in", args)
        self.assertEqual(result.text_results[0], "prompt goes in_1")
        self.assertEqual(result.text_results[1], "model_name")
        self.assertEqual(result.text_results[2], 0.42)
        self.assertEqual(result.text_results[3], 5)

    def test_retry(self):
        model = TestModel()

        with mock.patch.object(model, "_generate_text") as erroneous_generator:
            erroneous_generator.side_effect = [
                exceptions.ResourceExhausted("Over quota"),
                mock.DEFAULT,
            ]

            _ = model.call_model("phew it worked")

        self.assertEqual(erroneous_generator.call_count, 2)


if __name__ == "__main__":
    absltest.main()
