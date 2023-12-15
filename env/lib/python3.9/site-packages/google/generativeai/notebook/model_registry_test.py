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
"""Unittest for model_registry."""
from __future__ import annotations

from absl.testing import absltest
from google.generativeai.notebook import model_registry


class ModelRegistryTest(absltest.TestCase):
    def test_get_model_echo_model(self):
        registry = model_registry.ModelRegistry()
        model = registry.get_model(model_registry.ModelName.ECHO_MODEL)
        results = model.call_model(model_input="this_is_a_test")
        self.assertEqual("this_is_a_test", results.model_input)

        # Echo model returns the model_input as text results.
        self.assertEqual(["this_is_a_test"], results.text_results)


if __name__ == "__main__":
    absltest.main()
