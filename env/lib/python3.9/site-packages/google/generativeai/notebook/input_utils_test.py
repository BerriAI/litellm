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
"""Unittest for py_utils."""
from __future__ import annotations

import sys
from unittest import mock

from absl.testing import absltest
from google.generativeai.notebook import input_utils

_EMPTY_INPUT_VAR_ONE = {}
_EMPTY_INPUT_VAR_TWO = {"word": []}
_INPUT_VAR_ONE = {"word": ["lukewarm"]}
_INPUT_VAR_TWO = {"word": ["hot", "cold"]}
_MULTI_INPUTS_VAR_ONE = {"a": ["apple"], "b": ["banana"]}
_MULTI_INPUTS_VAR_TWO = {"a": ["australia", "alpha"], "b": ["brazil", "beta"]}


# `unittest discover` does not run via __main__, so patch this context in.
@mock.patch.dict(sys.modules, {"__main__": sys.modules[__name__]})
class InputUtilsTest(absltest.TestCase):
    def test_get_inputs_source_from_py_var_invalid_name(self):
        with self.assertRaisesRegex(NameError, "UnknownVar"):
            input_utils.get_inputs_source_from_py_var("UnknownVar")

    def test_get_inputs_source_from_py_var_empty_one(self):
        source = input_utils.get_inputs_source_from_py_var("_EMPTY_INPUT_VAR_ONE")
        results = source.to_normalized_inputs()
        self.assertEmpty(results)

    def test_get_inputs_source_from_py_var_empty_two(self):
        source = input_utils.get_inputs_source_from_py_var("_EMPTY_INPUT_VAR_TWO")
        results = source.to_normalized_inputs()
        self.assertEmpty(results)

    def test_get_inputs_source_from_py_var_single_input_one(self):
        source = input_utils.get_inputs_source_from_py_var("_INPUT_VAR_ONE")
        results = source.to_normalized_inputs()
        self.assertEqual([{"word": "lukewarm"}], results)

    def test_get_inputs_source_from_py_var_single_input_two(self):
        source = input_utils.get_inputs_source_from_py_var("_INPUT_VAR_TWO")
        results = source.to_normalized_inputs()
        self.assertEqual([{"word": "hot"}, {"word": "cold"}], results)


if __name__ == "__main__":
    absltest.main()
