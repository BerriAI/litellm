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
from google.generativeai.notebook import py_utils

_INPUT_VAR = "hello world"
_OUTPUT_VAR = None


# `unittest discover` does not run via __main__, so patch this context in.
@mock.patch.dict(sys.modules, {"__main__": sys.modules[__name__]})
class PyUtilsTest(absltest.TestCase):
    def test_get_py_var(self):
        # get_py_var() with an invalid var should raise an error.
        with self.assertRaisesRegex(NameError, "IncorrectVar"):
            py_utils.get_py_var("IncorrectVar")

        results = py_utils.get_py_var("_INPUT_VAR")
        self.assertEqual("hello world", results)

    def test_set_py_var(self):
        py_utils.set_py_var("_OUTPUT_VAR", "world hello")
        self.assertEqual("world hello", _OUTPUT_VAR)

        # Calling with a new variable name creates a new variable.
        py_utils.set_py_var("_NEW_VAR", "world hello world")
        # pylint: disable-next=undefined-variable
        self.assertEqual("world hello world", _NEW_VAR)  # type: ignore


if __name__ == "__main__":
    absltest.main()
