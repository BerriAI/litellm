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

from absl.testing import absltest
from google.generativeai.notebook.lib import prompt_utils


class PromptUtilsTest(absltest.TestCase):
    def test_get_placeholders_empty(self):
        placeholders = prompt_utils.get_placeholders("")
        self.assertEmpty(placeholders)

        placeholders = prompt_utils.get_placeholders("There are no placeholders here")
        self.assertEmpty(placeholders)

    def test_get_placeholders(self):
        placeholders = prompt_utils.get_placeholders("today {hello} world")
        self.assertEqual(frozenset({"hello"}), placeholders)

        placeholders = prompt_utils.get_placeholders("{hello} {world}")
        self.assertEqual(frozenset({"hello", "world"}), placeholders)

        placeholders = prompt_utils.get_placeholders("{hello} {hello}")
        self.assertEqual(frozenset({"hello"}), placeholders)


if __name__ == "__main__":
    absltest.main()
