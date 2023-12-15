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

from absl.testing import absltest
from google.generativeai.notebook import sheets_id


class SheetsIdentifierTest(absltest.TestCase):
    def test_constructor(self):
        sid = sheets_id.SheetsIdentifier(name="hello")
        self.assertEqual("name=hello", str(sid))
        sid = sheets_id.SheetsIdentifier(key=sheets_id.SheetsKey("hello"))
        self.assertEqual("key=hello", str(sid))
        sid = sheets_id.SheetsIdentifier(url=sheets_id.SheetsURL("https://docs.google.com/"))
        self.assertEqual("url=https://docs.google.com/", str(sid))

    def test_constructor_error(self):
        with self.assertRaisesRegex(ValueError, "Must set exactly one of name, key or url"):
            sheets_id.SheetsIdentifier()

        # Empty "name" is also considered an invalid name.
        with self.assertRaisesRegex(ValueError, "Must set exactly one of name, key or url"):
            sheets_id.SheetsIdentifier(name="")

        with self.assertRaisesRegex(ValueError, "Must set exactly one of name, key or url"):
            sheets_id.SheetsIdentifier(name="hello", key=sheets_id.SheetsKey("hello"))

        with self.assertRaisesRegex(ValueError, "Must set exactly one of name, key or url"):
            sheets_id.SheetsIdentifier(
                name="hello",
                key=sheets_id.SheetsKey("hello"),
                url=sheets_id.SheetsURL("https://docs.google.com/"),
            )


if __name__ == "__main__":
    absltest.main()
