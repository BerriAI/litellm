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
"""Unittest for ArgumentParser."""
from __future__ import annotations

import argparse
from absl.testing import absltest
from google.generativeai.notebook import argument_parser as parser_lib


class ArgumentParserTest(absltest.TestCase):
    def test_help(self):
        """Verify that help messages raise ParserNormalExit."""
        parser = parser_lib.ArgumentParser()
        with self.assertRaisesRegex(parser_lib.ParserNormalExit, "show this help message and exit"):
            parser.parse_args(["-h"])

    def test_parse_arg_errors(self):
        def new_parser() -> argparse.ArgumentParser:
            parser = parser_lib.ArgumentParser()
            parser.add_argument("--value", type=int, required=True)
            return parser

        # Normal case: no error.
        results = new_parser().parse_args(["--value", "42"])
        self.assertEqual(42, results.value)

        with self.assertRaisesRegex(parser_lib.ParserError, "invalid int value"):
            new_parser().parse_args(["--value", "forty-two"])

        with self.assertRaisesRegex(parser_lib.ParserError, "the following arguments are required"):
            new_parser().parse_args([])

        with self.assertRaisesRegex(parser_lib.ParserError, "expected one argument"):
            new_parser().parse_args(["--value"])


if __name__ == "__main__":
    absltest.main()
