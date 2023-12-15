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
"""Unittest for Argument Definition."""
from __future__ import annotations

import argparse
import enum
import math
from absl.testing import absltest
from google.generativeai.notebook import argument_parser
from google.generativeai.notebook import flag_def


def _new_parser(flag: flag_def.FlagDef) -> argparse.ArgumentParser:
    """Returns a new ArgumentParser with `flag` added as an argument."""
    parser = argument_parser.ArgumentParser()
    flag.add_argument_to_parser(parser)
    return parser


class SingleValueFlagDefTest(absltest.TestCase):
    def test_short_name(self):
        flag = flag_def.SingleValueFlagDef(
            name="value", short_name="v", parse_type=str, required=True
        )

        results = _new_parser(flag).parse_args(["--value=forty-one"])
        self.assertEqual("forty-one", results.value)
        results = _new_parser(flag).parse_args(["-v", "forty-two"])
        self.assertEqual("forty-two", results.value)

    def test_cardinality(self):
        flag = flag_def.SingleValueFlagDef(name="value", parse_type=str, required=True)

        # Parser should not accept flag without no values.
        with self.assertRaisesRegex(argument_parser.ParserError, "expected 1 argument"):
            _new_parser(flag).parse_args(["--value"])

        # Parser should not accept flag with more-than-one value.
        with self.assertRaisesRegex(
            argument_parser.ParserError, "unrecognized arguments: forty-two"
        ):
            _new_parser(flag).parse_args(["--value", "forty-one", "forty-two"])

        # Parser should not accept a command-line with the flag specified
        # more-than-once.
        with self.assertRaisesRegex(
            argument_parser.ParserError, "Cannot set --value more than once"
        ):
            _new_parser(flag).parse_args(["--value", "forty-one", "--value", "forty-two"])

        results = _new_parser(flag).parse_args(["--value", "forty-one"])
        self.assertEqual("forty-one", results.value)

    def test_required(self):
        req_flag = flag_def.SingleValueFlagDef(
            name="value",
            parse_type=str,
            required=True,
        )

        # Parser is able to parse a command line containing the required argument.
        results = _new_parser(req_flag).parse_args(["--value=forty-two"])
        self.assertEqual("forty-two", results.value)

        # Parser should raise an Exception if the commandline does not contain the
        # required argument.
        with self.assertRaisesRegex(
            argument_parser.ParserError, "the following arguments are required"
        ):
            _new_parser(req_flag).parse_args([])

    def test_optional(self):
        # Optional flags should have a default value
        with self.assertRaisesRegex(ValueError, "Optional flags must have a default value"):
            flag_def.SingleValueFlagDef(
                name="value",
                parse_type=str,
                required=False,
            )

        opt_flag = flag_def.SingleValueFlagDef(
            name="value",
            parse_type=str,
            required=False,
            default_value="zero",
        )

        # Parser is able to parse a command line containing the optional argument.
        results = _new_parser(opt_flag).parse_args(["--value=forty-two"])
        self.assertEqual("forty-two", results.value)

        # Parser should return the default value if the command line does not
        # contain the optional argument.
        results = _new_parser(opt_flag).parse_args([])
        self.assertEqual("zero", results.value)

        # Parser should not accept flag without a value.
        with self.assertRaisesRegex(argument_parser.ParserError, "expected 1 argument"):
            _new_parser(opt_flag).parse_args(["--value"])

    def test_default_is_none(self):
        """Make sure None can be accepted as a default value."""
        opt_flag = flag_def.SingleValueFlagDef(
            name="value",
            parse_type=str,
            required=False,
            default_value=None,
        )

        # Parser is able to parse a command line containing the optional argument.
        results = _new_parser(opt_flag).parse_args(["--value=forty-two"])
        self.assertEqual("forty-two", results.value)

        # Parser should return the default value if the command line does not
        # contain the optional argument.
        results = _new_parser(opt_flag).parse_args([])
        self.assertIsNone(results.value)

        # Parser should not accept flag without a value.
        with self.assertRaisesRegex(argument_parser.ParserError, "expected 1 argument"):
            _new_parser(opt_flag).parse_args(["--value"])

    def test_type_conversion(self):
        # Default value must be of the same type as destination type.
        with self.assertRaisesRegex(
            ValueError,
            "Default value must be of the same type as the destination type",
        ):
            flag_def.SingleValueFlagDef(
                name="value",
                parse_type=int,
                required=False,
                default_value="zero",
            )

        int_flag_def = flag_def.SingleValueFlagDef(
            name="value",
            parse_type=int,
            required=False,
            default_value=0,
        )

        # Parser should not accept a value of the wrong type.
        with self.assertRaisesRegex(argument_parser.ParserError, "invalid int value: 'forty-two'"):
            _new_parser(int_flag_def).parse_args(["--value", "forty-two"])

        results = _new_parser(int_flag_def).parse_args(["--value", "42"])
        self.assertEqual(42, results.value)

    def test_validation(self):
        def _check_is_not_nan(x: float) -> float:
            if math.isnan(x):
                raise ValueError("Must not be NAN")
            return x

        float_flag_def = flag_def.SingleValueFlagDef(
            name="value",
            parse_type=float,
            parse_to_dest_type_fn=_check_is_not_nan,
            required=True,
        )

        results = _new_parser(float_flag_def).parse_args(["--value", "0.25"])
        self.assertEqual(0.25, results.value)

        with self.assertRaisesRegex(argument_parser.ParserError, "Must not be NAN"):
            _new_parser(float_flag_def).parse_args(["--value", "nan"])


class ColorsEnum(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class EnumFlagDefTest(absltest.TestCase):
    def test_construction(self):
        # "enum_type" must be provided.
        with self.assertRaisesRegex(TypeError, "missing 1 required keyword-only argument"):
            # pylint: disable-next=missing-kwoa
            flag_def.EnumFlagDef(name="color", required=True)  # type: ignore

        # "parse_type" cannot be provided.
        with self.assertRaisesRegex(ValueError, 'Cannot set "parse_type" for EnumFlagDef'):
            flag_def.EnumFlagDef(
                name="color",
                required=True,
                enum_type=ColorsEnum,
                parse_type=int,
            )
        # "dest_type" cannot be provided.
        with self.assertRaisesRegex(ValueError, 'Cannot set "dest_type" for EnumFlagDef'):
            flag_def.EnumFlagDef(name="color", required=True, enum_type=ColorsEnum, dest_type=str)
        # This should succeed.
        flag_def.EnumFlagDef(name="color", required=True, enum_type=ColorsEnum)

    def test_parsing(self):
        flag = flag_def.EnumFlagDef(
            name="color",
            required=False,
            enum_type=ColorsEnum,
            default_value=ColorsEnum.RED,
        )

        # "teal" is not one of the enum values.
        with self.assertRaisesRegex(argument_parser.ParserError, "invalid choice: 'teal'"):
            _new_parser(flag).parse_args(["--color=teal"])

        results = _new_parser(flag).parse_args(["--color=red"])
        self.assertEqual(ColorsEnum.RED, results.color)
        results = _new_parser(flag).parse_args(["--color=green"])
        self.assertEqual(ColorsEnum.GREEN, results.color)
        results = _new_parser(flag).parse_args(["--color=blue"])
        self.assertEqual(ColorsEnum.BLUE, results.color)

    def test_choices(self):
        # If `choices` is provided, all values must be valid.
        with self.assertRaisesRegex(ValueError, 'Invalid value in "choices"'):
            flag_def.EnumFlagDef(
                name="color",
                required=True,
                enum_type=ColorsEnum,
                choices=["red", "green", "teal"],
            )

        # Exclude "blue".
        flag = flag_def.EnumFlagDef(
            name="color",
            required=True,
            enum_type=ColorsEnum,
            choices=["red", "green"],
        )

        # "blue" is no longer one of the choices.
        with self.assertRaisesRegex(argument_parser.ParserError, "invalid choice: 'blue'"):
            _new_parser(flag).parse_args(["--color=blue"])

        results = _new_parser(flag).parse_args(["--color=red"])
        self.assertEqual(ColorsEnum.RED, results.color)
        results = _new_parser(flag).parse_args(["--color=green"])
        self.assertEqual(ColorsEnum.GREEN, results.color)


class MultiValuesFlagDefTest(absltest.TestCase):
    def test_basic(self):
        # Default value is not needed even if optional; the value would just be the
        # empty list.
        flag = flag_def.MultiValuesFlagDef(name="colors", parse_type=str, required=False)

        # Default value is the empty list.
        results = _new_parser(flag).parse_args([])
        self.assertEmpty(results.colors)

        results = _new_parser(flag).parse_args(["--colors", "red"])
        self.assertEqual(["red"], results.colors)

        results = _new_parser(flag).parse_args(["--colors", "red", "green"])
        self.assertEqual(["red", "green"], results.colors)

    def test_required(self):
        flag = flag_def.MultiValuesFlagDef(
            name="colors",
            parse_type=str,
            required=True,
        )

        # Parser is able to parse a command line containing the required argument.
        results = _new_parser(flag).parse_args(["--colors", "red"])
        self.assertEqual(["red"], results.colors)

        # Parser should raise an Exception if the commandline does not contain the
        # required argument.
        with self.assertRaisesRegex(
            argument_parser.ParserError, "the following arguments are required"
        ):
            _new_parser(flag).parse_args([])

    def test_cannot_set_default_value(self):
        # `default_value` is not a field for MultiValueFlagsDef.
        with self.assertRaisesRegex(TypeError, "got an unexpected keyword argument"):
            # pylint: disable-next=unexpected-keyword-arg
            flag_def.MultiValuesFlagDef(  # type: ignore
                name="colors",
                parse_type=str,
                required=False,
                default_value="fuschia",
            )

    def test_values_must_be_unique(self):
        flag = flag_def.MultiValuesFlagDef(name="colors")

        # Cannot specify "red" more than once.
        with self.assertRaisesRegex(argument_parser.ParserError, 'Duplicate values "red"'):
            _new_parser(flag).parse_args(["--colors", "red", "green", "red"])

    def test_cardinality(self):
        flag = flag_def.MultiValuesFlagDef(
            name="colors",
            parse_type=str,
            required=False,
        )

        # Must have at least one argument.
        with self.assertRaisesRegex(argument_parser.ParserError, "expected at least one argument"):
            _new_parser(flag).parse_args(["--colors"])

        # Cannot specify "--colors" more than once.
        with self.assertRaisesRegex(
            argument_parser.ParserError, "Cannot set --colors more than once"
        ):
            _new_parser(flag).parse_args(["--colors", "red", "--colors", "blue"])

    def test_dest_type_conversion(self):
        flag = flag_def.MultiValuesFlagDef(
            name="colors",
            parse_type=str,
            dest_type=ColorsEnum,
            required=False,
            choices=[x.value for x in ColorsEnum],
        )

        # "fuschia" is not a valid value for enum.
        with self.assertRaisesRegex(argument_parser.ParserError, "invalid choice: 'fuschia'"):
            _new_parser(flag).parse_args(["--colors", "fuschia"])

        # Results are converted to a list of enums.
        results = _new_parser(flag).parse_args(["--colors", "red", "green"])
        self.assertEqual([ColorsEnum.RED, ColorsEnum.GREEN], results.colors)

    def test_validation(self):
        def _check_is_not_nan(x: float) -> float:
            if math.isnan(x):
                raise ValueError("Must not be NAN")
            return x

        flag = flag_def.MultiValuesFlagDef(
            name="values",
            parse_type=float,
            parse_to_dest_type_fn=_check_is_not_nan,
        )

        results = _new_parser(flag).parse_args(["--value", "0.25", "0.5"])
        self.assertEqual([0.25, 0.5], results.values)

        with self.assertRaisesRegex(argument_parser.ParserError, "Must not be NAN"):
            _new_parser(flag).parse_args(["--value", "0.25", "nan"])


class BooleanFlagDefTest(absltest.TestCase):
    def test_basic(self):
        flag = flag_def.BooleanFlagDef(name="unique")

        results = _new_parser(flag).parse_args([])
        self.assertFalse(results.unique)

        results = _new_parser(flag).parse_args(["--unique"])
        self.assertTrue(results.unique)

    def test_constructor(self):
        """Check that invalid constructor arguments are rejected."""
        with self.assertRaisesRegex(ValueError, "dest_type cannot be set for BooleanFlagDef"):
            flag_def.BooleanFlagDef(name="unique", dest_type=bool)
        with self.assertRaisesRegex(
            ValueError, "parse_to_dest_type_fn cannot be set for BooleanFlagDef"
        ):
            flag_def.BooleanFlagDef(name="unique", parse_to_dest_type_fn=lambda x: True)
        with self.assertRaisesRegex(ValueError, "choices cannot be set for BooleanFlagDef"):
            flag_def.BooleanFlagDef(name="unique", choices=[True])

    def test_cardinality(self):
        flag = flag_def.BooleanFlagDef(name="unique")

        with self.assertRaisesRegex(
            argument_parser.ParserError, "error: unrecognized arguments: True"
        ):
            _new_parser(flag).parse_args(["--unique", "True"])

        with self.assertRaisesRegex(
            argument_parser.ParserError, "Cannot set --unique more than once"
        ):
            _new_parser(flag).parse_args(["--unique", "--unique"])


if __name__ == "__main__":
    absltest.main()
