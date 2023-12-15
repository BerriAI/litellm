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
"""Classes that define arguments for populating ArgumentParser.

The argparse module's ArgumentParser.add_argument() takes several parameters and
is quite customizable. However this can lead to bugs where arguments do not
behave as expected.

For better ease-of-use and better testability, define a set of classes for the
types of flags used by LLM Magics.

Sample usage:

  str_flag = SingleValueFlagDef(name="title", required=True)
  enum_flag = EnumFlagDef(name="colors", required=True, enum_type=ColorsEnum)

  str_flag.add_argument_to_parser(my_parser)
  enum_flag.add_argument_to_parser(my_parser)
"""
from __future__ import annotations

import abc
import argparse
import dataclasses
import enum
from typing import Any, Callable, Sequence, Tuple, Union

from google.generativeai.notebook.lib import llmfn_inputs_source
from google.generativeai.notebook.lib import llmfn_outputs

# These are the intermediate types that argparse.ArgumentParser.parse_args()
# will pass command line arguments into.
_PARSETYPES = Union[str, int, float]
# These are the final result types that the intermediate parsed values will be
# converted into. It is a superset of _PARSETYPES because we support converting
# the parsed type into a more precise type, e.g. from str to Enum.
_DESTTYPES = Union[
    _PARSETYPES,
    enum.Enum,
    Tuple[str, Callable[[str, str], Any]],
    Sequence[str],  # For --compare_fn
    llmfn_inputs_source.LLMFnInputsSource,  # For --ground_truth
    llmfn_outputs.LLMFnOutputsSink,  # For --inputs  # For --outputs
]

# The signature of a function that converts a command line argument from the
# intermediate parsed type to the result type.
_PARSEFN = Callable[[_PARSETYPES], _DESTTYPES]


def _get_type_name(x: type[Any]) -> str:
    try:
        return x.__name__
    except AttributeError:
        return str(x)


def _validate_flag_name(name: str) -> str:
    """Validation for long and short names for flags."""
    if not name:
        raise ValueError("Cannot be empty")
    if name[0] == "-":
        raise ValueError("Cannot start with dash")
    return name


@dataclasses.dataclass(frozen=True)
class FlagDef(abc.ABC):
    """Abstract base class for flag definitions.

    Attributes:
      name: Long name, e.g. "colors" will define the flag "--colors".
      required: Whether the flag must be provided on the command line.
      short_name: Optional short name.
      parse_type: The type that ArgumentParser should parse the command line
        argument to.
      dest_type: The type that the parsed value is converted to. This is used when
        we want ArgumentParser to parse as one type, then convert to a different
        type. E.g. for enums we parse as "str" then convert to the desired enum
        type in order to provide cleaner help messages.
      parse_to_dest_type_fn: If provided, this function will be used to convert
        the value from `parse_type` to `dest_type`. This can be used for
        validation as well.
      choices: If provided, limit the set of acceptable values to these choices.
      help_msg: If provided, adds help message when -h is used in the command
        line.
    """

    name: str
    required: bool = False

    short_name: str | None = None

    parse_type: type[_PARSETYPES] = str
    dest_type: type[_DESTTYPES] | None = None
    parse_to_dest_type_fn: _PARSEFN | None = None

    choices: list[_PARSETYPES] | None = None
    help_msg: str | None = None

    @abc.abstractmethod
    def add_argument_to_parser(self, parser: argparse.ArgumentParser) -> None:
        """Adds this flag as an argument to `parser`.

        Child classes should implement this as a call to parser.add_argument()
        with the appropriate parameters.

        Args:
          parser: The parser to which this argument will be added.
        """

    @abc.abstractmethod
    def _do_additional_validation(self) -> None:
        """For child classes to do additional validation."""

    def _get_dest_type(self) -> type[_DESTTYPES]:
        """Returns the final converted type."""
        return self.parse_type if self.dest_type is None else self.dest_type

    def _get_parse_to_dest_type_fn(
        self,
    ) -> _PARSEFN:
        """Returns a function to convert from parse_type to dest_type."""
        if self.parse_to_dest_type_fn is not None:
            return self.parse_to_dest_type_fn

        dest_type = self._get_dest_type()
        if dest_type == self.parse_type:
            return lambda x: x
        else:
            return dest_type

    def __post_init__(self):
        _validate_flag_name(self.name)
        if self.short_name is not None:
            _validate_flag_name(self.short_name)

        self._do_additional_validation()


def _has_non_default_value(
    namespace: argparse.Namespace,
    dest: str,
    has_default: bool = False,
    default_value: Any = None,
) -> bool:
    """Returns true if `namespace.dest` is set to a non-default value.

    Args:
      namespace: The Namespace that is populated by ArgumentParser.
      dest: The attribute in the Namespacde to be populated.
      has_default: "None" is a valid default value so we use an additional
        `has_default` boolean to indicate that `default_value` is present.
      default_value: The default value to use when `has_default` is True.

    Returns:
      Whether namespace.dest is set to something other than the default value.
    """
    if not hasattr(namespace, dest):
        return False

    if not has_default:
        # No default value provided so `namespace.dest` cannot possibly be equal to
        # the default value.
        return True

    return getattr(namespace, dest) != default_value


class _SingleValueStoreAction(argparse.Action):
    """Custom Action for storing a value in an argparse.Namespace.

    This action checks that the flag is specified at-most once.
    """

    def __init__(
        self,
        option_strings,
        dest,
        dest_type: type[Any],
        parse_to_dest_type_fn: _PARSEFN,
        **kwargs,
    ):
        super().__init__(option_strings, dest, **kwargs)
        self._dest_type = dest_type
        self._parse_to_dest_type_fn = parse_to_dest_type_fn

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ):
        # Because `nargs` is set to 1, `values` must be a Sequence, rather
        # than a string.
        assert not isinstance(values, str) and not isinstance(values, bytes)

        if _has_non_default_value(
            namespace,
            self.dest,
            has_default=hasattr(self, "default"),
            default_value=getattr(self, "default"),
        ):
            raise argparse.ArgumentError(self, "Cannot set {} more than once".format(option_string))

        try:
            converted_value = self._parse_to_dest_type_fn(values[0])
        except Exception as e:
            raise argparse.ArgumentError(
                self,
                'Error with value "{}", got {}: {}'.format(values[0], _get_type_name(type(e)), e),
            )

        if not isinstance(converted_value, self._dest_type):
            raise RuntimeError(
                "Converted to wrong type, expected {} got {}".format(
                    _get_type_name(self._dest_type),
                    _get_type_name(type(converted_value)),
                )
            )
        setattr(namespace, self.dest, converted_value)


class _MultiValuesAppendAction(argparse.Action):
    """Custom Action for appending values in an argparse.Namespace.

    This action checks that the flag is specified at-most once.
    """

    def __init__(
        self,
        option_strings,
        dest,
        dest_type: type[Any],
        parse_to_dest_type_fn: _PARSEFN,
        **kwargs,
    ):
        super().__init__(option_strings, dest, **kwargs)
        self._dest_type = dest_type
        self._parse_to_dest_type_fn = parse_to_dest_type_fn

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ):
        # Because `nargs` is set to "+", `values` must be a Sequence, rather
        # than a string.
        assert not isinstance(values, str) and not isinstance(values, bytes)

        curr_value = getattr(namespace, self.dest)
        if curr_value:
            raise argparse.ArgumentError(self, "Cannot set {} more than once".format(option_string))

        for value in values:
            try:
                converted_value = self._parse_to_dest_type_fn(value)
            except Exception as e:
                raise argparse.ArgumentError(
                    self,
                    'Error with value "{}", got {}: {}'.format(
                        values[0], _get_type_name(type(e)), e
                    ),
                )

            if not isinstance(converted_value, self._dest_type):
                raise RuntimeError(
                    "Converted to wrong type, expected {} got {}".format(
                        self._dest_type, type(converted_value)
                    )
                )
            if converted_value in curr_value:
                raise argparse.ArgumentError(self, 'Duplicate values "{}"'.format(value))

            curr_value.append(converted_value)


class _BooleanValueStoreAction(argparse.Action):
    """Custom Action for setting a boolean value in argparse.Namespace.

    The boolean flag expects the default to be False and will set the value to
    True.
    This action checks that the flag is specified at-most once.
    """

    def __init__(
        self,
        option_strings,
        dest,
        **kwargs,
    ):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ):
        if _has_non_default_value(
            namespace,
            self.dest,
            has_default=True,
            default_value=False,
        ):
            raise argparse.ArgumentError(self, "Cannot set {} more than once".format(option_string))

        setattr(namespace, self.dest, True)


@dataclasses.dataclass(frozen=True)
class SingleValueFlagDef(FlagDef):
    """Definition for a flag that takes a single value.

    Sample usage:
      # This defines a flag that can be specified on the command line as:
      #   --count=10
      flag = SingleValueFlagDef(name="count", parse_type=int, required=True)
      flag.add_argument_to_parser(argument_parser)

    Attributes:
      default_value: Default value for optional flags.
    """

    class _DefaultValue(enum.Enum):
        """Special value to represent "no value provided".

        "None" can be used as a default value, so in order to differentiate between
        "None" and "no value provided", create a special value for "no value
        provided".
        """

        NOT_SET = None

    default_value: _DESTTYPES | _DefaultValue | None = _DefaultValue.NOT_SET

    def _has_default_value(self) -> bool:
        """Returns whether `default_value` has been provided."""
        return self.default_value != SingleValueFlagDef._DefaultValue.NOT_SET

    def add_argument_to_parser(self, parser: argparse.ArgumentParser) -> None:
        args = ["--" + self.name]
        if self.short_name is not None:
            args += ["-" + self.short_name]

        kwargs = {}
        if self._has_default_value():
            kwargs["default"] = self.default_value
        if self.choices is not None:
            kwargs["choices"] = self.choices
        if self.help_msg is not None:
            kwargs["help"] = self.help_msg

        parser.add_argument(
            *args,
            action=_SingleValueStoreAction,
            type=self.parse_type,
            dest_type=self._get_dest_type(),
            parse_to_dest_type_fn=self._get_parse_to_dest_type_fn(),
            required=self.required,
            nargs=1,
            **kwargs,
        )

    def _do_additional_validation(self) -> None:
        if self.required:
            if self._has_default_value():
                raise ValueError("Required flags cannot have default value")
        else:
            if not self._has_default_value():
                raise ValueError("Optional flags must have a default value")

        if self._has_default_value() and self.default_value is not None:
            if not isinstance(self.default_value, self._get_dest_type()):
                raise ValueError("Default value must be of the same type as the destination type")


class EnumFlagDef(SingleValueFlagDef):
    """Definition for a flag that takes a value from an Enum.

    Sample usage:
      # This defines a flag that can be specified on the command line as:
      #   --color=red
      flag = SingleValueFlagDef(name="color", enum_type=ColorsEnum,
                                required=True)
      flag.add_argument_to_parser(argument_parser)
    """

    def __init__(self, *args, enum_type: type[enum.Enum], **kwargs):
        if not issubclass(enum_type, enum.Enum):
            raise TypeError('"enum_type" must be of type Enum')

        # These properties are set by "enum_type" so don"t let the caller set them.
        if "parse_type" in kwargs:
            raise ValueError('Cannot set "parse_type" for EnumFlagDef; set "enum_type" instead')
        kwargs["parse_type"] = str

        if "dest_type" in kwargs:
            raise ValueError('Cannot set "dest_type" for EnumFlagDef; set "enum_type" instead')
        kwargs["dest_type"] = enum_type

        if "choices" in kwargs:
            # Verify that entries in `choices` are valid enum values.
            for x in kwargs["choices"]:
                try:
                    enum_type(x)
                except ValueError:
                    raise ValueError('Invalid value in "choices": "{}"'.format(x)) from None
        else:
            kwargs["choices"] = [x.value for x in enum_type]

        super().__init__(*args, **kwargs)


class MultiValuesFlagDef(FlagDef):
    """Definition for a flag that takes multiple values.

    Sample usage:
      # This defines a flag that can be specified on the command line as:
      #   --colors=red green blue
      flag = MultiValuesFlagDef(name="colors", parse_type=str, required=True)
      flag.add_argument_to_parser(argument_parser)
    """

    def add_argument_to_parser(self, parser: argparse.ArgumentParser) -> None:
        args = ["--" + self.name]
        if self.short_name is not None:
            args += ["-" + self.short_name]

        kwargs = {}
        if self.choices is not None:
            kwargs["choices"] = self.choices
        if self.help_msg is not None:
            kwargs["help"] = self.help_msg

        parser.add_argument(
            *args,
            action=_MultiValuesAppendAction,
            type=self.parse_type,
            dest_type=self._get_dest_type(),
            parse_to_dest_type_fn=self._get_parse_to_dest_type_fn(),
            required=self.required,
            default=[],
            nargs="+",
            **kwargs,
        )

    def _do_additional_validation(self) -> None:
        # No additional validation needed.
        pass


@dataclasses.dataclass(frozen=True)
class BooleanFlagDef(FlagDef):
    """Definition for a Boolean flag.

    A boolean flag is always optional with a default value of False. The flag does
    not take any values. Specifying the flag on the commandline will set it to
    True.
    """

    def _do_additional_validation(self) -> None:
        if self.dest_type is not None:
            raise ValueError("dest_type cannot be set for BooleanFlagDef")
        if self.parse_to_dest_type_fn is not None:
            raise ValueError("parse_to_dest_type_fn cannot be set for BooleanFlagDef")
        if self.choices is not None:
            raise ValueError("choices cannot be set for BooleanFlagDef")

    def add_argument_to_parser(self, parser: argparse.ArgumentParser) -> None:
        args = ["--" + self.name]
        if self.short_name is not None:
            args += ["-" + self.short_name]

        kwargs = {}
        if self.help_msg is not None:
            kwargs["help"] = self.help_msg

        parser.add_argument(
            *args,
            action=_BooleanValueStoreAction,
            type=bool,
            required=False,
            default=False,
            nargs=0,
            **kwargs,
        )
