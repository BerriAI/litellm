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
"""LLMFnOutputRow."""
from __future__ import annotations

import abc
from typing import Any, Iterator, Mapping


# The type of value stored in a cell.
_CELLVALUETYPE = Any


def _get_name_of_type(x: type[Any]) -> str:
    if hasattr(x, "__name__"):
        return x.__name__
    return str(x)


def _validate_is_result_type(value: Any, result_type: type[Any]) -> None:
    if result_type == Any:
        return
    if not isinstance(value, result_type):
        raise ValueError(
            'Value of last entry must be of type "{}", got "{}"'.format(
                _get_name_of_type(result_type),
                _get_name_of_type(type(value)),
            )
        )


class LLMFnOutputRowView(Mapping[str, _CELLVALUETYPE], metaclass=abc.ABCMeta):
    """Immutable view of LLMFnOutputRow."""

    # Additional methods (not required by Mapping[str, _CELLVALUETYPE])
    @abc.abstractmethod
    def __contains__(self, k: str) -> bool:
        """For expressions like: x in this_instance."""

    @abc.abstractmethod
    def __str__(self) -> str:
        """For expressions like: str(this_instance)."""

    # Own methods.
    @abc.abstractmethod
    def result_type(self) -> type[Any]:
        """Returns the type enforced for the result cell."""

    @abc.abstractmethod
    def result_value(self) -> Any:
        """Get the value of the result cell."""

    @abc.abstractmethod
    def result_key(self) -> str:
        """Get the key of the result cell."""


class LLMFnOutputRow(LLMFnOutputRowView):
    """Container that represents a single row in a table of outputs.

    We represent outputs as a table. This class represents a single row in the
    table like a dictionary, where the key is the column name and the value is the
    cell value.

    A single cell is designated the "result". This contains the output of the LLM
    model after running any post-processing functions specified by the user.

    In addition to behaving like a dictionary, this class provides additional
    methods, including:
    - Getting the value of the "result" cell
    - Setting the value (and optionally the key) of the "result" cell.
    - Add a new non-result cell

    Notes: As an implementation detail, the result-cell is always kept as the
    rightmost cell.
    """

    def __init__(self, data: Mapping[str, _CELLVALUETYPE], result_type: type[Any]):
        """Constructor.

        Args:
          data: The initial value of the row. The last entry will be treated as the
            result. Cannot be empty. The value of the last entry must be `str`.
          result_type: The type of the result cell. This will be enforced at
            runtime.
        """
        self._data: dict[str, _CELLVALUETYPE] = dict(data)
        if not self._data:
            raise ValueError("Must provide non-empty data")

        self._result_type = result_type
        result_value = list(self._data.values())[-1]
        _validate_is_result_type(result_value, self._result_type)

    # Methods needed for Mapping[str, _CELLVALUETYPE]:
    def __iter__(self) -> Iterator[str]:
        return self._data.__iter__()

    def __len__(self) -> int:
        return self._data.__len__()

    def __getitem__(self, k: str) -> _CELLVALUETYPE:
        return self._data.__getitem__(k)

    # Additional methods for LLMFnOutputRowView.
    def __contains__(self, k: str) -> bool:
        return self._data.__contains__(k)

    def __str__(self) -> str:
        return "LLMFnOutputRow: {}".format(self._data.__str__())

    def result_type(self) -> type[Any]:
        return self._result_type

    def result_value(self) -> Any:
        return self._data[self.result_key()]

    def result_key(self) -> str:
        # Our invariant is that the result-cell is always the rightmost cell.
        return list(self._data.keys())[-1]

    # Mutable methods.
    def set_result_value(self, value: Any, key: str | None = None) -> None:
        """Set the value of the result cell.

        Sets the value (and optionally the key) of the result cell.

        Args:
          value: The value to set the result cell today.
          key: Optionally change the key as well.
        """
        _validate_is_result_type(value, self._result_type)

        current_key = self.result_key()
        if key is None or key == current_key:
            self._data[current_key] = value
            return

        del self._data[current_key]
        self._data[key] = value

    def add(self, key: str, value: _CELLVALUETYPE) -> None:
        """Add a non-result cell.

        Adds a new non-result cell. This does not affect the result cell.

        Args:
          key: The key of the new cell to add.
          value: The value of the new cell to add.
        """
        # Handle collisions with `key`.
        if key in self._data:
            idx = 1
            candidate_key = key
            while candidate_key in self._data:
                candidate_key = "{}_{}".format(key, idx)
                idx = idx + 1
            key = candidate_key

        # Insert the new key/value into the second rightmost position to keep
        # the result cell as the rightmost cell.
        result_key = self.result_key()
        result_value = self._data.pop(result_key)
        self._data[key] = value
        self._data[result_key] = result_value
