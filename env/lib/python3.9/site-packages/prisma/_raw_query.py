from __future__ import annotations

import json
from typing import (
    Any,
    Callable,
    overload,
)

from ._types import BaseModelT
from ._compat import model_parse


# From: https://github.com/prisma/prisma/blob/main/packages/client/src/runtime/utils/deserializeRawResults.ts
# Last checked: 2022-12-04
"""
type PrismaType =
  | 'int'
  | 'bigint'
  | 'float'
  | 'double'
  | 'string'
  | 'enum'
  | 'bytes'
  | 'bool'
  | 'char'
  | 'decimal'
  | 'json'
  | 'xml'
  | 'uuid'
  | 'datetime'
  | 'date'
  | 'time'
  | 'array'
  | 'null'
"""


@overload
def deserialize_raw_results(
    raw_list: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    ...


@overload
def deserialize_raw_results(
    raw_list: list[dict[str, object]],
    model: type[BaseModelT],
) -> list[BaseModelT]:
    ...


def deserialize_raw_results(
    raw_list: list[dict[str, Any]],
    model: type[BaseModelT] | None = None,
) -> list[BaseModelT] | list[dict[str, Any]]:
    """Deserialize a list of raw query results into their rich Python types.

    If `model` is given, convert each result into the corresponding model.
    Otherwise results are returned as a dictionary
    """
    if model is not None:
        return [
            _deserialize_prisma_object(obj, model=model, for_model=True)
            for obj in raw_list
        ]

    return [
        _deserialize_prisma_object(obj, for_model=False) for obj in raw_list
    ]


# NOTE: this very weird `for_model` API is simply here as a workaround for
# https://github.com/RobertCraigie/prisma-client-py/issues/638
#
# This should hopefully be removed soon.


@overload
def _deserialize_prisma_object(
    raw_obj: dict[str, Any],
    *,
    for_model: bool,
) -> dict[str, Any]:
    ...


@overload
def _deserialize_prisma_object(
    raw_obj: dict[str, object],
    *,
    for_model: bool,
    model: type[BaseModelT],
) -> BaseModelT:
    ...


def _deserialize_prisma_object(
    raw_obj: dict[Any, Any],
    *,
    for_model: bool,
    model: type[BaseModelT] | None = None,
) -> BaseModelT | dict[str, Any]:
    # create a local reference to avoid performance penalty of global
    # lookups on some python versions
    _deserializers = DESERIALIZERS

    new_obj = {}
    for key, raw_value in raw_obj.items():
        value = raw_value['prisma__value']
        prisma_type = raw_value['prisma__type']

        new_obj[key] = (
            _deserializers[prisma_type](value, for_model)
            if prisma_type in _deserializers
            else value
        )

    if model is not None:
        return model_parse(model, new_obj)

    return new_obj


def _deserialize_bigint(value: str, _for_model: bool) -> int:
    return int(value)


def _deserialize_decimal(value: str, _for_model: bool) -> float:
    return float(value)


def _deserialize_array(value: list[Any], for_model: bool) -> list[Any]:
    # create a local reference to avoid performance penalty of global
    # lookups on some python versions
    _deserializers = DESERIALIZERS

    arr = []
    for entry in value:
        prisma_type = entry['prisma__type']
        prisma_value = entry['prisma__value']
        arr.append(
            (
                _deserializers[prisma_type](prisma_value, for_model)
                if prisma_type in _deserializers
                else prisma_value
            )
        )

    return arr


def _deserialize_json(value: object, for_model: bool) -> object:
    # TODO: this may break if someone inserts just a string into the database
    if not isinstance(value, str) and for_model:
        # TODO: this is very bad
        #
        # Pydantic expects Json fields to be a `str`, we should implement
        # an actual workaround for this validation instead of wasting compute
        # on re-serializing the data.
        return json.dumps(value)

    # This may or may not have already been deserialized by the database
    return value


DESERIALIZERS: dict[str, Callable[[Any, bool], object]] = {
    'bigint': _deserialize_bigint,
    'decimal': _deserialize_decimal,
    'array': _deserialize_array,
    'json': _deserialize_json,
}
