# Copyright 2024 Google LLC
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
#
"""Shared utilities for working with function schemas."""

import inspect
import typing
from typing import Any, Callable, Dict
import warnings

from google.cloud.aiplatform_v1beta1 import types as aiplatform_types

Struct = Dict[str, Any]


def _generate_json_schema_from_function_using_pydantic(
    func: Callable,
) -> Struct:
    """Generates JSON Schema for a callable object.

    The `func` function needs to follow specific rules.
    All parameters must be names explicitly (`*args` and `**kwargs` are not supported).

    Args:
        func: Function for which to generate schema

    Returns:
        The JSON Schema for the function as a dict.
    """
    # FIX(b/331534434): Workaround for a breaking change.
    try:
        from pydantic import v1 as pydantic
        from pydantic.v1 import fields as pydantic_fields
    except ImportError:
        import pydantic
        from pydantic import fields as pydantic_fields

    try:
        import docstring_parser  # pylint: disable=g-import-not-at-top
    except ImportError:
        warnings.warn("Unable to import docstring_parser")
        docstring_parser = None

    function_description = func.__doc__

    # Parse parameter descriptions from the docstring.
    # Also parse the function descripton in a better way.
    parameter_descriptions = {}
    if docstring_parser:
        parsed_docstring = docstring_parser.parse(func.__doc__)
        function_description = (
            parsed_docstring.long_description or parsed_docstring.short_description
        )
        for meta in parsed_docstring.meta:
            if isinstance(meta, docstring_parser.DocstringParam):
                parameter_descriptions[meta.arg_name] = meta.description

    defaults = dict(inspect.signature(func).parameters)
    fields_dict = {
        name: (
            # 1. We infer the argument type here: use Any rather than None so
            # it will not try to auto-infer the type based on the default value.
            (param.annotation if param.annotation != inspect.Parameter.empty else Any),
            pydantic.Field(
                # 2. We do not support default values for now.
                default=(
                    param.default
                    if param.default != inspect.Parameter.empty
                    # ! Need to use Undefined instead of None
                    else pydantic_fields.Undefined
                ),
                # 3. We support user-provided descriptions.
                description=parameter_descriptions.get(name, None),
            ),
        )
        for name, param in defaults.items()
        # We do not support *args or **kwargs
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_ONLY,
        )
    }
    function_schema = pydantic.create_model(func.__name__, **fields_dict).schema()

    function_schema["title"] = func.__name__
    function_schema["description"] = function_description
    # Postprocessing
    for name, property_schema in function_schema.get("properties", {}).items():
        annotation = defaults[name].annotation
        # 5. Nullable fields:
        #     * https://github.com/pydantic/pydantic/issues/1270
        #     * https://stackoverflow.com/a/58841311
        #     * https://github.com/pydantic/pydantic/discussions/4872
        if typing.get_origin(annotation) is typing.Union and type(
            None
        ) in typing.get_args(annotation):
            # for "typing.Optional" arguments, function_arg might be a
            # dictionary like
            #
            #   {'anyOf': [{'type': 'integer'}, {'type': 'null'}]
            for schema in property_schema.pop("anyOf", []):
                schema_type = schema.get("type")
                if schema_type and schema_type != "null":
                    property_schema["type"] = schema_type
                    break
            property_schema["nullable"] = True
    # 6. Annotate required fields.
    function_schema["required"] = [
        k
        for k in defaults
        if (
            defaults[k].default == inspect.Parameter.empty
            and defaults[k].kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_ONLY,
            )
        )
    ]
    return function_schema


def adapt_json_schema_to_google_tool_schema(schema: Struct) -> Struct:
    """Adapts JSON schema to Google tool schema."""
    fixed_schema = dict(schema)
    # `$schema` is one of the basic/most common fields of the real JSON Schema.
    # But Google's Schema proto does not support it.
    # Common attributes that we remove:
    # $schema, additionalProperties
    for key in list(fixed_schema):
        # Warning: The proto-plus library removes attributes from the class,
        # so `hasattr` does not work.
        if (
            key not in aiplatform_types.Schema.meta.fields
            and key + "_" not in aiplatform_types.Schema.meta.fields
        ):
            fixed_schema.pop(key, None)
    property_schemas = fixed_schema.get("properties")
    if property_schemas:
        for k, v in property_schemas.items():
            property_schemas[k] = adapt_json_schema_to_google_tool_schema(v)
    return fixed_schema


generate_json_schema_from_function = _generate_json_schema_from_function_using_pydantic
