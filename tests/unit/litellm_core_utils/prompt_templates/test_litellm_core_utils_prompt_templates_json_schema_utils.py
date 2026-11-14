import functools
import json
import sys
from collections.abc import Mapping
from typing import Final

import pytest

from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
    schema_grammar_branch_count,
    schema_has_open_object,
    schema_has_unresolvable_ref,
    schema_property_count,
)

_ARTIFACT_FIELD_PATTERN: Final = r'^(?!__.*__$)[^\p{Cc}\p{Cf}\p{Zl}\p{Zp}"\\./[\]]{1,200}$'
_ARTIFACT_DATA_ID_PATTERN: Final = r"^(?!\.\.?(?:\/|$))[A-Za-z0-9_\-.~:@+]{1,200}$"

class TestFlattenTopLevelSchemaCombinators:
    def _customer_anyof_schema(self):
        return {
            "type": "object",
            "anyOf": [
                {
                    "properties": {"id": {"type": "string"}, "enabled": {"type": "boolean"}},
                    "required": ["id", "enabled"],
                },
                {
                    "properties": {"id": {"type": "string"}, "schedule": {"type": "string"}},
                    "required": ["id", "schedule"],
                },
            ],
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        }

    def test_merges_anyof_branches_into_object_schema(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_top_level_schema_combinators,
        )

        result = flatten_top_level_schema_combinators(self._customer_anyof_schema())

        assert "anyOf" not in result
        assert result["type"] == "object"
        assert set(result["properties"]) == {"id", "enabled", "schedule"}
        assert result["properties"]["enabled"] == {"type": "boolean"}
        assert result["required"] == ["id"]

    def test_typeless_anyof_of_object_branches_gets_intersected_required(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_top_level_schema_combinators,
        )

        schema = {
            "anyOf": [
                {
                    "properties": {"id": {"type": "string"}, "enabled": {"type": "boolean"}},
                    "required": ["id", "enabled"],
                },
                {
                    "properties": {"id": {"type": "string"}, "schedule": {"type": "string"}},
                    "required": ["id", "schedule"],
                },
            ]
        }

        result = flatten_top_level_schema_combinators(schema)

        assert result["type"] == "object"
        assert "anyOf" not in result
        assert result["required"] == ["id"]

    def test_allof_required_is_the_union_of_branches(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_top_level_schema_combinators,
        )

        schema = {
            "type": "object",
            "allOf": [
                {"properties": {"id": {"type": "string"}}, "required": ["id"]},
                {"properties": {"enabled": {"type": "boolean"}}, "required": ["enabled"]},
            ],
        }

        result = flatten_top_level_schema_combinators(schema)

        assert "allOf" not in result
        assert result["required"] == ["enabled", "id"]
        assert set(result["properties"]) == {"id", "enabled"}

    def test_top_level_schema_wins_property_collisions(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_top_level_schema_combinators,
        )

        schema = {
            "type": "object",
            "anyOf": [
                {"properties": {"id": {"type": "integer"}}},
                {"properties": {"id": {"type": "number"}}},
            ],
            "properties": {"id": {"type": "string"}},
        }

        result = flatten_top_level_schema_combinators(schema)

        assert result["properties"]["id"] == {"type": "string"}

    def test_drops_openai_rejected_scalar_keys_on_object_schema(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_top_level_schema_combinators,
        )

        schema = {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "enum": [{"id": "a"}],
            "const": {"id": "a"},
            "not": {"required": ["other"]},
        }

        result = flatten_top_level_schema_combinators(schema)

        assert "enum" not in result
        assert "const" not in result
        assert "not" not in result
        assert result["properties"] == {"id": {"type": "string"}}

    def test_resolves_local_ref_branches_from_defs(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_top_level_schema_combinators,
        )

        schema = {
            "anyOf": [{"$ref": "#/$defs/Enable"}, {"$ref": "#/$defs/Schedule"}],
            "$defs": {
                "Enable": {
                    "properties": {"id": {"type": "string"}, "enabled": {"type": "boolean"}},
                    "required": ["id", "enabled"],
                },
                "Schedule": {
                    "properties": {"id": {"type": "string"}, "schedule": {"type": "string"}},
                    "required": ["id", "schedule"],
                },
            },
        }

        result = flatten_top_level_schema_combinators(schema)

        assert "anyOf" not in result
        assert result["type"] == "object"
        assert set(result["properties"]) == {"id", "enabled", "schedule"}
        assert result["required"] == ["id"]
        assert "$defs" in result

    def test_flattens_nested_combinator_branch_from_definitions(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_top_level_schema_combinators,
        )

        schema = {
            "type": "object",
            "oneOf": [
                {"$ref": "#/definitions/Toggle"},
                {"allOf": [{"properties": {"schedule": {"type": "string"}}, "required": ["schedule"]}]},
            ],
            "definitions": {"Toggle": {"properties": {"enabled": {"type": "boolean"}}, "required": ["enabled"]}},
        }

        result = flatten_top_level_schema_combinators(schema)

        assert "oneOf" not in result
        assert set(result["properties"]) == {"enabled", "schedule"}
        assert "required" not in result

    def test_unresolvable_ref_branch_leaves_schema_untouched(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_top_level_schema_combinators,
        )

        schema = {
            "type": "object",
            "anyOf": [{"$ref": "https://example.com/schemas/automation.json"}],
            "properties": {"id": {"type": "string"}},
        }

        assert flatten_top_level_schema_combinators(schema) is schema

    def test_self_referencing_ref_branch_leaves_schema_untouched(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_top_level_schema_combinators,
        )

        schema = {
            "type": "object",
            "anyOf": [{"$ref": "#/$defs/Node"}],
            "$defs": {"Node": {"type": "object", "anyOf": [{"$ref": "#/$defs/Node"}]}},
        }

        assert flatten_top_level_schema_combinators(schema) is schema

    @pytest.mark.parametrize("boolean_branch", [True, False])
    def test_boolean_branch_leaves_schema_untouched(self, boolean_branch):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_top_level_schema_combinators,
        )

        schema = {
            "type": "object",
            "anyOf": [boolean_branch, {"properties": {"id": {"type": "string"}}, "required": ["id"]}],
        }

        assert flatten_top_level_schema_combinators(schema) is schema

    def test_root_required_is_combined_with_branch_required(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_top_level_schema_combinators,
        )

        allof_schema = {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"type": "string"}},
            "allOf": [{"properties": {"enabled": {"type": "boolean"}}, "required": ["enabled"]}],
        }
        anyof_schema = {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"type": "string"}},
            "anyOf": [
                {"properties": {"name": {"type": "string"}, "a": {"type": "string"}}, "required": ["name", "a"]},
                {"properties": {"name": {"type": "string"}, "b": {"type": "string"}}, "required": ["name", "b"]},
            ],
        }

        assert flatten_top_level_schema_combinators(allof_schema)["required"] == ["enabled", "id"]
        assert flatten_top_level_schema_combinators(anyof_schema)["required"] == ["id", "name"]

    def test_repeated_refs_are_expanded_once(self):
        import time

        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_top_level_schema_combinators,
        )

        fan_out, chain_length = 8, 8
        schema = {
            "type": "object",
            "anyOf": [{"$ref": "#/$defs/Level0"}],
            "$defs": {
                **{
                    f"Level{level}": {"anyOf": [{"$ref": f"#/$defs/Level{level + 1}"}] * fan_out}
                    for level in range(chain_length)
                },
                f"Level{chain_length}": {"type": "object", "properties": {"id": {"type": "string"}}},
            },
        }

        started = time.perf_counter()
        result = flatten_top_level_schema_combinators(schema)

        assert time.perf_counter() - started < 5
        assert "anyOf" not in result
        assert result["properties"] == {"id": {"type": "string"}}

    def test_nesting_past_the_depth_cap_leaves_schema_untouched(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_top_level_schema_combinators,
        )

        def nested(levels):
            leaf = {"type": "object", "properties": {"id": {"type": "string"}}}
            return functools.reduce(lambda inner, _: {"type": "object", "anyOf": [inner]}, range(levels), leaf)

        shallow, deep = nested(20), nested(40)

        assert "anyOf" not in flatten_top_level_schema_combinators(shallow)
        assert flatten_top_level_schema_combinators(deep) is deep

    @pytest.mark.parametrize(
        "branches",
        [
            [{"required": ["enabled"]}, {"required": ["schedule"]}],
            [{"type": "object", "required": ["enabled"]}, {"type": "object", "required": ["schedule"]}],
        ],
    )
    def test_typeless_root_with_properties_flattens_branches_without_properties(self, branches):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_top_level_schema_combinators,
        )

        schema = {
            "properties": {"id": {"type": "string"}, "enabled": {"type": "boolean"}, "schedule": {"type": "string"}},
            "required": ["id"],
            "anyOf": branches,
        }

        result = flatten_top_level_schema_combinators(schema)

        assert "anyOf" not in result
        assert result["type"] == "object"
        assert set(result["properties"]) == {"id", "enabled", "schedule"}
        assert result["required"] == ["id"]

    def test_typeless_root_flattens_typed_object_branches_without_properties(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_top_level_schema_combinators,
        )

        schema = {
            "anyOf": [
                {"type": "object", "properties": {"id": {"type": "string"}}},
                {"type": "object", "required": ["id"]},
            ]
        }

        result = flatten_top_level_schema_combinators(schema)

        assert "anyOf" not in result
        assert result["type"] == "object"
        assert result["properties"] == {"id": {"type": "string"}}
        assert "required" not in result

    def test_non_object_union_passes_through_unchanged(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_top_level_schema_combinators,
        )

        schema = {"anyOf": [{"type": "string"}, {"type": "number"}]}

        assert flatten_top_level_schema_combinators(schema) is schema

    def test_schema_without_rejected_keys_is_returned_as_is(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_top_level_schema_combinators,
        )

        schema = {"type": "object", "properties": {"nested": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}

        assert flatten_top_level_schema_combinators(schema) is schema

    def test_input_schema_is_never_mutated(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_top_level_schema_combinators,
        )

        schema = self._customer_anyof_schema()
        snapshot = json.loads(json.dumps(schema))

        flatten_top_level_schema_combinators(schema)

        assert schema == snapshot


class TestToolWithSanitizedParameters:
    def _anyof_tool(self):
        return {
            "type": "function",
            "function": {
                "name": "automation_update",
                "description": "Update an automation",
                "parameters": {
                    "type": "object",
                    "anyOf": [
                        {
                            "properties": {"id": {"type": "string"}, "enabled": {"type": "boolean"}},
                            "required": ["id", "enabled"],
                        },
                        {
                            "properties": {"id": {"type": "string"}, "schedule": {"type": "string"}},
                            "required": ["id", "schedule"],
                        },
                    ],
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            },
        }

    def test_flattens_anyof_parameters_into_new_tool(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_combinators_and_drop_non_python_regex_patterns,
            tool_with_sanitized_parameters,
        )

        tool = self._anyof_tool()
        result = tool_with_sanitized_parameters(tool, flatten_combinators_and_drop_non_python_regex_patterns)

        assert result is not tool
        parameters = result["function"]["parameters"]
        assert "anyOf" not in parameters
        assert parameters["type"] == "object"
        assert set(parameters["properties"]) == {"id", "enabled", "schedule"}
        assert parameters["required"] == ["id"]
        assert result["function"]["name"] == "automation_update"
        assert tool == self._anyof_tool()

    def test_clean_parameters_return_the_same_tool_object(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_combinators_and_drop_non_python_regex_patterns,
            tool_with_sanitized_parameters,
        )

        tool = {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
            },
        }

        assert tool_with_sanitized_parameters(tool, flatten_combinators_and_drop_non_python_regex_patterns) is tool

    def test_pattern_only_sanitizer_drops_the_regex_and_keeps_the_union(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            drop_non_python_regex_patterns,
            tool_with_sanitized_parameters,
        )

        tool = self._anyof_tool()
        tool["function"]["parameters"]["properties"]["id"]["pattern"] = _ARTIFACT_FIELD_PATTERN

        result = tool_with_sanitized_parameters(tool, drop_non_python_regex_patterns)

        parameters = result["function"]["parameters"]
        assert parameters["properties"]["id"] == {"type": "string"}
        assert parameters["anyOf"] == self._anyof_tool()["function"]["parameters"]["anyOf"]
        assert tool["function"]["parameters"]["properties"]["id"]["pattern"] == _ARTIFACT_FIELD_PATTERN

    @pytest.mark.parametrize(
        "tool",
        [
            {"type": "function"},
            {"type": "function", "function": "not-a-dict"},
            {"type": "function", "function": {"name": "no_params"}},
            {"type": "function", "function": {"name": "bad_params", "parameters": "not-a-dict"}},
        ],
    )
    def test_non_dict_function_or_parameters_return_the_same_tool_object(self, tool):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            flatten_combinators_and_drop_non_python_regex_patterns,
            tool_with_sanitized_parameters,
        )

        assert tool_with_sanitized_parameters(tool, flatten_combinators_and_drop_non_python_regex_patterns) is tool

    def test_sanitizes_the_input_schema_of_an_anthropic_tool(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            drop_lookaround_regex_patterns,
            tool_with_sanitized_parameters,
        )

        tool = {
            "name": "ArtifactData",
            "description": "Read a shared database",
            "input_schema": {
                "type": "object",
                "properties": {"doc_id": {"type": "string", "pattern": _ARTIFACT_DATA_ID_PATTERN}},
            },
        }

        result = tool_with_sanitized_parameters(tool, drop_lookaround_regex_patterns)

        assert result == {
            "name": "ArtifactData",
            "description": "Read a shared database",
            "input_schema": {"type": "object", "properties": {"doc_id": {"type": "string"}}},
        }
        assert tool["input_schema"]["properties"]["doc_id"]["pattern"] == _ARTIFACT_DATA_ID_PATTERN

    def test_returns_the_same_anthropic_tool_when_its_schema_has_nothing_to_drop(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            drop_lookaround_regex_patterns,
            tool_with_sanitized_parameters,
        )

        tool = {"name": "Read", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}

        assert tool_with_sanitized_parameters(tool, drop_lookaround_regex_patterns) is tool


def _regex_schema(pattern):
    return {
        "type": "object",
        "properties": {
            "field": {"type": "string", "pattern": pattern},
            "writes": {
                "type": "array",
                "items": {"properties": {"doc_id": {"type": "string", "pattern": pattern}}},
            },
            "query": {"anyOf": [{"type": "string", "pattern": pattern}, {"type": "null"}]},
            "pair": {"type": "array", "prefixItems": [{"type": "string", "pattern": pattern}]},
            "extra": {"type": "object", "additionalProperties": {"type": "string", "pattern": pattern}},
            "tagged": {
                "type": "object",
                "patternProperties": {pattern: {"type": "string"}, "^x_": {"type": "integer"}},
            },
        },
        "$defs": {"segment": {"type": "string", "pattern": pattern}},
        "required": ["field"],
    }


class TestDropNonPythonRegexPatterns:
    """Claude Code's Artifact tool declares ECMA-262 ``\\p{..}`` escapes that OpenAI's
    validator, which compiles ``pattern`` values and ``patternProperties`` keys with
    Python ``re``, refuses as "not a 'regex'"."""

    def test_drops_every_regex_python_re_rejects_from_every_schema_position(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            drop_non_python_regex_patterns,
        )

        schema = _regex_schema(_ARTIFACT_FIELD_PATTERN)

        result = drop_non_python_regex_patterns(schema)

        assert '"pattern"' not in json.dumps(result)
        properties = result["properties"]
        assert properties["field"] == {"type": "string"}
        assert properties["writes"]["items"]["properties"]["doc_id"] == {"type": "string"}
        assert properties["query"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
        assert properties["pair"]["prefixItems"] == [{"type": "string"}]
        assert properties["extra"]["additionalProperties"] == {"type": "string"}
        assert properties["tagged"]["patternProperties"] == {"^x_": {"type": "integer"}}
        assert result["$defs"]["segment"] == {"type": "string"}
        assert result["required"] == ["field"]
        assert schema == _regex_schema(_ARTIFACT_FIELD_PATTERN)

    def test_keeps_regexes_python_re_compiles_and_returns_the_same_object(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            drop_non_python_regex_patterns,
        )

        schema = _regex_schema(r'^(?!__.*__$)[^"\\./[\]]{1,200}$')

        assert drop_non_python_regex_patterns(schema) is schema

    def test_pattern_keys_inside_data_positions_are_not_regexes(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            drop_non_python_regex_patterns,
        )

        schema = {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "template": {"type": "object", "default": {"pattern": _ARTIFACT_FIELD_PATTERN}},
                "samples": {"type": "array", "examples": [{"pattern": _ARTIFACT_FIELD_PATTERN}]},
                "fixed": {"const": {"pattern": _ARTIFACT_FIELD_PATTERN}},
                "vendor": {"type": "string", "x-litellm": {"pattern": _ARTIFACT_FIELD_PATTERN}},
            },
            "required": ["pattern"],
        }

        assert drop_non_python_regex_patterns(schema) is schema

    def test_regex_nested_past_what_python_re_can_parse_is_dropped_not_raised(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            drop_non_python_regex_patterns,
        )

        schema = {
            "type": "object",
            "properties": {"deep": {"type": "string", "pattern": "(" * 2000 + "a" + ")" * 2000}},
        }

        assert drop_non_python_regex_patterns(schema)["properties"]["deep"] == {"type": "string"}

    def test_walks_schemas_deeper_than_the_interpreter_recursion_limit(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            drop_non_python_regex_patterns,
        )

        depth = sys.getrecursionlimit()
        leaf = {"type": "string", "pattern": _ARTIFACT_FIELD_PATTERN}
        schema = functools.reduce(
            lambda inner, _: {"type": "object", "properties": {"child": inner}}, range(depth), leaf
        )

        result = drop_non_python_regex_patterns(schema)

        assert functools.reduce(lambda node, _: node["properties"]["child"], range(depth), result) == {"type": "string"}
        assert functools.reduce(lambda node, _: node["properties"]["child"], range(depth), schema) is leaf

    def test_leaves_levels_past_the_json_nesting_limit_alone(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            drop_non_python_regex_patterns,
        )

        leaf = {"type": "string", "pattern": _ARTIFACT_FIELD_PATTERN}
        schema = functools.reduce(
            lambda inner, _: {"type": "object", "properties": {"child": inner}}, range(1100), leaf
        )

        assert drop_non_python_regex_patterns(schema) is schema


class TestDropLookaroundRegexPatterns:
    """Kimi K3 and Grok 4.6/4.7 on Bedrock Converse reject every tool schema regex that
    uses a lookaround assertion, Claude Code's ``ArtifactData`` ``pattern`` included."""

    @pytest.mark.parametrize(
        "pattern",
        [r"^(?!x).*$", r"^(?=.*a).*$", r"^.*(?<!x)$", r"^.*(?<=a)$", _ARTIFACT_DATA_ID_PATTERN],
        ids=["negative-lookahead", "positive-lookahead", "negative-lookbehind", "positive-lookbehind", "ArtifactData"],
    )
    def test_drops_every_lookaround_regex_from_every_schema_position(self, pattern):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            drop_lookaround_regex_patterns,
        )

        schema = _regex_schema(pattern)

        result = drop_lookaround_regex_patterns(schema)

        assert '"pattern"' not in json.dumps(result)
        properties = result["properties"]
        assert properties["field"] == {"type": "string"}
        assert properties["writes"]["items"]["properties"]["doc_id"] == {"type": "string"}
        assert properties["query"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
        assert properties["pair"]["prefixItems"] == [{"type": "string"}]
        assert properties["extra"]["additionalProperties"] == {"type": "string"}
        assert properties["tagged"]["patternProperties"] == {"^x_": {"type": "integer"}}
        assert result["$defs"]["segment"] == {"type": "string"}
        assert result["required"] == ["field"]
        assert schema == _regex_schema(pattern)

    @pytest.mark.parametrize(
        "pattern",
        [r"^[A-Za-z0-9_\-.~:@+]{1,200}$", r"^(?:a|b)+$", r"^(?P<name>\w+)$", r"^(?i)abc$", r"^[^\p{Cc}\p{Cf}]{1,200}$"],
        ids=["plain", "non-capturing-group", "named-group", "inline-flag", "non-python-without-lookaround"],
    )
    def test_keeps_regexes_without_lookaround_and_returns_the_same_object(self, pattern):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            drop_lookaround_regex_patterns,
        )

        schema = _regex_schema(pattern)

        assert drop_lookaround_regex_patterns(schema) is schema

    def test_lookaround_inside_data_positions_is_not_a_regex(self):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            drop_lookaround_regex_patterns,
        )

        schema = {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "template": {"type": "object", "default": {"pattern": _ARTIFACT_DATA_ID_PATTERN}},
                "hint": {"type": "string", "description": "ids match " + _ARTIFACT_DATA_ID_PATTERN},
            },
            "required": ["pattern"],
        }

        assert drop_lookaround_regex_patterns(schema) is schema


@pytest.mark.parametrize(
    ("dropper", "patterns"),
    [
        ("drop_non_python_regex_patterns", (_ARTIFACT_FIELD_PATTERN, r"^\p{L}+$")),
        ("drop_lookaround_regex_patterns", (_ARTIFACT_DATA_ID_PATTERN, r"^(?=.*[a-z])\w+$")),
    ],
    ids=["non-python", "lookaround"],
)
class TestDroppedPatternPropertiesKeepTheirNamesAllowed:
    """Dropping a ``patternProperties`` key from an object closed by ``additionalProperties:
    false`` must not ban the names that key allowed: its value schema takes over as the
    object's ``additionalProperties``."""

    @staticmethod
    def _drop(dropper):
        from litellm.litellm_core_utils.prompt_templates.json_schema_utils import (
            drop_lookaround_regex_patterns,
            drop_non_python_regex_patterns,
        )

        return {
            "drop_non_python_regex_patterns": drop_non_python_regex_patterns,
            "drop_lookaround_regex_patterns": drop_lookaround_regex_patterns,
        }[dropper]

    def test_closed_object_takes_the_dropped_value_schema(self, dropper, patterns):
        schema = {
            "type": "object",
            "patternProperties": {patterns[0]: {"type": "string", "pattern": patterns[0]}},
            "additionalProperties": False,
        }

        assert self._drop(dropper)(schema) == {
            "type": "object",
            "patternProperties": {},
            "additionalProperties": {"type": "string"},
        }

    def test_closed_object_losing_two_entries_accepts_either_value_schema(self, dropper, patterns):
        schema = {
            "type": "object",
            "patternProperties": {
                patterns[0]: {"type": "string"},
                patterns[1]: {"type": "integer"},
                "^x_": {"type": "boolean"},
            },
            "additionalProperties": False,
        }

        assert self._drop(dropper)(schema) == {
            "type": "object",
            "patternProperties": {"^x_": {"type": "boolean"}},
            "additionalProperties": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        }

    def test_object_with_its_own_additional_properties_schema_keeps_it(self, dropper, patterns):
        schema = {
            "type": "object",
            "patternProperties": {patterns[0]: {"type": "string"}},
            "additionalProperties": {"type": "integer"},
        }

        assert self._drop(dropper)(schema) == {
            "type": "object",
            "patternProperties": {},
            "additionalProperties": {"type": "integer"},
        }


@pytest.mark.parametrize(
    ("schema", "is_open"),
    [
        ({"type": "object", "properties": {"a": {"type": "string"}}}, False),
        ({"type": "object", "properties": {}}, False),
        ({"type": "object", "additionalProperties": False}, False),
        ({"type": "string"}, False),
        ({"type": "object"}, True),
        ({"type": ["object", "null"]}, True),
        ({"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": True}, True),
        ({"type": "object", "additionalProperties": {"type": "integer"}}, True),
        ({"type": "object", "properties": {}, "patternProperties": {"^x": {"type": "string"}}}, True),
        ({"type": "object", "properties": {"inner": {"anyOf": [{"type": "object"}, {"type": "null"}]}}}, True),
        ({"type": "array", "items": {"type": "object", "additionalProperties": {"type": "string"}}}, True),
    ],
)
def test_schema_has_open_object_flags_any_node_that_accepts_unlisted_keys(schema, is_open):
    assert schema_has_open_object(schema) is is_open


def _object_with(properties: Mapping[str, object], **defs: Mapping[str, object]) -> Mapping[str, object]:
    return {"type": "object", "properties": properties, **defs}


@pytest.mark.parametrize(
    ("schema", "unresolvable"),
    [
        (_object_with({"a": {"type": "string"}}), False),
        (_object_with({"a": {"$ref": "#/$defs/A"}, "b": {"$ref": "#/$defs/A"}}, **{"$defs": {"A": {"type": "string"}}}), False),
        (_object_with({"a": {"$ref": "#/definitions/A"}}, definitions={"A": {"$ref": "#/definitions/B"}, "B": {"type": "integer"}}), False),
        (_object_with({"a": {"$ref": "/$defs/A"}}, **{"$defs": {"A": {"items": {"$ref": "/$defs/B"}}, "B": {"type": "string"}}}), False),
        (_object_with({"a": {"$ref": "#/$defs/Missing"}}, **{"$defs": {"A": {"type": "string"}}}), True),
        (_object_with({"a": {"$ref": "other.json#/A"}}), True),
        (_object_with({"parent": {"$ref": "#"}}), True),
        (_object_with({"a": {"$ref": "#/$defs/A"}}, **{"$defs": {"A": {"type": "array", "items": {"$ref": "#/$defs/A"}}}}), True),
        (_object_with({"a": {"$ref": "/$defs/A"}}, **{"$defs": {"A": {"type": "array", "items": {"$ref": "/$defs/A"}}}}), True),
        (
            _object_with(
                {"a": {"$ref": "#/$defs/A"}},
                **{"$defs": {"A": {"properties": {"b": {"$ref": "#/$defs/B"}}}, "B": {"properties": {"a": {"$ref": "#/$defs/A"}}}}},
            ),
            True,
        ),
    ],
    ids=["no-refs", "shared-def", "chained-definitions", "root-path-defs", "missing-def", "external", "root", "self-cycle", "root-path-self-cycle", "mutual-cycle"],
)
def test_schema_has_unresolvable_ref_flags_refs_that_cannot_be_inlined(schema, unresolvable):
    assert schema_has_unresolvable_ref(schema) is unresolvable


@pytest.mark.parametrize(
    ("schema", "properties", "branches"),
    [
        (_object_with({"a": {"type": "string"}, "b": {"type": "string"}, "c": {"type": "string"}}, required=["a"]), 3, 8),
        (
            _object_with(
                {"x": {"$ref": "#/$defs/Pair"}, "y": {"$ref": "#/$defs/Pair"}},
                required=["x", "y"],
                **{"$defs": {"Pair": _object_with({"l": {"type": "string"}, "r": {"type": "string"}}, required=["l"])}},
            ),
            4,
            7,
        ),
        (_object_with({"inner": _object_with({"z": {"type": "string"}})}, required=["inner"]), 2, 4),
    ],
    ids=["flat", "shared-def-counted-once", "nested"],
)
def test_schema_size_counts_every_written_object_once(schema, properties, branches):
    assert schema_property_count(schema) == properties
    assert schema_grammar_branch_count(schema) == branches
