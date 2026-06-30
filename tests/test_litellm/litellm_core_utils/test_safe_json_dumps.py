import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps, strip_null_bytes


def test_primitive_types():
    # Test basic primitive types
    assert safe_dumps("test") == '"test"'
    assert safe_dumps(123) == "123"
    assert safe_dumps(3.14) == "3.14"
    assert safe_dumps(True) == "true"
    assert safe_dumps(None) == "null"


def test_nested_structures():
    # Test nested dictionaries and lists
    data = {"name": "test", "numbers": [1, 2, 3], "nested": {"a": 1, "b": 2}}
    result = json.loads(safe_dumps(data))
    assert result["name"] == "test"
    assert result["numbers"] == [1, 2, 3]
    assert result["nested"] == {"a": 1, "b": 2}


def test_circular_reference():
    # Test circular reference detection
    d = {}
    d["self"] = d
    result = json.loads(safe_dumps(d))
    assert result["self"] == "CircularReference Detected"


def test_max_depth():
    # Test maximum depth handling
    deep_dict = {}
    current = deep_dict
    for i in range(15):
        current["deeper"] = {}
        current = current["deeper"]

    result = json.loads(safe_dumps(deep_dict, max_depth=5))
    assert "MaxDepthExceeded" in str(result)


def test_default_max_depth():
    # Test that default max depth still prevents infinite recursion
    deep_dict = {}
    current = deep_dict
    for i in range(1000):  # Create a very deep dictionary
        current["deeper"] = {}
        current = current["deeper"]

    result = json.loads(safe_dumps(deep_dict))  # No max_depth parameter provided
    assert "MaxDepthExceeded" in str(result)


def test_complex_types():
    # Test handling of sets and tuples
    data = {"set": {1, 2, 3}, "tuple": (4, 5, 6)}
    result = json.loads(safe_dumps(data))
    assert result["set"] == [1, 2, 3]  # Sets are converted to sorted lists
    assert result["tuple"] == [4, 5, 6]  # Tuples are converted to lists


def test_unserializable_object():
    # Test handling of unserializable objects
    class TestClass:
        def __str__(self):
            raise Exception("Cannot convert to string")

    obj = TestClass()
    result = json.loads(safe_dumps(obj))
    assert result == "Unserializable Object"


def test_non_standard_dict_keys():
    try:
        # Test handling of dictionaries with non-standard keys
        class GCCollector:
            def __str__(self):
                return "GCCollector"

        data = {GCCollector(): "value", "test": "test"}
        json_dump = safe_dumps(data)
        print(json_dump)
        result = json.loads(json_dump)
        assert result["test"] == "test"
    except Exception as e:
        print(e)
        import traceback

        traceback.print_exc()
        raise e


def test_non_standard_dict_keys_complex():
    try:
        # Test handling of dictionaries with non-standard keys
        class GCCollector:
            def __str__(self):
                return "GCCollector"

        data = [
            {"test": "test"},
            GCCollector(),
            {
                "bad_key": "bad_value",
            },
            {
                GCCollector(): "value",
            },
            {
                "bad_key": GCCollector(),
            },
            (GCCollector(), GCCollector()),
        ]
        json_dump = safe_dumps(data)
        print(json_dump)
        result = json.loads(json_dump)
        print("result=", json.dumps(result, indent=4))
        assert result[0]["test"] == "test"
        assert result[1] == "GCCollector"
        assert result[2]["bad_key"] == "bad_value"
        assert result[3] == {}
        assert result[4]["bad_key"] == "GCCollector"
        assert result[5][0] == "GCCollector"
        assert result[5][1] == "GCCollector"
    except Exception as e:
        print(e)
        import traceback

        traceback.print_exc()
        raise e


def test_strip_null_bytes_helper():
    assert strip_null_bytes("hello\x00world") == "helloworld"
    assert strip_null_bytes("\x00\x00abc\x00") == "abc"
    assert strip_null_bytes("no null here") == "no null here"


def test_null_byte_stripped_from_string():
    out = safe_dumps("hello\x00world")
    assert "\\u0000" not in out
    assert json.loads(out) == "helloworld"


def test_null_byte_stripped_in_nested_structure():
    data = {
        "messages": [{"role": "user", "content": "bad\x00content"}],
        "nested": {"k\x00ey": "v\x00alue"},
    }
    out = safe_dumps(data)
    assert "\\u0000" not in out
    result = json.loads(out)
    assert result["messages"][0]["content"] == "badcontent"
    assert result["nested"] == {"key": "value"}


def test_null_byte_stripped_in_fallback_str():
    class WithNullStr:
        def __str__(self):
            return "obj\x00repr"

    out = safe_dumps({"obj": WithNullStr()})
    assert "\\u0000" not in out
    assert json.loads(out)["obj"] == "objrepr"


def test_clean_strings_are_not_run_through_replace():
    """Regression for LIT-3910.

    safe_dumps must not call ``str.replace`` on NUL-free strings. Running the
    NUL strip unconditionally on every value and dict key (the v1.89.x behavior)
    added per-request serialization overhead that scaled with payload size and
    showed up under ``store_prompts_in_spend_logs``. Clean strings, which are the
    overwhelming majority, must be returned untouched.
    """

    class ReplaceForbidden(str):
        def replace(self, *args, **kwargs):
            raise AssertionError("safe_dumps ran str.replace on a NUL-free string")

    data = {
        ReplaceForbidden("clean_key"): ReplaceForbidden("clean_value"),
        "nested": [ReplaceForbidden("a"), {"deep": ReplaceForbidden("b")}],
    }
    result = json.loads(safe_dumps(data))
    assert result["clean_key"] == "clean_value"
    assert result["nested"][0] == "a"
    assert result["nested"][1]["deep"] == "b"


def test_pydantic_base_model():
    from pydantic import BaseModel

    class InnerModel(BaseModel):
        value: int
        label: str

    class OuterModel(BaseModel):
        name: str
        inner: InnerModel
        tags: list

    outer = OuterModel(
        name="test", inner=InnerModel(value=42, label="hello"), tags=["a", "b"]
    )

    # Test a pydantic model at the top level
    result = json.loads(safe_dumps(outer))
    assert result["name"] == "test"
    assert result["inner"] == {"value": 42, "label": "hello"}
    assert result["tags"] == ["a", "b"]

    # Test pydantic models nested inside dicts and lists
    data = {
        "healthy_endpoints": [outer, InnerModel(value=1, label="one")],
        "count": 2,
    }
    result = json.loads(safe_dumps(data))
    assert result["count"] == 2
    assert len(result["healthy_endpoints"]) == 2
    assert result["healthy_endpoints"][0]["name"] == "test"
    assert result["healthy_endpoints"][1] == {"value": 1, "label": "one"}


# --- single-pass fast path: equivalence with the recursive sanitizer ---

import random

from pydantic import BaseModel

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH


def _reference_safe_dumps(data, max_depth=DEFAULT_MAX_RECURSE_DEPTH):
    """The pre-optimization recursive sanitizer, kept verbatim as a test oracle."""

    def _serialize(obj, seen, depth):
        if depth > max_depth:
            return "MaxDepthExceeded"
        if isinstance(obj, str):
            return obj.replace("\x00", "") if "\x00" in obj else obj
        if isinstance(obj, (int, float, bool, type(None))):
            return obj
        if id(obj) in seen:
            return "CircularReference Detected"
        seen.add(id(obj))
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if isinstance(k, str):
                    clean_k = k.replace("\x00", "") if "\x00" in k else k
                    result[clean_k] = _serialize(v, seen, depth + 1)
            seen.remove(id(obj))
            return result
        elif isinstance(obj, list):
            result = [_serialize(item, seen, depth + 1) for item in obj]
            seen.remove(id(obj))
            return result
        elif isinstance(obj, tuple):
            result = tuple(_serialize(item, seen, depth + 1) for item in obj)
            seen.remove(id(obj))
            return result
        elif isinstance(obj, set):
            result = sorted([_serialize(item, seen, depth + 1) for item in obj])
            seen.remove(id(obj))
            return result
        elif isinstance(obj, BaseModel):
            dumped = obj.model_dump()
            result = _serialize(dumped, seen, depth + 1)
            seen.remove(id(obj))
            return result
        else:
            try:
                return strip_null_bytes(str(obj))
            except Exception:
                return "Unserializable Object"

    return json.dumps(_serialize(data, set(), 0), default=str)


class _FuzzModel(BaseModel):
    a: int
    b: str
    c: list


def _random_payload(rng, depth):
    if depth >= 5:
        return rng.choice([rng.randint(-9, 9), "leaf", None, True, "with\x00nul"])
    kind = rng.choice(
        ["int", "float", "str", "nul_str", "none", "bool", "list", "tuple", "dict", "set", "model"]
    )
    if kind == "int":
        return rng.randint(-1000, 1000)
    if kind == "float":
        return rng.random() * 100
    if kind == "str":
        return rng.choice(["hello", "", "a b c", "unicode_é", 'quote"here', "tab\tnl\n"])
    if kind == "nul_str":
        return rng.choice(["pre\x00post", "\x00lead", "trail\x00", "lit \\u0000 text"])
    if kind == "none":
        return None
    if kind == "bool":
        return rng.choice([True, False])
    if kind == "list":
        return [_random_payload(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    if kind == "tuple":
        return tuple(_random_payload(rng, depth + 1) for _ in range(rng.randint(0, 3)))
    if kind == "dict":
        # string keys only (the realistic shape; non-str keys are covered separately)
        return {f"k{i}\x00x" if rng.random() < 0.2 else f"k{i}": _random_payload(rng, depth + 1) for i in range(rng.randint(0, 4))}
    if kind == "set":
        return set(rng.sample(range(30), rng.randint(0, 4)))  # one comparable type, sortable
    return _FuzzModel(a=rng.randint(0, 9), b=rng.choice(["x", "y\x00z"]), c=[rng.randint(0, 3)])


def test_fuzz_matches_reference_sanitizer():
    """The single-pass path must be byte-identical to the recursive sanitizer for
    realistic (string-keyed) payloads, including NUL bytes, sets, tuples, models,
    and nesting."""
    rng = random.Random(20240630)
    for _ in range(600):
        data = _random_payload(rng, 0)
        assert safe_dumps(data) == _reference_safe_dumps(data)


def test_tightened_max_depth_still_truncates():
    """A caller-tightened max_depth must bypass the fast path and truncate exactly."""
    data = {"a": {"b": {"c": {"d": "deep"}}}}
    assert "MaxDepthExceeded" in safe_dumps(data, max_depth=2)
    assert safe_dumps(data, max_depth=2) == _reference_safe_dumps(data, max_depth=2)


def test_non_string_primitive_keys_are_kept():
    """With the single-pass path, dict keys json can stringify (int/bool/None) are
    kept with their JSON key form instead of being silently dropped. Keys json
    cannot encode still fall back to the sanitizer and are dropped."""
    assert json.loads(safe_dumps({1: "a", "b": "c"})) == {"1": "a", "b": "c"}

    class Unhashable:
        def __str__(self):
            return "obj"

    # object key -> json raises -> sanitizer fallback drops it (legacy behavior)
    assert json.loads(safe_dumps({Unhashable(): "v", "ok": 1})) == {"ok": 1}
