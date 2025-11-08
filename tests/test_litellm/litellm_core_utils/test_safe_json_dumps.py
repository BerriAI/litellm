import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps


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


def test_pydantic_model_serialization():
    """
    Test that safe_dumps properly serializes Pydantic models to dictionaries
    instead of converting them to string representations.
    This prevents the validation error: "Input should be a valid dictionary or instance of ContentFilterPattern"
    """
    from litellm.types.guardrails import ContentFilterAction, ContentFilterPattern
    
    pattern = ContentFilterPattern(
        pattern_type="prebuilt",
        pattern_name="email",
        name="email",
        action=ContentFilterAction.BLOCK
    )
    
    result = safe_dumps({"patterns": [pattern]})
    
    parsed = json.loads(result)
    assert "patterns" in parsed
    assert isinstance(parsed["patterns"], list)
    assert len(parsed["patterns"]) == 1
    assert isinstance(parsed["patterns"][0], dict)
    assert parsed["patterns"][0]["pattern_type"] == "prebuilt"
    assert parsed["patterns"][0]["pattern_name"] == "email"
    assert parsed["patterns"][0]["name"] == "email"
    assert parsed["patterns"][0]["action"] == "BLOCK"
    
    assert "pattern_type='prebuilt'" not in result


def test_list_of_pydantic_models():
    """
    Test that safe_dumps properly serializes a list of Pydantic models.
    """
    from litellm.types.guardrails import ContentFilterAction, ContentFilterPattern
    
    patterns = [
        ContentFilterPattern(
            pattern_type="prebuilt",
            pattern_name="email",
            name="email",
            action=ContentFilterAction.BLOCK
        ),
        ContentFilterPattern(
            pattern_type="regex",
            pattern=r"\d{3}-\d{2}-\d{4}",
            name="ssn",
            action=ContentFilterAction.MASK
        ),
    ]
    
    result = safe_dumps({"patterns": patterns})
    
    parsed = json.loads(result)
    assert "patterns" in parsed
    assert isinstance(parsed["patterns"], list)
    assert len(parsed["patterns"]) == 2
    
    assert parsed["patterns"][0]["pattern_type"] == "prebuilt"
    assert parsed["patterns"][0]["pattern_name"] == "email"
    assert parsed["patterns"][0]["action"] == "BLOCK"
    
    assert parsed["patterns"][1]["pattern_type"] == "regex"
    assert parsed["patterns"][1]["pattern"] == r"\d{3}-\d{2}-\d{4}"
    assert parsed["patterns"][1]["action"] == "MASK"
    
    assert "ContentFilterPattern" not in result


def test_nested_pydantic_models_in_dict():
    """
    Test that safe_dumps properly serializes nested dictionaries containing Pydantic models.
    This simulates the guardrail litellm_params structure.
    """
    from litellm.types.guardrails import ContentFilterAction, ContentFilterPattern
    
    litellm_params = {
        "guardrail": "litellm_content_filter",
        "mode": "pre_call",
        "patterns": [
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="email",
                name="email",
                action=ContentFilterAction.BLOCK
            )
        ],
        "metadata": {
            "nested": True,
            "count": 1
        }
    }
    
    result = safe_dumps(litellm_params)
    
    parsed = json.loads(result)
    assert parsed["guardrail"] == "litellm_content_filter"
    assert parsed["mode"] == "pre_call"
    assert isinstance(parsed["patterns"], list)
    assert len(parsed["patterns"]) == 1
    assert isinstance(parsed["patterns"][0], dict)
    assert parsed["patterns"][0]["pattern_type"] == "prebuilt"
    
    assert "ContentFilterPattern" not in result
    assert "pattern_type='prebuilt'" not in result
