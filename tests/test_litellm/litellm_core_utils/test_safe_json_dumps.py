import json
import os
import sys
import threading

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


def test_no_runtime_error_on_concurrent_dict_mutation():
    """safe_dumps must not raise RuntimeError when another thread mutates the dict."""
    errors: list = []

    for _ in range(50):
        data = {f"key_{i}": f"value_{i}" for i in range(200)}
        barrier = threading.Barrier(2, timeout=5)
        results: list = []

        def mutator():
            barrier.wait()
            for i in range(200, 400):
                data[f"key_{i}"] = f"value_{i}"

        def serializer():
            barrier.wait()
            try:
                result = safe_dumps(data)
                results.append(result)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=mutator)
        t2 = threading.Thread(target=serializer)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        if results:
            assert isinstance(results[0], str)
            assert len(results[0]) > 0

    assert not errors, f"safe_dumps raised {errors[0]!r} on concurrent mutation"


def test_no_runtime_error_on_concurrent_list_mutation():
    """safe_dumps must not raise RuntimeError when another thread mutates a nested list."""
    errors: list = []

    for _ in range(50):
        data = {"messages": [f"msg_{i}" for i in range(100)]}
        barrier = threading.Barrier(2, timeout=5)
        results: list = []

        def mutator():
            barrier.wait()
            for i in range(100, 200):
                data["messages"].append(f"msg_{i}")

        def serializer():
            barrier.wait()
            try:
                result = safe_dumps(data)
                results.append(result)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=mutator)
        t2 = threading.Thread(target=serializer)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        if results:
            assert isinstance(results[0], str)
            assert '"messages"' in results[0]

    assert not errors, f"safe_dumps raised {errors[0]!r} on concurrent list mutation"
