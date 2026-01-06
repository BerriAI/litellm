"""
Tests for litellm.litellm_core_utils.dot_notation_indexing module.
"""

import pytest

from litellm.litellm_core_utils.dot_notation_indexing import (
    get_nested_value,
    delete_nested_value,
)


class TestGetNestedValue:
    """Tests for get_nested_value function."""

    def test_simple_key(self):
        """Test accessing a simple top-level key."""
        data = {"name": "test"}
        assert get_nested_value(data, "name") == "test"

    def test_nested_key(self):
        """Test accessing nested keys with dot notation."""
        data = {"a": {"b": {"c": "value"}}}
        assert get_nested_value(data, "a.b.c") == "value"

    def test_missing_key_returns_default(self):
        """Test that missing keys return the default value."""
        data = {"a": {"b": "value"}}
        assert get_nested_value(data, "a.b", "default") == "value"
        assert get_nested_value(data, "a.c", "default") == "default"
        assert get_nested_value(data, "x.y.z") is None

    def test_empty_key_path(self):
        """Test that empty key path returns default."""
        data = {"a": "value"}
        assert get_nested_value(data, "") is None
        assert get_nested_value(data, "", "default") == "default"

    def test_metadata_prefix_removal(self):
        """Test that metadata. prefix is properly removed."""
        data = {"user": {"email": "test@example.com"}}
        assert get_nested_value(data, "metadata.user.email") == "test@example.com"

    def test_escaped_dot_in_key(self):
        """Test accessing keys that contain dots using escape sequence."""
        data = {"kubernetes.io": {"namespace": "default"}}
        assert get_nested_value(data, "kubernetes\\.io.namespace") == "default"

    def test_escaped_dot_nested(self):
        """Test multiple levels with escaped dots."""
        data = {
            "kubernetes.io": {
                "pod.info": {
                    "name": "my-pod"
                }
            }
        }
        assert get_nested_value(data, "kubernetes\\.io.pod\\.info.name") == "my-pod"

    def test_kubernetes_jwt_example(self):
        """Test with a realistic Kubernetes JWT structure."""
        jwt_token = {
            "aud": ["https://kubernetes.default.svc"],
            "exp": "1234567890",
            "iat": "123456789",
            "iss": "https://oidc.eks.region.amazonaws.com/id/randomstring",
            "jti": "randomstring",
            "kubernetes.io": {
                "namespace": "namespace",
                "node": {
                    "name": "node-name",
                    "uid": "node-uid"
                },
                "pod": {
                    "name": "pod-name",
                    "uid": "pod-uid"
                },
                "serviceaccount": {
                    "name": "serviceaccount-name",
                    "uid": "serviceaccount-uid"
                },
                "warnafter": 1234567880
            },
            "nbf": 123456789,
            "sub": "system:serviceaccount:namespace:serviceaccount-name"
        }
        
        # Test accessing kubernetes.io.namespace
        assert get_nested_value(jwt_token, "kubernetes\\.io.namespace") == "namespace"
        
        # Test accessing nested values within kubernetes.io
        assert get_nested_value(jwt_token, "kubernetes\\.io.pod.name") == "pod-name"
        assert get_nested_value(jwt_token, "kubernetes\\.io.serviceaccount.name") == "serviceaccount-name"
        
        # Test accessing regular keys still works
        assert get_nested_value(jwt_token, "sub") == "system:serviceaccount:namespace:serviceaccount-name"

    def test_mixed_escaped_and_regular_dots(self):
        """Test path with both escaped dots (in keys) and regular dots (separators)."""
        data = {
            "config.v1": {
                "settings": {
                    "feature.enabled": True
                }
            }
        }
        assert get_nested_value(data, "config\\.v1.settings.feature\\.enabled") is True


class TestDeleteNestedValue:
    """Tests for delete_nested_value function."""

    def test_delete_simple_key(self):
        """Test deleting a simple top-level key."""
        data = {"a": 1, "b": 2}
        result = delete_nested_value(data, "a")
        assert result == {"b": 2}
        # Original should be unchanged
        assert data == {"a": 1, "b": 2}

    def test_delete_nested_key(self):
        """Test deleting a nested key."""
        data = {"a": {"b": {"c": 1, "d": 2}}}
        result = delete_nested_value(data, "a.b.c")
        assert result == {"a": {"b": {"d": 2}}}

    def test_delete_array_wildcard(self):
        """Test deleting a field from all array elements."""
        data = {"tools": [{"name": "t1", "secret": "s1"}, {"name": "t2", "secret": "s2"}]}
        result = delete_nested_value(data, "tools[*].secret")
        assert result == {"tools": [{"name": "t1"}, {"name": "t2"}]}

    def test_delete_array_index(self):
        """Test deleting a field from a specific array element."""
        data = {"items": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}
        result = delete_nested_value(data, "items[0].b")
        assert result == {"items": [{"a": 1}, {"a": 3, "b": 4}]}

