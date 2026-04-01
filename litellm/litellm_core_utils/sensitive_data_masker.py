"""Utilities for masking sensitive data (API keys, tokens, passwords) in dicts."""

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Set

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH_SENSITIVE_DATA_MASKER


class SensitiveDataMasker:
    """Masks sensitive values (API keys, tokens, passwords) in dictionaries.

    Recursively traverses nested dicts/lists and replaces string values whose
    keys match known sensitive patterns with partially-redacted versions, e.g.
    ``"sk-1234567890abcdef"`` becomes ``"sk-1****cdef"``.

    Args:
        sensitive_patterns: Set of key-name substrings that indicate a sensitive
            field (matched against underscore-split segments of each key).
            Defaults to common secret-related terms like ``"key"``, ``"token"``,
            ``"password"``, etc.
        non_sensitive_overrides: Set of key-name substrings that, when present,
            override a sensitive match.  For example ``"cost"`` prevents
            ``"input_cost_per_token"`` from being treated as sensitive.
        visible_prefix: Number of leading characters to leave unmasked.
        visible_suffix: Number of trailing characters to leave unmasked.
        mask_char: Character used for the masked portion of the value.
    """

    def __init__(
        self,
        sensitive_patterns: Optional[Set[str]] = None,
        non_sensitive_overrides: Optional[Set[str]] = None,
        visible_prefix: int = 4,
        visible_suffix: int = 4,
        mask_char: str = "*",
    ):
        self.sensitive_patterns = sensitive_patterns or {
            "password",
            "secret",
            "key",
            "token",
            "auth",
            "authorization",
            "credential",
            "access",
            "private",
            "certificate",
            "fingerprint",
            "tenancy",
        }
        # If any key segment matches one of these, the key is not considered sensitive
        # even if it also matches a sensitive pattern. For example, "input_cost_per_token"
        # contains "token" but "cost" overrides that — it's a pricing field, not a secret.
        self.non_sensitive_overrides = non_sensitive_overrides or {"cost"}

        self.visible_prefix = visible_prefix
        self.visible_suffix = visible_suffix
        self.mask_char = mask_char

    def _mask_value(self, value: str) -> str:
        """Return *value* with its middle characters replaced by ``mask_char``.

        Characters at the start (``visible_prefix``) and end (``visible_suffix``)
        are preserved; the rest is replaced.  Values shorter than
        ``visible_prefix + visible_suffix`` are returned unchanged.
        """
        if not value or len(str(value)) < (self.visible_prefix + self.visible_suffix):
            return value

        value_str = str(value)
        masked_length = len(value_str) - (self.visible_prefix + self.visible_suffix)

        # Handle the case where visible_suffix is 0 to avoid showing the entire string
        if self.visible_suffix == 0:
            return f"{value_str[:self.visible_prefix]}{self.mask_char * masked_length}"
        else:
            return f"{value_str[:self.visible_prefix]}{self.mask_char * masked_length}{value_str[-self.visible_suffix:]}"

    def is_sensitive_key(
        self, key: str, excluded_keys: Optional[Set[str]] = None
    ) -> bool:
        """Determine whether *key* refers to a sensitive field.

        The key is split on underscores/hyphens into segments and each segment
        is checked against ``sensitive_patterns``.  If any segment also appears
        in ``non_sensitive_overrides``, the key is treated as non-sensitive.

        Args:
            key: The dictionary key to evaluate.
            excluded_keys: Optional set of keys to unconditionally treat as
                non-sensitive (exact match).

        Returns:
            ``True`` if the key is considered sensitive, ``False`` otherwise.
        """
        # Check if key is in excluded_keys first (exact match)
        if excluded_keys and key in excluded_keys:
            return False

        key_lower = str(key).lower()
        # Split on underscores/hyphens and check if any segment matches the pattern
        # This avoids false positives like "max_tokens" matching "token"
        # but still catches "api_key", "access_token", etc.
        key_segments = key_lower.replace("-", "_").split("_")

        # If any segment matches a non-sensitive override, the key is not sensitive.
        # For example, "input_cost_per_token" contains "token" but also "cost",
        # so it should not be masked — it's a pricing field, not a secret.
        if any(override in key_segments for override in self.non_sensitive_overrides):
            return False

        result = any(pattern in key_segments for pattern in self.sensitive_patterns)
        return result

    def _mask_sequence(
        self,
        values: List[Any],
        depth: int,
        max_depth: int,
        excluded_keys: Optional[Set[str]],
        key_is_sensitive: bool,
    ) -> List[Any]:
        """Recursively mask sensitive string items within a list.

        Args:
            values: The list to process.
            depth: Current recursion depth.
            max_depth: Maximum recursion depth to prevent infinite loops.
            excluded_keys: Keys to skip when evaluating nested dicts.
            key_is_sensitive: Whether the parent key was classified as sensitive.

        Returns:
            A new list with sensitive string items masked.
        """
        masked_items: List[Any] = []
        if depth >= max_depth:
            return values

        for item in values:
            if isinstance(item, Mapping):
                masked_items.append(
                    self.mask_dict(dict(item), depth + 1, max_depth, excluded_keys)
                )
            elif isinstance(item, list):
                masked_items.append(
                    self._mask_sequence(
                        item, depth + 1, max_depth, excluded_keys, key_is_sensitive
                    )
                )
            elif key_is_sensitive and isinstance(item, str):
                masked_items.append(self._mask_value(item))
            else:
                masked_items.append(
                    item
                    if isinstance(item, (int, float, bool, str, list))
                    else str(item)
                )
        return masked_items

    def mask_dict(
        self,
        data: Dict[str, Any],
        depth: int = 0,
        max_depth: int = DEFAULT_MAX_RECURSE_DEPTH_SENSITIVE_DATA_MASKER,
        excluded_keys: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Return a copy of *data* with sensitive values masked.

        Recursively walks nested dicts, lists, and objects with ``__dict__``.
        String values under keys that match ``sensitive_patterns`` are redacted
        via ``_mask_value``.  Non-serialisable values are converted to strings.

        Args:
            data: The dictionary to mask.
            depth: Current recursion depth (used internally).
            max_depth: Maximum recursion depth to prevent infinite loops.
            excluded_keys: Optional set of key names to skip when checking
                sensitivity (passed through to ``is_sensitive_key``).

        Returns:
            A new dictionary with sensitive values masked.
        """
        if depth >= max_depth:
            return data

        masked_data: Dict[str, Any] = {}
        for k, v in data.items():
            try:
                key_is_sensitive = self.is_sensitive_key(k, excluded_keys)
                if isinstance(v, Mapping):
                    masked_data[k] = self.mask_dict(
                        dict(v), depth + 1, max_depth, excluded_keys
                    )
                elif isinstance(v, list):
                    masked_data[k] = self._mask_sequence(
                        v, depth + 1, max_depth, excluded_keys, key_is_sensitive
                    )
                elif hasattr(v, "__dict__") and not isinstance(v, type):
                    masked_data[k] = self.mask_dict(
                        vars(v), depth + 1, max_depth, excluded_keys
                    )
                elif key_is_sensitive:
                    str_value = str(v) if v is not None else ""
                    masked_data[k] = self._mask_value(str_value)
                else:
                    masked_data[k] = (
                        v if isinstance(v, (int, float, bool, str, list)) else str(v)
                    )
            except Exception:
                masked_data[k] = "<unable to serialize>"

        return masked_data


# Usage example:
"""
masker = SensitiveDataMasker()
data = {
    "api_key": "sk-1234567890abcdef",
    "redis_password": "very_secret_pass",
    "port": 6379,
    "tags": ["East US 2", "production", "test"]
}
masked = masker.mask_dict(data)
# Result: {
#    "api_key": "sk-1****cdef",
#    "redis_password": "very****pass",
#    "port": 6379,
#    "tags": ["East US 2", "production", "test"]
# }
"""
