from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Set

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH_SENSITIVE_DATA_MASKER


class SensitiveDataMasker:
    def __init__(
        self,
        sensitive_patterns: Optional[Set[str]] = None,
        non_sensitive_overrides: Optional[Set[str]] = None,
        visible_prefix: int = 4,
        visible_suffix: int = 4,
        mask_char: str = "*",
        mask_short_values: bool = True,
    ):
        self.sensitive_patterns = sensitive_patterns or {
            "password",
            "secret",
            "key",
            "token",
            "auth",
            "authorization",
            "credential",
            # Plural form: Vertex uses ``vertex_credentials``; segment-exact
            # matching otherwise misses it because "credential" != "credentials".
            "credentials",
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
        self.mask_short_values = mask_short_values

    def _mask_value(self, value: str) -> str:
        value_str = str(value)
        if not value_str:
            return value
        if len(value_str) <= (self.visible_prefix + self.visible_suffix):
            return self.mask_char * len(value_str) if self.mask_short_values else value_str

        masked_length = len(value_str) - (self.visible_prefix + self.visible_suffix)

        # Handle the case where visible_suffix is 0 to avoid showing the entire string
        if self.visible_suffix == 0:
            return f"{value_str[: self.visible_prefix]}{self.mask_char * masked_length}"
        else:
            return (
                f"{value_str[: self.visible_prefix]}{self.mask_char * masked_length}{value_str[-self.visible_suffix :]}"
            )

    def is_sensitive_key(self, key: str, excluded_keys: Optional[Set[str]] = None) -> bool:
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
        masked_items: List[Any] = []
        if depth >= max_depth:
            return values

        for item in values:
            if isinstance(item, Mapping):
                masked_items.append(self.mask_dict(dict(item), depth + 1, max_depth, excluded_keys))
            elif isinstance(item, list):
                masked_items.append(self._mask_sequence(item, depth + 1, max_depth, excluded_keys, key_is_sensitive))
            elif key_is_sensitive and isinstance(item, str):
                masked_items.append(self._mask_value(item))
            else:
                masked_items.append(item if isinstance(item, (int, float, bool, str, list)) else str(item))
        return masked_items

    def mask_dict(
        self,
        data: Dict[str, Any],
        depth: int = 0,
        max_depth: int = DEFAULT_MAX_RECURSE_DEPTH_SENSITIVE_DATA_MASKER,
        excluded_keys: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        if depth >= max_depth:
            return data

        masked_data: Dict[str, Any] = {}
        for k, v in data.items():
            try:
                key_is_sensitive = self.is_sensitive_key(k, excluded_keys)
                if isinstance(v, Mapping):
                    masked_data[k] = self.mask_dict(dict(v), depth + 1, max_depth, excluded_keys)
                elif isinstance(v, list):
                    masked_data[k] = self._mask_sequence(v, depth + 1, max_depth, excluded_keys, key_is_sensitive)
                elif hasattr(v, "__dict__") and not isinstance(v, type):
                    masked_data[k] = self.mask_dict(vars(v), depth + 1, max_depth, excluded_keys)
                elif key_is_sensitive:
                    str_value = str(v) if v is not None else ""
                    masked_data[k] = self._mask_value(str_value)
                else:
                    masked_data[k] = v if isinstance(v, (int, float, bool, str, list)) else str(v)
            except Exception:
                masked_data[k] = "<unable to serialize>"

        return masked_data

    def mask(self, data: object) -> object:
        if isinstance(data, Mapping):
            return self.mask_dict(dict(data))
        if isinstance(data, list):
            return self._mask_sequence(
                data,
                0,
                DEFAULT_MAX_RECURSE_DEPTH_SENSITIVE_DATA_MASKER,
                None,
                False,
            )
        return data


_default_masker = SensitiveDataMasker()
_error_masker = SensitiveDataMasker(visible_prefix=4, visible_suffix=0)


def mask_sensitive_structure(data: object) -> object:
    return _error_masker.mask(data)


def mask_sensitive_keys(data: Dict[str, Any], sensitive_fields: Set[str]) -> Dict[str, Any]:
    """Return a new dict with values masked for keys listed in ``sensitive_fields``.

    Unlike :meth:`SensitiveDataMasker.mask_dict`, this does exact key-name
    matching (not segment matching), so callers explicitly enumerate which
    fields to mask. Non-string and None values are passed through unchanged.

    Values shorter than ``visible_prefix + visible_suffix`` (8 by default)
    fall outside :meth:`SensitiveDataMasker._mask_value`'s partial-reveal
    range and are replaced with a fixed-length all-mask string, so a short
    credential is never returned verbatim.
    """
    masked: Dict[str, Any] = {}
    mask_char = _default_masker.mask_char
    min_visible = _default_masker.visible_prefix + _default_masker.visible_suffix
    for key, value in data.items():
        if value is not None and key in sensitive_fields and isinstance(value, str):
            if len(value) < min_visible:
                masked[key] = mask_char * len(value) if value else value
            else:
                masked[key] = _default_masker._mask_value(value)
        else:
            masked[key] = value
    return masked


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
