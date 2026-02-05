"""
Shared configuration for Bedrock anthropic-beta header handling.

This module provides centralized whitelist-based filtering for anthropic-beta
headers across all Bedrock APIs (Invoke Chat, Invoke Messages, Converse).

## Architecture

All three Bedrock APIs use BedrockBetaHeaderFilter to ensure consistent filtering:
- Invoke Chat API: BedrockAPI.INVOKE_CHAT
- Invoke Messages API: BedrockAPI.INVOKE_MESSAGES (with advanced-tool-use translation)
- Converse API: BedrockAPI.CONVERSE

## Future-Proof Design

The filter uses version-based model support instead of hardcoded model lists:
- New Claude models (e.g., Opus 5, Sonnet 5) require ZERO code changes
- Beta headers specify minimum version (e.g., "requires 4.5+")
- Family restrictions (opus/sonnet/haiku) when needed

## Adding New Beta Headers

When AWS Bedrock adds support for a new Anthropic beta header:

**Scenario 1: Works on all models**
```python
BEDROCK_CORE_SUPPORTED_BETAS.add("new-feature-2027-01-15")
# Done! Works on all models automatically.
```

**Scenario 2: Requires specific version**
```python
BEDROCK_CORE_SUPPORTED_BETAS.add("advanced-reasoning-2027-06-15")
BETA_HEADER_MINIMUM_VERSION["advanced-reasoning-2027-06-15"] = 5.0
# Done! Works on all Claude 5.0+ models (Opus, Sonnet, Haiku).
```

**Scenario 3: Version + family restriction**
```python
BEDROCK_CORE_SUPPORTED_BETAS.add("ultra-context-2027-12-15")
BETA_HEADER_MINIMUM_VERSION["ultra-context-2027-12-15"] = 5.5
BETA_HEADER_FAMILY_RESTRICTIONS["ultra-context-2027-12-15"] = ["opus"]
# Done! Works on Opus 5.5+ only.
```

**Always add tests** in `tests/test_litellm/llms/bedrock/test_beta_headers_config.py`

## Testing

Run the test suite to verify changes:
```bash
poetry run pytest tests/test_litellm/llms/bedrock/test_beta_headers_config.py -v
poetry run pytest tests/test_litellm/llms/bedrock/test_anthropic_beta_support.py -v
```

## Debug Logging

Enable debug logging to see filtering decisions:
```bash
LITELLM_LOG=DEBUG
```

Reference:
- AWS Bedrock Documentation: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Set

from litellm._logging import verbose_logger


class BedrockAPI(Enum):
    """Enum for different Bedrock API types."""

    INVOKE_CHAT = "invoke_chat"
    INVOKE_MESSAGES = "invoke_messages"
    CONVERSE = "converse"


# Core whitelist of beta headers supported by ALL Bedrock APIs
BEDROCK_CORE_SUPPORTED_BETAS: Set[str] = {
    "computer-use-2024-10-22",  # Legacy computer use
    "computer-use-2025-01-24",  # Current computer use (Claude 3.7 Sonnet)
    "computer-use-2025-11-24",  # Latest computer use (Claude Opus 4.5+)
    "token-efficient-tools-2025-02-19",  # Tool use (Claude 3.7+ and Claude 4+)
    "interleaved-thinking-2025-05-14",  # Interleaved thinking (Claude 4+)
    "output-128k-2025-02-19",  # 128K output tokens (Claude 3.7 Sonnet)
    "dev-full-thinking-2025-05-14",  # Developer mode for raw thinking (Claude 4+)
    "context-1m-2025-08-07",  # 1 million tokens (Claude Sonnet 4)
    "context-management-2025-06-27",  # Context management (Claude Sonnet/Haiku 4.5)
    "effort-2025-11-24",  # Effort parameter (Claude Opus 4.5)
    "tool-search-tool-2025-10-19",  # Tool search (Claude Opus 4.5)
    "tool-examples-2025-10-29",  # Tool use examples (Claude Opus 4.5)
}

# API-specific exclusions (headers NOT supported by specific APIs)
BEDROCK_API_EXCLUSIONS: Dict[BedrockAPI, Set[str]] = {
    BedrockAPI.CONVERSE: set(),  # No additional exclusions
    BedrockAPI.INVOKE_CHAT: set(),  # No additional exclusions
    BedrockAPI.INVOKE_MESSAGES: set(),  # No additional exclusions
}

# Model version extraction regex pattern
# Matches Bedrock model IDs in both formats:
#   New:    claude-{family}-{major}-{minor}-{date}  (e.g., claude-opus-4-5-20250514-v1:0)
#   Legacy: claude-{major}-{minor}-{family}-{date}  (e.g., claude-3-5-sonnet-20240620-v1:0)
# Minor version is a single digit followed by a hyphen (to avoid capturing the date).
MODEL_VERSION_PATTERN = r"claude-(?:(?:opus|sonnet|haiku)-)?(\d+)(?:-(\d)-)?"

# Minimum model version required for each beta header (major.minor format)
# Default behavior: If a beta header is NOT in this dict, it's supported by ALL Anthropic models
# This approach is future-proof - new models automatically support all headers unless excluded
BETA_HEADER_MINIMUM_VERSION: Dict[str, float] = {
    # Extended thinking features require Claude 4.0+
    "interleaved-thinking-2025-05-14": 4.0,
    "dev-full-thinking-2025-05-14": 4.0,
    # 1M context requires Claude 4.0+
    "context-1m-2025-08-07": 4.0,
    # Context management requires Claude 4.5+
    "context-management-2025-06-27": 4.5,
    # Effort parameter requires Claude 4.5+ (but only Opus 4.5, see family restrictions)
    "effort-2025-11-24": 4.5,
    # Tool search requires Claude 4.5+
    "tool-search-tool-2025-10-19": 4.5,
    "tool-examples-2025-10-29": 4.5,
}

# Model family restrictions for specific beta headers
# Only enforced if the version requirement is met
# Example: "effort-2025-11-24" requires Claude 4.5+ AND Opus family
BETA_HEADER_FAMILY_RESTRICTIONS: Dict[str, List[str]] = {
    "effort-2025-11-24": ["opus"],  # Only Opus 4.5+ supports effort
    # Tool search works on Opus 4.5+ and Sonnet 4.5+, but not Haiku
    "tool-search-tool-2025-10-19": ["opus", "sonnet"],
    "tool-examples-2025-10-29": ["opus", "sonnet"],
}

# Beta headers that should be translated for backward compatibility
# Maps input header pattern to output headers
# Uses version-based approach for future-proofing
BETA_HEADER_TRANSLATIONS: Dict[str, Dict] = {
    "advanced-tool-use": {
        "target_headers": ["tool-search-tool-2025-10-19", "tool-examples-2025-10-29"],
        "minimum_version": 4.5,  # Requires Claude 4.5+
        "allowed_families": ["opus", "sonnet"],  # Not available on Haiku
    },
}


class BedrockBetaHeaderFilter:
    """
    Centralized filter for anthropic-beta headers across all Bedrock APIs.

    Uses a whitelist-based approach to ensure only supported headers are sent to AWS.
    """

    def __init__(self, api_type: BedrockAPI):
        """
        Initialize the filter for a specific Bedrock API.

        Args:
            api_type: The Bedrock API type (Invoke Chat, Invoke Messages, or Converse)
        """
        self.api_type = api_type
        self.supported_betas = self._get_supported_betas()

    def _get_supported_betas(self) -> Set[str]:
        """Get the set of supported beta headers for this API type."""
        # Start with core supported headers
        supported = BEDROCK_CORE_SUPPORTED_BETAS.copy()

        # Remove API-specific exclusions
        exclusions = BEDROCK_API_EXCLUSIONS.get(self.api_type, set())
        supported -= exclusions

        return supported

    def _extract_model_version(self, model: str) -> Optional[float]:
        """
        Extract Claude model version from Bedrock model ID.

        Args:
            model: Bedrock model ID (e.g., "anthropic.claude-opus-4-5-20250514-v1:0")

        Returns:
            Version as float (e.g., 4.5), or None if unable to parse

        Examples:
            "anthropic.claude-opus-4-5-20250514-v1:0" -> 4.5
            "anthropic.claude-sonnet-4-20250514-v1:0" -> 4.0
            "anthropic.claude-3-5-sonnet-20240620-v1:0" -> 3.5
            "anthropic.claude-3-sonnet-20240229-v1:0" -> 3.0
        """
        match = re.search(MODEL_VERSION_PATTERN, model)
        if not match:
            return None

        major = int(match.group(1))
        minor = int(match.group(2)) if match.group(2) else 0

        return float(f"{major}.{minor}")

    def _extract_model_family(self, model: str) -> Optional[str]:
        """
        Extract Claude model family (opus, sonnet, haiku) from Bedrock model ID.

        Args:
            model: Bedrock model ID

        Returns:
            Family name (opus/sonnet/haiku) or None if unable to parse

        Examples:
            "anthropic.claude-opus-4-5-20250514-v1:0" -> "opus"
            "anthropic.claude-3-5-sonnet-20240620-v1:0" -> "sonnet"
        """
        model_lower = model.lower()
        if "opus" in model_lower:
            return "opus"
        elif "sonnet" in model_lower:
            return "sonnet"
        elif "haiku" in model_lower:
            return "haiku"
        return None

    def _model_supports_beta(self, model: str, beta: str) -> bool:
        """
        Check if a model supports a specific beta header.

        Uses a future-proof approach:
        1. If beta has no version requirement -> ALLOW (supports all models)
        2. If beta has version requirement -> Extract model version and compare
        3. If beta has family restriction -> Check model family

        This means NEW models automatically support all beta headers unless explicitly
        restricted by version/family requirements.

        Args:
            model: The Bedrock model ID (e.g., "anthropic.claude-sonnet-4-20250514-v1:0")
            beta: The beta header to check

        Returns:
            True if the model supports the beta header, False otherwise
        """
        # Default: If no version requirement specified, ALL Anthropic models support it
        # This makes the system future-proof for new models
        if beta not in BETA_HEADER_MINIMUM_VERSION:
            return True

        # Extract model version
        model_version = self._extract_model_version(model)
        if model_version is None:
            # If we can't parse version, be conservative and reject
            # (This should rarely happen with well-formed Bedrock model IDs)
            return False

        # Check minimum version requirement
        required_version = BETA_HEADER_MINIMUM_VERSION[beta]
        if model_version < required_version:
            return False  # Model version too old

        # Check family restrictions (if any)
        if beta in BETA_HEADER_FAMILY_RESTRICTIONS:
            model_family = self._extract_model_family(model)
            if model_family is None:
                # Can't determine family, be conservative
                return False

            allowed_families = BETA_HEADER_FAMILY_RESTRICTIONS[beta]
            if model_family not in allowed_families:
                return False  # Wrong family

        # All checks passed
        return True

    def _translate_beta_headers(self, beta_headers: Set[str], model: str) -> Set[str]:
        """
        Translate beta headers for backward compatibility.

        Uses version-based checks to determine if model supports translation.
        Future-proof: new models at the required version automatically support translations.

        Args:
            beta_headers: Set of beta headers to translate
            model: The Bedrock model ID

        Returns:
            Set of translated beta headers
        """
        translated = beta_headers.copy()

        for input_pattern, translation_info in BETA_HEADER_TRANSLATIONS.items():
            # Check if any beta header matches the input pattern
            matching_headers = [h for h in beta_headers if input_pattern in h.lower()]

            if matching_headers:
                # Check if model supports the translation using version-based logic
                model_version = self._extract_model_version(model)
                if model_version is None:
                    continue  # Can't determine version, skip translation

                # Check minimum version
                required_version = translation_info.get("minimum_version")
                if required_version and model_version < required_version:
                    continue  # Model too old for this translation

                # Check family restrictions (if any)
                allowed_families = translation_info.get("allowed_families")
                if allowed_families:
                    model_family = self._extract_model_family(model)
                    if model_family not in allowed_families:
                        continue  # Wrong family

                # Model supports translation - apply it
                for header in matching_headers:
                    translated.discard(header)
                    verbose_logger.debug(
                        f"Bedrock {self.api_type.value}: Translating beta header '{header}' for model {model}"
                    )

                for target_header in translation_info["target_headers"]:
                    translated.add(target_header)
                    verbose_logger.debug(
                        f"Bedrock {self.api_type.value}: Added translated header '{target_header}'"
                    )

        return translated

    def filter_beta_headers(
        self, beta_headers: List[str], model: str, translate: bool = True
    ) -> List[str]:
        """
        Filter and translate beta headers for Bedrock.

        This is the main entry point for filtering beta headers.

        Args:
            beta_headers: List of beta headers from user request
            model: The Bedrock model ID
            translate: Whether to apply header translations (default: True)

        Returns:
            Filtered and translated list of beta headers
        """
        if not beta_headers:
            return []

        # Convert to set for efficient operations
        beta_set = set(beta_headers)

        # Apply translations if enabled
        if translate:
            beta_set = self._translate_beta_headers(beta_set, model)

        # Filter: Keep only whitelisted headers
        filtered = set()
        for beta in beta_set:
            # Check if header is in whitelist
            if beta not in self.supported_betas:
                verbose_logger.debug(
                    f"Bedrock {self.api_type.value}: Filtered out unsupported beta header: {beta}"
                )
                continue

            # Check if model supports this header
            if not self._model_supports_beta(model, beta):
                verbose_logger.debug(
                    f"Bedrock {self.api_type.value}: Filtered out beta header '{beta}' (not supported on model {model})"
                )
                continue

            filtered.add(beta)

        verbose_logger.debug(
            f"Bedrock {self.api_type.value}: Final beta headers for {model}: {sorted(filtered)}"
        )
        return sorted(list(filtered))  # Sort for deterministic output


def get_bedrock_beta_filter(api_type: BedrockAPI) -> BedrockBetaHeaderFilter:
    """
    Factory function to get a beta header filter for a specific API.

    Args:
        api_type: The Bedrock API type

    Returns:
        BedrockBetaHeaderFilter instance
    """
    return BedrockBetaHeaderFilter(api_type)
