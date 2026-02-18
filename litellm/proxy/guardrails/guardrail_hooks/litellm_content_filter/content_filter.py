"""
Content Filter Guardrail for LiteLLM.

This guardrail provides regex pattern matching and keyword filtering
to detect and block/mask sensitive content.
"""

import asyncio
import json
import os
import re
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Literal,
    Optional,
    Pattern,
    Tuple,
    Union,
    cast,
)

import yaml
from fastapi import HTTPException

from litellm import Router
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import GuardrailTracingDetail, ModelResponseStream

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import GenericGuardrailAPIInputs, GuardrailStatus

from litellm.types.guardrails import (
    BlockedWord,
    ContentFilterAction,
    ContentFilterPattern,
    GuardrailEventHooks,
    Mode,
)
from litellm.types.proxy.guardrails.guardrail_hooks.litellm_content_filter import (
    BlockedWordDetection,
    CategoryKeywordDetection,
    ContentFilterCategoryConfig,
    ContentFilterDetection,
    PatternDetection,
)

from .patterns import PATTERN_EXTRA_CONFIG, get_compiled_pattern

MAX_KEYWORD_VALUE_GAP_WORDS = 1
GAP_WORD_TOKENIZER = re.compile(r"\b\w+\b")


WORD_NUMBER_MAP = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

WORD_NUMBER_TOKEN_REGEX = "|".join(WORD_NUMBER_MAP.keys())
WORD_NUMBER_SEQUENCE_PATTERN = re.compile(
    rf"(?<![A-Za-z])(?:{WORD_NUMBER_TOKEN_REGEX})(?:[\s\-]+(?:{WORD_NUMBER_TOKEN_REGEX}))+(?![A-Za-z])",
    re.IGNORECASE,
)
WORD_NUMBER_TOKEN_FINDER = re.compile(rf"(?:{WORD_NUMBER_TOKEN_REGEX})", re.IGNORECASE)


# Helper data structure for category-based detection
class CategoryConfig:
    """Configuration for a content category."""

    def __init__(
        self,
        category_name: str,
        description: str,
        default_action: ContentFilterAction,
        keywords: List[Dict[str, str]],
        exceptions: List[str],
        identifier_words: Optional[List[str]] = None,
        always_block_keywords: Optional[List[Dict[str, str]]] = None,
        inherit_from: Optional[str] = None,
        additional_block_words: Optional[List[str]] = None,
    ):
        self.category_name = category_name
        self.description = description
        self.default_action = default_action
        self.keywords = keywords
        self.exceptions = [e.lower() for e in exceptions]
        # New fields for conditional child safety logic
        self.identifier_words = (
            [w.lower() for w in identifier_words] if identifier_words else []
        )
        self.always_block_keywords = always_block_keywords or []
        self.inherit_from = inherit_from
        self.additional_block_words = (
            [w.lower() for w in additional_block_words]
            if additional_block_words
            else []
        )


class ContentFilterGuardrail(CustomGuardrail):
    """
    Content filter guardrail that detects sensitive information using:
    - Prebuilt regex patterns (SSN, credit cards, API keys, etc.)
    - Custom user-defined regex patterns
    - Dictionary-based keyword matching

    Actions:
    - BLOCK: Reject the request with an error
    - MASK: Replace the sensitive content with a redacted placeholder
    """

    # Redaction format constants
    PATTERN_REDACTION_FORMAT = "[{pattern_name}_REDACTED]"
    KEYWORD_REDACTION_STR = "[KEYWORD_REDACTED]"

    def __init__(
        self,
        guardrail_name: Optional[str] = None,
        guardrail_id: Optional[str] = None,
        policy_template: Optional[str] = None,
        patterns: Optional[List[ContentFilterPattern]] = None,
        blocked_words: Optional[List[BlockedWord]] = None,
        blocked_words_file: Optional[str] = None,
        event_hook: Optional[
            Union[GuardrailEventHooks, List[GuardrailEventHooks], Mode]
        ] = None,
        default_on: bool = False,
        pattern_redaction_format: Optional[str] = None,
        keyword_redaction_tag: Optional[str] = None,
        categories: Optional[List[ContentFilterCategoryConfig]] = None,
        severity_threshold: str = "medium",
        llm_router: Optional[Router] = None,
        image_model: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Content Filter Guardrail.

        Args:
            guardrail_name: Name of this guardrail instance
            patterns: List of ContentFilterPattern objects to detect
            blocked_words: List of BlockedWord objects with keywords and actions
            blocked_words_file: Path to YAML file containing blocked_words list
            event_hook: When to run this guardrail (pre_call, post_call, etc.)
            default_on: If True, runs on all requests by default
            pattern_redaction_format: Format string for pattern redaction (use {pattern_name} placeholder)
            keyword_redaction_tag: Tag to use for keyword redaction
            categories: List of category configurations with enabled/action/severity settings
            severity_threshold: Minimum severity to block ("high", "medium", "low")
        """

        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=[
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.during_call,
            ],
            event_hook=event_hook or GuardrailEventHooks.pre_call,
            default_on=default_on,
            **kwargs,
        )

        self.guardrail_provider = "litellm_content_filter"
        self.config_guardrail_id = guardrail_id
        self.config_policy_template = policy_template
        self.pattern_redaction_format = (
            pattern_redaction_format or self.PATTERN_REDACTION_FORMAT
        )
        self.keyword_redaction_tag = keyword_redaction_tag or self.KEYWORD_REDACTION_STR
        self.severity_threshold = severity_threshold
        self.llm_router = llm_router
        self.image_model = image_model
        # Store loaded categories
        self.loaded_categories: Dict[str, CategoryConfig] = {}
        self.category_keywords: Dict[str, Tuple[str, str, ContentFilterAction]] = (
            {}
        )  # keyword -> (category, severity, action)
        # Store conditional categories (identifier_words + block_words)
        self.conditional_categories: Dict[str, Dict[str, Any]] = (
            {}
        )  # category_name -> {identifier_words, block_words, action, severity}

        # Load categories if provided
        if categories:
            self._load_categories(categories)
        else:
            verbose_proxy_logger.warning(
                "ContentFilterGuardrail has no content categories configured. "
                "Toxic/abuse and other category-based keyword filtering will not run. "
                "Add categories (e.g. harm_toxic_abuse) in the guardrail config to enable them."
            )

        # Normalize inputs: convert dicts to Pydantic models for consistent handling
        normalized_patterns: List[ContentFilterPattern] = []
        if patterns:
            for pattern_config in patterns:
                if isinstance(pattern_config, dict):
                    normalized_patterns.append(ContentFilterPattern(**pattern_config))
                else:
                    normalized_patterns.append(pattern_config)

        normalized_blocked_words: List[BlockedWord] = []
        if blocked_words:
            for word in blocked_words:
                if isinstance(word, dict):
                    normalized_blocked_words.append(BlockedWord(**word))
                else:
                    normalized_blocked_words.append(word)

        # Compile regex patterns
        self.compiled_patterns: List[Dict[str, Any]] = []
        for pattern_config in normalized_patterns:
            self._add_pattern(pattern_config)

        # Warn if using during_call with MASK action (unstable)
        if self.event_hook == GuardrailEventHooks.during_call and any(
            p["action"] == ContentFilterAction.MASK for p in self.compiled_patterns
        ):
            verbose_proxy_logger.warning(
                f"ContentFilterGuardrail '{self.guardrail_name}': 'during_call' mode with 'MASK' action is unstable due to race conditions. "
                "Use 'pre_call' mode for reliable request masking."
            )

        # Load blocked words - always initialize as dict
        self.blocked_words: Dict[str, Tuple[ContentFilterAction, Optional[str]]] = {}
        for word in normalized_blocked_words:
            self.blocked_words[word.keyword.lower()] = (word.action, word.description)

        # Defensive check: ensure blocked_words is a dict (not a list)
        if not isinstance(self.blocked_words, dict):
            verbose_proxy_logger.error(
                f"blocked_words is not a dict, got {type(self.blocked_words)}. Resetting to empty dict."
            )
            self.blocked_words = {}

        # Load blocked words from file if provided
        if blocked_words_file:
            self._load_blocked_words_file(blocked_words_file)

        verbose_proxy_logger.debug(
            f"ContentFilterGuardrail initialized with {len(self.compiled_patterns)} patterns "
            f"and {len(self.blocked_words)} blocked words"
        )
        verbose_proxy_logger.debug(
            f"Loaded {len(self.loaded_categories)} categories with "
            f"{len(self.category_keywords)} keywords"
        )

    @staticmethod
    def _resolve_category_file_path(file_path: str) -> str:
        """
        Resolve a category file path that may be relative.

        Paths in policy templates (e.g. category_file) are often stored as
        relative paths like "litellm/proxy/.../policy_templates/file.yaml".
        These only work when the CWD is the project root. In production
        (Docker, installed packages, etc.) the CWD is different, so the
        file isn't found.

        Resolution order:
        1. Return as-is if absolute or already exists.
        2. Try joining the full path relative to this module's directory.
        3. Progressively strip leading path components and try each suffix
           relative to this module's directory (handles paths like
           "litellm/proxy/.../policy_templates/file.yaml" by finding the
           "policy_templates/file.yaml" suffix that exists).

        Args:
            file_path: The file path to resolve (absolute or relative).

        Returns:
            The resolved absolute-ish path, or the original path if
            resolution fails (caller should check existence).
        """
        if os.path.isabs(file_path) or os.path.exists(file_path):
            return file_path

        module_dir = os.path.dirname(__file__)

        # Try the full relative path joined to the module directory
        candidate = os.path.join(module_dir, file_path)
        if os.path.exists(candidate):
            return candidate

        # Progressively strip leading components to find a matching suffix
        parts = file_path.split("/")
        for i in range(1, len(parts)):
            suffix = os.path.join(*parts[i:])
            candidate = os.path.join(module_dir, suffix)
            if os.path.exists(candidate):
                return candidate

        return file_path

    def _load_categories(self, categories: List[ContentFilterCategoryConfig]) -> None:
        """
        Load content categories from configuration.

        Args:
            categories: List of category configurations with format:
                - category: "harmful_self_harm"
                  enabled: true
                  action: "BLOCK"
                  severity_threshold: "medium"
                  category_file: "/path/to/custom_file.yaml"  # optional override
        """
        categories_dir = os.path.join(os.path.dirname(__file__), "categories")

        for cat_config in categories:
            category_name = cat_config.get("category")
            if not category_name or not isinstance(category_name, str):
                verbose_proxy_logger.warning(
                    "Category name missing or invalid in config, skipping"
                )
                continue

            enabled = cat_config.get("enabled", True)
            action = cat_config.get("action")
            severity_threshold = (
                cat_config.get("severity_threshold", self.severity_threshold)
                or self.severity_threshold
            )
            custom_file = cat_config.get("category_file")

            if not enabled:
                verbose_proxy_logger.debug(
                    f"Category {category_name} is disabled, skipping"
                )
                continue

            # Load category file (custom or default)
            if custom_file:
                category_file_path = self._resolve_category_file_path(custom_file)
            else:
                # Try .yaml first, then .json (e.g. harm_toxic_abuse.json)
                yaml_path = os.path.join(categories_dir, f"{category_name}.yaml")
                json_path = os.path.join(categories_dir, f"{category_name}.json")
                if os.path.exists(yaml_path):
                    category_file_path = yaml_path
                elif os.path.exists(json_path):
                    category_file_path = json_path
                else:
                    category_file_path = yaml_path  # will trigger "not found" below

            if not os.path.exists(category_file_path):
                verbose_proxy_logger.warning(
                    f"Category file not found: {category_file_path}, skipping"
                )
                continue

            try:
                category_config_obj = self._load_category_file(category_file_path)
                self.loaded_categories[category_name] = category_config_obj

                # Use action from config, or default from category file
                category_action = ContentFilterAction(
                    action if action else category_config_obj.default_action
                )

                # Handle conditional categories (with identifier_words + block words)
                if category_config_obj.identifier_words and (
                    category_config_obj.inherit_from
                    or category_config_obj.additional_block_words
                ):
                    self._load_conditional_category(
                        category_name,
                        category_config_obj,
                        category_action,
                        severity_threshold,
                        categories_dir,
                    )

                # Add always_block_keywords if present
                if category_config_obj.always_block_keywords:
                    for keyword_data in category_config_obj.always_block_keywords:
                        keyword = keyword_data["keyword"].lower()
                        severity = keyword_data.get("severity", "high")
                        if self._should_apply_severity(severity, severity_threshold):
                            self.category_keywords[keyword] = (
                                category_name,
                                severity,
                                category_action,
                            )

                # Add regular keywords from this category
                for keyword_data in category_config_obj.keywords:
                    keyword = keyword_data["keyword"].lower()
                    severity = keyword_data["severity"]

                    # Check if keyword meets severity threshold
                    if self._should_apply_severity(severity, severity_threshold):
                        self.category_keywords[keyword] = (
                            category_name,
                            severity,
                            category_action,
                        )

                verbose_proxy_logger.info(
                    f"Loaded category {category_name}: "
                    f"{len(category_config_obj.keywords)} keywords, "
                    f"{len(category_config_obj.always_block_keywords)} always-block keywords, "
                    f"conditional: {bool(category_config_obj.identifier_words)}"
                )
            except Exception as e:
                verbose_proxy_logger.error(
                    f"Error loading category {category_name}: {e}"
                )

    def _load_conditional_category(
        self,
        category_name: str,
        category_config_obj: CategoryConfig,
        category_action: ContentFilterAction,
        severity_threshold: str,
        categories_dir: str,
    ) -> None:
        """
        Load a conditional category that uses identifier_words + block_words.
        Block words can come from inherited category or additional_block_words.

        Args:
            category_name: Name of the category
            category_config_obj: CategoryConfig object with identifier_words
            category_action: Action to take when match is found
            severity_threshold: Minimum severity threshold
            categories_dir: Directory containing category files
        """
        try:
            block_words = []
            inherit_from = category_config_obj.inherit_from

            # Load inherited block words if specified
            if inherit_from:
                # Remove .json or .yaml extension if included
                inherit_base = inherit_from.replace(".json", "").replace(".yaml", "")

                # Find the inherited category file
                inherit_yaml_path = os.path.join(categories_dir, f"{inherit_base}.yaml")
                inherit_json_path = os.path.join(categories_dir, f"{inherit_base}.json")

                inherit_file_path = None
                if os.path.exists(inherit_yaml_path):
                    inherit_file_path = inherit_yaml_path
                elif os.path.exists(inherit_json_path):
                    inherit_file_path = inherit_json_path
                else:
                    verbose_proxy_logger.warning(
                        f"Category {category_name}: inherit_from '{inherit_from}' file not found at {categories_dir}"
                    )
                    verbose_proxy_logger.debug(
                        f"Tried paths: {inherit_yaml_path}, {inherit_json_path}"
                    )

                if inherit_file_path:
                    # Load the inherited category
                    inherited_category = self._load_category_file(inherit_file_path)

                    # Extract block words from inherited category that meet severity threshold
                    for keyword_data in inherited_category.keywords:
                        keyword = keyword_data["keyword"].lower()
                        severity = keyword_data["severity"]
                        if self._should_apply_severity(severity, severity_threshold):
                            block_words.append(keyword)
                else:
                    # If inherit file not found, set inherit_from to None for logging
                    inherit_from = None

            # Add additional block words specific to this category
            if category_config_obj.additional_block_words:
                block_words.extend(category_config_obj.additional_block_words)

            # Store the conditional category configuration
            self.conditional_categories[category_name] = {
                "identifier_words": category_config_obj.identifier_words,
                "block_words": block_words,
                "action": category_action,
                "severity": "high",  # Combinations are always high severity
            }

            # Build log message
            log_msg = (
                f"Loaded conditional category {category_name}: "
                f"{len(category_config_obj.identifier_words)} identifiers + "
                f"{len(block_words)} block words"
            )
            if inherit_from and category_config_obj.additional_block_words:
                inherited_count = len(block_words) - len(
                    category_config_obj.additional_block_words
                )
                log_msg += (
                    f" ({len(category_config_obj.additional_block_words)} additional + "
                    f"{inherited_count} from {inherit_from})"
                )
            elif inherit_from:
                log_msg += f" (from {inherit_from})"
            elif category_config_obj.additional_block_words:
                log_msg += f" ({len(block_words)} from additional_block_words)"

            verbose_proxy_logger.info(log_msg)
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error loading conditional category for {category_name}: {e}"
            )

    def _load_category_file(self, file_path: str) -> CategoryConfig:
        """
        Load a category definition from a YAML or JSON file.

        YAML format: category_name, description, default_action, keywords (list of
        {keyword, severity}), exceptions.
        Optional: identifier_words, always_block_keywords, inherit_from.
        JSON format: list of {id, match, tags, severity}; match is pipe-separated
        phrases; severity 1-4 mapped to low/medium/high. Used for harm_toxic_abuse.

        Args:
            file_path: Path to category YAML or JSON file

        Returns:
            CategoryConfig object
        """
        if file_path.lower().endswith(".json"):
            return self._load_category_file_json(file_path)
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        # Handle always_block_keywords if present
        always_block = data.get("always_block_keywords", [])

        return CategoryConfig(
            category_name=data.get("category_name", "unknown"),
            description=data.get("description", ""),
            default_action=ContentFilterAction(data.get("default_action", "BLOCK")),
            keywords=data.get("keywords", []),
            exceptions=data.get("exceptions", []),
            identifier_words=data.get("identifier_words"),
            always_block_keywords=always_block,
            inherit_from=data.get("inherit_from"),
            additional_block_words=data.get("additional_block_words"),
        )

    def _load_category_file_json(self, file_path: str) -> CategoryConfig:
        """
        Load a category from the harm_toxic_abuse-style JSON format.

        Each entry has: id, match (pipe-separated phrases), tags, severity (1-4).
        Severity mapping: 4,3 -> high; 2 -> medium; 1 -> low.
        """
        with open(file_path, "r") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            entries = [entries]
        # Derive category name from filename (e.g. harm_toxic_abuse.json -> harm_toxic_abuse)
        category_name = os.path.splitext(os.path.basename(file_path))[0]
        severity_map = {4: "high", 3: "high", 2: "medium", 1: "low"}
        keywords: List[Dict[str, str]] = []
        seen = set()
        for item in entries:
            if not isinstance(item, dict):
                continue
            match_str = item.get("match") or ""
            raw_severity = item.get("severity", 2)
            severity = severity_map.get(
                raw_severity if isinstance(raw_severity, int) else 2, "medium"
            )
            for phrase in match_str.split("|"):
                phrase = phrase.strip().lower()
                if not phrase or phrase in seen:
                    continue
                seen.add(phrase)
                keywords.append({"keyword": phrase, "severity": severity})
        return CategoryConfig(
            category_name=category_name,
            description="Detects harmful, toxic, or abusive language and content",
            default_action=ContentFilterAction("BLOCK"),
            keywords=keywords,
            exceptions=[],
        )

    def _should_apply_severity(self, severity: str, threshold: str) -> bool:
        """
        Check if a given severity meets the threshold.

        Args:
            severity: The severity level of the item ("high", "medium", "low")
            threshold: The minimum severity threshold

        Returns:
            True if severity meets or exceeds threshold
        """
        severity_order = {"low": 0, "medium": 1, "high": 2}
        return severity_order.get(severity, 0) >= severity_order.get(threshold, 1)

    def _add_pattern(self, pattern_config: ContentFilterPattern) -> None:
        """
        Add a pattern to the compiled patterns list.

        Args:
            pattern_config: ContentFilterPattern configuration
        """
        try:
            extra_config: Dict[str, Any] = {}
            if pattern_config.pattern_type == "prebuilt":
                if not pattern_config.pattern_name:
                    raise ValueError("pattern_name is required for prebuilt patterns")
                compiled = get_compiled_pattern(pattern_config.pattern_name)
                pattern_name = pattern_config.pattern_name
                extra_config = PATTERN_EXTRA_CONFIG.get(pattern_name, {}) or {}
            elif pattern_config.pattern_type == "regex":
                if not pattern_config.pattern:
                    raise ValueError("pattern is required for regex patterns")
                compiled = re.compile(pattern_config.pattern, re.IGNORECASE)
                pattern_name = pattern_config.name or "custom_regex"
            else:
                raise ValueError(f"Unknown pattern_type: {pattern_config.pattern_type}")

            keyword_regex: Optional[Pattern] = None
            if extra_config.get("keyword_pattern"):
                keyword_regex = re.compile(
                    extra_config["keyword_pattern"], re.IGNORECASE
                )

            self.compiled_patterns.append(
                {
                    "regex": compiled,
                    "pattern_name": pattern_name,
                    "action": pattern_config.action,
                    "keyword_regex": keyword_regex,
                    "allow_word_numbers": bool(extra_config.get("allow_word_numbers")),
                }
            )
            verbose_proxy_logger.debug(
                f"Added pattern: {pattern_name} with action {pattern_config.action}"
            )
        except Exception as e:
            verbose_proxy_logger.error(f"Error adding pattern {pattern_config}: {e}")
            raise

    def _load_blocked_words_file(self, file_path: str) -> None:
        """
        Load blocked words from a YAML file.

        Args:
            file_path: Path to YAML file containing blocked_words list

        Expected format:
        ```yaml
        blocked_words:
          - keyword: "sensitive_term"
            action: "BLOCK"
            description: "Optional description"
        ```
        """
        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict) or "blocked_words" not in data:
                raise ValueError(
                    "Invalid format: file must contain 'blocked_words' key with list of words"
                )

            for word_data in data["blocked_words"]:
                if (
                    not isinstance(word_data, dict)
                    or "keyword" not in word_data
                    or "action" not in word_data
                ):
                    verbose_proxy_logger.warning(
                        f"Skipping invalid word entry: {word_data}"
                    )
                    continue

                keyword = word_data["keyword"].lower()
                action = ContentFilterAction(word_data["action"])
                description = word_data.get("description")

                self.blocked_words[keyword] = (action, description)

            verbose_proxy_logger.info(
                f"Loaded {len(data['blocked_words'])} blocked words from {file_path}"
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Blocked words file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading blocked words file {file_path}: {str(e)}")

    def _find_pattern_spans(
        self, text: str, pattern_entry: Dict[str, Any]
    ) -> List[Tuple[int, int]]:
        """Return all match spans for a pattern, applying contextual rules if required."""

        regex: Pattern = pattern_entry["regex"]
        keyword_regex: Optional[Pattern] = pattern_entry.get("keyword_regex")
        allow_word_numbers: bool = pattern_entry.get("allow_word_numbers", False)

        keyword_matches: Optional[List[re.Match]] = None
        if keyword_regex is not None:
            keyword_matches = list(keyword_regex.finditer(text))
            if not keyword_matches:
                return []

        match_spans: List[Tuple[int, int]] = []

        for match in regex.finditer(text):
            if keyword_matches is not None and not self._match_near_keyword(
                match.start(), match.end(), keyword_matches, text
            ):
                continue
            match_spans.append((match.start(), match.end()))

        if allow_word_numbers:
            for word_match in WORD_NUMBER_SEQUENCE_PATTERN.finditer(text):
                digits = self._convert_word_number_sequence(word_match.group())
                if not digits:
                    continue
                if not regex.fullmatch(digits):
                    continue
                if keyword_matches is not None and not self._match_near_keyword(
                    word_match.start(), word_match.end(), keyword_matches, text
                ):
                    continue
                match_spans.append((word_match.start(), word_match.end()))

        return self._merge_spans(match_spans)

    def _match_near_keyword(
        self,
        value_start: int,
        value_end: int,
        keyword_matches: List[re.Match],
        text: str,
    ) -> bool:
        """Check if a value is separated from a keyword by an allowed gap."""

        for keyword_match in keyword_matches:
            keyword_start = keyword_match.start()
            keyword_end = keyword_match.end()

            if value_start >= keyword_end:
                gap_text = text[keyword_end:value_start]
            elif keyword_start >= value_end:
                gap_text = text[value_end:keyword_start]
            else:
                return True  # overlapping

            if self._gap_text_allowed(gap_text):
                return True
        return False

    def _gap_text_allowed(self, gap_text: str) -> bool:
        """Return True if the gap between keyword and value meets word-count rules."""

        if not gap_text.strip():
            return True
        if any(char.isdigit() for char in gap_text):
            return False

        words = GAP_WORD_TOKENIZER.findall(gap_text)
        return len(words) <= MAX_KEYWORD_VALUE_GAP_WORDS

    def _merge_spans(self, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping spans to avoid double-masking."""

        if not spans:
            return []

        spans.sort(key=lambda item: item[0])
        merged: List[Tuple[int, int]] = [spans[0]]

        for start, end in spans[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    def _mask_spans(
        self, text: str, spans: List[Tuple[int, int]], redaction: str
    ) -> str:
        """Apply masking for the provided spans using the given redaction tag."""

        if not spans:
            return text

        result_parts: List[str] = []
        previous_end = 0
        for start, end in spans:
            result_parts.append(text[previous_end:start])
            result_parts.append(redaction)
            previous_end = end
        result_parts.append(text[previous_end:])
        return "".join(result_parts)

    def _convert_word_number_sequence(self, sequence: str) -> Optional[str]:
        """Convert a spelled-out digit sequence (e.g., 'One-Two') into digits."""

        tokens = WORD_NUMBER_TOKEN_FINDER.findall(sequence)
        if not tokens:
            return None

        digits: List[str] = []
        for token in tokens:
            digit = WORD_NUMBER_MAP.get(token.lower())
            if digit is None:
                return None
            digits.append(digit)

        return "".join(digits) if digits else None

    def _check_patterns(
        self, text: str
    ) -> Optional[Tuple[str, str, ContentFilterAction]]:
        """
        Check text against all compiled regex patterns.

        Args:
            text: Text to check

        Returns:
            Tuple of (matched_text, pattern_name, action) if match found, None otherwise
        """
        for pattern_entry in self.compiled_patterns:
            spans = self._find_pattern_spans(text, pattern_entry)
            if spans:
                start, end = spans[0]
                matched_text = text[start:end]
                pattern_name = pattern_entry["pattern_name"]
                action = pattern_entry["action"]
                verbose_proxy_logger.debug(
                    f"Pattern '{pattern_name}' matched: {matched_text[:20]}..."
                )
                return (matched_text, pattern_name, action)
        return None

    def _check_conditional_categories(
        self, text: str, exceptions: List[str]
    ) -> Optional[Tuple[str, str, str, ContentFilterAction]]:
        """
        Check text for conditional category matches (identifier + block word in same sentence).

        This implements logic like: if text contains both an identifier word (e.g., "minor")
        AND a block word (e.g., "romantic"), then block it.

        Args:
            text: Text to check
            exceptions: List of exception phrases to ignore

        Returns:
            Tuple of (matched_phrase, category, severity, action) if match found, None otherwise
        """
        text_lower = text.lower()

        # First check if any exception applies
        for exception in exceptions:
            if exception in text_lower:
                return None

        # Split text into sentences for more precise matching
        # Simple sentence splitting on common terminators
        sentences = re.split(r"[.!?]+", text)

        for category_name, config in self.conditional_categories.items():
            identifier_words = config["identifier_words"]
            block_words = config["block_words"]
            action = config["action"]
            severity = config["severity"]

            # Check category-specific exceptions
            category_obj = self.loaded_categories.get(category_name)
            if category_obj:
                exception_found = False
                for exception in category_obj.exceptions:
                    if exception in text_lower:
                        verbose_proxy_logger.debug(
                            f"Category exception '{exception}' found for {category_name}, skipping"
                        )
                        exception_found = True
                        break
                if exception_found:
                    continue

            # Check each sentence for identifier + block word combination
            for sentence in sentences:
                sentence_lower = sentence.lower().strip()
                if not sentence_lower:
                    continue

                # Check if sentence contains ANY identifier word
                identifier_found = None
                for identifier in identifier_words:
                    if identifier in sentence_lower:
                        identifier_found = identifier
                        break

                if not identifier_found:
                    continue

                # Check if sentence also contains ANY block word
                block_word_found = None
                for block_word in block_words:
                    # Use word boundary for single words to avoid false positives
                    if " " in block_word:
                        # Multi-word phrase
                        if block_word in sentence_lower:
                            block_word_found = block_word
                            break
                    else:
                        # Single word - use word boundary
                        pattern = r"\b" + re.escape(block_word) + r"\b"
                        if re.search(pattern, sentence_lower):
                            block_word_found = block_word
                            break

                if block_word_found:
                    matched_phrase = f"{identifier_found} + {block_word_found}"
                    verbose_proxy_logger.warning(
                        f"Conditional match in {category_name}: '{matched_phrase}' in sentence"
                    )
                    return (matched_phrase, category_name, severity, action)

        return None

    def _check_category_keywords(
        self, text: str, exceptions: List[str]
    ) -> Optional[Tuple[str, str, str, ContentFilterAction]]:
        """
        Check text for category keywords.

        Args:
            text: Text to check
            exceptions: List of exception phrases to ignore

        Returns:
            Tuple of (keyword, category, severity, action) if match found, None otherwise
        """
        text_lower = text.lower()

        # First check if any exception applies
        for exception in exceptions:
            if exception in text_lower:
                verbose_proxy_logger.debug(
                    f"Exception phrase '{exception}' found, skipping category keyword check"
                )
                return None

        # Check category keywords
        for keyword, (category, severity, action) in self.category_keywords.items():
            # Convert asterisks (*) in keywords to regex wildcards
            # Asterisks are used in the source data to obfuscate profanity (e.g., "fu*c*k" -> "fuck")
            # We treat * as a wildcard matching zero or one character
            keyword_pattern_str = keyword.replace("*", ".?")

            # Use word boundary matching for single words to avoid false positives
            # (e.g., "men" should not match "recommend")
            # For multi-word phrases, use substring matching
            if " " in keyword:
                # Multi-word phrase - use substring matching with wildcards
                keyword_pattern = keyword_pattern_str
                keyword_found = bool(re.search(keyword_pattern, text_lower))
            else:
                # Single word - use word boundary matching to match whole words only
                keyword_pattern = r"\b" + keyword_pattern_str + r"\b"
                keyword_found = bool(re.search(keyword_pattern, text_lower))

            if keyword_found:
                # Check if this keyword has exceptions
                category_obj = self.loaded_categories.get(category)
                if category_obj:
                    # Check category-specific exceptions
                    exception_found = False
                    for exception in category_obj.exceptions:
                        if exception in text_lower:
                            verbose_proxy_logger.debug(
                                f"Category exception '{exception}' found for keyword '{keyword}', skipping"
                            )
                            exception_found = True
                            break
                    if exception_found:
                        continue

                verbose_proxy_logger.debug(
                    f"Category keyword '{keyword}' found in category '{category}' with severity {severity}"
                )
                return (keyword, category, severity, action)
        return None

    def _check_blocked_words(
        self, text: str
    ) -> Optional[Tuple[str, ContentFilterAction, Optional[str]]]:
        """
        Check text for blocked keywords.

        Args:
            text: Text to check

        Returns:
            Tuple of (keyword, action, description) if match found, None otherwise
        """
        # Handle case where blocked_words might still be a list (old instances)
        if isinstance(self.blocked_words, list):
            verbose_proxy_logger.warning(
                "blocked_words is a list instead of dict. Re-initializing. "
                "This suggests an old guardrail instance is still in use. Please restart the server."
            )
            # Convert list to dict on-the-fly
            temp_dict: Dict[str, Tuple[ContentFilterAction, Optional[str]]] = {}
            for word in self.blocked_words:
                if isinstance(word, dict):
                    temp_dict[word.get("keyword", "").lower()] = (
                        word.get("action", ContentFilterAction.BLOCK),
                        word.get("description"),
                    )
            self.blocked_words = temp_dict

        if not self.blocked_words:
            return None

        text_lower = text.lower()
        for keyword, (action, description) in self.blocked_words.items():
            if keyword in text_lower:
                verbose_proxy_logger.debug(
                    f"Blocked word '{keyword}' found with action {action}"
                )
                return (keyword, action, description)
        return None

    def _handle_conditional_match(
        self,
        matched_phrase: str,
        category_name: str,
        severity: str,
        action: ContentFilterAction,
        detections: Optional[List[ContentFilterDetection]],
    ) -> None:
        """Handle conditional category match detection and action."""
        if detections is not None:
            category_detection: CategoryKeywordDetection = {
                "type": "category_keyword",
                "category": category_name,
                "keyword": matched_phrase,
                "severity": severity,
                "action": action.value,
            }
            detections.append(category_detection)

        if action == ContentFilterAction.BLOCK:
            error_msg = (
                f"Content blocked: {category_name} conditional match '{matched_phrase}' detected "
                f"(severity: {severity})"
            )
            verbose_proxy_logger.warning(error_msg)
            raise HTTPException(
                status_code=403,
                detail={
                    "error": error_msg,
                    "category": category_name,
                    "matched_phrase": matched_phrase,
                    "severity": severity,
                },
            )
        elif action == ContentFilterAction.MASK:
            verbose_proxy_logger.warning(
                f"Conditional match '{matched_phrase}' from {category_name} detected but MASK action not supported for conditional categories"
            )

    def _handle_category_keyword_match(
        self,
        keyword: str,
        category_name: str,
        severity: str,
        action: ContentFilterAction,
        text: str,
        detections: Optional[List[ContentFilterDetection]],
    ) -> str:
        """Handle category keyword match detection and action."""
        if detections is not None:
            category_detection: CategoryKeywordDetection = {
                "type": "category_keyword",
                "category": category_name,
                "keyword": keyword,
                "severity": severity,
                "action": action.value,
            }
            detections.append(category_detection)

        if action == ContentFilterAction.BLOCK:
            error_msg = (
                f"Content blocked: {category_name} category keyword '{keyword}' detected "
                f"(severity: {severity})"
            )
            verbose_proxy_logger.warning(error_msg)
            raise HTTPException(
                status_code=403,
                detail={
                    "error": error_msg,
                    "category": category_name,
                    "keyword": keyword,
                    "severity": severity,
                },
            )
        elif action == ContentFilterAction.MASK:
            keyword_pattern_for_masking = keyword.replace("*", ".?")
            text = re.sub(
                keyword_pattern_for_masking,
                self.keyword_redaction_tag,
                text,
                flags=re.IGNORECASE,
            )
            verbose_proxy_logger.info(
                f"Masked category keyword '{keyword}' from {category_name} (severity: {severity})"
            )

        return text

    def _handle_pattern_match(
        self,
        pattern_name: str,
        action: ContentFilterAction,
        text: str,
        spans: List[Tuple[int, int]],
        detections: Optional[List[ContentFilterDetection]],
    ) -> str:
        """Handle regex pattern match detection and action."""
        if detections is not None:
            pattern_detection: PatternDetection = {
                "type": "pattern",
                "pattern_name": pattern_name,
                "action": action.value,
            }
            detections.append(pattern_detection)

        if action == ContentFilterAction.BLOCK:
            error_msg = f"Content blocked: {pattern_name} pattern detected"
            verbose_proxy_logger.warning(error_msg)
            raise HTTPException(
                status_code=403,
                detail={"error": error_msg, "pattern": pattern_name},
            )
        elif action == ContentFilterAction.MASK:
            redaction_tag = self.pattern_redaction_format.format(
                pattern_name=pattern_name.upper()
            )
            text = self._mask_spans(text, spans, redaction_tag)
            verbose_proxy_logger.info(
                f"Masked all {pattern_name} matches in content"
            )

        return text

    def _handle_blocked_word_match(
        self,
        keyword: str,
        action: ContentFilterAction,
        description: Optional[str],
        text: str,
        detections: Optional[List[ContentFilterDetection]],
    ) -> str:
        """Handle blocked word match detection and action."""
        verbose_proxy_logger.debug(
            f"Blocked word '{keyword}' found with action {action}"
        )

        if detections is not None:
            blocked_word_detection: BlockedWordDetection = {
                "type": "blocked_word",
                "keyword": keyword,
                "action": action.value,
                "description": description,
            }
            detections.append(blocked_word_detection)

        if action == ContentFilterAction.BLOCK:
            error_msg = f"Content blocked: keyword '{keyword}' detected"
            if description:
                error_msg += f" ({description})"
            verbose_proxy_logger.warning(error_msg)
            raise HTTPException(
                status_code=403,
                detail={
                    "error": error_msg,
                    "keyword": keyword,
                    "description": description,
                },
            )
        elif action == ContentFilterAction.MASK:
            keyword_pattern_for_masking = keyword.replace("*", ".?")
            text = re.sub(
                keyword_pattern_for_masking,
                self.keyword_redaction_tag,
                text,
                flags=re.IGNORECASE,
            )
            verbose_proxy_logger.info(f"Masked keyword '{keyword}' in content")

        return text

    def _filter_single_text(
        self, text: str, detections: Optional[List[ContentFilterDetection]] = None
    ) -> str:
        """
        Apply all content filtering checks to a single text.

        This method performs:
        1. Category keyword checks
        2. Regex pattern checks
        3. Blocked word checks

        Args:
            text: Text to filter
            detections: Optional list to append detection information

        Returns:
            Filtered text (with masking applied if action is MASK)

        Raises:
            HTTPException: If sensitive content is detected and action is BLOCK
        """
        # Collect all exceptions from loaded categories
        all_exceptions = []
        for category in self.loaded_categories.values():
            all_exceptions.extend(category.exceptions)

        # Check conditional categories first
        conditional_match = self._check_conditional_categories(text, all_exceptions)
        if conditional_match:
            matched_phrase, category_name, severity, action = conditional_match
            self._handle_conditional_match(
                matched_phrase, category_name, severity, action, detections
            )

        # Check category keywords
        category_keyword_match = self._check_category_keywords(text, all_exceptions)
        if category_keyword_match:
            keyword, category_name, severity, action = category_keyword_match
            text = self._handle_category_keyword_match(
                keyword, category_name, severity, action, text, detections
            )

        # Check regex patterns - process ALL patterns, not just first match
        for pattern_entry in self.compiled_patterns:
            spans = self._find_pattern_spans(text, pattern_entry)
            if spans:
                pattern_name = pattern_entry["pattern_name"]
                action = pattern_entry["action"]
                text = self._handle_pattern_match(
                    pattern_name, action, text, spans, detections
                )

        # Check blocked words - iterate through ALL blocked words
        text_lower = text.lower()
        for keyword, (action, description) in self.blocked_words.items():
            keyword_pattern_str = keyword.replace("*", ".?")
            if re.search(keyword_pattern_str, text_lower):
                text = self._handle_blocked_word_match(
                    keyword, action, description, text, detections
                )
                text_lower = text.lower()  # Update after masking

        return text

    def _mask_content(self, text: str, pattern_name: str) -> str:
        """
        Mask sensitive content in text.

        Args:
            text: Text containing sensitive content
            pattern_name: Name of the pattern that matched

        Returns:
            Text with sensitive content masked
        """
        redaction_tag = self.pattern_redaction_format.format(
            pattern_name=pattern_name.upper()
        )
        return redaction_tag

    async def _process_images(
        self, images: List[str], detections: List[ContentFilterDetection]
    ) -> None:
        """
        Process images by describing them and applying content filtering.

        Args:
            images: List of image URLs
            detections: List to append detection information
        """
        if not (images and self.image_model and self.llm_router):
            return

        tasks = []
        for image in images:
            task = self.llm_router.acompletion(
                model=self.image_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Describe the image in detail.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image}},
                        ],
                    },
                ],
                stream=False,
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        descriptions = []
        for response in responses:
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            if message and getattr(message, "content", None):
                image_description = message.content
                verbose_proxy_logger.debug(f"Image description: {image_description}")
                descriptions.append(image_description)
            else:
                verbose_proxy_logger.warning("No image description found")

        # Apply content filtering to image descriptions
        verbose_proxy_logger.debug(
            f"ContentFilterGuardrail: Applying guardrail to {len(descriptions)} image description(s)"
        )
        for description in descriptions:
            # This will raise HTTPException if BLOCK action is triggered
            try:
                self._filter_single_text(description, detections=detections)
            except HTTPException as e:
                # e.detail can be a string or dict
                if isinstance(e.detail, dict) and "error" in e.detail:
                    detail_dict = cast(Dict[str, Any], e.detail)
                    detail_dict["error"] = (
                        detail_dict["error"] + " (Image description): " + description
                    )
                elif isinstance(e.detail, str):
                    e.detail = e.detail + " (Image description): " + description
                else:
                    e.detail = (
                        "Content blocked: Image description detected" + description
                    )
                raise e

    def _count_masked_entities(
        self,
        detections: List[ContentFilterDetection],
        masked_entity_count: Dict[str, int],
    ) -> None:
        """
        Count masked entities by type from detections.

        Args:
            detections: List of detection dictionaries
            masked_entity_count: Dictionary to update with counts
        """
        for detection in detections:
            if detection["action"] == ContentFilterAction.MASK.value:
                detection_type = detection["type"]
                if detection_type == "pattern":
                    pattern_detection = cast(PatternDetection, detection)
                    pattern_name = pattern_detection["pattern_name"]
                    masked_entity_count[pattern_name] = (
                        masked_entity_count.get(pattern_name, 0) + 1
                    )
                elif detection_type == "blocked_word":
                    entity_type = "blocked_word"
                    masked_entity_count[entity_type] = (
                        masked_entity_count.get(entity_type, 0) + 1
                    )
                elif detection_type == "category_keyword":
                    category_detection = cast(CategoryKeywordDetection, detection)
                    category = category_detection["category"]
                    masked_entity_count[category] = (
                        masked_entity_count.get(category, 0) + 1
                    )

    def _build_match_details(
        self, detections: List[ContentFilterDetection]
    ) -> List[dict]:
        """Build match_details list from content filter detections."""
        match_details: List[dict] = []
        for detection in detections:
            detail: dict = {"type": detection["type"], "action_taken": detection["action"]}
            if detection["type"] == "pattern":
                detail["detection_method"] = "regex"
                detail["snippet"] = cast(PatternDetection, detection).get("pattern_name", "")
            elif detection["type"] == "blocked_word":
                detail["detection_method"] = "keyword"
                detail["snippet"] = cast(BlockedWordDetection, detection).get("keyword", "")
            elif detection["type"] == "category_keyword":
                detail["detection_method"] = "keyword"
                cat_det = cast(CategoryKeywordDetection, detection)
                detail["snippet"] = cat_det.get("keyword", "")
                detail["category"] = cat_det.get("category", "")
            match_details.append(detail)
        return match_details

    def _get_detection_methods(self, detections: List[ContentFilterDetection]) -> str:
        """Get comma-separated detection methods used."""
        methods: set = set()
        for detection in detections:
            if detection["type"] == "pattern":
                methods.add("regex")
            else:
                methods.add("keyword")
        return ",".join(sorted(methods)) if methods else ""

    def _get_patterns_checked_count(self) -> int:
        """Get total number of patterns and keywords that were evaluated."""
        return len(self.compiled_patterns) + len(self.blocked_words) + len(self.category_keywords)

    def _get_policy_templates(self) -> Optional[str]:
        """Get comma-separated policy template names from loaded categories."""
        if not self.loaded_categories:
            return None
        names = [cat.description or cat.category_name for cat in self.loaded_categories.values()]
        return ", ".join(names) if names else None

    def _compute_risk_score(
        self,
        detections: List[ContentFilterDetection],
        masked_entity_count: Dict[str, int],
        status: "GuardrailStatus",
    ) -> float:
        """
        Compute a risk score from 0-10 for this guardrail evaluation.

        Factors:
        - Match ratio: how many patterns matched vs total checked
        - Number of entities masked
        - Whether the guardrail blocked the request (max risk)
        """
        if status == "guardrail_intervened":
            return 10.0

        total_masked = sum(masked_entity_count.values()) if masked_entity_count else 0
        patterns_checked = self._get_patterns_checked_count()

        # Match ratio contribution (0-7 points)
        match_ratio = total_masked / patterns_checked if patterns_checked > 0 else 0.0
        ratio_score = match_ratio * 7.0

        # Detection count contribution (0-3 points, capped)
        detection_score = min(len(detections), 5) * 0.6

        score = ratio_score + detection_score

        # Floor: if anything matched, minimum risk is 2
        if total_masked > 0 and score < 2.0:
            score = 2.0

        return round(min(10.0, score), 1)

    def _log_guardrail_information(
        self,
        request_data: dict,
        detections: List[ContentFilterDetection],
        status: "GuardrailStatus",
        start_time: datetime,
        masked_entity_count: Dict[str, int],
        exception_str: str,
    ) -> None:
        """
        Log guardrail information to request_data metadata.

        Args:
            request_data: Request data dictionary
            detections: List of detection dictionaries
            status: Guardrail status
            start_time: Start time of guardrail execution
            masked_entity_count: Count of masked entities by type
            exception_str: Exception string if guardrail failed
        """
        # Convert TypedDict detections to regular dicts for JSON serialization
        guardrail_json_response: Union[Exception, str, dict, List[dict]] = [
            dict(detection) for detection in detections
        ]
        if status != "success":
            guardrail_json_response = (
                exception_str
                if exception_str
                else [dict(detection) for detection in detections]
            )

        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_provider=self.guardrail_provider,
            guardrail_json_response=guardrail_json_response,
            request_data=request_data,
            guardrail_status=status,
            start_time=start_time.timestamp(),
            end_time=datetime.now().timestamp(),
            duration=(datetime.now() - start_time).total_seconds(),
            masked_entity_count=masked_entity_count,
            tracing_detail=GuardrailTracingDetail(
                guardrail_id=self.config_guardrail_id or self.guardrail_name,
                policy_template=self.config_policy_template or self._get_policy_templates(),
                detection_method=self._get_detection_methods(detections) if detections else None,
                match_details=self._build_match_details(detections) if detections else None,
                patterns_checked=self._get_patterns_checked_count(),
                risk_score=self._compute_risk_score(detections, masked_entity_count, status),
            ),
        )

    async def apply_guardrail(
        self,
        inputs: "GenericGuardrailAPIInputs",
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> "GenericGuardrailAPIInputs":
        """
        Apply content filtering guardrail to a batch of texts.

        This method checks for sensitive patterns and blocked keywords,
        either blocking the request or masking the sensitive content.

        Args:
            inputs: Dictionary containing texts and optional images
            request_data: Request data dictionary for logging metadata
            input_type: Whether this is a "request" or "response"
            logging_obj: Optional logging object

        Returns:
            GenericGuardrailAPIInputs - processed_texts may be masked, images unchanged

        Raises:
            HTTPException: If sensitive content is detected and action is BLOCK
        """
        from litellm.types.utils import GuardrailStatus

        start_time = datetime.now()
        detections: List[ContentFilterDetection] = []
        masked_entity_count: Dict[str, int] = {}
        status: GuardrailStatus = "success"
        exception_str: str = ""

        try:
            texts = inputs.get("texts", [])
            images = inputs.get("images", [])

            # Process images if present
            await self._process_images(images, detections)

            # Process texts
            verbose_proxy_logger.debug(
                f"ContentFilterGuardrail: Applying guardrail to {len(texts)} text(s)"
            )

            processed_texts = []
            for text in texts:
                filtered_text = self._filter_single_text(text, detections=detections)
                processed_texts.append(filtered_text)

            verbose_proxy_logger.debug(
                "ContentFilterGuardrail: Guardrail applied successfully"
            )
            inputs["texts"] = processed_texts

            # Count masked entities by type
            self._count_masked_entities(detections, masked_entity_count)

            return inputs
        except HTTPException:
            status = "guardrail_intervened"
            raise
        except Exception as e:
            status = "guardrail_failed_to_respond"
            exception_str = str(e)
            raise e
        finally:
            # Log guardrail information
            self._log_guardrail_information(
                request_data=request_data,
                detections=detections,
                status=status,
                start_time=start_time,
                masked_entity_count=masked_entity_count,
                exception_str=exception_str,
            )

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Process streaming response chunks and check for blocked content.

        For BLOCK action: Raises HTTPException immediately when blocked content is detected.
        For MASK action: Content is buffered to handle patterns split across chunks.
        """
        accumulated_full_text = ""
        yielded_masked_text_len = 0
        buffer_size = 50  # Increased buffer to catch patterns split across many chunks

        verbose_proxy_logger.info(
            f"ContentFilterGuardrail: Starting robust streaming masking for model {request_data.get('model')}"
        )

        async for item in response:
            if isinstance(item, ModelResponseStream) and item.choices:
                delta_content = ""
                is_final = False
                for choice in item.choices:
                    if hasattr(choice, "delta") and choice.delta:
                        content = getattr(choice.delta, "content", None)
                        if content and isinstance(content, str):
                            delta_content += content
                    if getattr(choice, "finish_reason", None):
                        is_final = True

                accumulated_full_text += delta_content

                # Check for blocking or apply masking
                # Add a space at the end if it's the final chunk to trigger word boundaries (\b)
                text_to_check = accumulated_full_text
                if is_final:
                    text_to_check += " "

                try:
                    masked_text = self._filter_single_text(text_to_check)
                    if is_final and masked_text.endswith(" "):
                        masked_text = masked_text[:-1]
                except HTTPException:
                    raise
                except Exception as e:
                    verbose_proxy_logger.error(
                        f"ContentFilterGuardrail: Error in masking: {e}"
                    )
                    masked_text = text_to_check  # Fallback to current text

                # Determine how much can be safely yielded
                if is_final:
                    safe_to_yield_len = len(masked_text)
                else:
                    safe_to_yield_len = max(0, len(masked_text) - buffer_size)

                if safe_to_yield_len > yielded_masked_text_len:
                    new_masked_content = masked_text[
                        yielded_masked_text_len:safe_to_yield_len
                    ]
                    # Modify the chunk to contain only the new masked content
                    if (
                        item.choices
                        and hasattr(item.choices[0], "delta")
                        and item.choices[0].delta
                    ):
                        item.choices[0].delta.content = new_masked_content
                        yielded_masked_text_len = safe_to_yield_len
                        yield item
                else:
                    # Hold content by yielding empty content chunk (keeps metadata/structure)
                    if (
                        item.choices
                        and hasattr(item.choices[0], "delta")
                        and item.choices[0].delta
                    ):
                        item.choices[0].delta.content = ""
                    yield item
            else:
                # Not a ModelResponseStream or no choices - yield as is
                yield item

        # Any remaining content (should have been handled by is_final, but just in case)
        if yielded_masked_text_len < len(accumulated_full_text):
            # We already reached the end of the generator
            pass

    @staticmethod
    def get_config_model():
        from litellm.types.proxy.guardrails.guardrail_hooks.litellm_content_filter import (
            LitellmContentFilterGuardrailConfigModel,
        )

        return LitellmContentFilterGuardrailConfigModel
