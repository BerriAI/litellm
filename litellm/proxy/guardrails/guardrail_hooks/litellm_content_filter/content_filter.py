"""
Content Filter Guardrail for LiteLLM.

This guardrail provides regex pattern matching and keyword filtering
to detect and block/mask sensitive content.
"""

import asyncio
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
from litellm.types.utils import ModelResponseStream

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

from .patterns import get_compiled_pattern


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
    ):
        self.category_name = category_name
        self.description = description
        self.default_action = default_action
        self.keywords = keywords
        self.exceptions = [e.lower() for e in exceptions]


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

        # Load categories if provided
        if categories:
            self._load_categories(categories)

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
        self.compiled_patterns: List[Tuple[Pattern, str, ContentFilterAction]] = []
        for pattern_config in normalized_patterns:
            self._add_pattern(pattern_config)

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
                category_file_path = custom_file
            else:
                category_file_path = os.path.join(
                    categories_dir, f"{category_name}.yaml"
                )

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

                # Add keywords from this category
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
                    f"{len(category_config_obj.keywords)} keywords"
                )
            except Exception as e:
                verbose_proxy_logger.error(
                    f"Error loading category {category_name}: {e}"
                )

    def _load_category_file(self, file_path: str) -> CategoryConfig:
        """
        Load a category definition from a YAML file.

        Args:
            file_path: Path to category YAML file

        Returns:
            CategoryConfig object
        """
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        return CategoryConfig(
            category_name=data.get("category_name", "unknown"),
            description=data.get("description", ""),
            default_action=ContentFilterAction(data.get("default_action", "BLOCK")),
            keywords=data.get("keywords", []),
            exceptions=data.get("exceptions", []),
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
            if pattern_config.pattern_type == "prebuilt":
                if not pattern_config.pattern_name:
                    raise ValueError("pattern_name is required for prebuilt patterns")
                compiled = get_compiled_pattern(pattern_config.pattern_name)
                pattern_name = pattern_config.pattern_name
            elif pattern_config.pattern_type == "regex":
                if not pattern_config.pattern:
                    raise ValueError("pattern is required for regex patterns")
                compiled = re.compile(pattern_config.pattern, re.IGNORECASE)
                pattern_name = pattern_config.name or "custom_regex"
            else:
                raise ValueError(f"Unknown pattern_type: {pattern_config.pattern_type}")

            self.compiled_patterns.append(
                (compiled, pattern_name, pattern_config.action)
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
        for compiled_pattern, pattern_name, action in self.compiled_patterns:
            match = compiled_pattern.search(text)
            if match:
                matched_text = match.group(0)
                verbose_proxy_logger.debug(
                    f"Pattern '{pattern_name}' matched: {matched_text[:20]}..."
                )
                return (matched_text, pattern_name, action)
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
            # Use word boundary matching for single words to avoid false positives
            # (e.g., "men" should not match "recommend")
            # For multi-word phrases, use substring matching
            if " " in keyword:
                # Multi-word phrase - use substring matching
                keyword_found = keyword in text_lower
            else:
                # Single word - use word boundary matching to match whole words only
                keyword_pattern = r"\b" + re.escape(keyword) + r"\b"
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

        # Check category keywords
        category_keyword_match = self._check_category_keywords(text, all_exceptions)
        if category_keyword_match:
            keyword, category_name, severity, action = category_keyword_match
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
                # Replace keyword with redaction tag
                text = re.sub(
                    re.escape(keyword),
                    self.keyword_redaction_tag,
                    text,
                    flags=re.IGNORECASE,
                )
                verbose_proxy_logger.info(
                    f"Masked category keyword '{keyword}' from {category_name} (severity: {severity})"
                )

        # Check regex patterns - process ALL patterns, not just first match
        for compiled_pattern, pattern_name, action in self.compiled_patterns:
            match = compiled_pattern.search(text)
            if not match:
                continue

            if detections is not None:
                # Don't log matched_text to avoid exposing sensitive content (emails, credit cards, etc.)
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
                # Replace ALL matches of this pattern with redaction tag
                redaction_tag = self.pattern_redaction_format.format(
                    pattern_name=pattern_name.upper()
                )
                text = compiled_pattern.sub(redaction_tag, text)
                verbose_proxy_logger.info(
                    f"Masked all {pattern_name} matches in content"
                )

        # Check blocked words - iterate through ALL blocked words
        # to ensure all matching keywords are processed, not just the first one
        text_lower = text.lower()
        for keyword, (action, description) in self.blocked_words.items():
            if keyword not in text_lower:
                continue

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
                # Replace keyword with redaction tag (case-insensitive)
                text = re.sub(
                    re.escape(keyword),
                    self.keyword_redaction_tag,
                    text,
                    flags=re.IGNORECASE,
                )
                # Update text_lower after masking to avoid re-matching
                text_lower = text.lower()
                verbose_proxy_logger.info(f"Masked keyword '{keyword}' in content")

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
                    e.detail = "Content blocked: Image description detected" + description
                raise e

    def _count_masked_entities(
        self, detections: List[ContentFilterDetection], masked_entity_count: Dict[str, int]
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
            guardrail_json_response = exception_str if exception_str else [
                dict(detection) for detection in detections
            ]

        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_provider=self.guardrail_provider,
            guardrail_json_response=guardrail_json_response,
            request_data=request_data,
            guardrail_status=status,
            start_time=start_time.timestamp(),
            end_time=datetime.now().timestamp(),
            duration=(datetime.now() - start_time).total_seconds(),
            masked_entity_count=masked_entity_count,
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
        For MASK action: Content passes through (masking streaming responses is not supported).
        """

        # Accumulate content as we iterate through chunks
        accumulated_content = ""

        async for item in response:
            # Accumulate content from this chunk before checking
            if isinstance(item, ModelResponseStream) and item.choices:
                for choice in item.choices:
                    if hasattr(choice, "delta") and choice.delta:
                        content = getattr(choice.delta, "content", None)
                        if content and isinstance(content, str):
                            accumulated_content += content

                # Check accumulated content for blocked patterns/keywords after processing all choices
                # Only check for BLOCK actions, not MASK (masking streaming is not supported)
                if accumulated_content:
                    try:
                        # Check patterns
                        pattern_match = self._check_patterns(accumulated_content)
                        if pattern_match:
                            matched_text, pattern_name, action = pattern_match
                            if action == ContentFilterAction.BLOCK:
                                error_msg = f"Content blocked: {pattern_name} pattern detected"
                                verbose_proxy_logger.warning(error_msg)
                                raise HTTPException(
                                    status_code=403,
                                    detail={"error": error_msg, "pattern": pattern_name},
                                )

                        # Check blocked words
                        blocked_word_match = self._check_blocked_words(accumulated_content)
                        if blocked_word_match:
                            keyword, action, description = blocked_word_match
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

                        # Check category keywords
                        all_exceptions = []
                        for category in self.loaded_categories.values():
                            all_exceptions.extend(category.exceptions)
                        category_match = self._check_category_keywords(
                            accumulated_content, all_exceptions
                        )
                        if category_match:
                            keyword, category_name, severity, action = category_match
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
                    except HTTPException:
                        # Re-raise HTTPException (blocked content detected)
                        raise
                    except Exception as e:
                        # Log other exceptions but don't block the stream
                        verbose_proxy_logger.warning(
                            f"Error checking content filter in streaming: {e}"
                        )

            # Yield the chunk (only if no exception was raised above)
            yield item

    @staticmethod
    def get_config_model():
        from litellm.types.proxy.guardrails.guardrail_hooks.litellm_content_filter import (
            LitellmContentFilterGuardrailConfigModel,
        )

        return LitellmContentFilterGuardrailConfigModel
