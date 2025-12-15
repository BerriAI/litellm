"""
Content Filter Guardrail for LiteLLM.

This guardrail provides regex pattern matching and keyword filtering
to detect and block/mask sensitive content.
"""

import re
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
)

import yaml
from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import GenericGuardrailAPIInputs
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import (
    BlockedWord,
    ContentFilterAction,
    ContentFilterPattern,
    GuardrailEventHooks,
    Mode,
)
from litellm.types.utils import ModelResponseStream

from .patterns import get_compiled_pattern


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
        texts = inputs.get("texts", [])

        verbose_proxy_logger.debug(
            f"ContentFilterGuardrail: Applying guardrail to {len(texts)} text(s)"
        )

        processed_texts = []

        for text in texts:
            # Check regex patterns - process ALL patterns, not just first match
            for compiled_pattern, pattern_name, action in self.compiled_patterns:
                match = compiled_pattern.search(text)
                if not match:
                    continue

                if action == ContentFilterAction.BLOCK:
                    error_msg = f"Content blocked: {pattern_name} pattern detected"
                    verbose_proxy_logger.warning(error_msg)
                    raise HTTPException(
                        status_code=400,
                        detail={"error": error_msg, "pattern": pattern_name},
                    )
                elif action == ContentFilterAction.MASK:
                    # Replace ALL matches of this pattern with redaction tag
                    redaction_tag = self.pattern_redaction_format.format(
                        pattern_name=pattern_name.upper()
                    )
                    text = compiled_pattern.sub(redaction_tag, text)
                    verbose_proxy_logger.info(f"Masked all {pattern_name} matches in content")

            # Check blocked words - iterate through ALL blocked words
            # to ensure all matching keywords are processed, not just the first one
            text_lower = text.lower()
            for keyword, (action, description) in self.blocked_words.items():
                if keyword not in text_lower:
                    continue

                verbose_proxy_logger.debug(
                    f"Blocked word '{keyword}' found with action {action}"
                )

                if action == ContentFilterAction.BLOCK:
                    error_msg = f"Content blocked: keyword '{keyword}' detected"
                    if description:
                        error_msg += f" ({description})"
                    verbose_proxy_logger.warning(error_msg)
                    raise HTTPException(
                        status_code=400,
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

            processed_texts.append(text)

        verbose_proxy_logger.debug(
            "ContentFilterGuardrail: Guardrail applied successfully"
        )
        inputs["texts"] = processed_texts
        return inputs

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Streaming hook to check each chunk as it's yielded.

        This implementation checks each chunk individually and yields it immediately,
        allowing for low-latency streaming with content filtering.

        Args:
            user_api_key_dict: User API key authentication
            response: Async generator of response chunks
            request_data: Original request data

        Yields:
            Checked and potentially masked chunks

        Raises:
            HTTPException: If chunk content should be blocked
        """
        verbose_proxy_logger.debug(
            "ContentFilterGuardrail: Running streaming check (per-chunk mode)"
        )

        # Process each chunk individually
        async for chunk in response:
            if isinstance(chunk, ModelResponseStream):
                for choice in chunk.choices:
                    if hasattr(choice, "delta") and choice.delta.content:
                        if isinstance(choice.delta.content, str):
                            # Check the chunk content using apply_guardrail
                            try:
                                guardrailed_inputs = await self.apply_guardrail(
                                    inputs={"texts": [choice.delta.content]},
                                    input_type="response",
                                    request_data=request_data,
                                )
                                processed_texts = guardrailed_inputs.get("texts", [])
                                processed_content = (
                                    processed_texts[0]
                                    if processed_texts
                                    else choice.delta.content
                                )
                                if processed_content != choice.delta.content:
                                    choice.delta.content = processed_content
                                    verbose_proxy_logger.debug(
                                        "ContentFilterGuardrail: Modified streaming chunk"
                                    )
                            except HTTPException as e:
                                # If content should be blocked, raise immediately
                                verbose_proxy_logger.warning(
                                    f"ContentFilterGuardrail: Blocked streaming chunk: {e.detail}"
                                )
                                raise

            yield chunk

        verbose_proxy_logger.debug("ContentFilterGuardrail: Streaming check completed")

    @staticmethod
    def get_config_model():
        from litellm.types.proxy.guardrails.guardrail_hooks.litellm_content_filter import (
            LitellmContentFilterGuardrailConfigModel,
        )

        return LitellmContentFilterGuardrailConfigModel
