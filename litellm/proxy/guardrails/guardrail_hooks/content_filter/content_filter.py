"""
Content Filter Guardrail for LiteLLM.

This guardrail provides regex pattern matching and keyword filtering
to detect and block/mask sensitive content before it reaches the LLM.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Pattern, Tuple, Union

import yaml
from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import (
    BlockedWord,
    ContentFilterAction,
    ContentFilterPattern,
    GuardrailEventHooks,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import CallTypes, GuardrailStatus

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

    def __init__(
        self,
        guardrail_name: Optional[str] = None,
        patterns: Optional[List[ContentFilterPattern]] = None,
        blocked_words: Optional[List[BlockedWord]] = None,
        blocked_words_file: Optional[str] = None,
        event_hook: Optional[Union[GuardrailEventHooks, List[GuardrailEventHooks]]] = None,
        default_on: bool = False,
        pattern_redaction_format: str = "[{pattern_name}_REDACTED]",
        keyword_redaction_tag: str = "[KEYWORD_REDACTED]",
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
            supported_event_hooks=[GuardrailEventHooks.pre_call],
            event_hook=event_hook or GuardrailEventHooks.pre_call,
            default_on=default_on,
            **kwargs,
        )
        
        self.guardrail_provider = "litellm_content_filter"
        self.pattern_redaction_format = pattern_redaction_format
        self.keyword_redaction_tag = keyword_redaction_tag
        
        # Compile regex patterns
        self.compiled_patterns: List[Tuple[Pattern, str, ContentFilterAction]] = []
        if patterns:
            for pattern_config in patterns:
                self._add_pattern(pattern_config)
        
        # Load blocked words
        self.blocked_words: Dict[str, Tuple[ContentFilterAction, Optional[str]]] = {}
        if blocked_words:
            for word in blocked_words:
                self.blocked_words[word.keyword.lower()] = (
                    word.action,
                    word.description,
                )
        
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
            
            self.compiled_patterns.append((compiled, pattern_name, pattern_config.action))
            verbose_proxy_logger.debug(f"Added pattern: {pattern_name} with action {pattern_config.action}")
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
                if not isinstance(word_data, dict) or "keyword" not in word_data or "action" not in word_data:
                    verbose_proxy_logger.warning(f"Skipping invalid word entry: {word_data}")
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

    def _check_patterns(self, text: str) -> Optional[Tuple[str, str, ContentFilterAction]]:
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

    def _check_blocked_words(self, text: str) -> Optional[Tuple[str, ContentFilterAction, Optional[str]]]:
        """
        Check text for blocked keywords.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (keyword, action, description) if match found, None otherwise
        """
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

    def _process_message_content(
        self,
        content: str,
        request_data: dict,
    ) -> str:
        """
        Process a single message content string.
        
        Args:
            content: Message content to check
            request_data: Original request data for logging
            
        Returns:
            Processed content (masked if needed)
            
        Raises:
            HTTPException: If content should be blocked
        """
        # Check regex patterns
        pattern_match = self._check_patterns(content)
        if pattern_match:
            matched_text, pattern_name, action = pattern_match
            
            if action == ContentFilterAction.BLOCK:
                error_msg = f"Content blocked: {pattern_name} pattern detected"
                verbose_proxy_logger.warning(error_msg)
                raise HTTPException(
                    status_code=400,
                    detail={"error": error_msg, "pattern": pattern_name},
                )
            elif action == ContentFilterAction.MASK:
                # Replace the matched text with redaction tag
                redaction_tag = self._mask_content(matched_text, pattern_name)
                content = content.replace(matched_text, redaction_tag)
                verbose_proxy_logger.info(f"Masked {pattern_name} in content")
        
        # Check blocked words
        word_match = self._check_blocked_words(content)
        if word_match:
            keyword, action, description = word_match
            
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
                # Use regex for case-insensitive replacement
                content = re.sub(
                    re.escape(keyword),
                    self.keyword_redaction_tag,
                    content,
                    flags=re.IGNORECASE,
                )
                verbose_proxy_logger.info(f"Masked keyword '{keyword}' in content")
        
        return content

    def _filter_messages(
        self,
        messages: List[AllMessageValues],
        request_data: dict,
    ) -> None:
        """
        Helper method to process all messages and apply content filtering.
        
        Modifies messages in-place, replacing sensitive content with redacted tags
        or raising exceptions if content should be blocked.
        
        Args:
            messages: List of message objects to process
            request_data: Original request data for logging
            
        Raises:
            HTTPException: If content should be blocked
        """
        for message in messages:
            content = message.get("content")
            if not content:
                continue
            
            if isinstance(content, str):
                # Process string content
                processed_content = self._process_message_content(content, request_data)
                if processed_content != content:
                    message["content"] = processed_content
            elif isinstance(content, list):
                # Process list of content items (multimodal)
                for i, item in enumerate(content):
                    if isinstance(item, dict) and "text" in item:
                        text_content = item["text"]
                        if isinstance(text_content, str):
                            processed_text = self._process_message_content(text_content, request_data)
                            if processed_text != text_content:
                                item["text"] = processed_text  # type: ignore

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Optional[dict]:
        """
        Pre-call hook to check content before it reaches the LLM.
        
        Args:
            user_api_key_dict: User API key authentication
            cache: Dual cache instance
            data: Request data containing messages
            call_type: Type of call (completion, embeddings, etc.)
            
        Returns:
            Modified data dict with masked content, or None
            
        Raises:
            HTTPException: If content should be blocked
        """
        verbose_proxy_logger.debug("ContentFilterGuardrail: Running pre-call check")
        
        # Get messages from the request
        messages: Optional[List[AllMessageValues]] = self.get_guardrails_messages_for_call_type(
            call_type=CallTypes(call_type),
            data=data,
        )
        
        if not messages:
            verbose_proxy_logger.debug("ContentFilterGuardrail: No messages to check")
            return data
        
        # Process all messages
        self._filter_messages(
            messages=messages,
            request_data=data,
        )
        
        verbose_proxy_logger.debug("ContentFilterGuardrail: Pre-call check completed")
        return data

