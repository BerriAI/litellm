"""Ovalix guardrail integration: pre- and post-call checks via the Tracker service.

Use Ovalix Guardrails for your LLM calls. Supports pre_call (user input) and
post_call (model output) checkpoints with optional correction/blocking.
"""

import datetime
import os
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Type

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


BLOCKED_BY_OVALIX_FALLBACK_MESSAGE = "This message was blocked by Ovalix"
BLOCKED_ACTION_TYPE = "block"
USER_MESSAGE_ROLE = "user"


class OvalixGuardrailMissingSecrets(Exception):
    """Raised when required Ovalix config (API base, key, application/checkpoint IDs) is missing."""

    pass


class OvalixGuardrailBlockedException(GuardrailRaisedException):
    """
    Raised when Ovalix blocks a message. Sets status_code=400 so the proxy
    returns 400 and HTTP clients do not retry (they retry on 5xx).
    """

    status_code = 400

    def __init__(
        self,
        guardrail_name: Optional[str] = None,
        message: str = "",
        should_wrap_with_default_message: bool = True,
    ):
        super().__init__(
            guardrail_name=guardrail_name,
            message=message,
            should_wrap_with_default_message=should_wrap_with_default_message,
        )


class OvalixGuardrail(CustomGuardrail):
    """
    Ovalix guardrail: pre-prompt (pre_call) and post-prompt (post_call) checks
    via the Tracker service, with application and checkpoint resolution from the
    Monolith backend.
    """

    def __init__(
        self,
        tracker_api_base: Optional[str] = None,
        tracker_api_key: Optional[str] = None,
        application_id: Optional[str] = None,
        pre_checkpoint_id: Optional[str] = None,
        post_checkpoint_id: Optional[str] = None,
        **kwargs: Any,
    ):
        self._tracker_api_base = tracker_api_base or os.environ.get(
            "OVALIX_TRACKER_API_BASE"
        )
        self._tracker_api_key = tracker_api_key or os.environ.get(
            "OVALIX_TRACKER_API_KEY"
        )
        self._application_id = application_id or os.environ.get("OVALIX_APPLICATION_ID")
        self._pre_checkpoint_id = pre_checkpoint_id or os.environ.get(
            "OVALIX_PRE_CHECKPOINT_ID"
        )
        self._post_checkpoint_id = post_checkpoint_id or os.environ.get(
            "OVALIX_POST_CHECKPOINT_ID"
        )

        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = []

        self._validate_config(kwargs["supported_event_hooks"])

        self._tracker_headers = httpx.Headers(
            {
                "Authorization": f"Bearer {self._tracker_api_key}",
                "Content-Type": "application/json",
            },
            encoding="utf-8",
        )

        self._async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        super().__init__(**kwargs)
        verbose_proxy_logger.debug(
            "Ovalix Guardrail initialized: tracker=%s, application_id=%s, pre_checkpoint_id=%s, post_checkpoint_id=%s",
            self._tracker_api_base,
            self._application_id,
            self._pre_checkpoint_id,
            self._post_checkpoint_id,
        )

    def _validate_config(
        self, supported_event_hooks: List[GuardrailEventHooks]
    ) -> None:
        """Ensure required secrets and checkpoint IDs are set; auto-add hooks when IDs are present."""
        if not self._tracker_api_base:
            raise OvalixGuardrailMissingSecrets(
                "Ovalix Tracker API base required. Set OVALIX_TRACKER_API_BASE or pass tracker_api_base in litellm_params."
            )
        if not self._tracker_api_key:
            raise OvalixGuardrailMissingSecrets(
                "Ovalix Tracker API key required. Set OVALIX_TRACKER_API_KEY or pass tracker_api_key in litellm_params."
            )
        if not self._application_id:
            raise OvalixGuardrailMissingSecrets(
                "Ovalix Application ID required. Set OVALIX_APPLICATION_ID or pass application_id in litellm_params."
            )

        if (
            not self._pre_checkpoint_id
            and GuardrailEventHooks.pre_call in supported_event_hooks
        ):
            raise OvalixGuardrailMissingSecrets(
                "Ovalix Pre-checkpoint ID required. Set OVALIX_PRE_CHECKPOINT_ID or pass pre_checkpoint_id in litellm_params."
            )
        elif (
            self._pre_checkpoint_id
            and GuardrailEventHooks.pre_call not in supported_event_hooks
        ):
            supported_event_hooks.append(GuardrailEventHooks.pre_call)

        if (
            not self._post_checkpoint_id
            and GuardrailEventHooks.post_call in supported_event_hooks
        ):
            raise OvalixGuardrailMissingSecrets(
                "Ovalix Post-checkpoint ID required. Set OVALIX_POST_CHECKPOINT_ID or pass post_checkpoint_id in litellm_params."
            )
        elif (
            self._post_checkpoint_id
            and GuardrailEventHooks.post_call not in supported_event_hooks
        ):
            supported_event_hooks.append(GuardrailEventHooks.post_call)

        if not self._pre_checkpoint_id and not self._post_checkpoint_id:
            raise OvalixGuardrailMissingSecrets(
                "Ovalix Pre-checkpoint ID or Post-checkpoint ID required. Set OVALIX_PRE_CHECKPOINT_ID or OVALIX_POST_CHECKPOINT_ID or pass pre_checkpoint_id or post_checkpoint_id in litellm_params."
            )

    def _get_actor(self, data: dict) -> str:
        """Return a stable actor identifier from request metadata (e.g. user email or id)."""
        metadata = data.get("metadata") or data.get("litellm_metadata") or {}
        if metadata.get("user_api_key_user_email"):
            return metadata["user_api_key_user_email"]
        if metadata.get("user_api_key_user_id"):
            return metadata["user_api_key_user_id"]
        return "unknown"

    def _get_session_id(self, data: dict) -> str:
        """Return a unique identifier for the chat/session (actor + date + application_id)."""
        actor = hash(self._get_actor(data))
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        return f"{actor}_{today}_{self._application_id}"

    async def _call_checkpoint(
        self,
        content: str,
        checkpoint_id: str,
        actor: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """Call the Ovalix Tracker checkpoint API and return the JSON response."""
        application_id = self._application_id
        if not application_id or not checkpoint_id:
            raise ValueError("Ovalix: application_id or checkpoint_id not resolved")

        url = f"{self._tracker_api_base}/tracking/custom_application/checkpoint"
        headers = dict(self._tracker_headers)
        payload = {
            "application_id": application_id,
            "checkpoint_id": checkpoint_id,
            "actor": actor,
            "session_id": session_id,
            "data_type": "TEXT",
            "data": {"content": content},
        }
        response = await self._async_handler.post(
            url, headers=headers, json=payload
        )
        response.raise_for_status()
        return response.json()

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Apply Ovalix guardrail to the given inputs (request or response text).

        Used by the unified guardrail flow and the /apply_guardrail API.
        For "request", uses the pre-checkpoint; for "response", uses the post-checkpoint.

        Args:
            inputs: Guardrail API inputs (e.g. texts to check).
            request_data: Full request payload (messages, metadata, response).
            input_type: "request" (pre_call) or "response" (post_call).
            logging_obj: Optional logging context.

        Returns:
            Updated inputs (e.g. with replaced/corrected texts, or unchanged).
        """
        if not self._pre_checkpoint_id and not self._post_checkpoint_id:
            return inputs

        actor = self._get_actor(request_data)
        session_id = self._get_session_id(request_data)

        if input_type == "response":
            llm_response = self._get_llm_response_text(
                request_data.get("response", None)
            )
            if llm_response:
                (
                    corrected_llm_response,
                    is_blocked,
                ) = await self._handle_post_llm_response(
                    llm_response, actor, session_id
                )
                # TODO: set the llm response text to `corrected_llm_response`. will be addressed later.
            return inputs

        messages = request_data.get("messages") or []
        if not messages:
            return inputs

        if self._pre_checkpoint_id:
            post_guardrail_texts = await self._generate_post_guardrail_text(
                messages, actor, session_id, request_data
            )
            return {**inputs, "texts": post_guardrail_texts}
        return inputs

    def _block_current_message(self, blocking_message: str) -> None:
        """Raise OvalixGuardrailBlockedException with the given message (no default wrapper)."""
        raise OvalixGuardrailBlockedException(
            guardrail_name=self.guardrail_name,
            message=blocking_message,
            should_wrap_with_default_message=False,
        )

    def _get_llm_response_text(
        self, response: Optional[litellm.ModelResponse]
    ) -> Optional[str]:
        """Extract the first assistant text content from a ModelResponse, or None."""
        if not response:
            return None
        if isinstance(response, litellm.ModelResponse):
            for choice in response.choices:
                if isinstance(choice, litellm.Choices):
                    if choice.message.content and isinstance(
                        choice.message.content, str
                    ):
                        return choice.message.content
        return None

    async def _handle_post_llm_response(
        self, llm_response: str, actor: str, session_id: str
    ) -> tuple[str, bool]:
        """Run post-call checkpoint on model output; return corrected text or raise if blocked."""
        if not self._post_checkpoint_id:
            raise ValueError(
                "Ovalix: post-checkpoint ID is required for post_call handling."
            )

        try:
            resp = await self._call_checkpoint(
                llm_response, self._post_checkpoint_id, actor, session_id
            )
        except Exception as e:
            verbose_proxy_logger.exception(
                "Ovalix apply_guardrail checkpoint call failed: %s", e
            )
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Ovalix guardrail error: {e!s}",
                should_wrap_with_default_message=False,
            ) from e

        action_type = (resp.get("action_type") or "").lower()
        if action_type == BLOCKED_ACTION_TYPE:
            blocking_message = (
                self._get_trackers_corrected_message(resp)
                or BLOCKED_BY_OVALIX_FALLBACK_MESSAGE
            )
            return blocking_message, True
        return self._get_trackers_corrected_message(resp) or llm_response, False

    async def _generate_post_guardrail_text(
        self,
        messages: List[Dict[str, Any]],
        actor: str,
        session_id: str,
        request_data: dict,
    ) -> List[str]:
        """
        Generate post-guardrail text for the given messages.

        Args:
            messages: List of messages
            actor: Actor
            session_id: Session ID
            request_data: Request data

        Returns:
            List of post-guardrail texts
        """
        is_last_prompt = True
        post_guardrail_texts: List[str] = []

        if not self._pre_checkpoint_id:
            # should not happen - if it does, the guardrail is not configured correctly and self._validate_config did not raise an error
            raise ValueError("Ovalix: pre-checkpoint ID is required")

        for message in reversed(messages):
            content = message.get("content", None) or ""
            if not isinstance(content, str):
                continue
            message_role = message.get("role", None)
            if message_role and message_role != USER_MESSAGE_ROLE:
                # we are not scanning the LLM/system/developer past responses, only the response that the user sent
                post_guardrail_texts.insert(0, content)
                continue
            try:
                resp = await self._call_checkpoint(
                    content, self._pre_checkpoint_id, actor, session_id
                )
            except Exception as e:
                verbose_proxy_logger.exception(
                    "Ovalix apply_guardrail checkpoint call failed: %s", e
                )
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=f"Ovalix guardrail error: {e!s}",
                    should_wrap_with_default_message=False,
                ) from e

            action_type = (resp.get("action_type") or "").lower()
            if action_type == BLOCKED_ACTION_TYPE:
                blocking_message = (
                    self._get_trackers_corrected_message(resp)
                    or BLOCKED_BY_OVALIX_FALLBACK_MESSAGE
                )
                if is_last_prompt:
                    self._block_current_message(blocking_message)
                else:
                    post_guardrail_texts.insert(0, blocking_message)
            else:
                new_content = self._get_trackers_corrected_message(resp) or content
                post_guardrail_texts.insert(0, new_content)
            is_last_prompt = False
        return post_guardrail_texts

    def _get_trackers_corrected_message(self, resp: dict) -> Optional[str]:
        """Extract corrected/blocking message content from Tracker checkpoint response."""
        modified = resp.get("modified_data")
        if isinstance(modified, dict) and "content" in modified:
            return modified["content"]
        return None

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.ovalix import (
            OvalixGuardrailConfigModel,
        )

        return OvalixGuardrailConfigModel
