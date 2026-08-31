# +-------------------------------------------------------------+
#
#           Llama Guard content-safety guardrail
#   Classifies request/response text with a Llama Guard model
#   (Meta's LLM-based content-safety classifier) and blocks
#   content that violates the configured MLCommons hazard
#   categories.
#
#   https://github.com/meta-llama/PurpleLlama/tree/main/Llama-Guard3
#
# +-------------------------------------------------------------+
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypesLiteral, Choices, LLMResponseTypes, ModelResponse

if TYPE_CHECKING:
    from litellm import DualCache

# The MLCommons hazard taxonomy Llama Guard 3 / 4 is trained on (S1-S14).
# https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/8B/MODEL_CARD.md
DEFAULT_UNSAFE_CATEGORIES: Final[dict[str, str]] = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}

# Roles as Llama Guard refers to them in its prompt / assessment target.
_USER_ROLE: Final = "User"
_AGENT_ROLE: Final = "Agent"


def _extract_text(content: Any) -> str:
    """Flatten a chat message ``content`` (str or multimodal parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


class LlamaGuardGuardrail(CustomGuardrail):
    """Content-safety guardrail backed by a Llama Guard classifier model.

    The configured ``model`` can be any Llama Guard model reachable through
    LiteLLM (e.g. ``together_ai/meta-llama/Llama-Guard-4-12B``,
    ``groq/llama-guard-3-8b``, ``ollama/llama-guard3``). On each event the
    guardrail renders the conversation into Llama Guard's classification
    prompt, asks the model whether the last turn is safe, and raises a
    content-policy error when it is flagged ``unsafe``.
    """

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
        ]

    def __init__(
        self,
        model: str,
        guardrail_name: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        categories: Sequence[str] | None = None,
        unsafe_content_categories: str | None = None,
        event_hook: GuardrailEventHooks | list[GuardrailEventHooks] | str | None = None,
        default_on: bool = False,
        **kwargs: Any,
    ) -> None:
        if not model:
            raise ValueError("llama_guard guardrail requires a `model`")
        self.model: Final = model
        self.api_base: Final = api_base
        self.api_key: Final = api_key

        # Restrict enforcement to a subset of hazard codes when configured,
        # otherwise enforce the full MLCommons taxonomy.
        if categories:
            selected = {
                code.upper(): DEFAULT_UNSAFE_CATEGORIES[code.upper()]
                for code in categories
                if code.upper() in DEFAULT_UNSAFE_CATEGORIES
            }
            if not selected:
                raise ValueError(f"llama_guard `categories` must be a subset of {sorted(DEFAULT_UNSAFE_CATEGORIES)}")
            self.categories = selected
        else:
            self.categories = dict(DEFAULT_UNSAFE_CATEGORIES)

        # Allow a fully custom category block to override the taxonomy text.
        self.unsafe_content_categories: Final = unsafe_content_categories

        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))
        super().__init__(
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Prompt construction + response parsing
    # ------------------------------------------------------------------ #
    def _category_block(self) -> str:
        if self.unsafe_content_categories:
            return self.unsafe_content_categories
        return "\n".join(f"{code}: {name}." for code, name in self.categories.items())

    def _render_conversation(self, messages: Sequence[dict]) -> str:
        lines: list[str] = []
        for message in messages:
            role = _AGENT_ROLE if message.get("role") == "assistant" else _USER_ROLE
            text = _extract_text(message.get("content"))
            if text:
                lines.append(f"{role}: {text}")
        return "\n\n".join(lines)

    def _build_prompt(self, messages: Sequence[dict], role_to_check: str) -> str:
        return (
            f"Task: Check if there is unsafe content in '{role_to_check}' messages in "
            "conversations according to our safety policy with the below categories.\n\n"
            "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
            f"{self._category_block()}\n"
            "<END UNSAFE CONTENT CATEGORIES>\n\n"
            "<BEGIN CONVERSATION>\n\n"
            f"{self._render_conversation(messages)}\n\n"
            "<END CONVERSATION>\n\n"
            f"Provide your safety assessment for ONLY THE LAST {role_to_check} message in the "
            "above conversation:\n"
            " - First line must read 'safe' or 'unsafe'.\n"
            " - If unsafe, a second line must include a comma-separated list of violated categories."
        )

    @staticmethod
    def _parse_response(content: str | None) -> tuple[bool, list[str]]:
        """Return ``(is_unsafe, [category_codes])`` from a Llama Guard completion."""
        if not content:
            return False, []
        lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
        if not lines or not lines[0].lower().startswith("unsafe"):
            return False, []
        codes: list[str] = []
        if len(lines) > 1:
            for token in lines[1].replace(",", " ").split():
                code = token.strip().upper()
                if code:
                    codes.append(code)
        return True, codes

    async def _classify(self, messages: Sequence[dict], role_to_check: str) -> tuple[bool, list[str]]:
        prompt: Final = self._build_prompt(messages, role_to_check)
        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            api_base=self.api_base,
            api_key=self.api_key,
            temperature=0.0,
            max_tokens=20,
        )
        content: str | None = None
        if isinstance(response, ModelResponse) and response.choices:
            choice = response.choices[0]
            if isinstance(choice, Choices):
                content = choice.message.content
        return self._parse_response(content)

    def _raise_violation(self, codes: Sequence[str]) -> None:
        named = [f"{code} ({self.categories[code]})" if code in self.categories else code for code in codes]
        detail = ", ".join(named) if named else "unspecified category"
        verbose_proxy_logger.info("Llama Guard: unsafe content detected, categories=%s", named)
        raise ProxyException(
            message=f"Violated Llama Guard content policy. Categories: {detail}",
            type="invalid_request_error",
            param=None,
            code=400,
            openai_code="content_policy_violation",
        )

    async def _guard_input(self, data: dict, event_type: GuardrailEventHooks) -> dict:
        if not self.should_run_guardrail(data, event_type):
            return data
        messages = data.get("messages") or []
        if not messages:
            return data
        try:
            is_unsafe, codes = await self._classify(messages, _USER_ROLE)
        except ProxyException:
            raise
        except Exception as e:  # noqa: BLE001 - fail open so a classifier outage does not drop traffic
            verbose_proxy_logger.warning("Llama Guard input classification failed, failing open: %s", e)
            return data
        if is_unsafe:
            self._raise_violation(codes)
        return data

    # ------------------------------------------------------------------ #
    # Guardrail hooks
    # ------------------------------------------------------------------ #
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> dict | None:
        return await self._guard_input(data, GuardrailEventHooks.pre_call)

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ) -> dict:
        return await self._guard_input(data, GuardrailEventHooks.during_call)

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ) -> LLMResponseTypes:
        if not self.should_run_guardrail(data, GuardrailEventHooks.post_call):
            return response
        if not (isinstance(response, ModelResponse) and response.choices):
            return response
        base_messages = list(data.get("messages") or [])
        for choice in response.choices:
            if not isinstance(choice, Choices):
                continue
            output_text = choice.message.content or ""
            if not output_text:
                continue
            conversation = base_messages + [{"role": "assistant", "content": output_text}]
            try:
                is_unsafe, codes = await self._classify(conversation, _AGENT_ROLE)
            except ProxyException:
                raise
            except Exception as e:  # noqa: BLE001 - fail open so a classifier outage does not drop traffic
                verbose_proxy_logger.warning("Llama Guard output classification failed, failing open: %s", e)
                continue
            if is_unsafe:
                self._raise_violation(codes)
        return response
