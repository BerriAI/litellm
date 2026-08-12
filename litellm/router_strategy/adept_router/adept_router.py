"""
ADEPT (Adaptive Deployment via Prompt Templates) Router.

Designed for single-turn, task-specific routing. An agent or tool sends a fixed system
prompt (the task definition) plus XML-tagged variable user content (the runtime input).
ADEPT extracts a structural skeleton from each prompt, hashes it together with the system
prompt for per-tool isolation, and routes to a task-specific SLM once one has been trained.

Until a trained SLM exists for a template, all traffic falls back to the default model
while conversations accumulate as training data in Postgres.
"""

import asyncio
import datetime
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_strategy.adept_router.config import DEFAULT_CONVERSATIONS_THRESHOLD
from litellm.router_strategy.adept_router.template.implementation.adept_template_router import (
    AdeptTemplateRouter,
)
from litellm.types.utils import ModelResponse, Usage

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.types.router import PreRoutingHookResponse
else:
    Router: Final = object
    PreRoutingHookResponse: Final = object


class _MessageContentBlock(BaseModel):
    """One block of an OpenAI multimodal message content list; non-text blocks parse with text=None."""

    model_config = ConfigDict(extra="ignore")
    type: str | None = None
    text: str | None = None


class _UsageEnvelope(BaseModel):
    """Reads the dynamically-set `usage` attribute off a ModelResponse in a typed way."""

    model_config = ConfigDict(extra="ignore")
    usage: Usage | None = None


_MessageList: TypeAlias = list[dict[str, object]]
_MESSAGES_ADAPTER: Final = TypeAdapter(_MessageList)
_CONTENT_BLOCKS_ADAPTER: Final = TypeAdapter(list[_MessageContentBlock])
_METADATA_ADAPTER: Final = TypeAdapter(dict[str, object])


class AdeptRouter(CustomLogger):
    """
    ADEPT routing strategy — matches incoming prompts to known templates via
    SHA-256 hashing of the masked template string, with no external vector DB.
    """

    def __init__(
        self,
        model_name: str,
        default_model: str,
        litellm_router_instance: "Router",
        pg_url: str,
        tag_prefix: str = "",
        conversations_threshold: int = DEFAULT_CONVERSATIONS_THRESHOLD,
        trainer_url: str | None = None,
        seed_config: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.default_model = default_model
        self.litellm_router_instance = litellm_router_instance
        self.template_router = AdeptTemplateRouter(
            model_name=model_name,
            litellm_router_instance=litellm_router_instance,
            pg_url=pg_url,
            tag_prefix=tag_prefix,
            conversations_threshold=conversations_threshold,
            trainer_url=trainer_url,
        )
        # Seeding hits the store, which is async, so it runs lazily on first use (the store
        # connects lazily too). __init__ stays sync for the router's construction path.
        self._seed_config = seed_config
        self._seeded = not seed_config
        self._seed_lock = asyncio.Lock()

    async def _ensure_seeded(self) -> None:
        """Pre-populate templates from seed_config once, on first use."""
        if self._seeded:
            return
        async with self._seed_lock:
            if self._seeded:
                return
            for entry in self._seed_config or ():
                description = entry.get("description", "")
                target_model = entry.get("target_model", self.default_model)
                if not description:
                    verbose_router_logger.warning(
                        "AdeptRouter: seed_config entry missing 'description', skipping: %s", str(entry)[:100]
                    )
                    continue
                if await self.template_router.seed_template(str(description), str(target_model)):
                    verbose_router_logger.info("AdeptRouter: seeded template for target_model=%s", target_model)
            self._seeded = True

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: Mapping[str, object],
        messages: _MessageList | None = None,
        input: str | Sequence[object] | None = None,
        specific_deployment: bool | None = False,
    ) -> "PreRoutingHookResponse | None":
        from litellm.types.router import PreRoutingHookResponse

        if messages is None:
            return None

        await self._ensure_seeded()

        message_content: Final = self._extract_user_text(messages)
        if not message_content:
            return PreRoutingHookResponse(model=self.default_model, messages=messages)

        system_prompt: Final = self._extract_system_prompt(messages)

        template_match: Final = await self.template_router.route(message_content, system_prompt)

        target: Final = template_match.get("target_model") if template_match is not None else None
        routed_model: Final = target or self.default_model
        routed_to_slm: Final = bool(target)
        if template_match is not None:
            verbose_router_logger.info(
                "AdeptRouter: matched template %s, routing to %s",
                template_match.get("template_id"),
                routed_model,
            )
        else:
            verbose_router_logger.info("AdeptRouter: no template match, falling back to %s", self.default_model)

        # Stash routing decision so async_log_success_event can record it without re-querying.
        for md_key in ("metadata", "litellm_metadata"):
            candidate = request_kwargs.get(md_key)
            if isinstance(candidate, dict):
                candidate["adept_routed_to_slm"] = routed_to_slm
                break

        return PreRoutingHookResponse(model=routed_model, messages=messages)

    @staticmethod
    def _read_request_metadata(litellm_params: object, key: str) -> object:
        """Read a key from the request's metadata / litellm_metadata dict, if present."""
        try:
            params = _METADATA_ADAPTER.validate_python(litellm_params)
        except ValidationError:
            return None
        for md_key in ("metadata", "litellm_metadata"):
            try:
                nested = _METADATA_ADAPTER.validate_python(params.get(md_key))
            except ValidationError:
                continue
            value = nested.get(key)
            if value is not None:
                return value
        return None

    async def async_log_success_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: ModelResponse,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        # Multiple ADEPT-router deployments each register an instance of this class
        # as a global LiteLLM callback. Without a model_group filter, every instance
        # would log every successful request — including requests routed through a
        # *different* ADEPT deployment — duplicating conversation rows across
        # router_ids. Gate on the requested model_group so only the router that
        # actually handled this request logs it.
        lp_raw: Final = kwargs.get("litellm_params")
        request_model_group: Final = self._read_request_metadata(lp_raw, "model_group")
        # Only log requests routed through *this* ADEPT model. A request whose model_group is
        # absent or belongs to a different deployment is not ours, so skipping it avoids both
        # storing non-ADEPT traffic and duplicating rows when several ADEPT deployments exist.
        if request_model_group != self.model_name:
            return

        try:
            messages: Final = _MESSAGES_ADAPTER.validate_python(kwargs.get("messages"))
        except ValidationError:
            return
        if not messages:
            return
        # Skip tool-result turns — the preceding assistant turn already captured this exchange.
        if messages[-1].get("role") == "tool":
            return

        try:
            prompt_text: Final = self._extract_user_text(messages)
            if not prompt_text:
                return

            usage: Final = _UsageEnvelope.model_validate(response_obj, from_attributes=True).usage
            if usage is None:
                return

            response_content: Final = self._response_text(response_obj)
            if response_content is None:
                return

            token_usage: Final[dict[str, object]] = {  # mutable-ok: JSON payload persisted to a Postgres JSON column
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }

            cost_raw: Final = kwargs.get("response_cost")
            cost_usd: Final = cost_raw if isinstance(cost_raw, (int, float)) else None
            latency_ms: Final = (end_time - start_time).total_seconds() * 1000
            system_prompt: Final = self._extract_system_prompt(messages)
            routed_to_slm_raw: Final = self._read_request_metadata(lp_raw, "adept_routed_to_slm")
            routed_to_slm: Final = routed_to_slm_raw if isinstance(routed_to_slm_raw, bool) else None
            actual_model: Final = str(kwargs.get("model", "unknown"))

            await self.template_router.store_conversation(
                prompt_text,
                response_content,
                actual_model,
                token_usage,
                cost_usd,
                latency_ms,
                system_prompt,
                routed_to_slm,
            )
            verbose_router_logger.info("AdeptRouter: stored interaction.")
        except (AttributeError, KeyError, TypeError, ValueError):
            verbose_router_logger.exception("AdeptRouter: failed to log success event")

    @staticmethod
    def _response_text(response_obj: ModelResponse) -> str | None:
        if not response_obj.choices:
            return None
        choice: Final = response_obj.choices[0]
        return choice.message.content

    @staticmethod
    def _content_to_text(content: object) -> str:
        """Flatten a message's content (string or OpenAI content-block list) to plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        try:
            blocks: Final = _CONTENT_BLOCKS_ADAPTER.validate_python(content)
        except ValidationError:
            return str(content)
        return " ".join(block.text or "" for block in blocks if block.type == "text")

    @staticmethod
    def _extract_system_prompt(messages: Sequence[Mapping[str, object]]) -> str | None:
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content")
                return str(content) if content else None
        return None

    @staticmethod
    def _extract_user_text(messages: Sequence[Mapping[str, object]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return AdeptRouter._content_to_text(msg.get("content"))
        return ""
