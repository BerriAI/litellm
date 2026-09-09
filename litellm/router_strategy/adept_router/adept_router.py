"""
ADEPT (Adaptive Deployment via Prompt Templates) Router.

Extracts a structural skeleton from each single-turn prompt, hashes it with the system prompt
for per-tool isolation, and routes to a task-specific SLM once one has been trained. Until then,
traffic falls back to the default model while conversations accumulate as training data.
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
    model_config = ConfigDict(extra="ignore")
    type: str | None = None
    text: str | None = None


class _UsageEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore")
    usage: Usage | None = None


_MessageList: TypeAlias = list[dict[str, object]]
_MESSAGES_ADAPTER: Final = TypeAdapter(_MessageList)
_CONTENT_BLOCKS_ADAPTER: Final = TypeAdapter(list[_MessageContentBlock])
_METADATA_ADAPTER: Final = TypeAdapter(dict[str, object])


class AdeptRouter(CustomLogger):
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
        self.pg_url = pg_url
        self.litellm_router_instance = litellm_router_instance
        self.template_router = AdeptTemplateRouter(
            model_name=model_name,
            litellm_router_instance=litellm_router_instance,
            pg_url=pg_url,
            tag_prefix=tag_prefix,
            conversations_threshold=conversations_threshold,
            trainer_url=trainer_url,
        )
        self._seed_config = seed_config
        self._seeded = not seed_config
        self._seed_lock = asyncio.Lock()
        self._seed_task: asyncio.Task[None] | None = None  # mutable-ok: task handle rotates on restart

    def _kick_off_seed(self) -> None:
        if self._seeded:
            return
        existing: Final = self._seed_task
        if existing is not None and not existing.done():
            return
        try:
            loop: Final = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._seed_task = loop.create_task(self._run_seed())

    def close(self) -> None:
        """Cancel background tasks so a retired router cannot mutate state or write to the DB."""
        if self._seed_task is not None:
            self._seed_task.cancel()
            self._seed_task = None
        self.template_router.stop_refresh()

    async def _run_seed(self) -> None:
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

        self._kick_off_seed()

        message_content: Final = self._extract_user_text(messages)
        if not message_content:
            await self._authorize_routed_model(request_kwargs, self.default_model)
            return PreRoutingHookResponse(model=self.default_model, messages=messages)

        system_prompt: Final = self._extract_system_prompt(messages)

        template_match: Final = await self.template_router.route(message_content, system_prompt)

        target: Final = template_match.get("target_model") if template_match is not None else None
        routed_model: Final = target or self.default_model
        routed_to_slm: Final = bool(target)
        await self._authorize_routed_model(request_kwargs, routed_model)
        if template_match is not None:
            verbose_router_logger.info(
                "AdeptRouter: matched template %s, routing to %s",
                template_match.get("template_id"),
                routed_model,
            )
        else:
            verbose_router_logger.info("AdeptRouter: no template match, falling back to %s", self.default_model)

        for md_key in ("metadata", "litellm_metadata"):
            candidate = request_kwargs.get(md_key)
            if isinstance(candidate, dict):
                candidate["adept_routed_to_slm"] = routed_to_slm
                break

        return PreRoutingHookResponse(model=routed_model, messages=messages)

    async def _authorize_routed_model(self, request_kwargs: Mapping[str, object], routed_model: str) -> None:
        """Re-run proxy model-access check against the swapped target so a caller cannot be silently
        upgraded from the ADEPT alias to a SLM they lack access to."""
        if routed_model == self.model_name:
            return
        auth_from_litellm_params: Final = self._read_request_metadata(
            request_kwargs.get("litellm_params"), "user_api_key_auth"
        )
        auth_obj: Final = (
            auth_from_litellm_params
            if auth_from_litellm_params is not None
            else self._extract_user_api_key_auth(request_kwargs)
        )
        if auth_obj is None:
            return
        try:
            from litellm.proxy._types import UserAPIKeyAuth
            from litellm.proxy.auth.auth_checks import can_key_call_resolved_model
        except ImportError:
            return
        if not isinstance(auth_obj, UserAPIKeyAuth):
            return
        await can_key_call_resolved_model(
            model=routed_model,
            llm_model_list=self.litellm_router_instance.model_list,
            valid_token=auth_obj,
            llm_router=self.litellm_router_instance,
        )

    @staticmethod
    def _extract_user_api_key_auth(request_kwargs: Mapping[str, object]) -> object:
        for md_key in ("metadata", "litellm_metadata"):
            candidate = request_kwargs.get(md_key)
            if isinstance(candidate, Mapping):
                value = candidate.get("user_api_key_auth")
                if value is not None:
                    return value
        return None

    @staticmethod
    def _read_request_metadata(litellm_params: object, key: str) -> object:
        try:
            params: Final = _METADATA_ADAPTER.validate_python(litellm_params)
        except ValidationError:
            return None
        for md_key in ("metadata", "litellm_metadata"):
            try:
                nested = _METADATA_ADAPTER.validate_python(params.get(md_key))  # rebind-ok: loop rebinds per iteration
            except ValidationError:
                continue
            value = nested.get(key)  # rebind-ok: loop rebinds per iteration
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
        # Callback fires for every request; gate on model_group to only log our own.
        lp_raw: Final = kwargs.get("litellm_params")
        request_model_group: Final = self._read_request_metadata(lp_raw, "model_group")
        if request_model_group != self.model_name:
            return

        try:
            messages: Final = _MESSAGES_ADAPTER.validate_python(kwargs.get("messages"))
        except ValidationError:
            return
        if not messages:
            return
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
