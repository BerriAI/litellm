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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_strategy.adept_router.config import DEFAULT_CONVERSATIONS_THRESHOLD
from litellm.router_strategy.adept_router.template.implementation.adept_template_router import (
    AdeptTemplateRouter,
)

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.types.router import PreRoutingHookResponse
else:
    Router = Any
    PreRoutingHookResponse = Any


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
        trainer_url: Optional[str] = None,
        seed_config: Optional[List[Dict[str, Any]]] = None,
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
        if seed_config:
            self._seed_templates(seed_config)

    def _seed_templates(self, seed_config: List[Dict[str, Any]]) -> None:
        """Pre-populate templates from a seed config list."""
        from uuid import uuid4

        router_id = self.template_router.get_router_id()
        for entry in seed_config:
            description = entry.get("description", "")
            target_model = entry.get("target_model", self.default_model)
            if not description:
                verbose_router_logger.warning(
                    f"AdeptRouter: seed_config entry missing 'description', skipping: "
                    f"{str(entry)[:100]}"
                )
                continue
            masked = self.template_router._normalize_text(description)
            masked = self.template_router._mask_text(masked)
            # Use the shared hash function so seeding stays consistent with live routing.
            template_hash = AdeptTemplateRouter._hash_template(masked)
            existing = self.template_router.template_store.match_by_hash(
                template_hash, router_id
            )
            if existing is None:
                self.template_router.template_store.store_template(
                    template_id=str(uuid4()),
                    template=masked,
                    template_hash=template_hash,
                    target_model=target_model,
                    router_id=router_id,
                )
                verbose_router_logger.info(
                    f"AdeptRouter: seeded template for target_model={target_model}"
                )

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: Dict[str, Any],
        messages: Optional[List[Dict[str, Any]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ) -> Optional["PreRoutingHookResponse"]:
        from litellm.types.router import PreRoutingHookResponse

        if messages is None:
            return None

        message_content = self._extract_text_from_messages(messages)
        if not message_content:
            return PreRoutingHookResponse(model=self.default_model, messages=messages)

        system_prompt = self._extract_system_prompt_from_messages(messages)

        # route() uses a sync SQLAlchemy session — run in a thread to avoid blocking the event loop.
        template_match = await asyncio.to_thread(
            self.template_router.route, message_content, system_prompt
        )

        if template_match is not None:
            routed_model = template_match.get("target_model") or self.default_model
            routed_to_slm = bool(template_match.get("target_model"))
            verbose_router_logger.info(
                f"AdeptRouter: matched template {template_match.get('template_id')}, "
                f"routing to {routed_model}"
            )
        else:
            routed_model = self.default_model
            routed_to_slm = False
            verbose_router_logger.info(
                f"AdeptRouter: no template match, falling back to {self.default_model}"
            )

        # Stash routing decision so async_log_success_event can record it without re-querying.
        request_kwargs["adept_routed_to_slm"] = routed_to_slm

        return PreRoutingHookResponse(model=routed_model, messages=messages)

    async def async_log_success_event(
        self, kwargs: Dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        # Multiple ADEPT-router deployments each register an instance of this class
        # as a global LiteLLM callback. Without a model_group filter, every instance
        # would log every successful request — including requests routed through a
        # *different* ADEPT deployment — duplicating conversation rows across
        # router_ids. Gate on the requested model_group so only the router that
        # actually handled this request logs it.
        litellm_params = kwargs.get("litellm_params") or {}
        request_model_group = (
            (litellm_params.get("metadata") or {}).get("model_group")
            or (litellm_params.get("litellm_metadata") or {}).get("model_group")
        )
        if request_model_group is not None and request_model_group != self.model_name:
            return

        try:
            messages = kwargs.get("messages")
            if not isinstance(messages, list):
                return

            # Skip tool-result turns — the preceding assistant turn already captured this exchange.
            if messages and messages[-1].get("role") == "tool":
                return

            prompt = None
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    prompt = message.get("content")
                    break

            if prompt is None:
                return

            response_content = None
            if (
                response_obj
                and hasattr(response_obj, "choices")
                and response_obj.choices
            ):
                choice = response_obj.choices[0]
                response_content = (
                    choice.message.content
                    if hasattr(choice, "message")
                    else str(choice)
                )

            if response_content is None:
                return

            token_usage = None
            if response_obj and hasattr(response_obj, "usage"):
                token_usage = {
                    "prompt_tokens": getattr(response_obj.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(
                        response_obj.usage, "completion_tokens", 0
                    ),
                    "total_tokens": getattr(response_obj.usage, "total_tokens", 0),
                }

            cost_usd = kwargs.get("response_cost")
            latency_ms = (
                (end_time - start_time).total_seconds() * 1000
                if start_time and end_time
                else None
            )
            system_prompt = self._extract_system_prompt_from_messages(messages)
            routed_to_slm: Optional[bool] = kwargs.get("adept_routed_to_slm")
            actual_model: str = kwargs.get("model", "unknown")

            # store_conversation uses a sync SQLAlchemy engine — run in a thread to avoid
            # blocking the event loop. The trainer HTTP call inside also runs in this thread.
            await asyncio.to_thread(
                self.template_router.store_conversation,
                prompt,
                response_content,
                actual_model,
                token_usage,
                cost_usd,
                latency_ms,
                system_prompt,
                routed_to_slm,
            )
            verbose_router_logger.info("AdeptRouter: stored interaction.")
        except Exception:
            verbose_router_logger.exception("AdeptRouter: failed to log success event")

    @staticmethod
    def _extract_system_prompt_from_messages(
        messages: List[Dict[str, Any]]
    ) -> Optional[str]:
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content")
                return str(content) if content else None
        return None

    @staticmethod
    def _extract_text_from_messages(messages: List[Dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if content is None:
                    return ""
                if isinstance(content, list):
                    return " ".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                return str(content)
        return ""
