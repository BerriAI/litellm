"""
LatentFactorRouter LiteLLM integration.

Wraps SuperClaw's LatentFactorRouter as a LiteLLM CustomLogger pre-routing hook,
following the same pattern as ComplexityRouter.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger

from .config import LatentFactorRouterConfig

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.types.router import PreRoutingHookResponse
else:
    Router = Any
    PreRoutingHookResponse = Any


class LatentFactorRouterLiteLLM(CustomLogger):
    """
    LiteLLM pre-routing hook backed by SuperClaw's LatentFactorRouter.

    Prerequisites:
      1. A trained artefacts bundle (.pkl) must exist at config.artefacts_path.
      2. The SuperClaw embedding server must be running before the first call
         (default: http://127.0.0.1:18104/v1). Configure via config.yaml_path YAML.
      3. Model names in the artefacts (e.g. "gpt-4o") must match model_name
         entries in LiteLLM's model_list. Mismatches cause silent fallback.
      4. The SuperClaw repo root must be on sys.path so that
         `custom_routers.latentfactorrouter.router` is importable.

    Usage:
        config = LatentFactorRouterConfig(
            artefacts_path="/path/to/artefacts.pkl",
            yaml_path="/path/to/router.yaml",
        )
        # Registered automatically via model_list prefix "auto_router/latent_factor_router"
        # See Router.init_latent_factor_router_deployment()
    """

    def __init__(
        self,
        model_name: str,
        litellm_router_instance: "Router",
        config: LatentFactorRouterConfig,
    ) -> None:
        self.model_name = model_name
        self.litellm_router_instance = litellm_router_instance
        self.config = config
        self._router: Any = None  # lazy-initialized on first call

    def _get_router(self) -> Any:
        """
        Lazily initialize the SuperClaw LatentFactorRouter.

        Returns the router instance, or None if initialization fails
        (artefacts missing, SuperClaw not importable, etc.).
        """
        if self._router is not None:
            return self._router

        try:
            from custom_routers.latentfactorrouter.router import LatentFactorRouter  # type: ignore
        except ImportError:
            verbose_router_logger.warning(
                "[LatentFactorRouterLiteLLM] SuperClaw LatentFactorRouter not importable. "
                "Ensure the SuperClaw repo root is on sys.path."
            )
            return None

        try:
            router = LatentFactorRouter(yaml_path=self.config.yaml_path)
            router.load_artefacts(self.config.artefacts_path)
            # Propagate top_k from config if the attribute exists
            if hasattr(router, "top_k"):
                router.top_k = self.config.top_k
            self._router = router
            verbose_router_logger.info(
                f"[LatentFactorRouterLiteLLM] Artefacts loaded from {self.config.artefacts_path}"
            )
            return self._router
        except Exception as e:
            verbose_router_logger.warning(
                f"[LatentFactorRouterLiteLLM] Failed to initialize router: {e}"
            )
            return None

    def _fallback_response(
        self, messages: Optional[List[Dict[str, Any]]], has_original_messages: bool
    ) -> Optional["PreRoutingHookResponse"]:
        """Return a fallback PreRoutingHookResponse if fallback_model is configured, else None."""
        from litellm.types.router import PreRoutingHookResponse

        if self.config.fallback_model:
            return PreRoutingHookResponse(
                model=self.config.fallback_model,
                messages=messages if has_original_messages else None,
            )
        return None

    def _resolve_messages(
        self,
        messages: Optional[List[Dict[str, Any]]],
        request_kwargs: Dict,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Resolve messages from the request, converting from other formats if needed.
        Copied from ComplexityRouter._resolve_messages.
        """
        if messages:
            return messages

        try:
            from litellm.litellm_core_utils.api_route_to_call_types import (
                get_call_types_for_route,
            )
            from litellm.llms import load_guardrail_translation_mappings
            from litellm.types.utils import CallTypes

            mappings = load_guardrail_translation_mappings()
            call_type: Optional[Any] = None

            route = request_kwargs.get("litellm_metadata", {}).get(
                "user_api_key_request_route"
            )
            if route:
                call_types_list = get_call_types_for_route(route)
                if call_types_list:
                    for ct in call_types_list:
                        if ct in mappings:
                            call_type = ct
                            break

            handlers_to_try: List[Any] = []
            if call_type is not None and call_type in mappings:
                handlers_to_try.append(mappings[call_type]())
            else:
                handlers_to_try.extend(handler_cls() for handler_cls in mappings.values())

            for handler in handlers_to_try:
                structured = handler.get_structured_messages(request_kwargs)
                if structured:
                    return [
                        msg if isinstance(msg, dict) else msg.model_dump()
                        for msg in structured
                    ]
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_user_message(
        messages: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Extract the last user message text from messages."""
        for msg in reversed(messages):
            role = msg.get("role", "")
            content = msg.get("content") or ""
            if isinstance(content, list):
                text_parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                content = " ".join(text_parts).strip()
            if isinstance(content, str) and content and role == "user":
                return content
        return None

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: Dict,
        messages: Optional[List[Dict[str, Any]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ) -> Optional["PreRoutingHookResponse"]:
        """
        Pre-routing hook called before the routing decision.

        Runs SuperClaw LatentFactorRouter inference and returns the best model.

        Args:
            model: The original model name requested.
            request_kwargs: The request kwargs.
            messages: The messages in the request.
            input: Optional input for Responses API or embeddings.
            specific_deployment: Whether a specific deployment was requested.

        Returns:
            PreRoutingHookResponse with the routed model, or None if no routing needed.
        """
        from litellm.types.router import PreRoutingHookResponse

        has_original_messages = messages is not None and len(messages) > 0

        try:
            resolved_messages = self._resolve_messages(messages, request_kwargs)

            if not resolved_messages:
                verbose_router_logger.debug(
                    "[LatentFactorRouterLiteLLM] No messages could be resolved, skipping routing"
                )
                return None

            user_message = self._extract_user_message(resolved_messages)

            if user_message is None:
                verbose_router_logger.debug(
                    "[LatentFactorRouterLiteLLM] No user message found, using fallback"
                )
                return self._fallback_response(messages, has_original_messages)

            router = self._get_router()
            if router is None:
                verbose_router_logger.warning(
                    "[LatentFactorRouterLiteLLM] Router unavailable, using fallback"
                )
                return self._fallback_response(messages, has_original_messages)

            result = await asyncio.to_thread(router.route_single, {"query": user_message})
            predicted_llm: Optional[str] = result.get("predicted_llm") if result else None

            if not predicted_llm:
                verbose_router_logger.warning(
                    "[LatentFactorRouterLiteLLM] No prediction returned, using fallback"
                )
                return self._fallback_response(messages, has_original_messages)

            verbose_router_logger.info(
                f"[LatentFactorRouterLiteLLM] Routed to: {predicted_llm}"
            )
            return PreRoutingHookResponse(
                model=predicted_llm,
                messages=messages if has_original_messages else None,
            )

        except Exception as e:
            verbose_router_logger.warning(
                f"[LatentFactorRouterLiteLLM] Routing failed with exception: {e}. Using fallback."
            )
            return self._fallback_response(messages, has_original_messages)
