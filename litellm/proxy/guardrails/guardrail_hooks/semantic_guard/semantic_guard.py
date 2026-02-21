"""
Semantic Guard — embedding-based prompt injection detection.

Uses semantic-router to match user prompts against known attack patterns
via embedding similarity. Smarter than regex (understands intent), lighter
than an LLM call (~20-50ms per request for embedding).
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from litellm._logging import verbose_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy.guardrails.guardrail_hooks.semantic_guard.route_loader import (
    SemanticGuardRouteLoader,
)
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.utils import CallTypes

try:
    from fastapi.exceptions import HTTPException
except ImportError:
    HTTPException = None  # type: ignore

if TYPE_CHECKING:
    from semantic_router.routers import SemanticRouter

    from litellm.caching import DualCache
    from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
    from litellm.router import Router


class SemanticGuardrail(CustomGuardrail):
    """
    Semantic matching guardrail that blocks requests matching known-bad patterns
    using embedding similarity via semantic-router.

    Unlike regex, this understands intent:
    - "how to make a bomb?" -> may match harmful route (BLOCKED)
    - "tell me the spelling of bomb" -> does NOT match (ALLOWED)
    """

    def __init__(
        self,
        guardrail_name: str,
        llm_router: "Router",
        embedding_model: str,
        similarity_threshold: float,
        route_templates: Optional[List[str]] = None,
        custom_routes_file: Optional[str] = None,
        custom_routes: Optional[List[Dict[str, Any]]] = None,
        on_flagged_action: str = "block",
        event_hook: Optional[Union[GuardrailEventHooks, List[GuardrailEventHooks], Mode]] = None,
        default_on: bool = False,
        **kwargs,
    ):
        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=[
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
            ],
            event_hook=event_hook or GuardrailEventHooks.pre_call,
            default_on=default_on,
            **kwargs,
        )

        self.guardrail_provider = "semantic_guard"
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.on_flagged_action = on_flagged_action
        self.llm_router = llm_router

        routes = SemanticGuardRouteLoader.build_routes(
            route_templates=route_templates,
            custom_routes_file=custom_routes_file,
            custom_routes=custom_routes,
            global_threshold=similarity_threshold,
        )

        if not routes:
            raise ValueError(
                "SemanticGuardrail: no routes configured. "
                "Provide route_templates or custom_routes."
            )

        self.semantic_router: "SemanticRouter" = SemanticGuardRouteLoader.build_semantic_router(
            routes=routes,
            litellm_router=llm_router,
            embedding_model=embedding_model,
            global_threshold=similarity_threshold,
        )

        self.route_count = len(routes)
        verbose_logger.info(
            f"SemanticGuardrail '{guardrail_name}' initialized with {self.route_count} routes, "
            f"embedding_model={embedding_model}, threshold={similarity_threshold}"
        )

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: "DualCache",
        data: dict,
        call_type: str,
    ):
        """Check user messages against semantic routes before LLM call."""
        messages = self.get_guardrails_messages_for_call_type(
            call_type=CallTypes(call_type), data=data
        )
        if not messages:
            return None

        user_text = _extract_user_text(messages)
        if not user_text:
            return None

        route_choice = _get_top_route_choice(self.semantic_router(text=user_text))
        if route_choice is not None and route_choice.name:
            _handle_match(
                guardrail=self,
                route_name=route_choice.name,
                similarity_score=getattr(route_choice, "similarity_score", None),
                user_text=user_text,
                data=data,
            )

        return None

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: "UserAPIKeyAuth",
        response,
    ):
        """Optionally check LLM response for attack patterns."""
        response_text = _extract_response_text(response)
        if not response_text:
            return response

        route_choice = _get_top_route_choice(self.semantic_router(text=response_text))
        if route_choice is not None and route_choice.name:
            _handle_match(
                guardrail=self,
                route_name=route_choice.name,
                similarity_score=getattr(route_choice, "similarity_score", None),
                user_text=response_text,
                data=data,
            )

        return response


def _get_top_route_choice(result: Any) -> Any:
    """Extract the top RouteChoice from SemanticRouter result.

    SemanticRouter.__call__ can return RouteChoice or List[RouteChoice].
    """
    if result is None:
        return None
    if isinstance(result, list):
        return result[0] if result else None
    return result


def _extract_user_text(messages: List) -> str:
    """Extract the latest user message text."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
    return ""


def _extract_response_text(response: Any) -> str:
    """Extract text from LLM response object."""
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "message") and choice.message:
            return choice.message.content or ""
    return ""


def _handle_match(
    guardrail: SemanticGuardrail,
    route_name: str,
    similarity_score: Optional[float],
    user_text: str,
    data: dict,
) -> None:
    """Block or passthrough based on config."""
    violation_msg = (
        f"Request blocked by semantic guardrail '{guardrail.guardrail_name}'. "
        f"Matched route: {route_name}"
    )

    detection_info = {
        "route_name": route_name,
        "similarity_score": similarity_score,
        "guardrail": guardrail.guardrail_name,
    }

    verbose_logger.warning(
        f"SemanticGuard match: route={route_name}, score={similarity_score}, "
        f"action={guardrail.on_flagged_action}"
    )

    if guardrail.on_flagged_action == "passthrough":
        guardrail.raise_passthrough_exception(
            violation_message=violation_msg,
            request_data=data,
            detection_info=detection_info,
        )
    else:
        raise HTTPException(  # type: ignore[reportOptionalCall]
            status_code=400,
            detail={
                "error": violation_msg,
                "route": route_name,
                "similarity_score": similarity_score,
                "type": "semantic_guard_violation",
            },
        )
