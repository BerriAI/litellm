"""
Auto-Routing Strategy that works with a Semantic Router Config
"""

from typing import TYPE_CHECKING, Any, Final, Optional

from litellm._logging import verbose_router_logger
from litellm.constants import DEFAULT_AUTO_ROUTER_MAX_INPUT_CHARS
from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from semantic_router.routers import SemanticRouter
    from semantic_router.routers.base import Route

    from litellm.router import Router
    from litellm.types.router import PreRoutingHookResponse
else:
    Router = Any
    PreRoutingHookResponse = Any
    Route = Any
    SemanticRouter = Any


class AutoRouter(CustomLogger):
    DEFAULT_AUTO_SYNC_VALUE = "local"

    def __init__(
        self,
        model_name: str,
        default_model: str,
        embedding_model: str,
        litellm_router_instance: "Router",
        auto_router_config_path: str | None = None,
        auto_router_config: str | None = None,
        max_input_chars: int = DEFAULT_AUTO_ROUTER_MAX_INPUT_CHARS,
    ):
        """
        Auto-Router class that uses a semantic router to route requests to the appropriate model.

        Args:
            model_name: The name of the model to use for the auto-router. eg. if model = "auto-router1" then us this router.
            auto_router_config_path: The path to the router config file.
            auto_router_config: The config to use for the auto-router. You can either use this or auto_router_config_path, not both.
            default_model: The default model to use if no route is found.
            embedding_model: The embedding model to use for the auto-router.
            litellm_router_instance: The instance of the LiteLLM Router.
            max_input_chars: Longest prompt, in characters, handed to the embedding model. Longer
                prompts are cut to this length so an embedding context window far smaller than the
                routed model's never fails the request.
        """
        from semantic_router.routers import SemanticRouter

        self.auto_router_config_path: str | None = auto_router_config_path
        self.auto_router_config: str | None = auto_router_config
        self.auto_sync_value = self.DEFAULT_AUTO_SYNC_VALUE
        self.loaded_routes: list[Route] = self._load_semantic_routing_routes()
        self.routelayer: SemanticRouter | None = None
        self.default_model = default_model
        self.embedding_model: str = embedding_model
        self.max_input_chars: int = max_input_chars
        self.litellm_router_instance: Router = litellm_router_instance

    def _load_semantic_routing_routes(self) -> list[Route]:
        from semantic_router.routers import SemanticRouter

        if self.auto_router_config_path:
            return SemanticRouter.from_json(self.auto_router_config_path).routes
        elif self.auto_router_config:
            return self._load_auto_router_routes_from_config_json()
        else:
            raise ValueError("No router config provided")

    def _load_auto_router_routes_from_config_json(self) -> list[Route]:
        import json

        from semantic_router.routers.base import Route

        if self.auto_router_config is None:
            raise ValueError("No auto router config provided")
        auto_router_routes: Final[list[Route]] = []
        loaded_config: Final = json.loads(self.auto_router_config)
        for route in loaded_config.get("routes", []):
            auto_router_routes.append(
                Route(
                    name=route.get("name"),
                    description=route.get("description"),
                    utterances=route.get("utterances", []),
                    score_threshold=route.get("score_threshold"),
                )
            )
        return auto_router_routes

    @staticmethod
    def _extract_text_from_messages(messages: list[dict[str, Any]]) -> str:
        """
        Extract text content from the last user message for routing.

        Handles tool-call conversations (where the last message may be an
        assistant or tool message with non-string content) and multimodal
        messages (where content is a list of content blocks).
        """
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

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: dict,
        messages: list[dict[str, Any]] | None = None,
        input: str | list | None = None,
        specific_deployment: bool | None = False,
    ) -> Optional["PreRoutingHookResponse"]:
        """
        This hook is called before the routing decision is made.

        Used for the litellm auto-router to modify the request before the routing decision is made.
        """
        from semantic_router.routers import SemanticRouter

        from litellm.router_strategy.auto_router.litellm_encoder import (
            LiteLLMRouterEncoder,
        )
        from litellm.types.router import PreRoutingHookResponse

        if messages is None:
            # do nothing, return same inputs
            return None

        routelayer = self.routelayer
        if routelayer is None:
            #######################
            # Create the route layer
            #######################
            routelayer = SemanticRouter(
                routes=self.loaded_routes,
                encoder=LiteLLMRouterEncoder(
                    litellm_router_instance=self.litellm_router_instance,
                    model_name=self.embedding_model,
                    max_input_chars=self.max_input_chars,
                ),
                auto_sync=self.auto_sync_value,
            )
            self.routelayer = routelayer

        message_content: Final = self._extract_text_from_messages(messages)
        route_name: Final = self._matched_route_name(routelayer, message_content)

        return PreRoutingHookResponse(
            model=route_name or self.default_model,
            messages=messages,
        )

    def _matched_route_name(self, routelayer: "SemanticRouter", text: str) -> str | None:
        """Name of the route `text` matches, or None when nothing matched or the match failed.

        The route layer embeds `text` to compare it against the routes, and that embedding call can
        fail (context limit, timeout, provider error). Choosing a model is a routing decision, so a
        failure here falls back to the default model rather than failing the user's request.
        """
        from semantic_router.schema import RouteChoice

        try:
            route_choice: Final = routelayer(text=text)
        except Exception as e:  # noqa: BLE001 -- the embedding call behind the route layer can fail many ways (context limit, timeout, provider/network error); none of them may fail the request
            verbose_router_logger.warning(
                "AutoRouter: semantic routing failed (%s), falling back to default model %s", e, self.default_model
            )
            return None
        verbose_router_logger.debug("route_choice: %s", route_choice)
        if isinstance(route_choice, RouteChoice):
            return route_choice.name
        if isinstance(route_choice, list) and route_choice:
            return route_choice[0].name
        return None
