"""
Auto-Routing Strategy that works with a Semantic Router Config
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from semantic_router.routers.base import Route

    from litellm.router import Router
    from litellm.types.router import PreRoutingHookResponse
else:
    Router = Any
    PreRoutingHookResponse = Any
    Route = Any


class AutoRouter(CustomLogger):
    DEFAULT_AUTO_SYNC_VALUE = "local"

    def __init__(
        self,
        model_name: str,
        default_model: str,
        embedding_model: str,
        litellm_router_instance: "Router",
        auto_router_config_path: Optional[str] = None,
        auto_router_config: Optional[str] = None,
        savings_baseline_model: str | None = None,
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
            savings_baseline_model: Overrides the counterfactual model the dashboard measures savings against; derived from this router's own candidates when unset.
        """
        from semantic_router.routers import SemanticRouter

        self.auto_router_config_path: Optional[str] = auto_router_config_path
        self.auto_router_config: Optional[str] = auto_router_config
        self.auto_sync_value = self.DEFAULT_AUTO_SYNC_VALUE
        self.loaded_routes: List[Route] = self._load_semantic_routing_routes()
        self.routelayer: Optional[SemanticRouter] = None
        self.default_model = default_model
        self.embedding_model: str = embedding_model
        self.litellm_router_instance: "Router" = litellm_router_instance
        self.configured_savings_baseline_model: str | None = savings_baseline_model
        self._derived_savings_baseline_model: str | None = None
        self._savings_baseline_derived = False

    def _deployment_model(self, index: int) -> str | None:
        """The model a deployment actually calls."""
        params = self.litellm_router_instance.model_list[index].get("litellm_params")
        return params.get("model") if isinstance(params, dict) else None

    def _models_for_group(self, group_name: str) -> tuple[str, ...]:
        """The models a route's model group actually calls, or the name itself when the
        parent router has no deployment under it."""
        indices = self.litellm_router_instance.model_name_to_deployment_indices.get(group_name)
        if not indices:
            return (group_name,)
        return tuple(model for index in indices if (model := self._deployment_model(index)))

    def _candidate_models(self) -> tuple[str, ...]:
        """Every model this router can route to, as pricable model names.

        Routes name the router's own model groups rather than models, so each is
        resolved through the parent router's deployments before anything is priced.
        """
        group_names = frozenset(
            name for name in (*(route.name for route in self.loaded_routes), self.default_model) if name
        )
        return tuple(model for group_name in group_names for model in self._models_for_group(group_name))

    @staticmethod
    def _priced_candidate(model: str) -> tuple[float, float, str] | None:
        """``(output_rate, input_rate, model)``, or ``None`` when the model has no pricing."""
        import litellm

        try:
            info = litellm.get_model_info(model=model)
        except Exception as e:  # noqa: BLE001  # unmapped candidates simply cannot be the baseline
            verbose_router_logger.debug("auto-router savings: no pricing for candidate %s (%s)", model, e)
            return None
        return (info.get("output_cost_per_token") or 0.0, info.get("input_cost_per_token") or 0.0, model)

    def _most_expensive_candidate(self) -> str | None:
        """The priciest candidate by output rate, input rate breaking the tie."""
        priced = tuple(
            candidate for model in self._candidate_models() if (candidate := self._priced_candidate(model)) is not None
        )
        if not priced:
            verbose_router_logger.debug("auto-router savings: no priceable candidates; savings driver disabled")
            return None
        return max(priced)[2]

    @property
    def savings_baseline_model(self) -> str | None:
        """The model this router's savings are measured against.

        Without the router a deployment has to pick one model, and it has to be one that
        can carry the hardest request, so the counterfactual is the priciest model this
        router could have chosen. Deriving it from the router's own candidates keeps it
        honest: a fixed flagship credits savings against a model the operator would
        never have run, and drifts the moment the routes change.

        Resolved lazily and cached, because the parent router's deployments are still
        being assembled while this router is constructed. ``None`` when nothing can be
        priced, which zeroes the driver rather than inventing a baseline.
        """
        if self.configured_savings_baseline_model:
            return self.configured_savings_baseline_model
        if not self._savings_baseline_derived:
            self._derived_savings_baseline_model = self._most_expensive_candidate()
            self._savings_baseline_derived = True
        return self._derived_savings_baseline_model

    def _load_semantic_routing_routes(self) -> List[Route]:
        from semantic_router.routers import SemanticRouter

        if self.auto_router_config_path:
            return SemanticRouter.from_json(self.auto_router_config_path).routes
        elif self.auto_router_config:
            return self._load_auto_router_routes_from_config_json()
        else:
            raise ValueError("No router config provided")

    def _load_auto_router_routes_from_config_json(self) -> List[Route]:
        import json

        from semantic_router.routers.base import Route

        if self.auto_router_config is None:
            raise ValueError("No auto router config provided")
        auto_router_routes: List[Route] = []
        loaded_config = json.loads(self.auto_router_config)
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
    def _extract_text_from_messages(messages: List[Dict[str, Any]]) -> str:
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
        request_kwargs: Dict,
        messages: Optional[List[Dict[str, Any]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ) -> Optional["PreRoutingHookResponse"]:
        """
        This hook is called before the routing decision is made.

        Used for the litellm auto-router to modify the request before the routing decision is made.
        """
        from semantic_router.routers import SemanticRouter
        from semantic_router.schema import RouteChoice

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
                ),
                auto_sync=self.auto_sync_value,
            )
            self.routelayer = routelayer

        message_content = self._extract_text_from_messages(messages)
        route_choice: Optional[Union[RouteChoice, List[RouteChoice]]] = routelayer(text=message_content)
        verbose_router_logger.debug(f"route_choice: {route_choice}")
        if isinstance(route_choice, RouteChoice):
            model = route_choice.name or self.default_model
        elif isinstance(route_choice, list):
            model = route_choice[0].name or self.default_model

        return PreRoutingHookResponse(
            model=model,
            messages=messages,
            savings_baseline_model=self.savings_baseline_model,
        )
