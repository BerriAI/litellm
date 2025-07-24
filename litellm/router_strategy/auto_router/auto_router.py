"""
Auto-Routing Strategy that works with a Semantic Router Config
"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.types.router import PreRoutingHookResponse
else:
    Router = Any
    PreRoutingHookResponse = Any


class AutoRouter(CustomLogger):
    DEFAULT_AUTO_SYNC_VALUE = "local"
    def __init__(
        self,
        model_name: str,
        router_config_path: str,
        default_model: str,
        embedding_model: str,
        litellm_router_instance: "Router",
    ):  
        """
        Auto-Router class that uses a semantic router to route requests to the appropriate model.

        Args:
            model_name: The name of the model to use for the auto-router. eg. if model = "auto-router1" then us this router.
            router_config_path: The path to the router config file.
            default_model: The default model to use if no route is found.
            embedding_model: The embedding model to use for the auto-router.
            litellm_router_instance: The instance of the LiteLLM Router.
        """
        from semantic_router import Route
        from semantic_router.routers import SemanticRouter

        from litellm.router_strategy.auto_router.litellm_encoder import (
            LiteLLMRouterEncoder,
        )
        self.router_config_path = router_config_path
        self.auto_sync_value = self.DEFAULT_AUTO_SYNC_VALUE
        loaded_router: SemanticRouter = SemanticRouter.from_json(self.router_config_path)
        self.routelayer: SemanticRouter = SemanticRouter(
                routes=loaded_router.routes,
                encoder=LiteLLMRouterEncoder(
                    litellm_router_instance=litellm_router_instance,
                    model_name=embedding_model,
                ),
                auto_sync=self.auto_sync_value,
            )
        self.default_model = default_model
        pass


    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: Dict,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ) -> Optional["PreRoutingHookResponse"]:
        """
        This hook is called before the routing decision is made.

        Used for the litellm auto-router to modify the request before the routing decision is made.
        """
        from semantic_router.schema import RouteChoice

        from litellm.types.router import PreRoutingHookResponse
        if messages is None:
            # do nothing, return same inputs
            return None
        user_message: Dict[str, str] = messages[-1]
        message_content: str = user_message.get("content", "")
        route_choice: Optional[Union[RouteChoice, List[RouteChoice]]] = self.routelayer(text=message_content)
        if isinstance(route_choice, RouteChoice):
            model = route_choice.name or self.default_model
        elif isinstance(route_choice, list):
            model = route_choice[0].name or self.default_model
        
        return PreRoutingHookResponse(
            model=model,
            messages=messages,
        )

