"""
Auto-Routing Strategy that works with a Semantic Router Config
"""

import json
import os
from typing import List, Literal, Optional, Union

from semantic_router.schema import RouteChoice

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import DualCache, UserAPIKeyAuth


class AutoRouter(CustomLogger):
    DEFAULT_AUTO_SYNC_VALUE = "local"
    def __init__(
        self,
        router_config_path: str,
        default_model: str
    ):  
        from semantic_router import Route
        from semantic_router.encoders import OpenAIEncoder
        from semantic_router.routers import SemanticRouter
        self.router_config_path = router_config_path
        self.auto_sync_value = self.DEFAULT_AUTO_SYNC_VALUE
        loaded_router: SemanticRouter = SemanticRouter.from_json(self.router_config_path)
        self.routelayer: SemanticRouter = SemanticRouter(
                routes=loaded_router.routes,
                encoder=loaded_router.encoder,
                auto_sync=self.auto_sync_value
            )
        self.default_model = default_model
        pass


    async def async_pre_routing_hook(
        self, 
        data: dict, call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ]):
        from semantic_router.schema import RouteChoice

        #self.routelayer.to_json("./config/router.json")
        # If the call type is embeddings, do not modify the model.
        if call_type in ["embeddings", "image_generation"]:
           #print("Call type is 'embeddings', using default behavior.")
           return data  # Return without modifying the data
        
        msg = data['messages'][-1]['content']
        route_choice: Optional[Union[RouteChoice, List[RouteChoice]]] = self.routelayer(text=msg)
        if isinstance(route_choice, RouteChoice):
            data["model"] = route_choice.name
        elif isinstance(route_choice, list):
            data["model"] = route_choice[0].name
        else:
            data["model"] = self.default_model

        return data

