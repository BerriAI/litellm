"""
Route loader for the Semantic Guard guardrail.

Loads route definitions from built-in YAML templates and custom configs,
then builds a SemanticRouter for prompt matching.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml

from litellm._logging import verbose_logger
from litellm.constants import DEFAULT_SEMANTIC_GUARD_SIMILARITY_THRESHOLD

if TYPE_CHECKING:
    from semantic_router.routers import SemanticRouter
    from semantic_router.routers.base import Route

    from litellm.router import Router


ROUTE_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "route_templates")


class SemanticGuardRouteLoader:
    """Loads route definitions from YAML templates and custom configs, builds SemanticRouter."""

    @staticmethod
    def load_builtin_template(template_name: str) -> Dict[str, Any]:
        """Load a built-in route template YAML by name."""
        file_path = os.path.join(ROUTE_TEMPLATES_DIR, f"{template_name}.yaml")
        if not os.path.exists(file_path):
            raise ValueError(
                f"SemanticGuard: unknown route template '{template_name}'. "
                f"Available templates: {SemanticGuardRouteLoader.list_builtin_templates()}"
            )
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def list_builtin_templates() -> List[str]:
        """List available built-in template names."""
        templates = []
        if os.path.isdir(ROUTE_TEMPLATES_DIR):
            for fname in os.listdir(ROUTE_TEMPLATES_DIR):
                if fname.endswith(".yaml"):
                    templates.append(fname.replace(".yaml", ""))
        return sorted(templates)

    @staticmethod
    def load_custom_routes_file(file_path: str) -> List[Dict[str, Any]]:
        """Load custom routes from a YAML file."""
        if not os.path.exists(file_path):
            raise ValueError(f"SemanticGuard: custom routes file not found: {file_path}")
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError(f"SemanticGuard: invalid custom routes file format in {file_path}")

    @classmethod
    def build_routes(
        cls,
        route_templates: Optional[List[str]],
        custom_routes_file: Optional[str],
        custom_routes: Optional[List[Dict[str, Any]]],
        global_threshold: float = DEFAULT_SEMANTIC_GUARD_SIMILARITY_THRESHOLD,
    ) -> List["Route"]:
        """Build semantic-router Route objects from templates + custom config."""
        from semantic_router.routers.base import Route

        routes: List[Route] = []

        if route_templates:
            for template_name in route_templates:
                template_data = cls.load_builtin_template(template_name)
                threshold = template_data.get("similarity_threshold", global_threshold)
                routes.append(
                    Route(
                        name=template_data["route_name"],
                        description=template_data.get("description", ""),
                        utterances=template_data["utterances"],
                        score_threshold=threshold,
                    )
                )

        if custom_routes_file:
            custom_defs = cls.load_custom_routes_file(custom_routes_file)
            for route_def in custom_defs:
                threshold = route_def.get("similarity_threshold", global_threshold)
                routes.append(
                    Route(
                        name=route_def["route_name"],
                        description=route_def.get("description", ""),
                        utterances=route_def["utterances"],
                        score_threshold=threshold,
                    )
                )

        if custom_routes:
            for route_def in custom_routes:
                threshold = route_def.get("similarity_threshold", global_threshold)
                routes.append(
                    Route(
                        name=route_def["route_name"],
                        description=route_def.get("description", ""),
                        utterances=route_def["utterances"],
                        score_threshold=threshold,
                    )
                )

        verbose_logger.info(f"SemanticGuard: built {len(routes)} routes")
        return routes

    @classmethod
    def build_semantic_router(
        cls,
        routes: List["Route"],
        litellm_router: "Router",
        embedding_model: str,
        global_threshold: float,
    ) -> "SemanticRouter":
        """Build the SemanticRouter with LiteLLMRouterEncoder."""
        from semantic_router.routers import SemanticRouter

        from litellm.router_strategy.auto_router.litellm_encoder import (
            LiteLLMRouterEncoder,
        )

        encoder = LiteLLMRouterEncoder(
            litellm_router_instance=litellm_router,
            model_name=embedding_model,
            score_threshold=global_threshold,
        )

        return SemanticRouter(
            routes=routes,
            encoder=encoder,
            auto_sync="local",
        )
