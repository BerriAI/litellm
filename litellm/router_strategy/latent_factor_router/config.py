"""
Configuration for the LatentFactorRouter LiteLLM integration.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LatentFactorRouterConfig(BaseModel):
    """
    Configuration for LatentFactorRouterLiteLLM.

    Prerequisites:
      - A trained artefacts bundle (.pkl) must exist at artefacts_path.
      - The SuperClaw embedding server must be running before the first call
        (default: http://127.0.0.1:18104/v1). Configure via yaml_path YAML.
      - Model names in the artefacts (e.g. "gpt-4o") must match model_name
        entries in LiteLLM's model_list. Mismatches cause silent fallback.
      - The SuperClaw repo root must be on sys.path so that
        `custom_routers.latentfactorrouter.router` is importable.
    """

    model_config = ConfigDict(extra="allow")

    artefacts_path: str = Field(
        ...,
        description=(
            "Absolute path to the trained LatentFactorRouter artefacts bundle (.pkl). "
            "Produced by LatentFactorTrainer.train()."
        ),
    )
    yaml_path: str = Field(
        ...,
        description=(
            "Absolute path to the SuperClaw router YAML config. "
            "Used to initialise the embedder (base_url, api_key, model_name, etc.)."
        ),
    )
    fallback_model: Optional[str] = Field(
        default=None,
        description=(
            "LiteLLM model_name to use when routing fails (artefacts missing, "
            "embedding server down, no prediction). If None, returns None and lets "
            "LiteLLM fall back to its default routing strategy."
        ),
    )
    top_k: int = Field(
        default=1,
        description=(
            "Number of top candidate models to return from route_single(). "
            "Currently only the first (index 0) is used for routing."
        ),
    )
