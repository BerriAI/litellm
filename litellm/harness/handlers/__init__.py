"""Handlers run a harness config: CLI runtimes as subprocesses, Deep Agents in-process."""

from __future__ import annotations

from litellm.harness.errors import HarnessError
from litellm.harness.handlers.base import BaseHarnessHandler
from litellm.harness.types import Harness, require_harness
from litellm.llms.base_llm.harness.transformation import (
    BaseCLIHarnessConfig,
    BaseHarnessConfig,
)
from litellm.utils import ProviderConfigManager


def get_harness_config(harness: Harness) -> BaseHarnessConfig:
    config = ProviderConfigManager.get_provider_harness_config(require_harness(harness))
    if config is None:
        raise HarnessError(f"No harness config registered for Harness.{harness.name}")
    return config


def get_harness_handler(config: BaseHarnessConfig) -> BaseHarnessHandler:
    """The handler that knows how to run this kind of config."""
    if isinstance(config, BaseCLIHarnessConfig):
        from litellm.harness.handlers.cli_handler import CLIHarnessHandler

        return CLIHarnessHandler(config)
    if config.harness is Harness.DEEPAGENTS:
        from litellm.harness.handlers.deepagents_handler import DeepAgentsHandler

        return DeepAgentsHandler(config)
    raise HarnessError(f"No handler for Harness.{config.harness.name}")


__all__ = ("BaseHarnessHandler", "get_harness_config", "get_harness_handler")
