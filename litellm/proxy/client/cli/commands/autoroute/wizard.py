from pathlib import Path
from typing import List, Optional, Tuple

import click
import yaml
from rich.console import Console
from rich.table import Table

from .... import Client
from .config import (
    TIER_NAMES,
    AutorouteConfig,
    ConfigGenerationError,
    DiscoveredModel,
    HeuristicClassifier,
    LLMClassifier,
    NoSemanticMatching,
    SemanticMatching,
    build_generated_model_list,
    chat_models,
    embedding_models,
    parse_discovered_models,
    validate_config,
)
from .process import CONFIG_PATH


def _render_model_table(models: Tuple[DiscoveredModel, ...], prompt_label: str) -> None:
    console = Console()
    table = Table(title=f"Pick model(s) for {prompt_label}")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Model", style="magenta")
    for i, model in enumerate(models):
        table.add_row(str(i + 1), model.name)
    console.print(table)


def _parse_indices(choice: str, count: int) -> Optional[Tuple[int, ...]]:
    raw_parts = [part.strip() for part in choice.split(",") if part.strip()]
    if not raw_parts:
        return None
    indices: List[int] = []
    for part in raw_parts:
        try:
            index = int(part) - 1
        except ValueError:
            return None
        if not (0 <= index < count):
            return None
        indices.append(index)
    return tuple(dict.fromkeys(indices))


def _render_and_prompt_for_model(models: Tuple[DiscoveredModel, ...], prompt_label: str) -> str:
    _render_model_table(models, prompt_label)
    while True:
        choice = click.prompt(f"\nSelect a model for {prompt_label} by index", type=str).strip()
        indices = _parse_indices(choice, len(models))
        if indices is not None and len(indices) == 1:
            return models[indices[0]].name
        click.echo(f"Invalid selection. Please enter a single number between 1 and {len(models)}")


def _render_and_prompt_for_models(models: Tuple[DiscoveredModel, ...], prompt_label: str) -> Tuple[str, ...]:
    _render_model_table(models, prompt_label)
    while True:
        choice = click.prompt(
            f"\nSelect model(s) for {prompt_label} by index (comma-separated for multiple)", type=str
        ).strip()
        indices = _parse_indices(choice, len(models))
        if indices is not None:
            return tuple(models[i].name for i in indices)
        click.echo(f"Invalid selection. Please enter number(s) between 1 and {len(models)}, comma-separated")


def run_configure_wizard(ctx: click.Context) -> Path:
    """Discover the caller's accessible models, walk them through tier assignment, write config."""
    base_url = ctx.obj["base_url"]
    api_key = ctx.obj["api_key"]
    client = Client(base_url=base_url, api_key=api_key)

    raw_groups = client.model_groups.info()
    assert isinstance(raw_groups, list)
    discovered = parse_discovered_models(raw_groups)
    chat_pool = chat_models(discovered)
    embedding_pool = embedding_models(discovered)

    if not chat_pool:
        raise click.ClickException("Your key has no chat-capable models available on this proxy.")

    click.echo("Assign model(s) to each complexity tier (from what your key can access):")
    tiers = {tier: _render_and_prompt_for_models(chat_pool, tier) for tier in TIER_NAMES}
    default_model = tiers["MEDIUM"][0]

    classifier = HeuristicClassifier()
    if click.confirm("\nUse an LLM classifier instead of the free heuristic scorer?", default=False):
        classifier_model = _render_and_prompt_for_model(chat_pool, "LLM classifier")
        classifier = LLMClassifier(model=classifier_model)

    semantic_matching = NoSemanticMatching()
    if embedding_pool and click.confirm("\nEnable semantic keyword matching?", default=False):
        embedding_model = _render_and_prompt_for_model(embedding_pool, "semantic embeddings")
        semantic_matching = SemanticMatching(embedding_model=embedding_model)

    adaptive = click.confirm("\nEnable adaptive (bandit) selection on top of tiering?", default=False)

    config = AutorouteConfig(
        base_url=base_url,
        api_key=api_key,
        tiers=tiers,
        default_model=default_model,
        classifier=classifier,
        semantic_matching=semantic_matching,
        adaptive=adaptive,
    )
    try:
        validate_config(config, discovered)
    except ConfigGenerationError as e:
        raise click.ClickException(str(e))

    model_list = build_generated_model_list(config)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump({"model_list": model_list}, f, sort_keys=False)
    CONFIG_PATH.chmod(0o600)

    click.echo(f"\nWrote {CONFIG_PATH}")
    for tier, models in tiers.items():
        click.echo(f"  {tier}: {', '.join(models)}")
    return CONFIG_PATH


__all__ = ["run_configure_wizard"]
