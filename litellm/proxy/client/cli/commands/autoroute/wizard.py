import sys
from pathlib import Path

import click
import yaml
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from pydantic import JsonValue, TypeAdapter, ValidationError

from .... import Client
from .config import (
    DEFAULT_KEYWORD_TIER_RULES,
    TIER_NAMES,
    AutorouteConfig,
    ConfigGenerationError,
    DiscoveredModel,
    HeuristicClassifier,
    KeywordTierRule,
    LLMClassifier,
    NoSemanticMatching,
    SemanticMatching,
    build_generated_model_list,
    chat_models,
    embedding_models,
    master_key_from_config,
    parse_discovered_models,
    validate_config,
)
from .process import CONFIG_PATH, secure_create


def _is_interactive() -> bool:
    return sys.stdin.isatty()


def _fuzzy_pick(models: tuple[DiscoveredModel, ...], prompt_label: str, multiselect: bool) -> list[str]:
    """Type-to-filter picker over a (possibly huge) model pool, using InquirerPy's fzf-style fuzzy prompt.

    A plain numbered table + typed index does not scale past a handful of models -- proxies with
    hundreds of model groups made that interaction unusable. This lets the user narrow the pool by
    typing a substring instead of scrolling/counting.

    Assumes the caller already checked interactivity (run_configure_wizard does, once, up front) --
    checking here too would check the wrong thing under test, where InquirerPy is driven through its
    own injected input/output rather than the real process stdin.
    """
    choices = [Choice(value=model.name, name=model.name) for model in models]
    toggle_hint = "tab to toggle, " if multiselect else ""
    while True:
        result = inquirer.fuzzy(
            message=f"{prompt_label}: type to filter, {toggle_hint}enter to confirm",
            choices=choices,
            multiselect=multiselect,
            max_height="70%",
        ).execute()
        selected = result if multiselect else [result]
        if selected:
            return selected
        click.echo("Select at least one model.")


def _render_and_prompt_for_model(models: tuple[DiscoveredModel, ...], prompt_label: str) -> str:
    return _fuzzy_pick(models, prompt_label, multiselect=False)[0]


def _render_and_prompt_for_models(models: tuple[DiscoveredModel, ...], prompt_label: str) -> tuple[str, ...]:
    return tuple(_fuzzy_pick(models, prompt_label, multiselect=True))


def _parse_keywords(raw: str) -> tuple[str, ...]:
    return tuple(keyword.strip() for keyword in raw.split(",") if keyword.strip())


def _prompt_for_keyword_tier_rules() -> tuple[KeywordTierRule, ...]:
    """Let the user supply the semantic-matching keywords per tier, since matching those
    keywords against the request is the whole point of enabling it. Each prompt is prefilled
    with the built-in default, so pressing enter keeps it."""
    click.echo("\nEnter example keywords/phrases per tier (comma-separated); press enter to keep the default:")
    defaults = {rule.tier: rule.keywords for rule in DEFAULT_KEYWORD_TIER_RULES}

    def _rule_for(tier: str) -> KeywordTierRule:
        default_keywords = defaults.get(tier, ())
        raw = click.prompt(f"  {tier} keywords", default=", ".join(default_keywords), show_default=True)
        return KeywordTierRule(keywords=_parse_keywords(raw) or default_keywords, tier=tier)

    return tuple(_rule_for(tier) for tier in TIER_NAMES)


_RAW_CONFIG_ADAPTER = TypeAdapter(dict[str, JsonValue])


def _load_persisted_master_key(config_path: Path) -> str | None:
    """The master key from an existing generated config, so a rewrite carries it forward.

    Lenient on a missing or corrupt file: configure is the regeneration path, so it must
    succeed from any prior state; a key that cannot be read is simply not carried and `up`
    mints a fresh one.
    """
    if not config_path.exists():
        return None
    try:
        raw = _RAW_CONFIG_ADAPTER.validate_python(yaml.safe_load(config_path.read_text()))
    except (yaml.YAMLError, ValidationError):
        return None
    return master_key_from_config(raw)


def run_configure_wizard(ctx: click.Context) -> Path:
    """Discover the caller's accessible models, walk them through tier assignment, write config."""
    base_url = ctx.obj["base_url"]
    api_key = ctx.obj["api_key"]
    client = Client(base_url=base_url, api_key=api_key)

    raw_groups = client.model_groups.info()
    if not isinstance(raw_groups, list):
        raise click.ClickException(
            f"Unexpected response from /model_group/info: expected a list, got {type(raw_groups).__name__}"
        )
    discovered = parse_discovered_models(raw_groups)
    chat_pool = chat_models(discovered)
    embedding_pool = embedding_models(discovered)

    if not chat_pool:
        raise click.ClickException("Your key has no chat-capable models available on this proxy.")

    if not _is_interactive():
        raise click.ClickException("`lite autoroute configure` requires an interactive terminal.")

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
        keyword_tier_rules = _prompt_for_keyword_tier_rules()
        semantic_matching = SemanticMatching(embedding_model=embedding_model, keyword_tier_rules=keyword_tier_rules)

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
    persisted_master_key = _load_persisted_master_key(CONFIG_PATH)
    generated: dict[str, JsonValue] = (
        {"model_list": model_list, "general_settings": {"master_key": persisted_master_key}}
        if persisted_master_key is not None
        else {"model_list": model_list}
    )
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with secure_create(CONFIG_PATH) as f:
        yaml.safe_dump(generated, f, sort_keys=False)

    click.echo(f"\nWrote {CONFIG_PATH}")
    for tier, models in tiers.items():
        click.echo(f"  {tier}: {', '.join(models)}")
    return CONFIG_PATH


__all__ = ["run_configure_wizard"]
