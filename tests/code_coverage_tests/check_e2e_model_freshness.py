"""Fail CI when the e2e model matrix (tests/e2e/model_matrix.py) rots.

Checks, in order:
- every pin resolves to an entry in model_prices_and_context_window.json
- no pin is within DEPRECATION_WINDOW_DAYS of its deprecation_date
- the gateway config inside tests/e2e/docker-compose.yml matches GATEWAY_MODELS
  exactly, and every fallback references a configured alias
- no test under tests/e2e/ hardcodes a versioned model literal instead of
  importing a pin from model_matrix.py (triple-quoted strings are exempt as
  docstrings/prose; the scan is token-based so it survives syntax newer than
  the interpreter running it)
"""

import datetime
import importlib.util
import io
import json
import re
import sys
import tokenize
from pathlib import Path
from types import ModuleType

import yaml
from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
E2E_DIR = REPO_ROOT / "tests" / "e2e"
MATRIX_PATH = E2E_DIR / "model_matrix.py"
COMPOSE_PATH = E2E_DIR / "docker-compose.yml"
PRICING_PATH = REPO_ROOT / "model_prices_and_context_window.json"
DEPRECATION_WINDOW_DAYS = 30
MODEL_LITERAL_RE = re.compile(
    r"(?:gpt|gemini|claude|haiku|sonnet|opus|text-embedding)-\d|rerank-v\d"
)


class PricingEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")
    deprecation_date: str | None = None


class GatewayLiteLLMParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str


class GatewayModelEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model_name: str
    litellm_params: GatewayLiteLLMParams


class GatewayRouterSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    fallbacks: list[dict[str, list[str]]] = []


class GatewayConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model_list: list[GatewayModelEntry]
    router_settings: GatewayRouterSettings = GatewayRouterSettings()


def load_matrix() -> ModuleType:
    spec = importlib.util.spec_from_file_location("e2e_model_matrix", MATRIX_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {MATRIX_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pricing() -> dict[str, PricingEntry]:
    raw = json.loads(PRICING_PATH.read_text(encoding="utf-8"))
    return {
        key: PricingEntry.model_validate(value)
        for key, value in raw.items()
        if key != "sample_spec"
    }


def load_gateway_config() -> GatewayConfig:
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    content = compose["configs"]["litellm_config"]["content"]
    return GatewayConfig.model_validate(yaml.safe_load(content))


def resolve_pricing_entry(
    pricing: dict[str, PricingEntry], canonical: str, model_id: str
) -> PricingEntry | None:
    return pricing.get(canonical) or pricing.get(model_id)


def existence_violations(matrix: ModuleType, pricing: dict[str, PricingEntry]) -> tuple[str, ...]:
    return tuple(
        f"{pin.backend}: neither '{pin.canonical}' nor '{pin.model_id}' is in "
        f"model_prices_and_context_window.json; the model was removed or the pin is a typo"
        for pin in matrix.ALL_PINS
        if resolve_pricing_entry(pricing, pin.canonical, pin.model_id) is None
    )


def deprecation_violations(
    matrix: ModuleType, pricing: dict[str, PricingEntry], today: datetime.date
) -> tuple[str, ...]:
    def violation(pin) -> str | None:
        entry = resolve_pricing_entry(pricing, pin.canonical, pin.model_id)
        if entry is None or entry.deprecation_date is None:
            return None
        deprecation = datetime.date.fromisoformat(entry.deprecation_date)
        if (deprecation - today).days > DEPRECATION_WINDOW_DAYS:
            return None
        return (
            f"{pin.backend}: deprecation_date {entry.deprecation_date} is within "
            f"{DEPRECATION_WINDOW_DAYS} days; bump this pin in tests/e2e/model_matrix.py"
        )

    return tuple(v for v in (violation(pin) for pin in matrix.ALL_PINS) if v is not None)


def gateway_sync_violations(matrix: ModuleType, config: GatewayConfig) -> tuple[str, ...]:
    expected = frozenset((pin.alias, pin.backend) for pin in matrix.GATEWAY_MODELS)
    actual = frozenset(
        (entry.model_name, entry.litellm_params.model) for entry in config.model_list
    )
    aliases = frozenset(entry.model_name for entry in config.model_list)
    missing = tuple(
        f"docker-compose.yml gateway config is missing model_name '{alias}' -> '{backend}' "
        f"from GATEWAY_MODELS"
        for alias, backend in sorted(expected - actual)
    )
    extra = tuple(
        f"docker-compose.yml gateway config has model_name '{alias}' -> '{backend}' "
        f"that is not in GATEWAY_MODELS; add a pin to tests/e2e/model_matrix.py"
        for alias, backend in sorted(actual - expected)
    )
    dangling_fallbacks = tuple(
        f"docker-compose.yml fallback references '{name}', which is not a configured model_name"
        for mapping in config.router_settings.fallbacks
        for source, targets in mapping.items()
        for name in (source, *targets)
        if name not in aliases
    )
    return missing + extra + dangling_fallbacks


def is_flagged_token(token: tokenize.TokenInfo) -> bool:
    fstring_middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    if token.type == fstring_middle:
        return bool(MODEL_LITERAL_RE.search(token.string))
    if token.type != tokenize.STRING:
        return False
    if token.string.lstrip("rbufRBUF").startswith(('"""', "'''")):
        return False
    return bool(MODEL_LITERAL_RE.search(token.string))


def literal_violations_in_source(source: str, relative_path: str) -> tuple[str, ...]:
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return tuple(
        f"{relative_path}:{token.start[0]}: hardcoded model literal in {token.string.strip()!r}; "
        f"import a pin from tests/e2e/model_matrix.py instead"
        for token in tokens
        if is_flagged_token(token)
    )


def hardcoded_literal_violations(e2e_dir: Path) -> tuple[str, ...]:
    return tuple(
        violation
        for path in sorted(e2e_dir.rglob("*.py"))
        if path != MATRIX_PATH
        for violation in literal_violations_in_source(
            path.read_text(encoding="utf-8"), str(path.relative_to(REPO_ROOT))
        )
    )


def main() -> int:
    matrix = load_matrix()
    pricing = load_pricing()
    config = load_gateway_config()
    violations = (
        existence_violations(matrix, pricing)
        + deprecation_violations(matrix, pricing, datetime.date.today())
        + gateway_sync_violations(matrix, config)
        + hardcoded_literal_violations(E2E_DIR)
    )
    if violations:
        print("e2e model freshness check failed:")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    print(f"e2e model freshness check passed for {len(matrix.ALL_PINS)} pins")
    return 0


if __name__ == "__main__":
    sys.exit(main())
