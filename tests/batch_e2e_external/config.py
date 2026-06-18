"""
Environment-driven configuration for the external batch e2e suite.

Everything that varies per deployment (proxy URL, the user API key, which
providers / models / routing scenarios to exercise) is supplied through the
environment. Nothing is hardcoded.

Hard-fail contract: if anything required is missing or malformed, loading the
config raises. There are no silent defaults that would let a misconfigured run
pass. This suite exists to catch regressions, so "not configured" is a failure,
not a skip.

Required environment variables:
  LITELLM_E2E_BASE_URL   Base URL of the published LiteLLM proxy image.
  LITELLM_E2E_API_KEY    The (user-specific) API key the external suite runs as.
  LITELLM_E2E_CONFIG     The case matrix. Either inline JSON, or "@/path/to.json"
                         to read it from a file. See README.md for the schema.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Routing scenarios this suite knows how to exercise. These map 1:1 to the
# proxy's file-create routing priority in
# litellm/proxy/openai_files_endpoints/files_endpoints.py.
STRATEGY_MODEL_PARAM = "model_param"  # `model=` -> model creds, id encodes model
STRATEGY_TARGET_MODEL_NAMES = "target_model_names"  # managed files, requires DB
STRATEGY_CUSTOM_LLM_PROVIDER = "custom_llm_provider"  # `custom-llm-provider` header

VALID_STRATEGIES = {
    STRATEGY_MODEL_PARAM,
    STRATEGY_TARGET_MODEL_NAMES,
    STRATEGY_CUSTOM_LLM_PROVIDER,
}

# Batch lifecycle operations, in execution order.
OP_FILE = "file"
OP_CREATE = "create"
OP_RETRIEVE = "retrieve"
OP_LIST = "list"
OP_CANCEL = "cancel"
OP_DELETE = "delete"

ALL_OPS = [OP_FILE, OP_CREATE, OP_RETRIEVE, OP_LIST, OP_CANCEL, OP_DELETE]


class ConfigError(RuntimeError):
    """Raised when the suite is not configured correctly. Always fatal."""


@dataclass
class ExpectedError:
    """A declared, known contract gap for a (provider, op).

    When present for an op, the suite asserts the call raises an error matching
    this spec. Anything else -- success, or a different error -- is a failure.
    """

    status: Optional[int] = None
    match: Optional[str] = None  # case-insensitive substring expected in the error


@dataclass
class Case:
    """One row of the test matrix: a routing strategy against one provider."""

    strategy: str
    provider: str
    request_model: str  # model string written into the .jsonl request body
    # strategy-specific routing handles
    model: Optional[str] = None  # STRATEGY_MODEL_PARAM
    target_model_names: Optional[str] = None  # STRATEGY_TARGET_MODEL_NAMES
    # known unsupported ops -> expected error (the only sanctioned non-success)
    expected_unsupported: Dict[str, ExpectedError] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.strategy}-{self.provider}"


@dataclass
class Settings:
    base_url: str
    api_key: str
    endpoint: str
    await_completion: bool
    completion_timeout_s: int
    cases: List[Case]


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(
            f"{name} is required for the external batch e2e suite but was not set."
        )
    return value


def _load_raw_config(raw: str) -> Dict[str, Any]:
    if raw.startswith("@"):
        path = raw[1:]
        try:
            with open(path, "r") as f:
                raw = f.read()
        except OSError as e:
            raise ConfigError(f"Could not read LITELLM_E2E_CONFIG file {path!r}: {e}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ConfigError(f"LITELLM_E2E_CONFIG is not valid JSON: {e}")
    if not isinstance(parsed, dict):
        raise ConfigError("LITELLM_E2E_CONFIG must be a JSON object.")
    return parsed


def _parse_expected_unsupported(
    raw: Dict[str, Any], case_id: str
) -> Dict[str, ExpectedError]:
    expected: Dict[str, ExpectedError] = {}
    for op, spec in raw.items():
        if op not in ALL_OPS:
            raise ConfigError(
                f"case {case_id!r}: unknown op {op!r} in expected_unsupported; "
                f"valid ops are {ALL_OPS}."
            )
        if not isinstance(spec, dict):
            raise ConfigError(
                f"case {case_id!r}: expected_unsupported[{op!r}] must be an object."
            )
        expected[op] = ExpectedError(status=spec.get("status"), match=spec.get("match"))
    return expected


def _parse_case(raw: Dict[str, Any], index: int) -> Case:
    strategy = raw.get("strategy")
    if strategy not in VALID_STRATEGIES:
        raise ConfigError(
            f"cases[{index}]: strategy must be one of {sorted(VALID_STRATEGIES)}, "
            f"got {strategy!r}."
        )
    provider = raw.get("provider")
    if not provider:
        raise ConfigError(f"cases[{index}]: 'provider' is required.")
    request_model = raw.get("request_model")
    if not request_model:
        raise ConfigError(f"cases[{index}]: 'request_model' is required.")

    model = raw.get("model")
    target_model_names = raw.get("target_model_names")
    if strategy == STRATEGY_MODEL_PARAM and not model:
        raise ConfigError(
            f"cases[{index}]: strategy '{STRATEGY_MODEL_PARAM}' requires 'model'."
        )
    if strategy == STRATEGY_TARGET_MODEL_NAMES and not target_model_names:
        raise ConfigError(
            f"cases[{index}]: strategy '{STRATEGY_TARGET_MODEL_NAMES}' requires "
            f"'target_model_names'."
        )

    case_id = f"{strategy}-{provider}"
    expected_unsupported = _parse_expected_unsupported(
        raw.get("expected_unsupported", {}) or {}, case_id
    )
    return Case(
        strategy=strategy,
        provider=provider,
        request_model=request_model,
        model=model,
        target_model_names=target_model_names,
        expected_unsupported=expected_unsupported,
    )


def load_settings() -> Settings:
    """Load and validate the suite configuration. Raises ConfigError on any gap."""
    base_url = _require_env("LITELLM_E2E_BASE_URL")
    api_key = _require_env("LITELLM_E2E_API_KEY")
    raw = _load_raw_config(_require_env("LITELLM_E2E_CONFIG"))

    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ConfigError(
            "LITELLM_E2E_CONFIG.cases must be a non-empty list of case objects."
        )
    cases = [_parse_case(c, i) for i, c in enumerate(cases_raw)]

    ids = [c.id for c in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ConfigError(f"duplicate cases in matrix: {sorted(duplicates)}.")

    return Settings(
        base_url=base_url,
        api_key=api_key,
        endpoint=raw.get("endpoint", "/v1/chat/completions"),
        await_completion=bool(raw.get("await_completion", False)),
        completion_timeout_s=int(raw.get("completion_timeout_s", 900)),
        cases=cases,
    )
