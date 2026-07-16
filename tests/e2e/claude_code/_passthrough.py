"""Shared body for the `passthrough` × <provider> compat cells.

Every other matrix row drives the proxy's `/v1/messages` translation
layer: Claude Code speaks the first-party Anthropic wire and LiteLLM
transforms the request per provider. This row instead exercises
LiteLLM's *native passthrough* routes -- the "LLM gateway"
configuration documented at https://code.claude.com/docs/en/gateway --
where Claude Code speaks each cloud's own wire format and the proxy
forwards it, attaching provider credentials on the way out:

    anthropic        ANTHROPIC_BASE_URL={proxy}/anthropic. The CLI's
                     first-party wire, forwarded verbatim to
                     api.anthropic.com, so the model ids are real
                     Anthropic ids rather than proxy aliases.
    bedrock_invoke   CLAUDE_CODE_USE_BEDROCK=1 +
                     ANTHROPIC_BEDROCK_BASE_URL={proxy}/bedrock. The
                     CLI POSTs /model/{model}/invoke-with-response-stream;
                     the proxy recognizes a router alias in the model
                     segment, rewrites it to the deployment's upstream
                     model id, and SigV4-signs with its own AWS creds.
    vertex_ai        CLAUDE_CODE_USE_VERTEX=1 +
                     ANTHROPIC_VERTEX_BASE_URL={proxy}/vertex_ai/v1.
                     The CLI POSTs
                     .../models/{model}:streamRawPredict; the proxy
                     resolves a router alias in the model segment and
                     takes project, location, and credentials from the
                     deployment (which is why the deployment must set
                     `use_in_pass_through: true` -- see
                     test_config.yaml).
    azure            CLAUDE_CODE_USE_FOUNDRY=1 +
                     ANTHROPIC_FOUNDRY_BASE_URL={proxy}/azure. Foundry
                     mode sends the model in the JSON body, not the
                     URL, so the proxy's /azure route cannot resolve a
                     router alias and falls back to the env-configured
                     AZURE_API_BASE / AZURE_API_KEY target.
    bedrock_converse not applicable -- Claude Code's bedrock mode is
                     InvokeModel-only; no Converse-wire client exists.

Auth is the same in every mode: the CLI's provider-native signing is
disabled via CLAUDE_CODE_SKIP_<PROVIDER>_AUTH, and the LiteLLM virtual
key travels as `Authorization: Bearer` (ANTHROPIC_AUTH_TOKEN), exactly
like the translation rows. The proxy holds the real provider
credentials.

The per-mode env vars and URL shapes above were captured from a real
`claude` CLI (2.1.210) run against a request-logging sink, not from
docs; if a CLI release changes them, the cells fail with the CLI's own
diagnostic rather than silently testing the wrong wire.

`run_models` and `env` are injection seams for
`_driver_unit_tests/test_passthrough.py`; production callers leave
them unset.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import pytest

from claude_code.cli_driver import (
    ClaudeCLIError,
    failure_diagnostic,
    run_claude_models_parallel,
)

PROXY_BASE_URL_ENV = "LITELLM_PROXY_BASE_URL"
PROXY_API_KEY_ENV = "LITELLM_PROXY_API_KEY"

ANTHROPIC_PASSTHROUGH_BASE_PATH = "/anthropic"

CLIENT_SIDE_AWS_REGION = "us-east-1"
"""Satisfies the CLI's embedded AWS SDK, which refuses to construct a
client without a region. The value never influences routing: the proxy
signs the upstream request with its own credentials and region."""

VERTEX_PLACEHOLDER_PROJECT = "proxy-resolved-project"
VERTEX_PLACEHOLDER_REGION = "us-east5"
"""The CLI refuses to build a Vertex URL without a project id and
region, but the proxy replaces both path segments with the resolved
deployment's `vertex_project` / `vertex_location` before forwarding,
so deliberately-fake values prove the resolution actually happened."""


def bedrock_extra_env(proxy_base_url: str) -> Dict[str, str]:
    return {
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "CLAUDE_CODE_SKIP_BEDROCK_AUTH": "1",
        "ANTHROPIC_BEDROCK_BASE_URL": f"{proxy_base_url}/bedrock",
        "AWS_REGION": CLIENT_SIDE_AWS_REGION,
    }


def vertex_extra_env(proxy_base_url: str) -> Dict[str, str]:
    """Vertex-mode CLI env pointed at the proxy's /vertex_ai route.

    The `/v1` suffix on ANTHROPIC_VERTEX_BASE_URL is load-bearing: the
    CLI's Vertex SDK ships its API version inside its *default* base
    URL (`https://{region}-aiplatform.googleapis.com/v1`), so
    overriding the base drops the version from the request path unless
    the override carries it. LiteLLM's /vertex_ai route reuses the
    incoming path verbatim when it contains `/projects/.../locations/...`,
    so a version-less path would reach Google as
    `aiplatform.googleapis.com/projects/...` and 404.
    """
    return {
        "CLAUDE_CODE_USE_VERTEX": "1",
        "CLAUDE_CODE_SKIP_VERTEX_AUTH": "1",
        "ANTHROPIC_VERTEX_BASE_URL": f"{proxy_base_url}/vertex_ai/v1",
        "ANTHROPIC_VERTEX_PROJECT_ID": VERTEX_PLACEHOLDER_PROJECT,
        "CLOUD_ML_REGION": VERTEX_PLACEHOLDER_REGION,
    }


def foundry_extra_env(proxy_base_url: str) -> Dict[str, str]:
    return {
        "CLAUDE_CODE_USE_FOUNDRY": "1",
        "CLAUDE_CODE_SKIP_FOUNDRY_AUTH": "1",
        "ANTHROPIC_FOUNDRY_BASE_URL": f"{proxy_base_url}/azure",
    }


def run_passthrough_cell(
    *,
    compat_result,
    models: Sequence[str],
    prompt: str,
    passthrough_base_path: str = "",
    build_extra_env: Optional[Callable[[str], Mapping[str, str]]] = None,
    run_models: Callable[..., Mapping[str, Any]] = run_claude_models_parallel,
    env: Optional[Mapping[str, str]] = None,
) -> None:
    """Run the shared `passthrough` × <provider> cell body.

    `passthrough_base_path` is appended to the proxy base URL and
    becomes the CLI's ANTHROPIC_BASE_URL (only the anthropic column
    uses it; the cloud columns ignore ANTHROPIC_BASE_URL entirely once
    their CLAUDE_CODE_USE_* flag is set). `build_extra_env` receives
    the trailing-slash-normalized proxy base URL and returns the
    provider-mode env for the CLI subprocess.
    """
    environ = env if env is not None else os.environ
    base_url = environ.get(PROXY_BASE_URL_ENV)
    api_key = environ.get(PROXY_API_KEY_ENV)
    if not base_url or not api_key:
        compat_result.set(
            {
                "status": "fail",
                "error": (
                    f"missing required env: set {PROXY_BASE_URL_ENV} and "
                    f"{PROXY_API_KEY_ENV} to point at a running LiteLLM proxy"
                ),
            }
        )
        pytest.fail(
            f"{PROXY_BASE_URL_ENV} / {PROXY_API_KEY_ENV} not configured",
            pytrace=False,
        )

    proxy_base = base_url.rstrip("/")
    extra_env = dict(build_extra_env(proxy_base)) if build_extra_env else None

    outcomes = run_models(
        models=models,
        prompt=prompt,
        base_url=proxy_base + passthrough_base_path,
        api_key=api_key,
        extra_env=extra_env,
    )

    failures = []
    for model in models:
        outcome = outcomes[model]
        if isinstance(outcome, ClaudeCLIError):
            error = f"[{model}] {outcome}"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        if outcome.exit_code != 0:
            error = f"[{model}] claude CLI failed: {failure_diagnostic(outcome)}"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        if not outcome.text.strip():
            error = f"[{model}] claude returned empty assistant text"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
