"""Build the request body for the inference-only key the compat
matrix hands to the ``claude`` CLI.

Background: every compat cell shells out to the real ``claude`` CLI as
a subprocess and passes its API key via ``ANTHROPIC_AUTH_TOKEN``. If
that key is the proxy's ``LITELLM_MASTER_KEY``, a compromised
``@anthropic-ai/claude-code`` release (which the cell executes with
network access to the proxy) has admin capabilities: it can create /
delete keys, register new deployments, read spend logs, and drain other
tenants' budgets. Nothing about running an LLM prompt requires any of
that.

Fix: mint one virtual key per session with
``allowed_routes=["llm_api_routes"]`` and ``models=[...the compat
matrix's 15 names]``. That collapses the capability surface to the
inference endpoints the CLI actually uses (``/v1/messages`` and its
``count_tokens`` sibling). Management routes, spend routes, and key
CRUD are all denied at the proxy layer regardless of what the CLI ends
up sending.

Kept as a pure function in its own module so the body shape can be
pinned by tests without any control-plane I/O.
"""

from __future__ import annotations

from typing import Iterable

from models import KeyGenerateBody

# Route group the proxy recognizes as "everything under the LLM data
# plane" - /v1/messages, /v1/chat/completions, /embeddings, and the
# passthrough surfaces. Defined in litellm/proxy/_types.py::LiteLLMRoutes
# as the sum of openai_routes, anthropic_routes, google_routes,
# passthrough routes, mcp_inference_routes, and litellm_native_routes.
# See that enum for the authoritative list.
_INFERENCE_ROUTE_GROUP = "llm_api_routes"


def build_compat_cli_key_body(model_names: Iterable[str]) -> KeyGenerateBody:
    """Return the ``/key/generate`` body for the compat CLI key.

    The key is scoped to:

    - only the compat deployments registered this session (``models``)
    - only the LLM inference routes (``allowed_routes``)

    so a compromised CLI cannot pivot to management endpoints even if
    it captures the returned key. The models list is what makes the
    scope non-empty: without it the proxy would default to "all models
    this key can access," which for a master-signed generate request
    is effectively wide-open."""
    return KeyGenerateBody(
        models=list(model_names),
        allowed_routes=[_INFERENCE_ROUTE_GROUP],
        key_alias="compat-cli-session-key",
    )
