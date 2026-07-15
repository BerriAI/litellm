"""passthrough x Azure (Microsoft Foundry).

Drive the real `claude` CLI in foundry mode (CLAUDE_CODE_USE_FOUNDRY=1)
with ANTHROPIC_FOUNDRY_BASE_URL aimed at the proxy's `/azure`
passthrough route. The CLI POSTs `/v1/messages` with the model in the
JSON body -- unlike the bedrock/vertex modes there is no model segment
in the URL, so the proxy's router-alias resolution cannot engage and
the `/azure` route falls back to its env-configured target: the proxy
must set AZURE_API_BASE to the Foundry resource's Anthropic surface
(`https://<resource>.services.ai.azure.com/anthropic`) and
AZURE_API_KEY to the Foundry key (see
cron_vm/litellm-compat-matrix.env.example). The model ids are the
Foundry deployment names, which this matrix provisions to match the
Anthropic ids.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/passthrough/test_azure.py
                       ^^^^^^^^^^^     ^^^^^
                       feature_id      provider

Wiring verified live at authoring time: through `{proxy}/azure` the
Foundry Anthropic surface accepted the `api-key` / `Authorization:
Bearer` headers the fallback sends (a bogus key 401s, the real key
proceeds to deployment lookup), so a red cell here means missing
AZURE_API_BASE/AZURE_API_KEY on the proxy, missing Foundry deployments
for the three tiers, or a genuine forwarding regression -- not an
auth-scheme mismatch.
"""

from __future__ import annotations

from claude_code._passthrough import foundry_extra_env, run_passthrough_cell

AZURE_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]


def test_passthrough_azure(compat_result):
    """Drive the `claude` CLI through `{proxy}/azure` and assert a reply."""
    run_passthrough_cell(
        compat_result=compat_result,
        models=AZURE_MODELS,
        prompt="Reply with the single word 'pong' and nothing else.",
        build_extra_env=foundry_extra_env,
    )
