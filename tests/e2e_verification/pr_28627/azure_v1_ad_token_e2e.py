"""
End-to-end demonstration for PR #28627
  fix(azure): preserve AD token refresh in v1 OpenAI client path

This makes REAL network calls to Azure OpenAI (no mocking).

Scenario reproduced: an Azure config that authenticates with an
`azure_ad_token_provider` (a callable returning a Bearer credential) and NO
static `api_key`, using a v1 api_version ("preview"/"latest"/"v1") which routes
through the plain OpenAI client path in `BaseAzureLLM.get_azure_openai_client`.

- BEFORE the fix: the v1 branch forwarded only `api_key` (None here), so the
  OpenAI client cannot be built / authenticated -> hard failure.
- AFTER the fix: the AD token provider is forwarded as the OpenAI client's
  callable `api_key`, so the client authenticates and the request succeeds.

We then USE the returned client to make a real Azure call to prove auth works
end to end. The async path additionally exercises the sync->async provider
wrapper that the fix adds.
"""

import asyncio
import os
import sys

# --- Capture real credentials locally, then remove every env var the OpenAI
# --- SDK could silently fall back to. This guarantees the ONLY way the client
# --- can authenticate is via the azure_ad_token_provider the fix forwards.
AZURE_API_BASE = os.environ["AZURE_AI_API_BASE"]
_REAL_BEARER = os.environ["AZURE_AI_API_KEY"]  # accepted as a Bearer token by /openai/v1/
MODEL = "gpt-5.3-codex"

for _k in (
    "OPENAI_API_KEY",
    "OPENAI_ADMIN_KEY",
    "AZURE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_AI_API_KEY",
    "AZURE_FOUNDRY_API_KEY",
):
    os.environ.pop(_k, None)

from litellm.llms.azure.common_utils import BaseAzureLLM  # noqa: E402

CALLS = 0


def azure_ad_token_provider() -> str:
    """Simulates azure-identity's get_bearer_token_provider(): a sync callable
    that returns a fresh Bearer token on each call (here, a real credential the
    Azure /openai/v1/ endpoint accepts)."""
    global CALLS
    CALLS += 1
    return _REAL_BEARER


def line(c="-"):
    print(c * 72)


def show_header(title):
    line("=")
    print(title)
    line("=")


def run_sync():
    show_header("SYNC v1 path  (api_version='preview', api_key=None, AD provider)")
    base_llm = BaseAzureLLM()
    client = base_llm.get_azure_openai_client(
        api_key=None,
        api_base=AZURE_API_BASE,
        api_version="preview",
        litellm_params={"azure_ad_token_provider": azure_ad_token_provider},
        _is_async=False,
    )
    print(f"  client type           : {type(client).__name__}")
    print(f"  client.base_url       : {client.base_url}")
    print(f"  client._api_key_provider forwarded : {client._api_key_provider is not None}")
    print("  -> making REAL call to Azure: client.responses.create(...)")
    resp = client.responses.create(
        model=MODEL,
        input="Reply with exactly: hello from azure",
    )
    print(f"  Azure replied         : {resp.output_text!r}")
    print(f"  provider was invoked   : {CALLS} time(s)")
    line()
    print("  SYNC RESULT: SUCCESS")


async def run_async():
    show_header("ASYNC v1 path (sync provider wrapped in async wrapper by the fix)")
    base_llm = BaseAzureLLM()
    client = base_llm.get_azure_openai_client(
        api_key=None,
        api_base=AZURE_API_BASE,
        api_version="preview",
        litellm_params={"azure_ad_token_provider": azure_ad_token_provider},
        _is_async=True,
    )
    print(f"  client type           : {type(client).__name__}")
    print(f"  async provider forwarded : {client._api_key_provider is not None}")
    print("  -> making REAL async call to Azure: await client.responses.create(...)")
    resp = await client.responses.create(
        model=MODEL,
        input="Reply with exactly: hello from azure async",
    )
    print(f"  Azure replied         : {resp.output_text!r}")
    line()
    print("  ASYNC RESULT: SUCCESS")


def main():
    print()
    print("PR #28627  e2e  | api_base=%s | model=%s" % (AZURE_API_BASE, MODEL))
    print("Auth = azure_ad_token_provider() ONLY (no api_key, no env fallback)")
    print()
    failed = False
    try:
        run_sync()
    except Exception as e:
        failed = True
        line()
        print(f"  SYNC RESULT: FAILED -> {type(e).__name__}: {str(e)[:200]}")
    print()
    try:
        asyncio.run(run_async())
    except Exception as e:
        failed = True
        line()
        print(f"  ASYNC RESULT: FAILED -> {type(e).__name__}: {str(e)[:200]}")
    print()
    line("=")
    print("OVERALL: " + ("FAILED (AD provider dropped on v1 path)" if failed else "PASSED (AD auth works end to end on v1 path)"))
    line("=")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
