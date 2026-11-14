"""Seed the pricing currency demo through the proxy's own management and chat APIs.

Safe to rerun: models are created only when their id is missing, the demo key only when it does not exist, and the
synthetic traffic only once, recorded on the key's metadata. Every request goes to the local mock upstream.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Final

PROXY: Final = os.environ.get("DEMO_PROXY_URL", "http://127.0.0.1:4000").rstrip("/")
MASTER_KEY: Final = os.environ["LITELLM_MASTER_KEY"]
DEMO_KEY: Final = os.environ["DEMO_VIRTUAL_KEY"]
MOCK_UPSTREAM: Final = {"api_base": "http://mock-upstream:8090/v1", "api_key": "mock-upstream-ignores-this"}
DEMO_GROUPS: Final = ("demo-currency-cheapest", "demo-currency-huf-model-info", "demo-currency-eur-mixed")
DEMO_KEY_BUDGET_USD: Final = 0.03
SEED_MARKER: Final = "pricing_currency_demo_seeded"

API_MODELS: Final[tuple[dict[str, object], ...]] = (
    {
        "model_name": "demo-currency-cheapest",
        "litellm_params": {
            "model": "openai/demo-eur-180",
            **MOCK_UPSTREAM,
            "pricing_currency": "EUR",
            "usd_per_currency_unit": "1.20",
            "input_cost_per_token": 0.0000018,
            "output_cost_per_token": 0.0000036,
        },
        "model_info": {"id": "demo-eur-180"},
    },
    {
        "model_name": "demo-currency-eur-mixed",
        "litellm_params": {
            "model": "openai/demo-eur-mixed",
            **MOCK_UPSTREAM,
            "pricing_currency": "EUR",
            "usd_per_currency_unit": "1.20",
            "input_cost_per_token": 0.0000018,
        },
        "model_info": {"id": "demo-eur-mixed", "output_cost_per_token": 0.0000036},
    },
)

SYNTHETIC_TRAFFIC: Final = (
    "demo-currency-cheapest",
    "demo-currency-cheapest",
    "demo-currency-huf-model-info",
    "demo-currency-huf-model-info",
    "demo-currency-eur-mixed",
    "demo-currency-eur-mixed",
)


def _call(method: str, path: str, key: str, body: object | None = None) -> tuple[int, dict[str, object], dict[str, str]]:
    request: Final = urllib.request.Request(
        f"{PROXY}{path}",
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read() or b"{}"), dict(response.headers.items())
    except urllib.error.HTTPError as error:
        payload: Final = error.read()
        try:
            parsed: Final = json.loads(payload or b"{}")
        except json.JSONDecodeError:
            return error.code, {"raw": payload.decode(errors="replace")}, {}
        return error.code, parsed if isinstance(parsed, dict) else {"raw": parsed}, {}


def _wait_until(description: str, ready: Callable[[], bool], attempts: int = 90) -> None:
    for _ in range(attempts):
        if ready():
            return
        time.sleep(1)
    sys.exit(f"Timed out waiting for {description}")


def _model_exists(model_id: str) -> bool:
    status, body, _ = _call("GET", f"/v2/model/info?modelId={model_id}", MASTER_KEY)
    data: Final = body.get("data")
    return status == 200 and isinstance(data, list) and len(data) > 0


def _ensure_models() -> None:
    for spec in API_MODELS:
        model_info: Final = spec["model_info"]
        model_id: Final = str(model_info["id"]) if isinstance(model_info, dict) else ""
        if _model_exists(model_id):
            print(f"model {model_id}: already exists")
            continue
        status, body, _ = _call("POST", "/model/new", MASTER_KEY, spec)
        if status != 200:
            sys.exit(f"model {model_id}: create failed with {status} {body}")
        print(f"model {model_id}: created")


def _key_info() -> dict[str, object] | None:
    status, body, _ = _call("GET", f"/key/info?key={DEMO_KEY}", MASTER_KEY)
    info: Final = body.get("info")
    return info if status == 200 and isinstance(info, dict) else None


def _ensure_key() -> dict[str, object]:
    existing: Final = _key_info()
    if existing is not None:
        print("demo key: already exists")
        return existing
    status, body, _ = _call(
        "POST",
        "/key/generate",
        MASTER_KEY,
        {
            "key": DEMO_KEY,
            "key_alias": "demo-currency-key",
            "models": list(DEMO_GROUPS),
            "max_budget": DEMO_KEY_BUDGET_USD,
            "metadata": {"demo": "pricing-currency"},
        },
    )
    if status != 200:
        sys.exit(f"demo key: create failed with {status} {body}")
    print(f"demo key: created with a ${DEMO_KEY_BUDGET_USD} USD budget")
    created: Final = _key_info()
    if created is None:
        sys.exit("demo key: created but not readable")
    return created


def _servable(group: str) -> bool:
    status, body, _ = _call("GET", f"/model_group/info?model_group={group}", MASTER_KEY)
    data: Final = body.get("data")
    return status == 200 and isinstance(data, list) and len(data) > 0


def _send_traffic() -> None:
    for group in SYNTHETIC_TRAFFIC:
        status, body, headers = _call(
            "POST",
            "/v1/chat/completions",
            DEMO_KEY,
            {
                "model": group,
                "messages": [{"role": "user", "content": "Synthetic demo request, not live model evidence."}],
                "user": "synthetic-demo",
                "metadata": {"tags": ["synthetic-demo"]},
            },
        )
        if status != 200:
            sys.exit(f"traffic to {group} failed with {status} {body}")
        print(
            f"{group}: served by {headers.get('x-litellm-model-id')} for ${headers.get('x-litellm-response-cost')} USD"
        )


def main() -> None:
    _wait_until("the proxy", lambda: _call("GET", "/health/readiness", MASTER_KEY)[0] == 200)
    _ensure_models()
    key: Final = _ensure_key()
    metadata: Final = key.get("metadata")
    if isinstance(metadata, dict) and metadata.get(SEED_MARKER) is True:
        print("synthetic traffic: already sent on an earlier run, skipping")
        return
    for group in DEMO_GROUPS:
        _wait_until(f"both workers to serve {group}", lambda group=group: all(_servable(group) for _ in range(4)))
    _send_traffic()
    status, body, _ = _call(
        "POST",
        "/key/update",
        MASTER_KEY,
        {"key": DEMO_KEY, "metadata": {"demo": "pricing-currency", SEED_MARKER: True}},
    )
    if status != 200:
        sys.exit(f"recording the seed on the demo key failed with {status} {body}")
    print("Seed complete. Spend rows appear on the Logs page within about a minute.")


if __name__ == "__main__":
    main()
