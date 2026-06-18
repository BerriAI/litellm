"""Generic HTTP client for live e2e tests against a running LiteLLM proxy.

Shared by every e2e suite under tests/e2e_tests/. Covers the proxy operations any
suite needs: key/customer management (so the shared ResourceManager can clean up),
OpenAI-compatible calls, route probing, and SpendLogs read-back. Suite-specific
clients subclass ProxyClient (see spend_tracking/, llm_translation/, budgets/).

Talks to the proxy over real HTTP so every test sees what a real client sees:
the x-litellm-call-id header, the response body, and the rows the proxy writes.
Writes are eventually consistent (proxy_batch_write_at ~60s), so read-backs poll
to a deadline rather than sleeping once.
"""

import json
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

import pytest
import requests

from e2e_config import (
    MASTER_KEY,
    POLL_INTERVAL,
    POLL_TIMEOUT,
    PROXY_BASE_URL,
    REQUEST_TIMEOUT,
)

SpendLogRow = Dict[str, object]


@runtime_checkable
class CallOutcome(Protocol):
    """Anything with an HTTP status and body that can pass the skip/fail boundary."""

    status_code: int
    body: str

    @property
    def ok(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Outcome of a single route probe: enough to see *why* it (mis)behaved."""

    url: str
    status_code: int
    body: str

    @property
    def healthy(self) -> bool:
        # Route exists (not 404), handler did not crash (not 5xx), request
        # completed (not -1). A 4xx (missing params/auth) still means it ran.
        return 200 <= self.status_code != 404 and self.status_code < 500

    def __str__(self) -> str:
        return f"GET {self.url} -> {self.status_code}\n{self.body[:600]}"


@dataclass(frozen=True, slots=True)
class CallResult:
    """Outcome of a single OpenAI-compatible call made through the proxy."""

    status_code: int
    call_id: Optional[str]  # x-litellm-call-id response header
    response_id: Optional[str]  # body "id"; SpendLogs.request_id is derived from this
    response_cost_header: Optional[str]  # x-litellm-response-cost header
    body: str
    content: Optional[str]

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


def _auth(key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


class ProxyClient:
    def __init__(
        self,
        base_url: str,
        master_key: str,
        *,
        request_timeout: float,
        poll_timeout: float,
        poll_interval: float,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._master_key = master_key
        self._request_timeout = request_timeout
        self._poll_timeout = poll_timeout
        self._poll_interval = poll_interval

    # ---- key / customer management (satisfies lifecycle.ResourceClient) ----

    def generate_key(
        self,
        *,
        models: Optional[List[str]] = None,
        max_budget: Optional[float] = None,
        metadata: Optional[Dict[str, object]] = None,
        extra_params: Optional[Dict[str, object]] = None,
    ) -> str:
        payload: Dict[str, object] = {"models": models or [], "duration": None}
        if max_budget is not None:
            payload["max_budget"] = max_budget
        if metadata is not None:
            payload["metadata"] = metadata
        if extra_params:
            payload.update(extra_params)
        resp = requests.post(
            f"{self._base_url}/key/generate",
            headers=_auth(self._master_key),
            json=payload,
            timeout=self._request_timeout,
        )
        resp.raise_for_status()
        return str(resp.json()["key"])

    def key_info(self, key: str) -> Dict[str, object]:
        resp = requests.get(
            f"{self._base_url}/key/info",
            headers=_auth(self._master_key),
            params={"key": key},
            timeout=self._request_timeout,
        )
        resp.raise_for_status()
        return dict(resp.json().get("info", {}))

    def delete_key(self, key: str) -> None:
        """Best-effort teardown; a failed cleanup must not fail the test."""
        try:
            requests.post(
                f"{self._base_url}/key/delete",
                headers=_auth(self._master_key),
                json={"keys": [key]},
                timeout=self._request_timeout,
            )
        except requests.RequestException:
            pass

    def delete_customers(self, user_ids: List[str]) -> None:
        """Best-effort teardown of end-user/customer rows the `user` param creates."""
        if not user_ids:
            return
        try:
            requests.post(
                f"{self._base_url}/customer/delete",
                headers=_auth(self._master_key),
                json={"user_ids": user_ids},
                timeout=self._request_timeout,
            )
        except requests.RequestException:
            pass

    # ---- OpenAI-compatible calls ----------------------------------------

    def chat(
        self,
        key: str,
        model: str,
        content: str,
        *,
        stream: bool = False,
        metadata: Optional[Dict[str, object]] = None,
        extra_body: Optional[Dict[str, object]] = None,
    ) -> CallResult:
        body: Dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": stream,
        }
        if metadata is not None:
            body["metadata"] = metadata
        if extra_body is not None:
            body.update(extra_body)
        url = f"{self._base_url}/chat/completions"
        if stream:
            return self._chat_stream(url, key, body)
        resp = requests.post(
            url, headers=_auth(key), json=body, timeout=self._request_timeout
        )
        parsed = resp.json() if resp.content else {}
        choices = parsed.get("choices") or [{}]
        message_content = (choices[0].get("message") or {}).get("content")
        return CallResult(
            status_code=resp.status_code,
            call_id=resp.headers.get("x-litellm-call-id"),
            response_id=parsed.get("id"),
            response_cost_header=resp.headers.get("x-litellm-response-cost"),
            body=resp.text,
            content=message_content,
        )

    def _chat_stream(self, url: str, key: str, body: Dict[str, object]) -> CallResult:
        resp = requests.post(
            url,
            headers=_auth(key),
            json=body,
            stream=True,
            timeout=self._request_timeout,
        )
        if not (200 <= resp.status_code < 300):
            return CallResult(
                status_code=resp.status_code,
                call_id=resp.headers.get("x-litellm-call-id"),
                response_id=None,
                response_cost_header=resp.headers.get("x-litellm-response-cost"),
                body=resp.text,
                content=None,
            )
        response_id: Optional[str] = None
        parts: List[str] = []
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8")
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            response_id = chunk.get("id", response_id)
            for choice in chunk.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    parts.append(piece)
        return CallResult(
            status_code=resp.status_code,
            call_id=resp.headers.get("x-litellm-call-id"),
            response_id=response_id,
            response_cost_header=resp.headers.get("x-litellm-response-cost"),
            body="<streamed>",
            content="".join(parts) or None,
        )

    def embed(self, key: str, model: str, text: str) -> CallResult:
        resp = requests.post(
            f"{self._base_url}/embeddings",
            headers=_auth(key),
            json={"model": model, "input": text},
            timeout=self._request_timeout,
        )
        parsed = resp.json() if resp.content else {}
        return CallResult(
            status_code=resp.status_code,
            call_id=resp.headers.get("x-litellm-call-id"),
            response_id=parsed.get("id"),
            response_cost_header=resp.headers.get("x-litellm-response-cost"),
            body=resp.text,
            content=None,
        )

    # ---- route discovery -------------------------------------------------

    def get_openapi(self) -> Dict[str, object]:
        """The proxy's live route schema from /openapi.json."""
        resp = requests.get(
            f"{self._base_url}/openapi.json", timeout=self._request_timeout
        )
        resp.raise_for_status()
        return dict(resp.json())

    def probe(self, path: str, params: Optional[Dict[str, str]] = None) -> ProbeResult:
        """GET a route with master-key auth; capture status + body to show why."""
        url = f"{self._base_url}{path}"
        try:
            resp = requests.get(
                url,
                headers=_auth(self._master_key),
                params=params or {},
                timeout=self._request_timeout,
            )
        except requests.RequestException as exc:
            return ProbeResult(url=url, status_code=-1, body=f"request error: {exc}")
        return ProbeResult(url=url, status_code=resp.status_code, body=resp.text)

    # ---- SpendLogs read-back --------------------------------------------

    def _get_logs(
        self, *, request_id: Optional[str] = None, api_key: Optional[str] = None
    ) -> List[SpendLogRow]:
        params: Dict[str, str] = {}
        if request_id is not None:
            params["request_id"] = request_id
        if api_key is not None:
            params["api_key"] = api_key
        resp = requests.get(
            f"{self._base_url}/spend/logs",
            headers=_auth(self._master_key),
            params=params,
            timeout=self._request_timeout,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        return [dict(row) for row in data] if isinstance(data, list) else []

    def poll_logs_for_key(
        self,
        key: str,
        *,
        min_rows: int = 1,
        predicate: Optional[Callable[[List[SpendLogRow]], bool]] = None,
    ) -> List[SpendLogRow]:
        return self._poll(lambda: self._get_logs(api_key=key), min_rows, predicate)

    def poll_logs_for_request_id(
        self,
        request_id: str,
        *,
        min_rows: int = 1,
        predicate: Optional[Callable[[List[SpendLogRow]], bool]] = None,
    ) -> List[SpendLogRow]:
        return self._poll(
            lambda: self._get_logs(request_id=request_id), min_rows, predicate
        )

    def _poll(
        self,
        fetch: Callable[[], List[SpendLogRow]],
        min_rows: int,
        predicate: Optional[Callable[[List[SpendLogRow]], bool]],
    ) -> List[SpendLogRow]:
        deadline = time.monotonic() + self._poll_timeout
        rows: List[SpendLogRow] = []
        while time.monotonic() < deadline:
            rows = fetch()
            satisfied = len(rows) >= min_rows and (
                predicate is None or predicate(rows)
            )
            if satisfied:
                return rows
            time.sleep(self._poll_interval)
        return rows


def proxy_client_kwargs() -> Dict[str, object]:
    """Constructor kwargs shared by every ProxyClient subclass."""
    return {
        "base_url": PROXY_BASE_URL,
        "master_key": MASTER_KEY,
        "request_timeout": REQUEST_TIMEOUT,
        "poll_timeout": POLL_TIMEOUT,
        "poll_interval": POLL_INTERVAL,
    }


def require_successful_call(result: CallOutcome) -> None:
    """The skip/fail boundary.

    A non-2xx here means the environment cannot make the call (missing provider
    key, upstream down) -> skip. Everything downstream of a 2xx is behavior that
    is allowed to fail.
    """
    if result.ok:
        return
    pytest.skip(
        f"upstream call unavailable (status {result.status_code}); "
        f"body={result.body[:300]}"
    )


def unique_marker() -> str:
    return uuid.uuid4().hex[:12]
