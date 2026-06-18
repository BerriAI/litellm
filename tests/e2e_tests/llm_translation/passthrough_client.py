"""Client for LLM-translation e2e tests over the proxy's passthrough endpoints.

Extends the shared ProxyClient with native provider passthrough calls. A
passthrough request is sent in the PROVIDER's native format (Gemini
generateContent, Anthropic /v1/messages) to the proxy, which forwards it to the
provider and still logs a SpendLogs row (call_type="pass_through_endpoint"). The
litellm virtual key is passed as the provider key; the proxy swaps in the real
env credential. SpendLogs.request_id == the x-litellm-call-id response header.
"""

from dataclasses import dataclass
from typing import List, Optional

import requests

from proxy_client import ProxyClient, proxy_client_kwargs


@dataclass(frozen=True, slots=True)
class PassthroughResult:
    """Outcome of a native passthrough call. ``call_id`` correlates to the row."""

    status_code: int
    call_id: Optional[str]  # x-litellm-call-id -> SpendLogs.request_id
    body: str
    chunks: int = 0  # number of streamed events (0 for non-streaming)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


def _tag_header(tags: Optional[List[str]]) -> dict:
    return {"tags": ",".join(tags)} if tags else {}


class PassthroughClient(ProxyClient):
    # ---- Gemini native passthrough (/gemini/v1beta/...) -----------------

    def gemini_generate(
        self,
        key: str,
        model: str,
        text: str,
        *,
        tools: Optional[list] = None,
        tags: Optional[List[str]] = None,
    ) -> PassthroughResult:
        body: dict = {"contents": [{"role": "user", "parts": [{"text": text}]}]}
        if tools is not None:
            body["tools"] = tools
        headers = {
            "x-goog-api-key": key,
            "Content-Type": "application/json",
            **_tag_header(tags),
        }
        resp = requests.post(
            f"{self._base_url}/gemini/v1beta/models/{model}:generateContent",
            headers=headers,
            json=body,
            timeout=self._request_timeout,
        )
        return PassthroughResult(
            resp.status_code, resp.headers.get("x-litellm-call-id"), resp.text
        )

    def gemini_stream(
        self, key: str, model: str, text: str, *, tags: Optional[List[str]] = None
    ) -> PassthroughResult:
        headers = {
            "x-goog-api-key": key,
            "Content-Type": "application/json",
            **_tag_header(tags),
        }
        resp = requests.post(
            f"{self._base_url}/gemini/v1beta/models/{model}:streamGenerateContent",
            headers=headers,
            params={"alt": "sse"},
            json={"contents": [{"role": "user", "parts": [{"text": text}]}]},
            stream=True,
            timeout=self._request_timeout,
        )
        call_id = resp.headers.get("x-litellm-call-id")
        if not (200 <= resp.status_code < 300):
            return PassthroughResult(resp.status_code, call_id, resp.text)
        chunks = sum(1 for line in resp.iter_lines() if line)
        return PassthroughResult(resp.status_code, call_id, "<streamed>", chunks)

    # ---- Anthropic native passthrough (/anthropic/v1/messages) ----------

    def anthropic_message(
        self,
        key: str,
        model: str,
        text: str,
        *,
        max_tokens: int = 64,
        tools: Optional[list] = None,
        stream: bool = False,
        tags: Optional[List[str]] = None,
    ) -> PassthroughResult:
        body: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": text}],
        }
        if tools is not None:
            body["tools"] = tools
        if stream:
            body["stream"] = True
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            **_tag_header(tags),
        }
        url = f"{self._base_url}/anthropic/v1/messages"
        if not stream:
            resp = requests.post(
                url, headers=headers, json=body, timeout=self._request_timeout
            )
            return PassthroughResult(
                resp.status_code, resp.headers.get("x-litellm-call-id"), resp.text
            )
        resp = requests.post(
            url, headers=headers, json=body, stream=True, timeout=self._request_timeout
        )
        call_id = resp.headers.get("x-litellm-call-id")
        if not (200 <= resp.status_code < 300):
            return PassthroughResult(resp.status_code, call_id, resp.text)
        chunks = sum(1 for line in resp.iter_lines() if line)
        return PassthroughResult(resp.status_code, call_id, "<streamed>", chunks)


def build_client() -> PassthroughClient:
    return PassthroughClient(**proxy_client_kwargs())
