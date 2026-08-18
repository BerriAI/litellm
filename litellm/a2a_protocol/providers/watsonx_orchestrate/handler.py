"""
Handler for IBM watsonx Orchestrate (WXO) agent provider.
"""

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator
from typing import Any, Final, NamedTuple, cast

import httpx

from litellm._logging import verbose_logger
from litellm.a2a_protocol.providers.watsonx_orchestrate.transformation import (
    WatsonxOrchestrateTransformation,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
)
from litellm.types.llms.custom_http import httpxSpecialProvider

_IBM_CLOUD_IAM_URL: Final = "https://iam.cloud.ibm.com/identity/token"
_POLL_INTERVAL_S: Final = 2.0
_MAX_POLL_ATTEMPTS: Final = 90
_TOKEN_CACHE_TTL_BUFFER_S: Final = 60
_token_cache: Final[dict[str, tuple[str, float]]] = {}


class WXORequestParams(NamedTuple):
    cp4d_host: str
    instance_id: str
    wxo_agent_id: str
    api_key: str
    username: str | None
    auth_mode: str
    thread_id: str | None


class WatsonxOrchestrateHandler:
    @staticmethod
    def _http_client(timeout: float = 90.0) -> AsyncHTTPHandler:
        return get_async_httpx_client(
            llm_provider=cast(Any, httpxSpecialProvider.A2AProvider),
            params={"timeout": timeout},
        )

    @staticmethod
    def _token_cache_key(
        auth_mode: str,
        cp4d_host: str,
        api_key: str,
        username: str | None,
    ) -> str:
        material: Final = f"{auth_mode}:{cp4d_host}:{username or ''}:{api_key}"
        return hashlib.sha256(material.encode()).hexdigest()

    @staticmethod
    def _cp4d_token_ttl_seconds(expiration: Any, now_wall: float | None = None) -> int:
        # CP4D returns expiration as absolute Unix epoch seconds, not a duration.
        expires_at: Final = int(expiration)
        wall: Final = now_wall if now_wall is not None else time.time()
        return max(expires_at - int(wall), 0)

    @staticmethod
    async def _get_bearer_token(
        cp4d_host: str,
        auth_mode: str,
        api_key: str,
        username: str | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> str:
        cache_key: Final = WatsonxOrchestrateHandler._token_cache_key(auth_mode, cp4d_host, api_key, username)
        now: Final = time.monotonic()
        cached: Final = _token_cache.get(cache_key)
        if cached and cached[1] > now:
            return cached[0]

        if client is None:
            client = WatsonxOrchestrateHandler._http_client(timeout=30.0)

        if auth_mode == "ibm_cloud":
            response = await client.post(
                _IBM_CLOUD_IAM_URL,
                data={
                    "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                    "apikey": api_key,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            payload = response.json()
            token = str(payload["access_token"])
            ttl_s = int(payload.get("expires_in", 3600))
        else:
            if not username:
                raise ValueError("'username' is required in litellm_params when auth_mode='cp4d'")
            token_url: Final = f"{cp4d_host.rstrip('/')}/icp4d-api/v1/authorize"
            response = await client.post(
                token_url,
                json={"username": username, "api_key": api_key},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            payload = response.json()
            token = str(payload["token"])
            expiration: Final = payload.get("expiration")
            if expiration is None:
                ttl_s = 3600
            else:
                ttl_s = WatsonxOrchestrateHandler._cp4d_token_ttl_seconds(expiration)

        expires_at: Final = now + max(ttl_s - _TOKEN_CACHE_TTL_BUFFER_S, 0)
        _token_cache[cache_key] = (token, expires_at)
        for stale_key, (_, stale_expires_at) in list(_token_cache.items()):
            if stale_expires_at <= now:
                del _token_cache[stale_key]
        return token

    @staticmethod
    async def _poll_run(
        base_url: str,
        run_id: str,
        auth_headers: dict[str, str],
        client: AsyncHTTPHandler,
        max_attempts: int = _MAX_POLL_ATTEMPTS,
        interval_s: float = _POLL_INTERVAL_S,
    ) -> dict[str, Any]:
        url: Final = f"{base_url}/v1/orchestrate/runs/{run_id}"

        for attempt in range(max_attempts):
            await asyncio.sleep(interval_s)
            response = await client.get(url, headers=auth_headers)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            status = result.get("status", "")
            verbose_logger.debug("WXO: Poll %s/%s run='%s' status='%s'", attempt + 1, max_attempts, run_id, status)
            if status in WatsonxOrchestrateTransformation.TERMINAL_STATES:
                return result

        raise asyncio.TimeoutError(
            f"WXO run '{run_id}' did not reach a terminal state after {max_attempts * interval_s:.0f}s"
        )

    @staticmethod
    async def _get_successful_run_data(
        run_data: dict[str, Any],
        base_url: str,
        auth_headers: dict[str, str],
        client: AsyncHTTPHandler,
    ) -> dict[str, Any]:
        status = run_data.get("status", "")
        if status not in WatsonxOrchestrateTransformation.TERMINAL_STATES:
            run_id: Final = run_data.get("run_id") or run_data.get("id") or ""
            if not run_id:
                raise ValueError(f"WXO: No run_id in response: {run_data}")
            run_data = await WatsonxOrchestrateHandler._poll_run(
                base_url=base_url,
                run_id=run_id,
                auth_headers=auth_headers,
                client=client,
            )
            status = run_data.get("status", "")

        if status not in WatsonxOrchestrateTransformation.SUCCESS_STATES:
            raise RuntimeError(f"WXO run ended with non-success status '{status}': {run_data}")

        return run_data

    @staticmethod
    async def _accumulate_wxo_sse_text(response: Any) -> str:
        accumulated_text = ""
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            chunk_text = WatsonxOrchestrateTransformation.extract_text_from_wxo_result(event)
            if chunk_text:
                accumulated_text += chunk_text
        return accumulated_text

    @staticmethod
    def _extract_litellm_params(litellm_params: dict[str, Any]) -> WXORequestParams:
        cp4d_host: Final = litellm_params.get("cp4d_host") or ""
        instance_id: Final = litellm_params.get("instance_id") or ""
        wxo_agent_id: Final = litellm_params.get("wxo_agent_id") or ""
        api_key: Final = litellm_params.get("api_key") or ""

        if not cp4d_host:
            raise ValueError("'cp4d_host' is required in litellm_params for WXO agents")
        if not instance_id:
            raise ValueError("'instance_id' is required in litellm_params for WXO agents")
        if not wxo_agent_id:
            raise ValueError("'wxo_agent_id' is required in litellm_params for WXO agents")
        if not api_key:
            raise ValueError("'api_key' is required in litellm_params for WXO agents")

        return WXORequestParams(
            cp4d_host=cp4d_host,
            instance_id=instance_id,
            wxo_agent_id=wxo_agent_id,
            api_key=api_key,
            username=litellm_params.get("username") or None,
            auth_mode=litellm_params.get("auth_mode") or "cp4d",
            thread_id=litellm_params.get("thread_id") or None,
        )

    @staticmethod
    async def handle_non_streaming(
        request_id: str,
        params: dict[str, Any],
        litellm_params: dict[str, Any],
    ) -> dict[str, Any]:
        wxo: Final = WatsonxOrchestrateHandler._extract_litellm_params(litellm_params)

        client: Final = WatsonxOrchestrateHandler._http_client(timeout=90.0)
        token: Final = await WatsonxOrchestrateHandler._get_bearer_token(
            cp4d_host=wxo.cp4d_host,
            auth_mode=wxo.auth_mode,
            api_key=wxo.api_key,
            username=wxo.username,
            client=client,
        )
        base_url: Final = WatsonxOrchestrateTransformation.get_api_base_url(wxo.cp4d_host, wxo.instance_id)
        auth_headers: Final = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        text: Final = WatsonxOrchestrateTransformation.extract_text_from_a2a_params(params)
        body: Final = WatsonxOrchestrateTransformation.build_wxo_run_body(
            wxo_agent_id=wxo.wxo_agent_id, text=text, thread_id=wxo.thread_id
        )

        run_response: Final = await client.post(
            f"{base_url}/v1/orchestrate/runs",
            json=body,
            headers=auth_headers,
        )
        run_response.raise_for_status()
        run_data: dict[str, Any] = run_response.json()

        run_data = await WatsonxOrchestrateHandler._get_successful_run_data(
            run_data=run_data,
            base_url=base_url,
            auth_headers=auth_headers,
            client=client,
        )

        response_text: Final = WatsonxOrchestrateTransformation.extract_text_from_wxo_result(run_data)
        return WatsonxOrchestrateTransformation.build_a2a_message_response(request_id=request_id, text=response_text)

    @staticmethod
    async def handle_streaming(
        request_id: str,
        params: dict[str, Any],
        litellm_params: dict[str, Any],
        chunk_size: int = 50,
        delay_ms: int = 10,
    ) -> AsyncIterator[dict[str, Any]]:
        wxo: Final = WatsonxOrchestrateHandler._extract_litellm_params(litellm_params)

        client: Final = WatsonxOrchestrateHandler._http_client(timeout=120.0)
        token: Final = await WatsonxOrchestrateHandler._get_bearer_token(
            cp4d_host=wxo.cp4d_host,
            auth_mode=wxo.auth_mode,
            api_key=wxo.api_key,
            username=wxo.username,
            client=client,
        )
        base_url: Final = WatsonxOrchestrateTransformation.get_api_base_url(wxo.cp4d_host, wxo.instance_id)
        auth_headers: Final = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json",
        }
        text: Final = WatsonxOrchestrateTransformation.extract_text_from_a2a_params(params)
        body: Final = WatsonxOrchestrateTransformation.build_wxo_run_body(
            wxo_agent_id=wxo.wxo_agent_id, text=text, thread_id=wxo.thread_id
        )

        try:
            response: Final = await client.post(
                f"{base_url}/v1/orchestrate/runs/stream",
                json=body,
                headers=auth_headers,
                stream=True,
            )
            response.raise_for_status()
        except httpx.TransportError as exc:
            verbose_logger.warning(
                "WXO: Streaming request failed before a run was submitted (%r), falling back to non-streaming + fake streaming",
                exc,
                exc_info=True,
            )
            result = await WatsonxOrchestrateHandler.handle_non_streaming(
                request_id=request_id,
                params=params,
                litellm_params=litellm_params,
            )
            response_text: Final = WatsonxOrchestrateTransformation.extract_text_from_a2a_message_response(result)
            async for chunk in WatsonxOrchestrateTransformation.fake_streaming_from_text(
                text=response_text,
                request_id=request_id,
                chunk_size=chunk_size,
                delay_ms=delay_ms,
            ):
                yield chunk
            return

        content_type: Final = response.headers.get("content-type", "").lower()
        if "text/event-stream" not in content_type:
            response_body: Final = await response.aread()
            result = json.loads(response_body)
            result = await WatsonxOrchestrateHandler._get_successful_run_data(
                run_data=result,
                base_url=base_url,
                auth_headers=auth_headers,
                client=client,
            )
            accumulated_text = WatsonxOrchestrateTransformation.extract_text_from_wxo_result(result)
        else:
            accumulated_text = await WatsonxOrchestrateHandler._accumulate_wxo_sse_text(response)

        async for chunk in WatsonxOrchestrateTransformation.fake_streaming_from_text(
            text=accumulated_text,
            request_id=request_id,
            chunk_size=chunk_size,
            delay_ms=delay_ms,
        ):
            yield chunk
