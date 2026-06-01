"""
Handler for IBM watsonx Orchestrate (WXO) agent provider.

Authentication:
  - CP4D:      POST <cp4d_host>/icp4d-api/v1/authorize  → {"token": "..."}
  - IBM Cloud: POST https://iam.cloud.ibm.com/identity/token → {"access_token": "..."}

Execution:
  - Non-streaming: POST /v1/orchestrate/runs, then poll GET /v1/orchestrate/runs/{run_id}
  - Streaming:     POST /v1/orchestrate/runs/stream (SSE), falls back to poll + fake streaming
"""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Optional, cast

from litellm._logging import verbose_logger
from litellm.a2a_protocol.providers.watsonx_orchestrate.transformation import (
    WatsonxOrchestrateTransformation,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider

_IBM_CLOUD_IAM_URL = "https://iam.cloud.ibm.com/identity/token"
_POLL_INTERVAL_S = 2.0
_MAX_POLL_ATTEMPTS = 90  # ~3 minutes at 2s per poll


class WatsonxOrchestrateHandler:
    """
    Handler for IBM watsonx Orchestrate agent requests.
    """

    @staticmethod
    async def _get_bearer_token(
        cp4d_host: str,
        auth_mode: str,
        api_key: str,
        username: Optional[str] = None,
    ) -> str:
        """
        Obtain a WXO bearer token.

        auth_mode="cp4d"       → POST <cp4d_host>/icp4d-api/v1/authorize
        auth_mode="ibm_cloud"  → POST https://iam.cloud.ibm.com/identity/token
        """
        client = get_async_httpx_client(
            llm_provider=cast(Any, httpxSpecialProvider.A2AProvider),
            params={"timeout": 30.0},
        )

        if auth_mode == "ibm_cloud":
            verbose_logger.debug("WXO: Authenticating via IBM Cloud IAM")
            response = await client.post(
                _IBM_CLOUD_IAM_URL,
                data={
                    "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                    "apikey": api_key,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            return str(response.json()["access_token"])

        # Default: CP4D
        if not username:
            raise ValueError(
                "'username' is required in litellm_params when auth_mode='cp4d'"
            )
        token_url = f"{cp4d_host.rstrip('/')}/icp4d-api/v1/authorize"
        verbose_logger.debug(f"WXO: Authenticating via CP4D at {token_url}")
        response = await client.post(
            token_url,
            json={"username": username, "api_key": api_key},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return str(response.json()["token"])

    @staticmethod
    async def _poll_run(
        base_url: str,
        run_id: str,
        auth_headers: Dict[str, str],
        max_attempts: int = _MAX_POLL_ATTEMPTS,
        interval_s: float = _POLL_INTERVAL_S,
    ) -> Dict[str, Any]:
        """
        Poll GET /v1/orchestrate/runs/{run_id} until a terminal state is reached.
        """
        client = get_async_httpx_client(
            llm_provider=cast(Any, httpxSpecialProvider.A2AProvider),
            params={"timeout": 30.0},
        )
        url = f"{base_url}/v1/orchestrate/runs/{run_id}"

        for attempt in range(max_attempts):
            await asyncio.sleep(interval_s)
            response = await client.get(url, headers=auth_headers)
            response.raise_for_status()
            result: Dict[str, Any] = response.json()
            status = result.get("status", "")
            verbose_logger.debug(
                f"WXO: Poll {attempt + 1}/{max_attempts} run='{run_id}' status='{status}'"
            )
            if status in WatsonxOrchestrateTransformation.TERMINAL_STATES:
                return result

        raise TimeoutError(
            f"WXO run '{run_id}' did not reach a terminal state after "
            f"{max_attempts * interval_s:.0f}s"
        )

    @staticmethod
    def _extract_litellm_params(litellm_params: Dict[str, Any]) -> tuple:
        """Validate and extract required WXO params from litellm_params."""
        cp4d_host = litellm_params.get("cp4d_host") or ""
        instance_id = litellm_params.get("instance_id") or ""
        wxo_agent_id = litellm_params.get("wxo_agent_id") or ""
        api_key = litellm_params.get("api_key") or ""
        username = litellm_params.get("username") or None
        auth_mode = litellm_params.get("auth_mode") or "cp4d"
        thread_id = litellm_params.get("thread_id") or None

        if not cp4d_host:
            raise ValueError("'cp4d_host' is required in litellm_params for WXO agents")
        if not instance_id:
            raise ValueError(
                "'instance_id' is required in litellm_params for WXO agents"
            )
        if not wxo_agent_id:
            raise ValueError(
                "'wxo_agent_id' is required in litellm_params for WXO agents"
            )
        if not api_key:
            raise ValueError("'api_key' is required in litellm_params for WXO agents")

        return (
            cp4d_host,
            instance_id,
            wxo_agent_id,
            api_key,
            username,
            auth_mode,
            thread_id,
        )

    @staticmethod
    async def handle_non_streaming(
        request_id: str,
        params: Dict[str, Any],
        litellm_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Submit a WXO run and poll for completion, then return a standard A2A message response.
        """
        (
            cp4d_host,
            instance_id,
            wxo_agent_id,
            api_key,
            username,
            auth_mode,
            thread_id,
        ) = WatsonxOrchestrateHandler._extract_litellm_params(litellm_params)

        token = await WatsonxOrchestrateHandler._get_bearer_token(
            cp4d_host=cp4d_host,
            auth_mode=auth_mode,
            api_key=api_key,
            username=username,
        )
        base_url = WatsonxOrchestrateTransformation.get_api_base_url(
            cp4d_host, instance_id
        )
        auth_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        text = WatsonxOrchestrateTransformation.extract_text_from_a2a_params(params)
        body = WatsonxOrchestrateTransformation.build_wxo_run_body(
            wxo_agent_id=wxo_agent_id, text=text, thread_id=thread_id
        )

        verbose_logger.info(
            f"WXO: Submitting run for agent='{wxo_agent_id}' at {base_url}"
        )
        client = get_async_httpx_client(
            llm_provider=cast(Any, httpxSpecialProvider.A2AProvider),
            params={"timeout": 90.0},
        )
        run_response = await client.post(
            f"{base_url}/v1/orchestrate/runs",
            json=body,
            headers=auth_headers,
        )
        run_response.raise_for_status()
        run_data: Dict[str, Any] = run_response.json()

        status = run_data.get("status", "")
        verbose_logger.debug(f"WXO: Run submitted, initial status='{status}'")

        if status not in WatsonxOrchestrateTransformation.TERMINAL_STATES:
            run_id = run_data.get("run_id") or run_data.get("id") or ""
            if not run_id:
                raise ValueError(f"WXO: No run_id in response: {run_data}")
            verbose_logger.info(f"WXO: Polling run '{run_id}' for completion...")
            run_data = await WatsonxOrchestrateHandler._poll_run(
                base_url=base_url,
                run_id=run_id,
                auth_headers=auth_headers,
            )
            status = run_data.get("status", "")

        if status not in WatsonxOrchestrateTransformation.SUCCESS_STATES:
            raise RuntimeError(
                f"WXO run ended with non-success status '{status}': {run_data}"
            )

        response_text = WatsonxOrchestrateTransformation.extract_text_from_wxo_result(
            run_data
        )
        verbose_logger.info(
            f"WXO: Run completed successfully for request_id={request_id}"
        )

        return WatsonxOrchestrateTransformation.build_a2a_message_response(
            request_id=request_id, text=response_text
        )

    @staticmethod
    async def handle_streaming(
        request_id: str,
        params: Dict[str, Any],
        litellm_params: Dict[str, Any],
        chunk_size: int = 50,
        delay_ms: int = 10,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream a WXO run.

        Tries native SSE via POST /v1/orchestrate/runs/stream first.
        If that fails or returns non-SSE content, falls back to non-streaming
        poll + fake A2A streaming events.
        """
        (
            cp4d_host,
            instance_id,
            wxo_agent_id,
            api_key,
            username,
            auth_mode,
            thread_id,
        ) = WatsonxOrchestrateHandler._extract_litellm_params(litellm_params)

        token = await WatsonxOrchestrateHandler._get_bearer_token(
            cp4d_host=cp4d_host,
            auth_mode=auth_mode,
            api_key=api_key,
            username=username,
        )
        base_url = WatsonxOrchestrateTransformation.get_api_base_url(
            cp4d_host, instance_id
        )
        auth_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        text = WatsonxOrchestrateTransformation.extract_text_from_a2a_params(params)
        body = WatsonxOrchestrateTransformation.build_wxo_run_body(
            wxo_agent_id=wxo_agent_id, text=text, thread_id=thread_id
        )

        verbose_logger.info(f"WXO: Submitting streaming run for agent='{wxo_agent_id}'")

        try:
            import httpx as _httpx

            accumulated_text = ""
            async with _httpx.AsyncClient(verify=False, timeout=120.0) as http_client:
                async with http_client.stream(
                    "POST",
                    f"{base_url}/v1/orchestrate/runs/stream",
                    json=body,
                    headers=auth_headers,
                ) as stream_resp:
                    stream_resp.raise_for_status()
                    content_type = stream_resp.headers.get("content-type", "")

                    if "text/event-stream" not in content_type:
                        # Provider returned a plain JSON response; fake-stream it
                        response_body = await stream_resp.aread()
                        result = json.loads(response_body)
                        accumulated_text = WatsonxOrchestrateTransformation.extract_text_from_wxo_result(
                            result
                        )
                    else:
                        # Parse SSE lines and accumulate text from WXO events
                        async for line in stream_resp.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if not data_str or data_str == "[DONE]":
                                continue
                            try:
                                event = json.loads(data_str)
                                chunk_text = WatsonxOrchestrateTransformation.extract_text_from_wxo_result(
                                    event
                                )
                                if chunk_text:
                                    accumulated_text += chunk_text
                            except json.JSONDecodeError:
                                pass

            async for (
                chunk
            ) in WatsonxOrchestrateTransformation.fake_streaming_from_text(
                text=accumulated_text,
                request_id=request_id,
                chunk_size=chunk_size,
                delay_ms=delay_ms,
            ):
                yield chunk

        except Exception as exc:
            verbose_logger.warning(
                f"WXO: Streaming request failed ({exc!r}), "
                "falling back to non-streaming + fake streaming"
            )
            # Fallback: poll then fake-stream
            result = await WatsonxOrchestrateHandler.handle_non_streaming(
                request_id=request_id,
                params=params,
                litellm_params=litellm_params,
            )
            # Extract the text from the A2A message response we built
            response_text = ""
            try:
                response_text = result["result"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                pass
            async for (
                chunk
            ) in WatsonxOrchestrateTransformation.fake_streaming_from_text(
                text=response_text,
                request_id=request_id,
                chunk_size=chunk_size,
                delay_ms=delay_ms,
            ):
                yield chunk
