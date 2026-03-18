import asyncio
import httpx
from typing import Any, Optional

import litellm
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import _get_httpx_client, AsyncHTTPHandler
from .transformation import GoogleCodeAssistConfig, GoogleCodeAssistError


class GoogleCodeAssistChat:
    """
    Handler for Google Code Assist API.
    Provides both synchronous and asynchronous completion methods.
    """

    def __init__(self) -> None:
        self.config = GoogleCodeAssistConfig()

    def completion(
        self,
        model: str,
        messages: list,
        model_response: litellm.utils.ModelResponse,
        print_verbose: Any,
        logging_obj: Any,
        optional_params: dict,
        litellm_params: dict,
        logger_fn=None,
    ) -> litellm.utils.ModelResponse:
        """
        Synchronous completion for Google Code Assist.
        """
        try:
            from litellm.llms.gemini.common_utils import get_gemini_oauth_token

            # 1. Get OAuth data
            gemini_auth_data = get_gemini_oauth_token()
            if not gemini_auth_data:
                raise GoogleCodeAssistError(
                    status_code=401,
                    message="Missing Gemini OAuth token. Run 'litellm-proxy gemini login' or set GEMINI_OAUTH_TOKEN.",
                )

            token = gemini_auth_data.get("token")
            initial_project_id = gemini_auth_data.get("project_id")
            if not token:
                raise GoogleCodeAssistError(
                    status_code=401,
                    message="Missing Gemini OAuth token value. Re-run 'litellm-proxy gemini login'.",
                )

            client = _get_httpx_client()

            # 2. MANDATORY HANDSHAKE: loadCodeAssist
            final_project_id = self._handle_handshake(client, token, initial_project_id)
            litellm_params["google_code_assist_project"] = final_project_id

            # 3. Transform request
            data = self.config.transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
            )

            # 4. Call Completion API
            url = "https://cloudcode-pa.googleapis.com/v1internal:generateContent"
            headers = self._get_headers(token)

            response = client.post(
                url=url,
                headers=headers,
                json=data,
            )
            response.raise_for_status()

            # 5. Transform response
            return self.config.transform_response(
                model=model,
                raw_response=response,
                model_response=model_response,
                logging_obj=logging_obj,
                request_data=data,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                encoding=None,
            )

        except Exception as e:
            raise self._handle_error(e)

    async def acompletion(
        self,
        model: str,
        messages: list,
        model_response: litellm.utils.ModelResponse,
        print_verbose: Any,
        logging_obj: Any,
        optional_params: dict,
        litellm_params: dict,
        logger_fn=None,
    ) -> litellm.utils.ModelResponse:
        """
        Asynchronous completion for Google Code Assist.
        """
        try:
            from litellm.llms.gemini.common_utils import get_gemini_oauth_token

            gemini_auth_data = await asyncio.to_thread(get_gemini_oauth_token)
            if not gemini_auth_data:
                raise GoogleCodeAssistError(
                    status_code=401,
                    message="Missing Gemini OAuth token. Run 'litellm-proxy gemini login' or set GEMINI_OAUTH_TOKEN.",
                )

            token = gemini_auth_data.get("token")
            initial_project_id = gemini_auth_data.get("project_id")
            if not token:
                raise GoogleCodeAssistError(
                    status_code=401,
                    message="Missing Gemini OAuth token value. Re-run 'litellm-proxy gemini login'.",
                )

            async_handler = AsyncHTTPHandler()
            try:
                final_project_id = await self._ahandle_handshake(
                    async_handler, token, initial_project_id
                )
                litellm_params["google_code_assist_project"] = final_project_id

                data = self.config.transform_request(
                    model, messages, optional_params, litellm_params
                )
                url = "https://cloudcode-pa.googleapis.com/v1internal:generateContent"
                headers = self._get_headers(token)

                response = await async_handler.post(url=url, headers=headers, json=data)
                response.raise_for_status()

                return self.config.transform_response(
                    model=model,
                    raw_response=response,
                    model_response=model_response,
                    logging_obj=logging_obj,
                    request_data=data,
                    messages=messages,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    encoding=None,
                )
            finally:
                try:
                    await async_handler.close()
                except Exception as close_error:
                    verbose_logger.debug(
                        f"Failed to close Google Code Assist async HTTP handler: {close_error}"
                    )

        except Exception as e:
            raise self._handle_error(e)

    def _handle_handshake(
        self, client, token: str, initial_project_id: Optional[str]
    ) -> Optional[str]:
        """Performs the loadCodeAssist handshake to establish session context."""
        load_url = "https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist"
        load_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "GeminiCLI/litellm",
        }
        load_payload = {
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            }
        }
        if initial_project_id:
            load_payload["cloudaicompanionProject"] = initial_project_id
            load_payload["metadata"]["duetProject"] = initial_project_id

        try:
            load_resp = client.post(load_url, headers=load_headers, json=load_payload)
            load_resp.raise_for_status()
            return load_resp.json().get("cloudaicompanionProject") or initial_project_id
        except Exception as e:
            verbose_logger.debug(f"Gemini Code Assist handshake failed: {e}")
            return initial_project_id

    async def _ahandle_handshake(
        self,
        async_handler: AsyncHTTPHandler,
        token: str,
        initial_project_id: Optional[str],
    ) -> Optional[str]:
        """Async version of loadCodeAssist handshake for non-blocking async calls."""
        load_url = "https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist"
        load_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "GeminiCLI/litellm",
        }
        load_payload = {
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            }
        }
        if initial_project_id:
            load_payload["cloudaicompanionProject"] = initial_project_id
            load_payload["metadata"]["duetProject"] = initial_project_id

        try:
            load_resp = await async_handler.post(
                url=load_url, headers=load_headers, json=load_payload
            )
            load_resp.raise_for_status()
            return load_resp.json().get("cloudaicompanionProject") or initial_project_id
        except Exception as e:
            verbose_logger.debug(f"Gemini Code Assist async handshake failed: {e}")
            return initial_project_id

    def _get_headers(self, token: str) -> dict:
        """Returns standard headers for Code Assist API calls."""
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "GeminiCLI/litellm",
            "X-Goog-Api-Client": "litellm-google-code-assist",
        }

    def _handle_error(self, e: Exception) -> Exception:
        """Centralized error mapping for Code Assist."""
        if isinstance(e, httpx.HTTPStatusError):
            return GoogleCodeAssistError(
                status_code=e.response.status_code, message=e.response.text
            )
        if isinstance(e, GoogleCodeAssistError):
            return e
        return GoogleCodeAssistError(status_code=500, message=str(e))
