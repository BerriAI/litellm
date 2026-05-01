"""
Handler for Soniox async speech-to-text transcription.

Soniox's async transcription API requires multiple HTTP calls:
  1. (optional) POST /v1/files                      — upload a local audio file
  2.            POST /v1/transcriptions             — create a transcription job
  3.            GET  /v1/transcriptions/{id}        — poll until status == "completed"
  4.            GET  /v1/transcriptions/{id}/transcript — fetch the transcript
  5. (optional) DELETE /v1/transcriptions/{id}      — cleanup
  6. (optional) DELETE /v1/files/{id}               — cleanup

Because this does not fit the single-request shape of
`base_llm_http_handler.audio_transcriptions`, the dispatch in
`litellm.main.transcription()` routes Soniox requests directly to this
handler (analogous to the OpenAI / Azure transcription handlers).
"""

import asyncio
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import httpx

from litellm.litellm_core_utils.audio_utils.utils import (
    get_audio_file_name,
    process_audio_file,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.soniox.audio_transcription.transformation import (
    SonioxAudioTranscriptionConfig,
)
from litellm.llms.soniox.common_utils import (
    SONIOX_DEFAULT_CLEANUP,
    SONIOX_DEFAULT_MAX_POLL_ATTEMPTS,
    SONIOX_DEFAULT_POLL_INTERVAL,
    SonioxException,
    get_soniox_api_base,
)
from litellm.types.utils import FileTypes, TranscriptionResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )
else:
    LiteLLMLoggingObj = Any


class SonioxAudioTranscriptionHandler:
    """Orchestrates the Soniox async transcription flow."""

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def audio_transcriptions(
        self,
        model: str,
        audio_file: Optional[FileTypes],
        optional_params: dict,
        litellm_params: dict,
        model_response: TranscriptionResponse,
        timeout: float,
        max_retries: int,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        api_base: Optional[str],
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        atranscription: bool = False,
        headers: Optional[Dict[str, Any]] = None,
        provider_config: Optional[SonioxAudioTranscriptionConfig] = None,
    ) -> Union[TranscriptionResponse, Coroutine[Any, Any, TranscriptionResponse]]:
        """Sync/async dispatch for Soniox transcription requests.

        Note: ``max_retries`` is accepted for signature compatibility with
        ``litellm.transcription`` but is **not yet implemented** for the Soniox
        async pipeline. Transient HTTP failures during upload, create, poll,
        or fetch will surface immediately. Wrap calls with the standard
        ``litellm.Router`` / ``num_retries`` mechanism for retry behaviour.
        """
        config = provider_config or SonioxAudioTranscriptionConfig()

        if atranscription is True:
            return self._async_audio_transcriptions(
                model=model,
                audio_file=audio_file,
                optional_params=optional_params,
                litellm_params=litellm_params,
                model_response=model_response,
                timeout=timeout,
                logging_obj=logging_obj,
                api_key=api_key,
                api_base=api_base,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
                headers=headers or {},
                provider_config=config,
            )

        return self._sync_audio_transcriptions(
            model=model,
            audio_file=audio_file,
            optional_params=optional_params,
            litellm_params=litellm_params,
            model_response=model_response,
            timeout=timeout,
            logging_obj=logging_obj,
            api_key=api_key,
            api_base=api_base,
            client=client if isinstance(client, HTTPHandler) else None,
            headers=headers or {},
            provider_config=config,
        )

    # ------------------------------------------------------------------
    # Helpers shared between sync and async paths
    # ------------------------------------------------------------------

    def _prepare(
        self,
        audio_file: Optional[FileTypes],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        provider_config: SonioxAudioTranscriptionConfig,
        headers: Dict[str, Any],
    ) -> Tuple[
        Dict[str, str],  # auth headers
        str,  # api_base (no trailing slash)
        Dict[str, Any],  # body for POST /v1/transcriptions (without file_id/audio_url)
        Dict[str, Any],  # handler-only options (poll interval, cleanup, ...)
    ]:
        # Validate env -> auth headers.
        auth_headers = provider_config.validate_environment(
            headers=headers,
            model="",  # unused
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )

        base_url = get_soniox_api_base(api_base)

        # Operate on a local copy so we don't mutate the caller's dict
        # (the caller may reuse `optional_params` for retries or logging).
        params = dict(optional_params)

        # Pull handler-only kwargs out of params so they aren't sent
        # to Soniox.
        poll_interval = float(
            params.pop("soniox_polling_interval", SONIOX_DEFAULT_POLL_INTERVAL)
        )
        max_attempts = int(
            params.pop("soniox_max_polling_attempts", SONIOX_DEFAULT_MAX_POLL_ATTEMPTS)
        )
        cleanup_raw = params.pop("soniox_cleanup", SONIOX_DEFAULT_CLEANUP)
        if cleanup_raw is None:
            cleanup: List[str] = []
        elif isinstance(cleanup_raw, str):
            cleanup = [cleanup_raw]
        else:
            cleanup = list(cleanup_raw)
        filename_override = params.pop("filename", None)

        handler_opts: Dict[str, Any] = {
            "poll_interval": max(poll_interval, 0.0),
            "max_attempts": max(max_attempts, 1),
            "cleanup": cleanup,
            "filename_override": filename_override,
            "audio_url": params.pop("audio_url", None),
            "file_id": params.pop("file_id", None),
        }

        # Soniox does not accept `language` directly; map_openai_params should
        # already have translated it, but drop any leftover to be safe.
        params.pop("language", None)

        return auth_headers, base_url, params, handler_opts

    def _build_create_body(
        self,
        model: str,
        optional_params: dict,
        handler_opts: Dict[str, Any],
        file_id: Optional[str],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model}
        # Soniox-native passthrough fields
        for key, value in optional_params.items():
            if value is None:
                continue
            body[key] = value

        if handler_opts.get("audio_url"):
            body["audio_url"] = handler_opts["audio_url"]
        if file_id:
            body["file_id"] = file_id

        return body

    @staticmethod
    def _safe_log_pre_call(
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        api_base: str,
        body: Dict[str, Any],
    ) -> None:
        try:
            logging_obj.pre_call(
                input=None,
                api_key=api_key,
                additional_args={
                    "api_base": f"{api_base}/v1/transcriptions",
                    "atranscription": True,
                    "complete_input_dict": body,
                },
            )
        except Exception:
            # Logging hooks are best-effort: a misbehaving callback or third-party
            # observability integration must never break a real Soniox call.
            pass

    @staticmethod
    def _safe_log_post_call(
        logging_obj: LiteLLMLoggingObj,
        audio_file: Optional[FileTypes],
        api_key: Optional[str],
        body: Dict[str, Any],
        original_response: Any,
    ) -> None:
        try:
            logging_obj.post_call(
                input=get_audio_file_name(audio_file) if audio_file else None,
                api_key=api_key,
                additional_args={"complete_input_dict": body},
                original_response=original_response,
            )
        except Exception:
            # Logging hooks are best-effort: a misbehaving callback or third-party
            # observability integration must never break a real Soniox call.
            pass

    @staticmethod
    def _raise_for_response(
        response: httpx.Response,
        provider_config: SonioxAudioTranscriptionConfig,
        action: str,
    ) -> None:
        if response.status_code >= 400:
            try:
                payload = response.json()
                message = (
                    payload.get("error_message")
                    or payload.get("error")
                    or response.text
                )
            except Exception:
                message = response.text
            raise provider_config.get_error_class(
                error_message=f"Soniox {action} failed (HTTP {response.status_code}): {message}",
                status_code=response.status_code,
                headers=response.headers,
            )

    # ------------------------------------------------------------------
    # Sync flow
    # ------------------------------------------------------------------

    def _sync_audio_transcriptions(
        self,
        model: str,
        audio_file: Optional[FileTypes],
        optional_params: dict,
        litellm_params: dict,
        model_response: TranscriptionResponse,
        timeout: float,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        api_base: Optional[str],
        client: Optional[HTTPHandler],
        headers: Dict[str, Any],
        provider_config: SonioxAudioTranscriptionConfig,
    ) -> TranscriptionResponse:
        auth_headers, base_url, opt_params, handler_opts = self._prepare(
            audio_file=audio_file,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
            provider_config=provider_config,
            headers=headers,
        )

        http_client = client if isinstance(client, HTTPHandler) else _get_httpx_client()

        file_id = handler_opts.get("file_id")
        uploaded_file_id: Optional[str] = None
        transcription_id: Optional[str] = None

        try:
            if not file_id and not handler_opts.get("audio_url"):
                if audio_file is None:
                    raise SonioxException(
                        message=(
                            "Soniox transcription requires one of: a file argument, "
                            "an `audio_url` kwarg, or a `file_id` kwarg."
                        ),
                        status_code=400,
                        headers=None,
                    )
                uploaded_file_id = self._sync_upload_file(
                    http_client=http_client,
                    base_url=base_url,
                    auth_headers=auth_headers,
                    audio_file=audio_file,
                    filename_override=handler_opts.get("filename_override"),
                    timeout=timeout,
                    provider_config=provider_config,
                )
                file_id = uploaded_file_id

            body = self._build_create_body(model, opt_params, handler_opts, file_id)
            self._safe_log_pre_call(logging_obj, api_key, base_url, body)

            create_resp = http_client.post(
                url=f"{base_url}/v1/transcriptions",
                headers=auth_headers,
                json=body,
                timeout=timeout,
            )
            self._raise_for_response(
                create_resp, provider_config, "create transcription"
            )
            transcription_id = create_resp.json()["id"]

            transcription_meta = self._sync_poll_until_completed(
                http_client=http_client,
                base_url=base_url,
                auth_headers=auth_headers,
                transcription_id=transcription_id,
                poll_interval=handler_opts["poll_interval"],
                max_attempts=handler_opts["max_attempts"],
                timeout=timeout,
                provider_config=provider_config,
            )

            transcript_resp = http_client.get(
                url=f"{base_url}/v1/transcriptions/{transcription_id}/transcript",
                headers=auth_headers,
                timeout=timeout,
            )
            self._raise_for_response(
                transcript_resp, provider_config, "fetch transcript"
            )
            transcript = transcript_resp.json()

            payload = {"transcription": transcription_meta, "transcript": transcript}
            response = provider_config._build_response_from_payload(payload)

            self._safe_log_post_call(logging_obj, audio_file, api_key, body, payload)

            # Carry through hidden_params hints expected by the rest of litellm.
            response._hidden_params.update(
                {"model": model, "custom_llm_provider": "soniox"}
            )
            return response
        finally:
            self._sync_cleanup(
                http_client=http_client,
                base_url=base_url,
                auth_headers=auth_headers,
                cleanup=handler_opts["cleanup"],
                file_id_to_cleanup=uploaded_file_id,
                transcription_id=transcription_id,
                timeout=timeout,
            )

    def _sync_upload_file(
        self,
        http_client: HTTPHandler,
        base_url: str,
        auth_headers: Dict[str, str],
        audio_file: FileTypes,
        filename_override: Optional[str],
        timeout: float,
        provider_config: SonioxAudioTranscriptionConfig,
    ) -> str:
        processed = process_audio_file(audio_file)
        filename = filename_override or processed.filename
        files = {
            "file": (filename, processed.file_content, processed.content_type),
        }
        # `Authorization` header is fine; httpx sets multipart Content-Type.
        upload_headers = {"Authorization": auth_headers["Authorization"]}
        resp = http_client.post(
            url=f"{base_url}/v1/files",
            headers=upload_headers,
            files=files,
            timeout=timeout,
        )
        self._raise_for_response(resp, provider_config, "upload file")
        return resp.json()["id"]

    def _sync_poll_until_completed(
        self,
        http_client: HTTPHandler,
        base_url: str,
        auth_headers: Dict[str, str],
        transcription_id: str,
        poll_interval: float,
        max_attempts: int,
        timeout: float,
        provider_config: SonioxAudioTranscriptionConfig,
    ) -> Dict[str, Any]:
        for _ in range(max_attempts):
            resp = http_client.get(
                url=f"{base_url}/v1/transcriptions/{transcription_id}",
                headers=auth_headers,
                timeout=timeout,
            )
            self._raise_for_response(resp, provider_config, "poll transcription")
            data = resp.json()
            status = data.get("status")
            if status == "completed":
                return data
            if status == "error":
                raise provider_config.get_error_class(
                    error_message=(
                        f"Soniox transcription {transcription_id} failed: "
                        f"{data.get('error_message') or data.get('error_type') or 'unknown error'}"
                    ),
                    status_code=500,
                    headers=resp.headers,
                )
            time.sleep(poll_interval)
        raise provider_config.get_error_class(
            error_message=(
                f"Soniox transcription {transcription_id} did not complete after "
                f"{max_attempts} polling attempts (interval={poll_interval}s)."
            ),
            status_code=504,
            headers=None,
        )

    def _sync_cleanup(
        self,
        http_client: HTTPHandler,
        base_url: str,
        auth_headers: Dict[str, str],
        cleanup: List[str],
        file_id_to_cleanup: Optional[str],
        transcription_id: Optional[str],
        timeout: float,
    ) -> None:
        if not cleanup:
            return
        if "transcription" in cleanup and transcription_id:
            try:
                http_client.delete(
                    url=f"{base_url}/v1/transcriptions/{transcription_id}",
                    headers=auth_headers,
                    timeout=timeout,
                )
            except Exception:
                # Cleanup is best-effort: a failed delete leaves stale data on
                # Soniox but must not mask the original transcription result
                # (or, on the error path, the original error).
                pass
        if "file" in cleanup and file_id_to_cleanup:
            try:
                http_client.delete(
                    url=f"{base_url}/v1/files/{file_id_to_cleanup}",
                    headers=auth_headers,
                    timeout=timeout,
                )
            except Exception:
                # Cleanup is best-effort; see comment above.
                pass

    # ------------------------------------------------------------------
    # Async flow
    # ------------------------------------------------------------------

    async def _async_audio_transcriptions(
        self,
        model: str,
        audio_file: Optional[FileTypes],
        optional_params: dict,
        litellm_params: dict,
        model_response: TranscriptionResponse,
        timeout: float,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        api_base: Optional[str],
        client: Optional[AsyncHTTPHandler],
        headers: Dict[str, Any],
        provider_config: SonioxAudioTranscriptionConfig,
    ) -> TranscriptionResponse:
        import litellm

        auth_headers, base_url, opt_params, handler_opts = self._prepare(
            audio_file=audio_file,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
            provider_config=provider_config,
            headers=headers,
        )

        http_client = (
            client
            if isinstance(client, AsyncHTTPHandler)
            else (
                get_async_httpx_client(
                    llm_provider=litellm.LlmProviders.SONIOX,
                    params={"ssl_verify": litellm_params.get("ssl_verify", None)},
                )
            )
        )

        file_id = handler_opts.get("file_id")
        uploaded_file_id: Optional[str] = None
        transcription_id: Optional[str] = None

        try:
            if not file_id and not handler_opts.get("audio_url"):
                if audio_file is None:
                    raise SonioxException(
                        message=(
                            "Soniox transcription requires one of: a file argument, "
                            "an `audio_url` kwarg, or a `file_id` kwarg."
                        ),
                        status_code=400,
                        headers=None,
                    )
                uploaded_file_id = await self._async_upload_file(
                    http_client=http_client,
                    base_url=base_url,
                    auth_headers=auth_headers,
                    audio_file=audio_file,
                    filename_override=handler_opts.get("filename_override"),
                    timeout=timeout,
                    provider_config=provider_config,
                )
                file_id = uploaded_file_id

            body = self._build_create_body(model, opt_params, handler_opts, file_id)
            self._safe_log_pre_call(logging_obj, api_key, base_url, body)

            create_resp = await http_client.post(
                url=f"{base_url}/v1/transcriptions",
                headers=auth_headers,
                json=body,
                timeout=timeout,
            )
            self._raise_for_response(
                create_resp, provider_config, "create transcription"
            )
            transcription_id = create_resp.json()["id"]

            transcription_meta = await self._async_poll_until_completed(
                http_client=http_client,
                base_url=base_url,
                auth_headers=auth_headers,
                transcription_id=transcription_id,
                poll_interval=handler_opts["poll_interval"],
                max_attempts=handler_opts["max_attempts"],
                timeout=timeout,
                provider_config=provider_config,
            )

            transcript_resp = await http_client.get(
                url=f"{base_url}/v1/transcriptions/{transcription_id}/transcript",
                headers=auth_headers,
                timeout=timeout,
            )
            self._raise_for_response(
                transcript_resp, provider_config, "fetch transcript"
            )
            transcript = transcript_resp.json()

            payload = {"transcription": transcription_meta, "transcript": transcript}
            response = provider_config._build_response_from_payload(payload)

            self._safe_log_post_call(logging_obj, audio_file, api_key, body, payload)

            response._hidden_params.update(
                {"model": model, "custom_llm_provider": "soniox"}
            )
            return response
        finally:
            await self._async_cleanup(
                http_client=http_client,
                base_url=base_url,
                auth_headers=auth_headers,
                cleanup=handler_opts["cleanup"],
                file_id_to_cleanup=uploaded_file_id,
                transcription_id=transcription_id,
                timeout=timeout,
            )

    async def _async_upload_file(
        self,
        http_client: AsyncHTTPHandler,
        base_url: str,
        auth_headers: Dict[str, str],
        audio_file: FileTypes,
        filename_override: Optional[str],
        timeout: float,
        provider_config: SonioxAudioTranscriptionConfig,
    ) -> str:
        processed = process_audio_file(audio_file)
        filename = filename_override or processed.filename
        files = {
            "file": (filename, processed.file_content, processed.content_type),
        }
        upload_headers = {"Authorization": auth_headers["Authorization"]}
        resp = await http_client.post(
            url=f"{base_url}/v1/files",
            headers=upload_headers,
            files=files,
            timeout=timeout,
        )
        self._raise_for_response(resp, provider_config, "upload file")
        return resp.json()["id"]

    async def _async_poll_until_completed(
        self,
        http_client: AsyncHTTPHandler,
        base_url: str,
        auth_headers: Dict[str, str],
        transcription_id: str,
        poll_interval: float,
        max_attempts: int,
        timeout: float,
        provider_config: SonioxAudioTranscriptionConfig,
    ) -> Dict[str, Any]:
        for _ in range(max_attempts):
            resp = await http_client.get(
                url=f"{base_url}/v1/transcriptions/{transcription_id}",
                headers=auth_headers,
                timeout=timeout,
            )
            self._raise_for_response(resp, provider_config, "poll transcription")
            data = resp.json()
            status = data.get("status")
            if status == "completed":
                return data
            if status == "error":
                raise provider_config.get_error_class(
                    error_message=(
                        f"Soniox transcription {transcription_id} failed: "
                        f"{data.get('error_message') or data.get('error_type') or 'unknown error'}"
                    ),
                    status_code=500,
                    headers=resp.headers,
                )
            await asyncio.sleep(poll_interval)
        raise provider_config.get_error_class(
            error_message=(
                f"Soniox transcription {transcription_id} did not complete after "
                f"{max_attempts} polling attempts (interval={poll_interval}s)."
            ),
            status_code=504,
            headers=None,
        )

    async def _async_cleanup(
        self,
        http_client: AsyncHTTPHandler,
        base_url: str,
        auth_headers: Dict[str, str],
        cleanup: List[str],
        file_id_to_cleanup: Optional[str],
        transcription_id: Optional[str],
        timeout: float,
    ) -> None:
        if not cleanup:
            return
        if "transcription" in cleanup and transcription_id:
            try:
                await http_client.delete(
                    url=f"{base_url}/v1/transcriptions/{transcription_id}",
                    headers=auth_headers,
                    timeout=timeout,
                )
            except Exception:
                # Cleanup is best-effort: a failed delete leaves stale data on
                # Soniox but must not mask the original transcription result
                # (or, on the error path, the original error).
                pass
        if "file" in cleanup and file_id_to_cleanup:
            try:
                await http_client.delete(
                    url=f"{base_url}/v1/files/{file_id_to_cleanup}",
                    headers=auth_headers,
                    timeout=timeout,
                )
            except Exception:
                # Cleanup is best-effort; see comment above.
                pass
