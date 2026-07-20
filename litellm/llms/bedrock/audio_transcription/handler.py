import json
from typing import Coroutine, Union

import httpx

import litellm
from litellm.llms.base_llm.audio_transcription.transformation import BaseAudioTranscriptionConfig
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.audio_transcription.transformation import (
    BedrockAudioTranscriptionConfig,
    get_bedrock_audio_model_id,
)
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.utils import FileTypes, TranscriptionResponse


class BedrockAudioTranscriptionHandler(BaseAWSLLM):
    def audio_transcriptions(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
        model_response: TranscriptionResponse,
        timeout: float,
        max_retries: int,
        logging_obj: object,
        api_key: str | None,
        api_base: str | None,
        client: Union[HTTPHandler, AsyncHTTPHandler] | None = None,
        atranscription: bool = False,
        headers: dict[str, str] | None = None,
        provider_config: BaseAudioTranscriptionConfig | None = None,
        shared_session: object | None = None,
    ) -> Union[TranscriptionResponse, Coroutine[object, object, TranscriptionResponse]]:
        config = provider_config or BedrockAudioTranscriptionConfig()
        if atranscription:
            return self.async_audio_transcriptions(
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
                headers=headers,
                provider_config=config,
                shared_session=shared_session,
            )

        request_headers, url, serialized_body = self._prepare_request(
            model=model,
            audio_file=audio_file,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
            headers=headers,
            provider_config=config,
        )
        http_client = client if isinstance(client, HTTPHandler) else _get_httpx_client()
        response = http_client.post(url=url, headers=request_headers, data=serialized_body, timeout=timeout)
        if response is None:
            raise BedrockError(status_code=500, message="Bedrock audio transcription returned no response")
        self._raise_for_status(response)
        return config.transform_audio_transcription_response(response)

    async def async_audio_transcriptions(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        model_response: TranscriptionResponse,
        timeout: float,
        logging_obj: object,
        api_key: str | None,
        api_base: str | None,
        client: AsyncHTTPHandler | None = None,
        headers: dict[str, str] | None = None,
        provider_config: BaseAudioTranscriptionConfig | None = None,
        shared_session: object | None = None,
    ) -> TranscriptionResponse:
        config = provider_config or BedrockAudioTranscriptionConfig()
        request_headers, url, serialized_body = self._prepare_request(
            model=model,
            audio_file=audio_file,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
            headers=headers,
            provider_config=config,
        )
        http_client = client or get_async_httpx_client(
            llm_provider=litellm.LlmProviders.BEDROCK,
            shared_session=shared_session,
        )
        response = await http_client.post(url=url, headers=request_headers, data=serialized_body, timeout=timeout)
        if response is None:
            raise BedrockError(status_code=500, message="Bedrock audio transcription returned no response")
        self._raise_for_status(response)
        return config.transform_audio_transcription_response(response)

    def _prepare_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        headers: dict[str, str] | None,
        provider_config: BaseAudioTranscriptionConfig,
    ) -> tuple[dict, str, str]:
        params = dict(optional_params)
        _, model_region = get_bedrock_audio_model_id(model)
        region = params.pop("aws_region_name", None) or model_region or self._get_aws_region_name(params, model=model)
        credentials = self.get_credentials(
            aws_access_key_id=params.pop("aws_access_key_id", None),
            aws_secret_access_key=params.pop("aws_secret_access_key", None),
            aws_session_token=params.pop("aws_session_token", None),
            aws_region_name=region,
            aws_session_name=params.pop("aws_session_name", None),
            aws_profile_name=params.pop("aws_profile_name", None),
            aws_role_name=params.pop("aws_role_name", None),
            aws_web_identity_token=params.pop("aws_web_identity_token", None),
            aws_sts_endpoint=params.pop("aws_sts_endpoint", None),
            aws_external_id=params.pop("aws_external_id", None),
        )
        url = provider_config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params={"aws_region_name": region, **params},
            litellm_params=litellm_params,
        )
        request_data = provider_config.transform_audio_transcription_request(
            model=model,
            audio_file=audio_file,
            optional_params=optional_params,
            litellm_params=litellm_params,
        ).data
        body = request_data if isinstance(request_data, dict) else {}
        serialized_body = json.dumps(body)
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        prepared = self.get_request_headers(
            credentials=credentials,
            aws_region_name=region,
            extra_headers=headers,
            endpoint_url=url,
            data=serialized_body,
            headers=request_headers,
            api_key=api_key,
        )
        return dict(prepared.headers), url, serialized_body

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        raise BedrockError(
            status_code=response.status_code,
            message=response.text,
            headers=response.headers,
        )
