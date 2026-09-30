from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, Final

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from pydantic import BaseModel

import litellm
from litellm._uuid import uuid
from litellm.litellm_core_utils.audio_utils.utils import get_audio_file_name
from litellm.llms.base_llm.audio_transcription.transformation import sdk_compatible_transcription_request_data
from litellm.types.utils import FileTypes
from litellm.utils import (
    TranscriptionResponse,
    convert_to_model_response_object,
    extract_duration_from_srt_or_vtt,
)

from .azure import AzureChatCompletion
from .common_utils import AzureOpenAIError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class AzureAudioTranscription(AzureChatCompletion):
    def audio_transcriptions(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        logging_obj: "LiteLLMLoggingObj",
        model_response: TranscriptionResponse,
        timeout: float,
        max_retries: int,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        client=None,
        azure_ad_token: str | None = None,
        atranscription: bool = False,
        litellm_params: dict | None = None,
        custom_llm_provider: str = "azure",
    ) -> TranscriptionResponse | Coroutine[Any, Any, TranscriptionResponse]:
        data: Final = {"model": model, "file": audio_file, **optional_params}
        sdk_data: Final = sdk_compatible_transcription_request_data(data)
        model_info: Final = litellm.model_cost.get(f"azure/{model}")
        provider_specific_entry: Final = model_info.get("provider_specific_entry") if model_info is not None else None
        requires_deployment_api: Final = model_info is None or (
            provider_specific_entry is not None and provider_specific_entry.get("transcription_deployment_api") == 1
        )
        resolved_api_version: Final = (
            litellm.AZURE_DEFAULT_API_VERSION
            if requires_deployment_api and api_version in ("v1", "latest", "preview")
            else api_version
        )

        if atranscription is True:
            return self.async_audio_transcriptions(
                audio_file=audio_file,
                data=sdk_data,
                model_response=model_response,
                timeout=timeout,
                api_key=api_key,
                api_base=api_base,
                api_version=resolved_api_version,
                client=client,
                max_retries=max_retries,
                logging_obj=logging_obj,
                model=model,
                litellm_params=litellm_params,
                custom_llm_provider=custom_llm_provider,
            )

        azure_client: Final = self.get_azure_openai_client(
            api_version=resolved_api_version,
            api_base=api_base,
            api_key=api_key,
            model=model,
            _is_async=False,
            client=client,
            litellm_params=litellm_params,
        )
        if not isinstance(azure_client, (AzureOpenAI, OpenAI)):
            raise AzureOpenAIError(
                status_code=500,
                message="azure_client is not an instance of AzureOpenAI",
            )

        ## LOGGING
        logging_obj.pre_call(
            input=f"audio_file_{uuid.uuid4()}",
            api_key=azure_client.api_key,
            additional_args={
                "headers": {"Authorization": f"Bearer {azure_client.api_key}"},
                "api_base": azure_client._base_url._uri_reference,
                "atranscription": True,
                "complete_input_dict": data,
            },
        )

        response: Final = azure_client.audio.transcriptions.create(
            **sdk_data,  # pyright: ignore[reportArgumentType]  # SDK TypedDict lags accepted transcription options
            timeout=timeout,
        )

        if data.get("stream") is True:
            return response

        if isinstance(response, BaseModel):
            stringified_response = response.model_dump()
        else:
            stringified_response = TranscriptionResponse(text=response).model_dump()

        ## LOGGING
        logging_obj.post_call(
            input=get_audio_file_name(audio_file),
            api_key=api_key,
            additional_args={"complete_input_dict": data},
            original_response=stringified_response,
        )
        hidden_params: Final = {"model": model, "custom_llm_provider": custom_llm_provider}
        final_response: Final[TranscriptionResponse] = convert_to_model_response_object(
            response_object=stringified_response,
            model_response_object=model_response,
            hidden_params=hidden_params,
            response_type="audio_transcription",
        )
        return final_response

    async def async_audio_transcriptions(
        self,
        audio_file: FileTypes,
        model: str,
        data: dict,
        model_response: TranscriptionResponse,
        timeout: float,
        logging_obj: "LiteLLMLoggingObj",
        api_version: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        client=None,
        max_retries=None,
        litellm_params: dict | None = None,
        custom_llm_provider: str = "azure",
    ) -> TranscriptionResponse:
        response = None
        try:
            async_azure_client: Final = self.get_azure_openai_client(
                api_version=api_version,
                api_base=api_base,
                api_key=api_key,
                model=model,
                _is_async=True,
                client=client,
                litellm_params=litellm_params,
            )
            if not isinstance(async_azure_client, (AsyncAzureOpenAI, AsyncOpenAI)):
                raise AzureOpenAIError(
                    status_code=500,
                    message="async_azure_client is not an instance of AsyncAzureOpenAI",
                )

            ## LOGGING
            logging_obj.pre_call(
                input=f"audio_file_{uuid.uuid4()}",
                api_key=async_azure_client.api_key,
                additional_args={
                    "headers": {"Authorization": f"Bearer {async_azure_client.api_key}"},
                    "api_base": async_azure_client._base_url._uri_reference,
                    "atranscription": True,
                    "complete_input_dict": data,
                },
            )

            if data.get("stream") is True:
                return await async_azure_client.audio.transcriptions.create(
                    **data,  # pyright: ignore[reportArgumentType]  # SDK TypedDict lags accepted transcription options
                    timeout=timeout,
                )

            raw_response: Final = await async_azure_client.audio.transcriptions.with_raw_response.create(
                **data,  # pyright: ignore[reportArgumentType]  # SDK TypedDict lags accepted transcription options
                timeout=timeout,
            )

            headers: Final = dict(raw_response.headers)
            response = raw_response.parse()

            if isinstance(response, BaseModel):
                stringified_response = response.model_dump()
            else:
                stringified_response = TranscriptionResponse(text=response).model_dump()
                duration: Final = extract_duration_from_srt_or_vtt(response)
                stringified_response["_audio_transcription_duration"] = duration

            ## LOGGING
            logging_obj.post_call(
                input=get_audio_file_name(audio_file),
                api_key=api_key,
                additional_args={
                    "headers": {"Authorization": f"Bearer {async_azure_client.api_key}"},
                    "api_base": async_azure_client._base_url._uri_reference,
                    "atranscription": True,
                    "complete_input_dict": data,
                },
                original_response=stringified_response,
            )
            hidden_params: Final = {"model": model, "custom_llm_provider": custom_llm_provider}
            response = convert_to_model_response_object(
                _response_headers=headers,
                response_object=stringified_response,
                model_response_object=model_response,
                hidden_params=hidden_params,
                response_type="audio_transcription",
            )
            if not isinstance(response, TranscriptionResponse):
                raise AzureOpenAIError(
                    status_code=500,
                    message="response is not an instance of TranscriptionResponse",
                )
            return response
        except Exception as e:
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                original_response=str(e),
            )
            raise e
