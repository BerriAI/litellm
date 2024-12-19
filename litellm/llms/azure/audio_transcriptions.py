import uuid
from typing import Any, Optional

from openai import AsyncAzureOpenAI, AzureOpenAI
from pydantic import BaseModel

import litellm
from litellm.litellm_core_utils.audio_utils.utils import get_audio_file_name
from litellm.types.utils import FileTypes
from litellm.utils import TranscriptionResponse, convert_to_model_response_object

from .azure import (
    AzureChatCompletion,
    get_azure_ad_token_from_oidc,
    select_azure_base_url_or_endpoint,
)


class AzureAudioTranscription(AzureChatCompletion):
    def audio_transcriptions(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        logging_obj: Any,
        model_response: TranscriptionResponse,
        timeout: float,
        max_retries: int,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        client=None,
        azure_ad_token: Optional[str] = None,
        atranscription: bool = False,
    ) -> TranscriptionResponse:
        data = {"model": model, "file": audio_file, **optional_params}

        # init AzureOpenAI Client
        azure_client_params = {
            "api_version": api_version,
            "azure_endpoint": api_base,
            "azure_deployment": model,
            "timeout": timeout,
        }

        azure_client_params = select_azure_base_url_or_endpoint(
            azure_client_params=azure_client_params
        )
        if api_key is not None:
            azure_client_params["api_key"] = api_key
        elif azure_ad_token is not None:
            if azure_ad_token.startswith("oidc/"):
                azure_ad_token = get_azure_ad_token_from_oidc(azure_ad_token)
            azure_client_params["azure_ad_token"] = azure_ad_token

        if max_retries is not None:
            azure_client_params["max_retries"] = max_retries

        if atranscription is True:
            return self.async_audio_transcriptions(  # type: ignore
                audio_file=audio_file,
                data=data,
                model_response=model_response,
                timeout=timeout,
                api_key=api_key,
                api_base=api_base,
                client=client,
                azure_client_params=azure_client_params,
                max_retries=max_retries,
                logging_obj=logging_obj,
            )
        if client is None:
            azure_client = AzureOpenAI(http_client=litellm.client_session, **azure_client_params)  # type: ignore
        else:
            azure_client = client

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

        response = azure_client.audio.transcriptions.create(
            **data, timeout=timeout  # type: ignore
        )

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
        hidden_params = {"model": "whisper-1", "custom_llm_provider": "azure"}
        final_response: TranscriptionResponse = convert_to_model_response_object(response_object=stringified_response, model_response_object=model_response, hidden_params=hidden_params, response_type="audio_transcription")  # type: ignore
        return final_response

    async def async_audio_transcriptions(
        self,
        audio_file: FileTypes,
        data: dict,
        model_response: TranscriptionResponse,
        timeout: float,
        azure_client_params: dict,
        logging_obj: Any,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        client=None,
        max_retries=None,
    ):
        response = None
        try:
            if client is None:
                async_azure_client = AsyncAzureOpenAI(
                    **azure_client_params,
                    http_client=litellm.aclient_session,
                )
            else:
                async_azure_client = client

            ## LOGGING
            logging_obj.pre_call(
                input=f"audio_file_{uuid.uuid4()}",
                api_key=async_azure_client.api_key,
                additional_args={
                    "headers": {
                        "Authorization": f"Bearer {async_azure_client.api_key}"
                    },
                    "api_base": async_azure_client._base_url._uri_reference,
                    "atranscription": True,
                    "complete_input_dict": data,
                },
            )

            raw_response = (
                await async_azure_client.audio.transcriptions.with_raw_response.create(
                    **data, timeout=timeout
                )
            )  # type: ignore

            headers = dict(raw_response.headers)
            response = raw_response.parse()

            if isinstance(response, BaseModel):
                stringified_response = response.model_dump()
            else:
                stringified_response = TranscriptionResponse(text=response).model_dump()

            ## LOGGING
            logging_obj.post_call(
                input=get_audio_file_name(audio_file),
                api_key=api_key,
                additional_args={
                    "headers": {
                        "Authorization": f"Bearer {async_azure_client.api_key}"
                    },
                    "api_base": async_azure_client._base_url._uri_reference,
                    "atranscription": True,
                    "complete_input_dict": data,
                },
                original_response=stringified_response,
            )
            hidden_params = {"model": "whisper-1", "custom_llm_provider": "azure"}
            response = convert_to_model_response_object(
                _response_headers=headers,
                response_object=stringified_response,
                model_response_object=model_response,
                hidden_params=hidden_params,
                response_type="audio_transcription",
            )  # type: ignore
            return response
        except Exception as e:
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                original_response=str(e),
            )
            raise e
