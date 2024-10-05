from typing import Optional, Union

import httpx
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

import litellm
from litellm.litellm_core_utils.audio_utils.utils import get_audio_file_name
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.utils import FileTypes
from litellm.utils import TranscriptionResponse, convert_to_model_response_object

from .openai import OpenAIChatCompletion


class OpenAIAudioTranscription(OpenAIChatCompletion):
    # Audio Transcriptions
    async def make_openai_audio_transcriptions_request(
        self,
        openai_aclient: AsyncOpenAI,
        data: dict,
        timeout: Union[float, httpx.Timeout],
    ):
        """
        Helper to:
        - call openai_aclient.audio.transcriptions.with_raw_response when litellm.return_response_headers is True
        - call openai_aclient.audio.transcriptions.create by default
        """
        try:
            if litellm.return_response_headers is True:
                raw_response = (
                    await openai_aclient.audio.transcriptions.with_raw_response.create(
                        **data, timeout=timeout
                    )
                )  # type: ignore
                headers = dict(raw_response.headers)
                response = raw_response.parse()
                return headers, response
            else:
                response = await openai_aclient.audio.transcriptions.create(**data, timeout=timeout)  # type: ignore
                return None, response
        except Exception as e:
            raise e

    def make_sync_openai_audio_transcriptions_request(
        self,
        openai_client: OpenAI,
        data: dict,
        timeout: Union[float, httpx.Timeout],
    ):
        """
        Helper to:
        - call openai_aclient.audio.transcriptions.with_raw_response when litellm.return_response_headers is True
        - call openai_aclient.audio.transcriptions.create by default
        """
        try:
            if litellm.return_response_headers is True:
                raw_response = (
                    openai_client.audio.transcriptions.with_raw_response.create(
                        **data, timeout=timeout
                    )
                )  # type: ignore
                headers = dict(raw_response.headers)
                response = raw_response.parse()
                return headers, response
            else:
                response = openai_client.audio.transcriptions.create(**data, timeout=timeout)  # type: ignore
                return None, response
        except Exception as e:
            raise e

    def audio_transcriptions(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        model_response: TranscriptionResponse,
        timeout: float,
        max_retries: int,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        api_base: Optional[str],
        client=None,
        atranscription: bool = False,
    ) -> TranscriptionResponse:
        data = {"model": model, "file": audio_file, **optional_params}
        if atranscription is True:
            return self.async_audio_transcriptions(  # type: ignore
                audio_file=audio_file,
                data=data,
                model_response=model_response,
                timeout=timeout,
                api_key=api_key,
                api_base=api_base,
                client=client,
                max_retries=max_retries,
                logging_obj=logging_obj,
            )

        openai_client: OpenAI = self._get_openai_client(  # type: ignore
            is_async=False,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
        )
        _, response = self.make_sync_openai_audio_transcriptions_request(
            openai_client=openai_client,
            data=data,
            timeout=timeout,
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
        hidden_params = {"model": "whisper-1", "custom_llm_provider": "openai"}
        final_response: TranscriptionResponse = convert_to_model_response_object(response_object=stringified_response, model_response_object=model_response, hidden_params=hidden_params, response_type="audio_transcription")  # type: ignore
        return final_response

    async def async_audio_transcriptions(
        self,
        audio_file: FileTypes,
        data: dict,
        model_response: TranscriptionResponse,
        timeout: float,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        client=None,
        max_retries=None,
    ):
        try:
            openai_aclient: AsyncOpenAI = self._get_openai_client(  # type: ignore
                is_async=True,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
            )

            headers, response = await self.make_openai_audio_transcriptions_request(
                openai_aclient=openai_aclient,
                data=data,
                timeout=timeout,
            )
            logging_obj.model_call_details["response_headers"] = headers
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
            hidden_params = {"model": "whisper-1", "custom_llm_provider": "openai"}
            return convert_to_model_response_object(response_object=stringified_response, model_response_object=model_response, hidden_params=hidden_params, response_type="audio_transcription")  # type: ignore
        except Exception as e:
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                original_response=str(e),
            )
            raise e
