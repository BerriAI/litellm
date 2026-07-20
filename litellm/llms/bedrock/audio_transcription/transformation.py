import base64

import httpx

from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import BedrockError, _get_all_bedrock_regions
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import FileTypes, TranscriptionResponse


def get_bedrock_audio_model_id(model: str) -> tuple[str, str | None]:
    stripped_model = model
    for prefix in ("bedrock/converse/", "bedrock/", "converse/"):
        if stripped_model.startswith(prefix):
            stripped_model = stripped_model[len(prefix) :]
            break

    model_parts = stripped_model.split("/", 1)
    region = model_parts[0] if len(model_parts) == 2 and model_parts[0] in _get_all_bedrock_regions() else None
    model_id = model_parts[1] if region is not None else stripped_model
    for prefix in ("nova-2/", "nova/"):
        if model_id.startswith(prefix):
            model_id = model_id[len(prefix) :]
            break
    return BaseAWSLLM.encode_model_id(model_id), region


def get_bedrock_audio_format(filename: str, content_type: str) -> str:
    content_type_formats = {
        "audio/flac": "flac",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/ogg": "ogg",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
    }
    if content_type in content_type_formats:
        return content_type_formats[content_type]

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension in {"flac", "mp3", "ogg", "wav"}:
        return extension
    raise ValueError(f"Unsupported Bedrock audio format for file {filename!r}")


class BedrockAudioTranscriptionConfig(BaseAudioTranscriptionConfig, BaseAWSLLM):
    def __init__(self) -> None:
        BaseAudioTranscriptionConfig.__init__(self)
        BaseAWSLLM.__init__(self)

    def get_supported_openai_params(self, model: str) -> list[OpenAIAudioTranscriptionOptionalParams]:
        return ["language", "prompt", "temperature", "response_format"]

    def map_openai_params(
        self,
        non_default_params: dict[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        supported_params = self.get_supported_openai_params(model)
        return {
            **optional_params,
            **{key: value for key, value in non_default_params.items() if key in supported_params},
        }

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,
    ) -> BaseLLMException:
        return BedrockError(status_code=status_code, message=error_message, headers=headers)

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:
        return {**headers, "Content-Type": "application/json"}

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        stream: bool | None = None,
    ) -> str:
        region = optional_params.get("aws_region_name")
        model_id, model_region = get_bedrock_audio_model_id(model)
        aws_region = region or model_region or self._get_aws_region_name(optional_params, model=model)
        endpoint, _ = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=optional_params.get("aws_bedrock_runtime_endpoint"),
            aws_region_name=aws_region,
        )
        return f"{endpoint.rstrip('/')}/model/{model_id}/converse"

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
    ) -> AudioTranscriptionRequestData:
        processed_audio = process_audio_file(audio_file)
        explicit_content_type = (
            audio_file[2]
            if isinstance(audio_file, tuple) and len(audio_file) > 2 and isinstance(audio_file[2], str)
            else None
        )
        audio_format = get_bedrock_audio_format(
            processed_audio.filename,
            explicit_content_type or processed_audio.content_type,
        )
        language = optional_params.get("language")
        prompt = optional_params.get("prompt")
        instruction_parts = [
            "Transcribe the audio. Respond with only the transcript.",
            f"The audio language is {language}." if language else None,
            f"Additional context: {prompt}" if prompt else None,
        ]
        instruction = " ".join(part for part in instruction_parts if part)
        inference_config = {"maxTokens": 4096}
        if optional_params.get("temperature") is not None:
            inference_config["temperature"] = optional_params["temperature"]
        request_body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "audio": {
                                "format": audio_format,
                                "source": {"bytes": base64.b64encode(processed_audio.file_content).decode("ascii")},
                            }
                        },
                        {"text": instruction},
                    ],
                }
            ],
            "system": [{"text": "You are a transcription assistant."}],
            "inferenceConfig": inference_config,
        }
        return AudioTranscriptionRequestData(data=request_body)

    def transform_audio_transcription_response(
        self,
        raw_response: httpx.Response,
    ) -> TranscriptionResponse:
        response_json = raw_response.json()
        output = response_json.get("output", {})
        message = output.get("message", {}) if isinstance(output, dict) else {}
        content = message.get("content", []) if isinstance(message, dict) else []
        text = "".join(
            block.get("text", "") for block in content if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
        response = TranscriptionResponse(text=text)
        response._hidden_params = response_json
        return response
