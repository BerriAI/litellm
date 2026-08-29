from typing import Final

import httpx
from typing_extensions import TypedDict

import litellm
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.openai.openai import HttpxBinaryResponseContent
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES


class VertexInput(TypedDict, total=False):
    text: str | None
    ssml: str | None


class VertexVoice(TypedDict, total=False):
    languageCode: str
    name: str


class VertexAudioConfig(TypedDict, total=False):
    audioEncoding: str
    speakingRate: str


class VertexTextToSpeechRequest(TypedDict, total=False):
    input: VertexInput
    voice: VertexVoice
    audioConfig: VertexAudioConfig | None


class VertexTextToSpeechAPI(VertexLLM):
    """
    Vertex methods to support for batches
    """

    def __init__(self) -> None:
        super().__init__()

    def audio_speech(
        self,
        logging_obj,
        vertex_project: str | None,
        vertex_location: str | None,
        vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        model: str,
        input: str,
        voice: dict | None = None,
        _is_async: bool | None = False,
        optional_params: dict | None = None,
        kwargs: dict | None = None,
    ) -> HttpxBinaryResponseContent:
        import base64

        ####### Authenticate with Vertex AI ########
        _auth_header, vertex_project = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai_beta",
        )

        auth_header, _ = self._get_token_and_url(
            model="",
            auth_header=_auth_header,
            gemini_api_key=None,
            vertex_credentials=vertex_credentials,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            stream=False,
            custom_llm_provider="vertex_ai_beta",
            api_base=api_base,
        )

        headers: Final = {
            "Authorization": f"Bearer {auth_header}",
            "x-goog-user-project": vertex_project,
            "Content-Type": "application/json",
            "charset": "UTF-8",
        }

        ######### End of Authentication ###########

        ####### Build the request ################
        # API Ref: https://cloud.google.com/text-to-speech/docs/reference/rest/v1/text/synthesize
        kwargs = kwargs or {}
        optional_params = optional_params or {}

        vertex_input: Final = VertexInput(text=input)
        validate_vertex_input(vertex_input, kwargs, optional_params)

        # required param
        if voice is not None:
            vertex_voice = VertexVoice(**voice)
        elif "voice" in kwargs:
            vertex_voice = VertexVoice(**kwargs["voice"])
        else:
            # use defaults to not fail the request
            vertex_voice = VertexVoice(
                languageCode="en-US",
                name="en-US-Studio-O",
            )

        if "audioConfig" in kwargs:
            vertex_audio_config = VertexAudioConfig(**kwargs["audioConfig"])
        else:
            # use defaults to not fail the request
            vertex_audio_config = VertexAudioConfig(
                audioEncoding="LINEAR16",
                speakingRate="1",
            )

        request: Final = VertexTextToSpeechRequest(
            input=vertex_input,
            voice=vertex_voice,
            audioConfig=vertex_audio_config,
        )

        url: Final = "https://texttospeech.googleapis.com/v1/text:synthesize"
        ########## End of building request ############

        ########## Log the request for debugging / logging ############
        logging_obj.pre_call(
            input=[],
            api_key="",
            additional_args={
                "complete_input_dict": request,
                "api_base": url,
                "headers": headers,
            },
        )

        ########## End of logging ############
        ####### Send the request ###################
        if _is_async is True:
            return self.async_audio_speech(logging_obj=logging_obj, url=url, headers=headers, request=request)
        sync_handler: Final = _get_httpx_client()

        response = sync_handler.post(
            url=url,
            headers=headers,
            json=request,
        )
        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}, {response.text}")
        ############ Process the response ############
        _json_response: Final = response.json()

        response_content: Final = _json_response["audioContent"]

        # Decode base64 to get binary content
        binary_data: Final = base64.b64decode(response_content)

        # Create an httpx.Response object
        response = httpx.Response(
            status_code=200,
            content=binary_data,
        )

        # Initialize the HttpxBinaryResponseContent instance
        http_binary_response: Final = HttpxBinaryResponseContent(response)
        return http_binary_response

    async def async_audio_speech(
        self,
        logging_obj,
        url: str,
        headers: dict,
        request: VertexTextToSpeechRequest,
    ) -> HttpxBinaryResponseContent:
        import base64

        async_handler: Final = get_async_httpx_client(llm_provider=litellm.LlmProviders.VERTEX_AI)

        response = await async_handler.post(
            url=url,
            headers=headers,
            json=request,
        )

        if response.status_code != 200:
            raise Exception(f"Request did not return a 200 status code: {response.status_code}, {response.text}")

        _json_response: Final = response.json()

        response_content: Final = _json_response["audioContent"]

        # Decode base64 to get binary content
        binary_data: Final = base64.b64decode(response_content)

        # Create an httpx.Response object
        response = httpx.Response(
            status_code=200,
            content=binary_data,
        )

        # Initialize the HttpxBinaryResponseContent instance
        http_binary_response: Final = HttpxBinaryResponseContent(response)
        return http_binary_response


def validate_vertex_input(input_data: VertexInput, kwargs: dict, optional_params: dict) -> None:
    # Remove None values
    if input_data.get("text") is None:
        input_data.pop("text", None)
    if input_data.get("ssml") is None:
        input_data.pop("ssml", None)

    # Check if use_ssml is set
    use_ssml: Final = kwargs.get("use_ssml", optional_params.get("use_ssml", False))

    if use_ssml:
        if "text" in input_data:
            input_data["ssml"] = input_data.pop("text")
        elif "ssml" not in input_data:
            raise ValueError("SSML input is required when use_ssml is True.")
    else:
        # LiteLLM will auto-detect if text is in ssml format
        # check if "text" is an ssml - in this case we should pass it as ssml instead of text
        if input_data:
            _text: Final = input_data.get("text", None) or ""
            if "<speak>" in _text:
                input_data["ssml"] = input_data.pop("text")

    if not input_data:
        raise ValueError("Either 'text' or 'ssml' must be provided.")
    if "text" in input_data and "ssml" in input_data:
        raise ValueError("Only one of 'text' or 'ssml' should be provided, not both.")
