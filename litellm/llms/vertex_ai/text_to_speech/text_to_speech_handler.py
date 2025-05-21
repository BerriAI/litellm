from typing import Optional, TypedDict, Union

import httpx

import litellm
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.openai.openai import HttpxBinaryResponseContent
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES


class VertexInput(TypedDict, total=False):
    text: Optional[str]
    ssml: Optional[str]


class VertexVoice(TypedDict, total=False):
    languageCode: str
    name: str


class VertexAudioConfig(TypedDict, total=False):
    audioEncoding: str
    speakingRate: str


class VertexTextToSpeechRequest(TypedDict, total=False):
    input: VertexInput
    voice: VertexVoice
    audioConfig: Optional[VertexAudioConfig]


class VertexTextToSpeechAPI(VertexLLM):
    """
    Vertex methods to support for batches
    """

    def __init__(self) -> None:
        super().__init__()

    def audio_speech(
        self,
        logging_obj,
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        model: str,
        input: str,
        voice: Optional[dict] = None,
        _is_async: Optional[bool] = False,
        optional_params: Optional[dict] = None,
        kwargs: Optional[dict] = None,
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

        headers = {
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

        vertex_input = VertexInput(text=input)
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

        request = VertexTextToSpeechRequest(
            input=vertex_input,
            voice=vertex_voice,
            audioConfig=vertex_audio_config,
        )

        url = "https://texttospeech.googleapis.com/v1/text:synthesize"
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
            return self.async_audio_speech(  # type:ignore
                logging_obj=logging_obj, url=url, headers=headers, request=request
            )
        sync_handler = _get_httpx_client()

        response = sync_handler.post(
            url=url,
            headers=headers,
            json=request,  # type: ignore
        )
        if response.status_code != 200:
            raise Exception(
                f"Request failed with status code {response.status_code}, {response.text}"
            )
        ############ Process the response ############
        _json_response = response.json()

        response_content = _json_response["audioContent"]

        # Decode base64 to get binary content
        binary_data = base64.b64decode(response_content)

        # Create an httpx.Response object
        response = httpx.Response(
            status_code=200,
            content=binary_data,
        )

        # Initialize the HttpxBinaryResponseContent instance
        http_binary_response = HttpxBinaryResponseContent(response)
        return http_binary_response

    async def async_audio_speech(
        self,
        logging_obj,
        url: str,
        headers: dict,
        request: VertexTextToSpeechRequest,
    ) -> HttpxBinaryResponseContent:
        import base64

        async_handler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VERTEX_AI
        )

        response = await async_handler.post(
            url=url,
            headers=headers,
            json=request,  # type: ignore
        )

        if response.status_code != 200:
            raise Exception(
                f"Request did not return a 200 status code: {response.status_code}, {response.text}"
            )

        _json_response = response.json()

        response_content = _json_response["audioContent"]

        # Decode base64 to get binary content
        binary_data = base64.b64decode(response_content)

        # Create an httpx.Response object
        response = httpx.Response(
            status_code=200,
            content=binary_data,
        )

        # Initialize the HttpxBinaryResponseContent instance
        http_binary_response = HttpxBinaryResponseContent(response)
        return http_binary_response


def validate_vertex_input(
    input_data: VertexInput, kwargs: dict, optional_params: dict
) -> None:
    # Remove None values
    if input_data.get("text") is None:
        input_data.pop("text", None)
    if input_data.get("ssml") is None:
        input_data.pop("ssml", None)

    # Check if use_ssml is set
    use_ssml = kwargs.get("use_ssml", optional_params.get("use_ssml", False))

    if use_ssml:
        if "text" in input_data:
            input_data["ssml"] = input_data.pop("text")
        elif "ssml" not in input_data:
            raise ValueError("SSML input is required when use_ssml is True.")
    else:
        # LiteLLM will auto-detect if text is in ssml format
        # check if "text" is an ssml - in this case we should pass it as ssml instead of text
        if input_data:
            _text = input_data.get("text", None) or ""
            if "<speak>" in _text:
                input_data["ssml"] = input_data.pop("text")

    if not input_data:
        raise ValueError("Either 'text' or 'ssml' must be provided.")
    if "text" in input_data and "ssml" in input_data:
        raise ValueError("Only one of 'text' or 'ssml' should be provided, not both.")
