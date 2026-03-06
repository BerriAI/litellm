"""
AWS Polly Text-to-Speech transformation

Maps OpenAI TTS spec to AWS Polly SynthesizeSpeech API
Reference: https://docs.aws.amazon.com/polly/latest/dg/API_SynthesizeSpeech.html
"""

import json
from typing import TYPE_CHECKING, Any, Coroutine, Dict, Optional, Tuple, Union

import httpx

from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj = Any
    HttpxBinaryResponseContent = Any


class AWSPollyTextToSpeechConfig(BaseTextToSpeechConfig, BaseAWSLLM):
    """
    Configuration for AWS Polly Text-to-Speech

    Reference: https://docs.aws.amazon.com/polly/latest/dg/API_SynthesizeSpeech.html
    """

    def __init__(self):
        BaseTextToSpeechConfig.__init__(self)
        BaseAWSLLM.__init__(self)

    # Default settings
    DEFAULT_VOICE = "Joanna"
    DEFAULT_ENGINE = "neural"
    DEFAULT_OUTPUT_FORMAT = "mp3"
    DEFAULT_REGION = "us-east-1"

    # Voice name mappings from OpenAI voices to Polly voices
    VOICE_MAPPINGS = {
        "alloy": "Joanna",      # US English female
        "echo": "Matthew",      # US English male
        "fable": "Amy",         # British English female
        "onyx": "Brian",        # British English male
        "nova": "Ivy",          # US English female (child)
        "shimmer": "Kendra",    # US English female
    }

    # Response format mappings from OpenAI to Polly
    FORMAT_MAPPINGS = {
        "mp3": "mp3",
        "opus": "ogg_vorbis",
        "aac": "mp3",           # Polly doesn't support AAC, use MP3
        "flac": "mp3",          # Polly doesn't support FLAC, use MP3
        "wav": "pcm",
        "pcm": "pcm",
    }

    # Valid Polly engines
    VALID_ENGINES = {"standard", "neural", "long-form", "generative"}

    def dispatch_text_to_speech(
        self,
        model: str,
        input: str,
        voice: Optional[Union[str, Dict]],
        optional_params: Dict,
        litellm_params_dict: Dict,
        logging_obj: "LiteLLMLoggingObj",
        timeout: Union[float, httpx.Timeout],
        extra_headers: Optional[Dict[str, Any]],
        base_llm_http_handler: Any,
        aspeech: bool,
        api_base: Optional[str],
        api_key: Optional[str],
        **kwargs: Any,
    ) -> Union[
        "HttpxBinaryResponseContent",
        Coroutine[Any, Any, "HttpxBinaryResponseContent"],
    ]:
        """
        Dispatch method to handle AWS Polly TTS requests

        This method encapsulates AWS-specific credential resolution and parameter handling

        Args:
            base_llm_http_handler: The BaseLLMHTTPHandler instance from main.py
        """
        # Get AWS region from kwargs or environment
        aws_region_name = kwargs.get("aws_region_name") or self._get_aws_region_name_for_polly(
            optional_params=optional_params
        )

        # Convert voice to string if it's a dict
        voice_str: Optional[str] = None
        if isinstance(voice, str):
            voice_str = voice
        elif isinstance(voice, dict):
            voice_str = voice.get("name") if voice else None

        # Update litellm_params with resolved values
        # Note: AWS credentials (aws_access_key_id, aws_secret_access_key, etc.)
        # are already in litellm_params_dict via get_litellm_params() in main.py
        litellm_params_dict["aws_region_name"] = aws_region_name
        litellm_params_dict["api_base"] = api_base
        litellm_params_dict["api_key"] = api_key

        # Call the text_to_speech_handler
        response = base_llm_http_handler.text_to_speech_handler(
            model=model,
            input=input,
            voice=voice_str,
            text_to_speech_provider_config=self,
            text_to_speech_optional_params=optional_params,
            custom_llm_provider="aws_polly",
            litellm_params=litellm_params_dict,
            logging_obj=logging_obj,
            timeout=timeout,
            extra_headers=extra_headers,
            client=None,
            _is_async=aspeech,
        )

        return response

    def _get_aws_region_name_for_polly(self, optional_params: Dict) -> str:
        """Get AWS region name for Polly API calls."""
        aws_region_name = optional_params.get("aws_region_name")
        if aws_region_name is None:
            aws_region_name = self.get_aws_region_name_for_non_llm_api_calls()
        return aws_region_name

    def get_supported_openai_params(self, model: str) -> list:
        """
        AWS Polly TTS supports these OpenAI parameters
        """
        return ["voice", "response_format", "speed"]

    def map_openai_params(
        self,
        model: str,
        optional_params: Dict,
        voice: Optional[Union[str, Dict]] = None,
        drop_params: bool = False,
        kwargs: Dict = {},
    ) -> Tuple[Optional[str], Dict]:
        """
        Map OpenAI parameters to AWS Polly parameters
        """
        mapped_params = {}

        # Map voice - support both native Polly voices and OpenAI voice mappings
        mapped_voice: Optional[str] = None
        if isinstance(voice, str):
            if voice in self.VOICE_MAPPINGS:
                # OpenAI voice -> Polly voice
                mapped_voice = self.VOICE_MAPPINGS[voice]
            else:
                # Assume it's already a Polly voice name
                mapped_voice = voice

        # Map response format
        if "response_format" in optional_params:
            format_name = optional_params["response_format"]
            if format_name in self.FORMAT_MAPPINGS:
                mapped_params["output_format"] = self.FORMAT_MAPPINGS[format_name]
            else:
                mapped_params["output_format"] = format_name
        else:
            mapped_params["output_format"] = self.DEFAULT_OUTPUT_FORMAT

        # Extract engine from model name (e.g., "aws_polly/neural" -> "neural")
        engine = self._extract_engine_from_model(model)
        mapped_params["engine"] = engine

        # Pass through Polly-specific parameters (use AWS API casing)
        if "language_code" in kwargs:
            mapped_params["LanguageCode"] = kwargs["language_code"]
        if "lexicon_names" in kwargs:
            mapped_params["LexiconNames"] = kwargs["lexicon_names"]
        if "sample_rate" in kwargs:
            mapped_params["SampleRate"] = kwargs["sample_rate"]

        return mapped_voice, mapped_params

    def _extract_engine_from_model(self, model: str) -> str:
        """
        Extract engine from model name.

        Examples:
            - aws_polly/neural -> neural
            - aws_polly/standard -> standard
            - aws_polly/long-form -> long-form
            - aws_polly -> neural (default)
        """
        if "/" in model:
            parts = model.split("/")
            if len(parts) >= 2:
                engine = parts[1].lower()
                if engine in self.VALID_ENGINES:
                    return engine
        return self.DEFAULT_ENGINE

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate AWS environment and set up headers.
        AWS SigV4 signing will be done in transform_text_to_speech_request.
        """
        validated_headers = headers.copy()
        validated_headers["Content-Type"] = "application/json"
        return validated_headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for AWS Polly SynthesizeSpeech request

        Polly endpoint format:
        https://polly.{region}.amazonaws.com/v1/speech
        """
        if api_base is not None:
            return api_base.rstrip("/") + "/v1/speech"

        aws_region_name = litellm_params.get("aws_region_name", self.DEFAULT_REGION)
        return f"https://polly.{aws_region_name}.amazonaws.com/v1/speech"

    def is_ssml_input(self, input: str) -> bool:
        """
        Returns True if input is SSML, False otherwise.

        Based on AWS Polly SSML requirements - must contain <speak> tag.
        """
        return "<speak>" in input or "<speak " in input

    def _sign_polly_request(
        self,
        request_body: Dict[str, Any],
        endpoint_url: str,
        litellm_params: Dict,
    ) -> Tuple[Dict[str, str], str]:
        """
        Sign the AWS Polly request using SigV4.

        Returns:
            Tuple of (signed_headers, json_body_string)
        """
        try:
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
        except ImportError:
            raise ImportError("Missing boto3 to call AWS Polly. Run 'pip install boto3'.")

        # Get AWS region
        aws_region_name = litellm_params.get("aws_region_name", self.DEFAULT_REGION)

        # Get AWS credentials
        credentials = self.get_credentials(
            aws_access_key_id=litellm_params.get("aws_access_key_id"),
            aws_secret_access_key=litellm_params.get("aws_secret_access_key"),
            aws_session_token=litellm_params.get("aws_session_token"),
            aws_region_name=aws_region_name,
            aws_session_name=litellm_params.get("aws_session_name"),
            aws_profile_name=litellm_params.get("aws_profile_name"),
            aws_role_name=litellm_params.get("aws_role_name"),
            aws_web_identity_token=litellm_params.get("aws_web_identity_token"),
            aws_sts_endpoint=litellm_params.get("aws_sts_endpoint"),
            aws_external_id=litellm_params.get("aws_external_id"),
        )

        # Serialize request body to JSON
        json_body = json.dumps(request_body)

        # Create headers for signing
        headers = {
            "Content-Type": "application/json",
        }

        # Create AWS request for signing
        aws_request = AWSRequest(
            method="POST",
            url=endpoint_url,
            data=json_body,
            headers=headers,
        )

        # Sign the request
        SigV4Auth(credentials, "polly", aws_region_name).add_auth(aws_request)

        # Return signed headers and body
        return dict(aws_request.headers), json_body

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: Optional[str],
        optional_params: Dict,
        litellm_params: Dict,
        headers: dict,
    ) -> TextToSpeechRequestData:
        """
        Transform OpenAI TTS request to AWS Polly SynthesizeSpeech format.

        Supports:
        - Native Polly voices (Joanna, Matthew, etc.)
        - OpenAI voice mapping (alloy, echo, etc.)
        - SSML input (auto-detected via <speak> tag)
        - Multiple engines (neural, standard, long-form, generative)

        Returns:
            TextToSpeechRequestData: Contains signed request for Polly API
        """
        # Get voice (already mapped in main.py, or use default)
        polly_voice = voice or self.DEFAULT_VOICE

        # Get output format
        output_format = optional_params.get("output_format", self.DEFAULT_OUTPUT_FORMAT)

        # Get engine
        engine = optional_params.get("engine", self.DEFAULT_ENGINE)

        # Build request body
        request_body: Dict[str, Any] = {
            "Engine": engine,
            "OutputFormat": output_format,
            "Text": input,
            "VoiceId": polly_voice,
        }

        # Auto-detect SSML
        if self.is_ssml_input(input):
            request_body["TextType"] = "ssml"
        else:
            request_body["TextType"] = "text"

        # Add optional Polly parameters (already in AWS casing from map_openai_params)
        for key in ["LanguageCode", "LexiconNames", "SampleRate"]:
            if key in optional_params:
                request_body[key] = optional_params[key]

        # Get endpoint URL
        endpoint_url = self.get_complete_url(
            model=model,
            api_base=litellm_params.get("api_base"),
            litellm_params=litellm_params,
        )

        # Sign the request with AWS SigV4
        signed_headers, json_body = self._sign_polly_request(
            request_body=request_body,
            endpoint_url=endpoint_url,
            litellm_params=litellm_params,
        )

        # Return as ssml_body so the handler uses data= instead of json=
        # This preserves the exact JSON string that was signed
        return TextToSpeechRequestData(
            ssml_body=json_body,
            headers=signed_headers,
        )

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> "HttpxBinaryResponseContent":
        """
        Transform AWS Polly response to standard format.

        Polly returns the audio data directly in the response body.
        """
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        return HttpxBinaryResponseContent(raw_response)

