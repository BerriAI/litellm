# coding=utf-8
# Copyright 2023-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# WARNING
# This entire file has been adapted from the sync-client code in `src/huggingface_hub/inference/_client.py`.
# Any change in InferenceClient will be automatically reflected in AsyncInferenceClient.
# To re-generate the code, run `make style` or `python ./utils/generate_async_inference_client.py --update`.
# WARNING
import logging
import time
import warnings
from dataclasses import asdict
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    Dict,
    List,
    Optional,
    Union,
    overload,
)

from requests.structures import CaseInsensitiveDict

from huggingface_hub.constants import INFERENCE_ENDPOINT
from huggingface_hub.inference._common import (
    ContentT,
    InferenceTimeoutError,
    _async_stream_text_generation_response,
    _b64_encode,
    _b64_to_image,
    _bytes_to_dict,
    _bytes_to_image,
    _get_recommended_model,
    _import_numpy,
    _is_tgi_server,
    _open_as_binary,
    _set_as_non_tgi,
)
from huggingface_hub.inference._text_generation import (
    TextGenerationParameters,
    TextGenerationRequest,
    TextGenerationResponse,
    TextGenerationStreamResponse,
    raise_text_generation_error,
)
from huggingface_hub.inference._types import ClassificationOutput, ConversationalOutput, ImageSegmentationOutput
from huggingface_hub.utils import (
    build_hf_headers,
)
from huggingface_hub.utils._typing import Literal

from .._common import _async_yield_from, _import_aiohttp


if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

logger = logging.getLogger(__name__)


class AsyncInferenceClient:
    """
    Initialize a new Inference Client.

    [`InferenceClient`] aims to provide a unified experience to perform inference. The client can be used
    seamlessly with either the (free) Inference API or self-hosted Inference Endpoints.

    Args:
        model (`str`, `optional`):
            The model to run inference with. Can be a model id hosted on the Hugging Face Hub, e.g. `bigcode/starcoder`
            or a URL to a deployed Inference Endpoint. Defaults to None, in which case a recommended model is
            automatically selected for the task.
        token (`str`, *optional*):
            Hugging Face token. Will default to the locally saved token. Pass `token=False` if you don't want to send
            your token to the server.
        timeout (`float`, `optional`):
            The maximum number of seconds to wait for a response from the server. Loading a new model in Inference
            API can take up to several minutes. Defaults to None, meaning it will loop until the server is available.
        headers (`Dict[str, str]`, `optional`):
            Additional headers to send to the server. By default only the authorization and user-agent headers are sent.
            Values in this dictionary will override the default values.
        cookies (`Dict[str, str]`, `optional`):
            Additional cookies to send to the server.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        token: Union[str, bool, None] = None,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model: Optional[str] = model
        self.headers = CaseInsensitiveDict(build_hf_headers(token=token))  # contains 'authorization' + 'user-agent'
        if headers is not None:
            self.headers.update(headers)
        self.cookies = cookies
        self.timeout = timeout

    def __repr__(self):
        return f"<InferenceClient(model='{self.model if self.model else ''}', timeout={self.timeout})>"

    @overload
    async def post(  # type: ignore
        self,
        *,
        json: Optional[Union[str, Dict, List]] = None,
        data: Optional[ContentT] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        stream: Literal[False] = ...,
    ) -> bytes:
        pass

    @overload
    async def post(  # type: ignore
        self,
        *,
        json: Optional[Union[str, Dict, List]] = None,
        data: Optional[ContentT] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        stream: Literal[True] = ...,
    ) -> AsyncIterable[bytes]:
        pass

    async def post(
        self,
        *,
        json: Optional[Union[str, Dict, List]] = None,
        data: Optional[ContentT] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        stream: bool = False,
    ) -> Union[bytes, AsyncIterable[bytes]]:
        """
        Make a POST request to the inference server.

        Args:
            json (`Union[str, Dict, List]`, *optional*):
                The JSON data to send in the request body. Defaults to None.
            data (`Union[str, Path, bytes, BinaryIO]`, *optional*):
                The content to send in the request body. It can be raw bytes, a pointer to an opened file, a local file
                path, or a URL to an online resource (image, audio file,...). If both `json` and `data` are passed,
                `data` will take precedence. At least `json` or `data` must be provided. Defaults to None.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. Will override the model defined at the instance level. Defaults to None.
            task (`str`, *optional*):
                The task to perform on the inference. Used only to default to a recommended model if `model` is not
                provided. At least `model` or `task` must be provided. Defaults to None.
            stream (`bool`, *optional*):
                Whether to iterate over streaming APIs.

        Returns:
            bytes: The raw bytes returned by the server.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.
        """

        aiohttp = _import_aiohttp()

        url = self._resolve_url(model, task)

        if data is not None and json is not None:
            warnings.warn("Ignoring `json` as `data` is passed as binary.")

        t0 = time.time()
        timeout = self.timeout
        while True:
            with _open_as_binary(data) as data_as_binary:
                # Do not use context manager as we don't want to close the connection immediately when returning
                # a stream
                client = aiohttp.ClientSession(
                    headers=self.headers, cookies=self.cookies, timeout=aiohttp.ClientTimeout(self.timeout)
                )

                try:
                    response = await client.post(url, headers=build_hf_headers(), json=json, data=data_as_binary)
                    response_error_payload = None
                    if response.status != 200:
                        try:
                            response_error_payload = await response.json()  # get payload before connection closed
                        except Exception:
                            pass
                    response.raise_for_status()
                    if stream:
                        return _async_yield_from(client, response)
                    else:
                        content = await response.read()
                        await client.close()
                        return content
                except TimeoutError as error:
                    await client.close()
                    # Convert any `TimeoutError` to a `InferenceTimeoutError`
                    raise InferenceTimeoutError(f"Inference call timed out: {url}") from error
                except aiohttp.ClientResponseError as error:
                    error.response_error_payload = response_error_payload
                    await client.close()
                    if response.status == 503:
                        # If Model is unavailable, either raise a TimeoutError...
                        if timeout is not None and time.time() - t0 > timeout:
                            raise InferenceTimeoutError(
                                f"Model not loaded on the server: {url}. Please retry with a higher timeout"
                                f" (current: {self.timeout})."
                            ) from error
                        # ...or wait 1s and retry
                        logger.info(f"Waiting for model to be loaded on the server: {error}")
                        time.sleep(1)
                        if timeout is not None:
                            timeout = max(self.timeout - (time.time() - t0), 1)  # type: ignore
                        continue
                    raise error

    async def audio_classification(
        self,
        audio: ContentT,
        *,
        model: Optional[str] = None,
    ) -> List[ClassificationOutput]:
        """
        Perform audio classification on the provided audio content.

        Args:
            audio (Union[str, Path, bytes, BinaryIO]):
                The audio content to classify. It can be raw audio bytes, a local audio file, or a URL pointing to an
                audio file.
            model (`str`, *optional*):
                The model to use for audio classification. Can be a model ID hosted on the Hugging Face Hub
                or a URL to a deployed Inference Endpoint. If not provided, the default recommended model for
                audio classification will be used.

        Returns:
            `List[Dict]`: The classification output containing the predicted label and its confidence.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.audio_classification("audio.flac")
        [{'score': 0.4976358711719513, 'label': 'hap'}, {'score': 0.3677836060523987, 'label': 'neu'},...]
        ```
        """
        response = await self.post(data=audio, model=model, task="audio-classification")
        return _bytes_to_dict(response)

    async def automatic_speech_recognition(
        self,
        audio: ContentT,
        *,
        model: Optional[str] = None,
    ) -> str:
        """
        Perform automatic speech recognition (ASR or audio-to-text) on the given audio content.

        Args:
            audio (Union[str, Path, bytes, BinaryIO]):
                The content to transcribe. It can be raw audio bytes, local audio file, or a URL to an audio file.
            model (`str`, *optional*):
                The model to use for ASR. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. If not provided, the default recommended model for ASR will be used.

        Returns:
            str: The transcribed text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.automatic_speech_recognition("hello_world.flac")
        "hello world"
        ```
        """
        response = await self.post(data=audio, model=model, task="automatic-speech-recognition")
        return _bytes_to_dict(response)["text"]

    async def conversational(
        self,
        text: str,
        generated_responses: Optional[List[str]] = None,
        past_user_inputs: Optional[List[str]] = None,
        *,
        parameters: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> ConversationalOutput:
        """
        Generate conversational responses based on the given input text (i.e. chat with the API).

        Args:
            text (`str`):
                The last input from the user in the conversation.
            generated_responses (`List[str]`, *optional*):
                A list of strings corresponding to the earlier replies from the model. Defaults to None.
            past_user_inputs (`List[str]`, *optional*):
                A list of strings corresponding to the earlier replies from the user. Should be the same length as
                `generated_responses`. Defaults to None.
            parameters (`Dict[str, Any]`, *optional*):
                Additional parameters for the conversational task. Defaults to None. For more details about the available
                parameters, please refer to [this page](https://huggingface.co/docs/api-inference/detailed_parameters#conversational-task)
            model (`str`, *optional*):
                The model to use for the conversational task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended conversational model will be used.
                Defaults to None.

        Returns:
            `Dict`: The generated conversational output.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> output = await client.conversational("Hi, who are you?")
        >>> output
        {'generated_text': 'I am the one who knocks.', 'conversation': {'generated_responses': ['I am the one who knocks.'], 'past_user_inputs': ['Hi, who are you?']}, 'warnings': ['Setting `pad_token_id` to `eos_token_id`:50256 async for open-end generation.']}
        >>> await client.conversational(
        ...     "Wow, that's scary!",
        ...     generated_responses=output["conversation"]["generated_responses"],
        ...     past_user_inputs=output["conversation"]["past_user_inputs"],
        ... )
        ```
        """
        payload: Dict[str, Any] = {"inputs": {"text": text}}
        if generated_responses is not None:
            payload["inputs"]["generated_responses"] = generated_responses
        if past_user_inputs is not None:
            payload["inputs"]["past_user_inputs"] = past_user_inputs
        if parameters is not None:
            payload["parameters"] = parameters
        response = await self.post(json=payload, model=model, task="conversational")
        return _bytes_to_dict(response)

    async def feature_extraction(self, text: str, *, model: Optional[str] = None) -> "np.ndarray":
        """
        Generate embeddings for a given text.

        Args:
            text (`str`):
                The text to embed.
            model (`str`, *optional*):
                The model to use for the conversational task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended conversational model will be used.
                Defaults to None.

        Returns:
            `np.ndarray`: The embedding representing the input text as a float32 numpy array.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.feature_extraction("Hi, who are you?")
        array([[ 2.424802  ,  2.93384   ,  1.1750331 , ...,  1.240499, -0.13776633, -0.7889173 ],
        [-0.42943227, -0.6364878 , -1.693462  , ...,  0.41978157, -2.4336355 ,  0.6162071 ],
        ...,
        [ 0.28552425, -0.928395  , -1.2077185 , ...,  0.76810825, -2.1069427 ,  0.6236161 ]], dtype=float32)
        ```
        """
        response = await self.post(json={"inputs": text}, model=model, task="feature-extraction")
        np = _import_numpy()
        return np.array(_bytes_to_dict(response)[0], dtype="float32")

    async def image_classification(
        self,
        image: ContentT,
        *,
        model: Optional[str] = None,
    ) -> List[ClassificationOutput]:
        """
        Perform image classification on the given image using the specified model.

        Args:
            image (`Union[str, Path, bytes, BinaryIO]`):
                The image to classify. It can be raw bytes, an image file, or a URL to an online image.
            model (`str`, *optional*):
                The model to use for image classification. Can be a model ID hosted on the Hugging Face Hub or a URL to a
                deployed Inference Endpoint. If not provided, the default recommended model for image classification will be used.

        Returns:
            `List[Dict]`: a list of dictionaries containing the predicted label and associated probability.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.image_classification("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
        [{'score': 0.9779096841812134, 'label': 'Blenheim spaniel'}, ...]
        ```
        """
        response = await self.post(data=image, model=model, task="image-classification")
        return _bytes_to_dict(response)

    async def image_segmentation(
        self,
        image: ContentT,
        *,
        model: Optional[str] = None,
    ) -> List[ImageSegmentationOutput]:
        """
        Perform image segmentation on the given image using the specified model.

        <Tip warning={true}>

        You must have `PIL` installed if you want to work with images (`pip install Pillow`).

        </Tip>

        Args:
            image (`Union[str, Path, bytes, BinaryIO]`):
                The image to segment. It can be raw bytes, an image file, or a URL to an online image.
            model (`str`, *optional*):
                The model to use for image segmentation. Can be a model ID hosted on the Hugging Face Hub or a URL to a
                deployed Inference Endpoint. If not provided, the default recommended model for image segmentation will be used.

        Returns:
            `List[Dict]`: A list of dictionaries containing the segmented masks and associated attributes.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.image_segmentation("cat.jpg"):
        [{'score': 0.989008, 'label': 'LABEL_184', 'mask': <PIL.PngImagePlugin.PngImageFile image mode=L size=400x300 at 0x7FDD2B129CC0>}, ...]
        ```
        """

        # Segment
        response = await self.post(data=image, model=model, task="image-segmentation")
        output = _bytes_to_dict(response)

        # Parse masks as PIL Image
        if not isinstance(output, list):
            raise ValueError(f"Server output must be a list. Got {type(output)}: {str(output)[:200]}...")
        for item in output:
            item["mask"] = _b64_to_image(item["mask"])
        return output

    async def image_to_image(
        self,
        image: ContentT,
        prompt: Optional[str] = None,
        *,
        negative_prompt: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> "Image":
        """
        Perform image-to-image translation using a specified model.

        <Tip warning={true}>

        You must have `PIL` installed if you want to work with images (`pip install Pillow`).

        </Tip>

        Args:
            image (`Union[str, Path, bytes, BinaryIO]`):
                The input image for translation. It can be raw bytes, an image file, or a URL to an online image.
            prompt (`str`, *optional*):
                The text prompt to guide the image generation.
            negative_prompt (`str`, *optional*):
                A negative prompt to guide the translation process.
            height (`int`, *optional*):
                The height in pixels of the generated image.
            width (`int`, *optional*):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*):
                Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.

        Returns:
            `Image`: The translated image.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> image = await client.image_to_image("cat.jpg", prompt="turn the cat into a tiger")
        >>> image.save("tiger.jpg")
        ```
        """
        parameters = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            **kwargs,
        }
        if all(parameter is None for parameter in parameters.values()):
            # Either only an image to send => send as raw bytes
            data = image
            payload: Optional[Dict[str, Any]] = None
        else:
            # Or an image + some parameters => use base64 encoding
            data = None
            payload = {"inputs": _b64_encode(image)}
            for key, value in parameters.items():
                if value is not None:
                    payload[key] = value

        response = await self.post(json=payload, data=data, model=model, task="image-to-image")
        return _bytes_to_image(response)

    async def image_to_text(self, image: ContentT, *, model: Optional[str] = None) -> str:
        """
        Takes an input image and return text.

        Models can have very different outputs depending on your use case (image captioning, optical character recognition
        (OCR), Pix2Struct, etc). Please have a look to the model card to learn more about a model's specificities.

        Args:
            image (`Union[str, Path, bytes, BinaryIO]`):
                The input image to caption. It can be raw bytes, an image file, or a URL to an online image..
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.

        Returns:
            `str`: The generated text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.image_to_text("cat.jpg")
        'a cat standing in a grassy field '
        >>> await client.image_to_text("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
        'a dog laying on the grass next to a flower pot '
        ```
        """
        response = await self.post(data=image, model=model, task="image-to-text")
        return _bytes_to_dict(response)[0]["generated_text"]

    async def sentence_similarity(
        self, sentence: str, other_sentences: List[str], *, model: Optional[str] = None
    ) -> List[float]:
        """
        Compute the semantic similarity between a sentence and a list of other sentences by comparing their embeddings.

        Args:
            sentence (`str`):
                The main sentence to compare to others.
            other_sentences (`List[str]`):
                The list of sentences to compare to.
            model (`str`, *optional*):
                The model to use for the conversational task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended conversational model will be used.
                Defaults to None.

        Returns:
            `List[float]`: The embedding representing the input text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.sentence_similarity(
        ...     "Machine learning is so easy.",
        ...     other_sentences=[
        ...         "Deep learning is so straightforward.",
        ...         "This is so difficult, like rocket science.",
        ...         "I can't believe how much I struggled with this.",
        ...     ],
        ... )
        [0.7785726189613342, 0.45876261591911316, 0.2906220555305481]
        ```
        """
        response = await self.post(
            json={"inputs": {"source_sentence": sentence, "sentences": other_sentences}},
            model=model,
            task="sentence-similarity",
        )
        return _bytes_to_dict(response)

    async def summarization(
        self,
        text: str,
        *,
        parameters: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Generate a summary of a given text using a specified model.

        Args:
            text (`str`):
                The input text to summarize.
            parameters (`Dict[str, Any]`, *optional*):
                Additional parameters for summarization. Check out this [page](https://huggingface.co/docs/api-inference/detailed_parameters#summarization-task)
                for more details.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.

        Returns:
            `str`: The generated summary text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()
        >>> await client.summarization("The Eiffel tower...")
        'The Eiffel tower is one of the most famous landmarks in the world....'
        ```
        """
        payload: Dict[str, Any] = {"inputs": text}
        if parameters is not None:
            payload["parameters"] = parameters
        response = await self.post(json=payload, model=model, task="summarization")
        return _bytes_to_dict(response)[0]["summary_text"]

    @overload
    async def text_generation(  # type: ignore
        self,
        prompt: str,
        *,
        details: Literal[False] = ...,
        stream: Literal[False] = ...,
        model: Optional[str] = None,
        do_sample: bool = False,
        max_new_tokens: int = 20,
        best_of: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
    ) -> str:
        ...

    @overload
    async def text_generation(  # type: ignore
        self,
        prompt: str,
        *,
        details: Literal[True] = ...,
        stream: Literal[False] = ...,
        model: Optional[str] = None,
        do_sample: bool = False,
        max_new_tokens: int = 20,
        best_of: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
    ) -> TextGenerationResponse:
        ...

    @overload
    async def text_generation(  # type: ignore
        self,
        prompt: str,
        *,
        details: Literal[False] = ...,
        stream: Literal[True] = ...,
        model: Optional[str] = None,
        do_sample: bool = False,
        max_new_tokens: int = 20,
        best_of: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
    ) -> AsyncIterable[str]:
        ...

    @overload
    async def text_generation(
        self,
        prompt: str,
        *,
        details: Literal[True] = ...,
        stream: Literal[True] = ...,
        model: Optional[str] = None,
        do_sample: bool = False,
        max_new_tokens: int = 20,
        best_of: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
    ) -> AsyncIterable[TextGenerationStreamResponse]:
        ...

    async def text_generation(
        self,
        prompt: str,
        *,
        details: bool = False,
        stream: bool = False,
        model: Optional[str] = None,
        do_sample: bool = False,
        max_new_tokens: int = 20,
        best_of: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
        decoder_input_details: bool = False,
    ) -> Union[str, TextGenerationResponse, AsyncIterable[str], AsyncIterable[TextGenerationStreamResponse]]:
        """
        Given a prompt, generate the following text.

        It is recommended to have Pydantic installed in order to get inputs validated. This is preferable as it allow
        early failures.

        API endpoint is supposed to run with the `text-generation-inference` backend (TGI). This backend is the
        go-to solution to run large language models at scale. However, for some smaller models (e.g. "gpt2") the
        default `transformers` + `api-inference` solution is still in use. Both approaches have very similar APIs, but
        not exactly the same. This method is compatible with both approaches but some parameters are only available for
        `text-generation-inference`. If some parameters are ignored, a warning message is triggered but the process
        continues correctly.

        To learn more about the TGI project, please refer to https://github.com/huggingface/text-generation-inference.

        Args:
            prompt (`str`):
                Input text.
            details (`bool`, *optional*):
                By default, text_generation returns a string. Pass `details=True` if you want a detailed output (tokens,
                probabilities, seed, finish reason, etc.). Only available for models running on with the
                `text-generation-inference` backend.
            stream (`bool`, *optional*):
                By default, text_generation returns the full generated text. Pass `stream=True` if you want a stream of
                tokens to be returned. Only available for models running on with the `text-generation-inference`
                backend.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.
            do_sample (`bool`):
                Activate logits sampling
            max_new_tokens (`int`):
                Maximum number of generated tokens
            best_of (`int`):
                Generate best_of sequences and return the one if the highest token logprobs
            repetition_penalty (`float`):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            return_full_text (`bool`):
                Whether to prepend the prompt to the generated text
            seed (`int`):
                Random sampling seed
            stop_sequences (`List[str]`):
                Stop generating tokens if a member of `stop_sequences` is generated
            temperature (`float`):
                The value used to module the logits distribution.
            top_k (`int`):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`):
                Truncate inputs tokens to the given size
            typical_p (`float`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
            decoder_input_details (`bool`):
                Return the decoder input token logprobs and ids. You must set `details=True` as well for it to be taken
                into account. Defaults to `False`.

        Returns:
            `Union[str, TextGenerationResponse, Iterable[str], Iterable[TextGenerationStreamResponse]]`:
            Generated text returned from the server:
            - if `stream=False` and `details=False`, the generated text is returned as a `str` (default)
            - if `stream=True` and `details=False`, the generated text is returned token by token as a `Iterable[str]`
            - if `stream=False` and `details=True`, the generated text is returned with more details as a [`~huggingface_hub.inference._text_generation.TextGenerationResponse`]
            - if `details=True` and `stream=True`, the generated text is returned token by token as a iterable of [`~huggingface_hub.inference._text_generation.TextGenerationStreamResponse`]

        Raises:
            `ValidationError`:
                If input values are not valid. No HTTP call is made to the server.
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()

        # Case 1: generate text
        >>> await client.text_generation("The huggingface_hub library is ", max_new_tokens=12)
        '100% open source and built to be easy to use.'

        # Case 2: iterate over the generated tokens. Useful async for large generation.
        >>> async for token in await client.text_generation("The huggingface_hub library is ", max_new_tokens=12, stream=True):
        ...     print(token)
        100
        %
        open
        source
        and
        built
        to
        be
        easy
        to
        use
        .

        # Case 3: get more details about the generation process.
        >>> await client.text_generation("The huggingface_hub library is ", max_new_tokens=12, details=True)
        TextGenerationResponse(
            generated_text='100% open source and built to be easy to use.',
            details=Details(
                finish_reason=<FinishReason.Length: 'length'>,
                generated_tokens=12,
                seed=None,
                prefill=[
                    InputToken(id=487, text='The', logprob=None),
                    InputToken(id=53789, text=' hugging', logprob=-13.171875),
                    (...)
                    InputToken(id=204, text=' ', logprob=-7.0390625)
                ],
                tokens=[
                    Token(id=1425, text='100', logprob=-1.0175781, special=False),
                    Token(id=16, text='%', logprob=-0.0463562, special=False),
                    (...)
                    Token(id=25, text='.', logprob=-0.5703125, special=False)
                ],
                best_of_sequences=None
            )
        )

        # Case 4: iterate over the generated tokens with more details.
        # Last object is more complete, containing the full generated text and the finish reason.
        >>> async for details in await client.text_generation("The huggingface_hub library is ", max_new_tokens=12, details=True, stream=True):
        ...     print(details)
        ...
        TextGenerationStreamResponse(token=Token(id=1425, text='100', logprob=-1.0175781, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=16, text='%', logprob=-0.0463562, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=1314, text=' open', logprob=-1.3359375, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=3178, text=' source', logprob=-0.28100586, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=273, text=' and', logprob=-0.5961914, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=3426, text=' built', logprob=-1.9423828, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=271, text=' to', logprob=-1.4121094, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=314, text=' be', logprob=-1.5224609, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=1833, text=' easy', logprob=-2.1132812, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=271, text=' to', logprob=-0.08520508, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(id=745, text=' use', logprob=-0.39453125, special=False), generated_text=None, details=None)
        TextGenerationStreamResponse(token=Token(
            id=25,
            text='.',
            logprob=-0.5703125,
            special=False),
            generated_text='100% open source and built to be easy to use.',
            details=StreamDetails(finish_reason=<FinishReason.Length: 'length'>, generated_tokens=12, seed=None)
        )
        ```
        """
        # NOTE: Text-generation integration is taken from the text-generation-inference project. It has more features
        # like input/output validation (if Pydantic is installed). See `_text_generation.py` header for more details.

        if decoder_input_details and not details:
            warnings.warn(
                "`decoder_input_details=True` has been passed to the server but `details=False` is set meaning that"
                " the output from the server will be truncated."
            )
            decoder_input_details = False

        # Validate parameters
        parameters = TextGenerationParameters(
            best_of=best_of,
            details=details,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            return_full_text=return_full_text,
            seed=seed,
            stop=stop_sequences if stop_sequences is not None else [],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            truncate=truncate,
            typical_p=typical_p,
            watermark=watermark,
            decoder_input_details=decoder_input_details,
        )
        request = TextGenerationRequest(inputs=prompt, stream=stream, parameters=parameters)
        payload = asdict(request)

        # Remove some parameters if not a TGI server
        if not _is_tgi_server(model):
            ignored_parameters = []
            for key in "watermark", "stop", "details", "decoder_input_details":
                if payload["parameters"][key] is not None:
                    ignored_parameters.append(key)
                del payload["parameters"][key]
            if len(ignored_parameters) > 0:
                warnings.warn(
                    (
                        "API endpoint/model for text-generation is not served via TGI. Ignoring parameters"
                        f" {ignored_parameters}."
                    ),
                    UserWarning,
                )
            if details:
                warnings.warn(
                    (
                        "API endpoint/model for text-generation is not served via TGI. Parameter `details=True` will"
                        " be ignored meaning only the generated text will be returned."
                    ),
                    UserWarning,
                )
                details = False
            if stream:
                raise ValueError(
                    "API endpoint/model for text-generation is not served via TGI. Cannot return output as a stream."
                    " Please pass `stream=False` as input."
                )

        # Handle errors separately for more precise error messages
        try:
            bytes_output = await self.post(json=payload, model=model, task="text-generation", stream=stream)  # type: ignore
        except _import_aiohttp().ClientResponseError as e:
            error_message = getattr(e, "response_error_payload", {}).get("error", "")
            if e.code == 400 and "The following `model_kwargs` are not used by the model" in error_message:
                _set_as_non_tgi(model)
                return await self.text_generation(  # type: ignore
                    prompt=prompt,
                    details=details,
                    stream=stream,
                    model=model,
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens,
                    best_of=best_of,
                    repetition_penalty=repetition_penalty,
                    return_full_text=return_full_text,
                    seed=seed,
                    stop_sequences=stop_sequences,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    truncate=truncate,
                    typical_p=typical_p,
                    watermark=watermark,
                    decoder_input_details=decoder_input_details,
                )
            raise_text_generation_error(e)

        # Parse output
        if stream:
            return _async_stream_text_generation_response(bytes_output, details)  # type: ignore

        data = _bytes_to_dict(bytes_output)[0]
        return TextGenerationResponse(**data) if details else data["generated_text"]

    async def text_to_image(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        height: Optional[float] = None,
        width: Optional[float] = None,
        num_inference_steps: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> "Image":
        """
        Generate an image based on a given text using a specified model.

        <Tip warning={true}>

        You must have `PIL` installed if you want to work with images (`pip install Pillow`).

        </Tip>

        Args:
            prompt (`str`):
                The prompt to generate an image from.
            negative_prompt (`str`, *optional*):
                An optional negative prompt for the image generation.
            height (`float`, *optional*):
                The height in pixels of the image to generate.
            width (`float`, *optional*):
                The width in pixels of the image to generate.
            num_inference_steps (`int`, *optional*):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*):
                Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.

        Returns:
            `Image`: The generated image.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()

        >>> image = await client.text_to_image("An astronaut riding a horse on the moon.")
        >>> image.save("astronaut.png")

        >>> image = await client.text_to_image(
        ...     "An astronaut riding a horse on the moon.",
        ...     negative_prompt="low resolution, blurry",
        ...     model="stabilityai/stable-diffusion-2-1",
        ... )
        >>> image.save("better_astronaut.png")
        ```
        """
        parameters = {
            "inputs": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            **kwargs,
        }
        payload = {}
        for key, value in parameters.items():
            if value is not None:
                payload[key] = value
        response = await self.post(json=payload, model=model, task="text-to-image")
        return _bytes_to_image(response)

    async def text_to_speech(self, text: str, *, model: Optional[str] = None) -> bytes:
        """
        Synthesize an audio of a voice pronouncing a given text.

        Args:
            text (`str`):
                The text to synthesize.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.

        Returns:
            `bytes`: The generated audio.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from pathlib import Path
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()

        >>> audio = await client.text_to_speech("Hello world")
        >>> Path("hello_world.flac").write_bytes(audio)
        ```
        """
        return await self.post(json={"inputs": text}, model=model, task="text-to-speech")

    async def zero_shot_image_classification(
        self, image: ContentT, labels: List[str], *, model: Optional[str] = None
    ) -> List[ClassificationOutput]:
        """
        Provide input image and text labels to predict text labels for the image.

        Args:
            image (`Union[str, Path, bytes, BinaryIO]`):
                The input image to caption. It can be raw bytes, an image file, or a URL to an online image.
            labels (`List[str]`):
                List of string possible labels. The `len(labels)` must be greater than 1.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.

        Returns:
            `List[Dict]`: List of classification outputs containing the predicted labels and their confidence.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `aiohttp.ClientResponseError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        # Must be run in an async context
        >>> from huggingface_hub import AsyncInferenceClient
        >>> client = AsyncInferenceClient()

        >>> await client.zero_shot_image_classification(
        ...     "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg",
        ...     labels=["dog", "cat", "horse"],
        ... )
        [{"label": "dog", "score": 0.956}, ...]
        ```
        """

        # Raise valueerror if input is less than 2 labels
        if len(labels) < 2:
            raise ValueError("You must specify at least 2 classes to compare. Please specify more than 1 class.")

        response = await self.post(
            json={"image": _b64_encode(image), "parameters": {"candidate_labels": ",".join(labels)}},
            model=model,
            task="zero-shot-image-classification",
        )
        return _bytes_to_dict(response)

    def _resolve_url(self, model: Optional[str] = None, task: Optional[str] = None) -> str:
        model = model or self.model

        # If model is already a URL, ignore `task` and return directly
        if model is not None and (model.startswith("http://") or model.startswith("https://")):
            return model

        # # If no model but task is set => fetch the recommended one for this task
        if model is None:
            if task is None:
                raise ValueError(
                    "You must specify at least a model (repo_id or URL) or a task, either when instantiating"
                    " `InferenceClient` or when making a request."
                )
            model = _get_recommended_model(task)

        # Compute InferenceAPI url
        return (
            # Feature-extraction and sentence-similarity are the only cases where we handle models with several tasks.
            f"{INFERENCE_ENDPOINT}/pipeline/{task}/{model}"
            if task in ("feature-extraction", "sentence-similarity")
            # Otherwise, we use the default endpoint
            else f"{INFERENCE_ENDPOINT}/models/{model}"
        )
