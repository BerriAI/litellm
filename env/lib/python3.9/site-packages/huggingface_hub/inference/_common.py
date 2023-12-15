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
"""Contains utilities used by both the sync and async inference clients."""
import base64
import io
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    BinaryIO,
    ContextManager,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Union,
    overload,
)

from requests import HTTPError

from ..constants import ENDPOINT
from ..utils import (
    build_hf_headers,
    get_session,
    hf_raise_for_status,
    is_aiohttp_available,
    is_numpy_available,
    is_pillow_available,
)
from ..utils._typing import Literal
from ._text_generation import (
    TextGenerationStreamResponse,
)


if TYPE_CHECKING:
    from aiohttp import ClientResponse, ClientSession
    from PIL import Image

# TYPES
UrlT = str
PathT = Union[str, Path]
BinaryT = Union[bytes, BinaryIO]
ContentT = Union[BinaryT, PathT, UrlT]

logger = logging.getLogger(__name__)


class InferenceTimeoutError(HTTPError, TimeoutError):
    """Error raised when a model is unavailable or the request times out."""


## IMPORT UTILS


def _import_aiohttp():
    # Make sure `aiohttp` is installed on the machine.
    if not is_aiohttp_available():
        raise ImportError("Please install aiohttp to use `AsyncInferenceClient` (`pip install aiohttp`).")
    import aiohttp

    return aiohttp


def _import_numpy():
    """Make sure `numpy` is installed on the machine."""
    if not is_numpy_available():
        raise ImportError("Please install numpy to use deal with embeddings (`pip install numpy`).")
    import numpy

    return numpy


def _import_pil_image():
    """Make sure `PIL` is installed on the machine."""
    if not is_pillow_available():
        raise ImportError(
            "Please install Pillow to use deal with images (`pip install Pillow`). If you don't want the image to be"
            " post-processed, use `client.post(...)` and get the raw response from the server."
        )
    from PIL import Image

    return Image


## RECOMMENDED MODELS

# Will be globally fetched only once (see '_fetch_recommended_models')
_RECOMMENDED_MODELS: Optional[Dict[str, Optional[str]]] = None


def _get_recommended_model(task: str) -> str:
    model = _fetch_recommended_models().get(task)
    if model is None:
        raise ValueError(
            f"Task {task} has no recommended task. Please specify a model explicitly. Visit"
            " https://huggingface.co/tasks for more info."
        )
    logger.info(
        f"Using recommended model {model} for task {task}. Note that it is encouraged to explicitly set"
        f" `model='{model}'` as the recommended models list might get updated without prior notice."
    )
    return model


def _fetch_recommended_models() -> Dict[str, Optional[str]]:
    global _RECOMMENDED_MODELS
    if _RECOMMENDED_MODELS is None:
        response = get_session().get(f"{ENDPOINT}/api/tasks", headers=build_hf_headers())
        hf_raise_for_status(response)
        _RECOMMENDED_MODELS = {
            task: _first_or_none(details["widgetModels"]) for task, details in response.json().items()
        }
    return _RECOMMENDED_MODELS


def _first_or_none(items: List[Any]) -> Optional[Any]:
    try:
        return items[0] or None
    except IndexError:
        return None


## ENCODING / DECODING UTILS


@overload
def _open_as_binary(content: ContentT) -> ContextManager[BinaryT]:
    ...  # means "if input is not None, output is not None"


@overload
def _open_as_binary(content: Literal[None]) -> ContextManager[Literal[None]]:
    ...  # means "if input is None, output is None"


@contextmanager  # type: ignore
def _open_as_binary(content: Optional[ContentT]) -> Generator[Optional[BinaryT], None, None]:
    """Open `content` as a binary file, either from a URL, a local path, or raw bytes.

    Do nothing if `content` is None,

    TODO: handle a PIL.Image as input
    TODO: handle base64 as input
    """
    # If content is a string => must be either a URL or a path
    if isinstance(content, str):
        if content.startswith("https://") or content.startswith("http://"):
            logger.debug(f"Downloading content from {content}")
            yield get_session().get(content).content  # TODO: retrieve as stream and pipe to post request ?
            return
        content = Path(content)
        if not content.exists():
            raise FileNotFoundError(
                f"File not found at {content}. If `data` is a string, it must either be a URL or a path to a local"
                " file. To pass raw content, please encode it as bytes first."
            )

    # If content is a Path => open it
    if isinstance(content, Path):
        logger.debug(f"Opening content from {content}")
        with content.open("rb") as f:
            yield f
    else:
        # Otherwise: already a file-like object or None
        yield content


def _b64_encode(content: ContentT) -> str:
    """Encode a raw file (image, audio) into base64. Can be byes, an opened file, a path or a URL."""
    with _open_as_binary(content) as data:
        data_as_bytes = data if isinstance(data, bytes) else data.read()
        return base64.b64encode(data_as_bytes).decode()


def _b64_to_image(encoded_image: str) -> "Image":
    """Parse a base64-encoded string into a PIL Image."""
    Image = _import_pil_image()
    return Image.open(io.BytesIO(base64.b64decode(encoded_image)))


def _bytes_to_dict(content: bytes) -> "Image":
    """Parse bytes from a Response object into a Python dictionary.

    Expects the response body to be encoded-JSON data.
    """
    return json.loads(content.decode())


def _bytes_to_image(content: bytes) -> "Image":
    """Parse bytes from a Response object into a PIL Image.

    Expects the response body to be raw bytes. To deal with b64 encoded images, use `_b64_to_image` instead.
    """
    Image = _import_pil_image()
    return Image.open(io.BytesIO(content))


## STREAMING UTILS


def _stream_text_generation_response(
    bytes_output_as_lines: Iterable[bytes], details: bool
) -> Union[Iterable[str], Iterable[TextGenerationStreamResponse]]:
    # Parse ServerSentEvents
    for byte_payload in bytes_output_as_lines:
        # Skip line
        if byte_payload == b"\n":
            continue

        payload = byte_payload.decode("utf-8")

        # Event data
        if payload.startswith("data:"):
            # Decode payload
            json_payload = json.loads(payload.lstrip("data:").rstrip("/n"))
            # Parse payload
            output = TextGenerationStreamResponse(**json_payload)
            yield output.token.text if not details else output


async def _async_stream_text_generation_response(
    bytes_output_as_lines: AsyncIterable[bytes], details: bool
) -> Union[AsyncIterable[str], AsyncIterable[TextGenerationStreamResponse]]:
    # Parse ServerSentEvents
    async for byte_payload in bytes_output_as_lines:
        # Skip line
        if byte_payload == b"\n":
            continue

        payload = byte_payload.decode("utf-8")

        # Event data
        if payload.startswith("data:"):
            # Decode payload
            json_payload = json.loads(payload.lstrip("data:").rstrip("/n"))
            # Parse payload
            output = TextGenerationStreamResponse(**json_payload)
            yield output.token.text if not details else output


async def _async_yield_from(client: "ClientSession", response: "ClientResponse") -> AsyncIterable[bytes]:
    async for byte_payload in response.content:
        yield byte_payload
    await client.close()


# "TGI servers" are servers running with the `text-generation-inference` backend.
# This backend is the go-to solution to run large language models at scale. However,
# for some smaller models (e.g. "gpt2") the default `transformers` + `api-inference`
# solution is still in use.
#
# Both approaches have very similar APIs, but not exactly the same. What we do first in
# the `text_generation` method is to assume the model is served via TGI. If we realize
# it's not the case (i.e. we receive an HTTP 400 Bad Request), we fallback to the
# default API with a warning message. We remember for each model if it's a TGI server
# or not using `_NON_TGI_SERVERS` global variable.
#
# For more details, see https://github.com/huggingface/text-generation-inference and
# https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task.

_NON_TGI_SERVERS: Set[Optional[str]] = set()


def _set_as_non_tgi(model: Optional[str]) -> None:
    _NON_TGI_SERVERS.add(model)


def _is_tgi_server(model: Optional[str]) -> bool:
    return model not in _NON_TGI_SERVERS
