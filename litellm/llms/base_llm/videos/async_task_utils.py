import base64
from collections.abc import Mapping
from io import BufferedReader, BytesIO
from itertools import accumulate, repeat
from pathlib import Path, PurePath
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.images.utils import ImageEditRequestUtils
from litellm.types.videos.utils import decode_video_id_with_provider

_FIELDS_ADAPTER: Final = TypeAdapter(dict[str, object])
_FILE_TUPLE_ADAPTER: Final = TypeAdapter(tuple[object, ...])


def _fields(value: object) -> Mapping[str, object] | None:
    try:
        return _FIELDS_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _logged_litellm_params(logging_obj: object) -> Mapping[str, object]:
    return _fields(getattr(logging_obj, "litellm_params", None)) or {}


def _id_safe_name(name: object) -> str | None:
    return name if isinstance(name, str) and name and ";" not in name else None


def _stamped_models(metadata: object) -> tuple[str, str | None] | None:
    fields: Final = _fields(metadata)
    if fields is None:
        return None
    requested: Final = _id_safe_name(fields.get("model_group"))
    served: Final = _id_safe_name(fields.get("deployment_model_name"))
    if requested is None:
        return None
    return requested, None if served is None or served == requested or "*" in served else served


def routed_video_models(logging_obj: object) -> tuple[str, str | None] | None:
    """
    A later status or content request checks the key's model access against the model encoded in the id, and
    routes by it. One upstream model name can be served by several deployments (regions, accounts), so only the
    group that served the task can send its polls back to the deployment that created it, while a key may be
    scoped to the alias, routing group or wildcard name it asked for instead. The id carries both names: the one
    the router was asked for on this hop, and the group that served it when that differs and is not a wildcard.
    The router stamps both into litellm_metadata, and metadata carries a copy only when the caller sent none.
    """
    params: Final = _logged_litellm_params(logging_obj)
    stamped: Final = (_stamped_models(params.get(key)) for key in ("litellm_metadata", "metadata"))
    return next((models for models in stamped if models), None)


def polled_video_models(logging_obj: object) -> tuple[str | None, str | None]:
    """
    The served group survives into the returned id only when the router sent this poll to it, so an edited id
    cannot write any name it likes back.
    """
    video_id: Final = _logged_litellm_params(logging_obj).get("video_id")
    if not isinstance(video_id, str):
        return None, None
    decoded: Final = decode_video_id_with_provider(video_id)
    served: Final = decoded.get("served_model")
    routed: Final = routed_video_models(logging_obj)
    return decoded.get("model_id") or None, served if routed is None or served in routed else None


def _tuple_payload(file_obj: object) -> object:
    if not isinstance(file_obj, tuple):
        return file_obj
    file_tuple: Final = _FILE_TUPLE_ADAPTER.validate_python(file_obj)
    return file_tuple[1] if len(file_tuple) >= 2 else None


def _file_content(file_obj: object) -> object:
    unwrapped: Final = accumulate(repeat(None), lambda current, _: _tuple_payload(current), initial=file_obj)
    return next(item for item in unwrapped if not isinstance(item, tuple))


def input_reference_bytes(file_obj: object) -> bytes:
    content: Final = _file_content(file_obj)
    if isinstance(content, PurePath):
        return Path(content).read_bytes()
    if isinstance(content, (BytesIO, BufferedReader)):
        current_position: Final = content.tell()
        _ = content.seek(0)
        data: Final = content.read()
        _ = content.seek(current_position)
        return data
    if isinstance(content, bytes):
        return content
    if isinstance(content, bytearray):
        return bytes(content)
    read: Final = getattr(content, "read", None)
    if callable(read):
        read_data: Final[object] = read()
        if isinstance(read_data, bytes):
            return read_data
    raise ValueError("input_reference must be a URL string, bytes, a path, a file object, or a file tuple")


def _reference_object_url(reference: Mapping[str, object]) -> str:
    image_url: Final = reference.get("image_url")
    if isinstance(image_url, str) and image_url:
        return image_url
    raise ValueError("input_reference as an object must carry an image_url, a file_id is not supported")


def input_reference_url(image: object) -> str:
    """A string is a URL the provider fetches itself and is never opened as a local path. OpenAI's object form
    carries that URL, or a data URL, in ``image_url``."""
    if isinstance(image, str):
        return image
    reference: Final = _fields(image)
    if reference is not None:
        return _reference_object_url(reference)
    content: Final = input_reference_bytes(image)
    content_type: Final = ImageEditRequestUtils.get_image_content_type(content)
    encoded: Final = base64.b64encode(content).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"
