import builtins
from collections.abc import Callable, Mapping, Sequence
from typing import Final, Literal

from openai.types.audio.transcription_create_params import FileTypes
from pydantic import Field, PrivateAttr, TypeAdapter
from pydantic.main import IncEx
from typing_extensions import ReadOnly, TypedDict, Unpack

from litellm.types.llms.base import LiteLLMBaseModel

_ATTRIBUTE_VALUE: Final = TypeAdapter(builtins.object)


class _ModelDumpOptions(TypedDict, total=False):
    """Keyword arguments `json` forwards to pydantic's `BaseModel.model_dump`."""

    mode: ReadOnly[str]
    include: ReadOnly[IncEx | None]
    exclude: ReadOnly[IncEx | None]
    context: ReadOnly[builtins.object]
    by_alias: ReadOnly[bool | None]
    exclude_unset: ReadOnly[bool]
    exclude_defaults: ReadOnly[bool]
    exclude_none: ReadOnly[bool]
    exclude_computed_fields: ReadOnly[bool]
    round_trip: ReadOnly[bool]
    warnings: ReadOnly[bool | Literal["none", "warn", "error"]]
    fallback: ReadOnly[Callable[[builtins.object], builtins.object] | None]
    serialize_as_any: ReadOnly[bool]
    polymorphic_serialization: ReadOnly[bool | None]


class VideoObject(LiteLLMBaseModel):
    """Represents a generated video object."""

    id: str
    object: Literal["video"]
    status: str
    created_at: int | None = None
    completed_at: int | None = None
    expires_at: int | None = None
    error: dict[str, builtins.object] | None = None
    progress: int | None = None
    remixed_from_video_id: str | None = None
    seconds: str | None = None
    size: str | None = None
    model: str | None = None
    usage: Mapping[str, builtins.object] | None = None
    _hidden_params: dict[str, builtins.object] = PrivateAttr(default={})

    @property
    def hidden_params(self) -> dict[str, builtins.object]:  # mutable-ok: API requires mutation
        return self._hidden_params

    @hidden_params.setter
    def hidden_params(self, hidden_params: dict[str, builtins.object]) -> None:  # mutable-ok: API requires mutation
        self._hidden_params = hidden_params

    def __contains__(self, key: str) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key: str, default: builtins.object = None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> builtins.object:
        # Allow dictionary-style access to attributes
        return _ATTRIBUTE_VALUE.validate_python(getattr(self, key))

    def json(self, **kwargs: Unpack[_ModelDumpOptions]):
        try:
            return self.model_dump(**kwargs)
        except Exception:
            # if using pydantic v1
            return self.dict()


class VideoResponse(LiteLLMBaseModel):
    """Response object for video generation requests."""

    data: list[VideoObject]
    hidden_params: dict[str, object] = Field(default={})

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: builtins.object = None):
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> object:
        return _ATTRIBUTE_VALUE.validate_python(getattr(self, key))

    def json(self, **kwargs: Unpack[_ModelDumpOptions]):
        try:
            return self.model_dump(**kwargs)
        except Exception:
            return self.dict()


class VideoImageURL(TypedDict):
    url: ReadOnly[str]


class VideoInputReference(TypedDict):
    type: ReadOnly[Literal["image_url"]]
    image_url: ReadOnly[VideoImageURL]


class VideoFrameImage(VideoInputReference):
    frame_type: ReadOnly[Literal["first_frame", "last_frame"]]


class VideoCreateOptionalRequestParams(TypedDict, total=False):
    """
    TypedDict for Optional parameters supported by OpenAI's video creation API.

    Params here: https://platform.openai.com/docs/api-reference/videos/create
    """

    input_reference: FileTypes | None  # File reference for input image
    image: object | None  # Image for image-to-video; dict with gcsUri/bytesBase64Encoded, or file-like object
    parameters: dict[str, object] | None  # Provider-specific parameters block passed directly to the API
    model: str | None
    resolution: ReadOnly[str | None]
    duration: ReadOnly[int]
    aspect_ratio: ReadOnly[str]
    frame_images: ReadOnly[Sequence[VideoFrameImage]]
    input_references: ReadOnly[Sequence[VideoInputReference]]
    generate_audio: ReadOnly[bool]
    seed: ReadOnly[int]
    seconds: str | None
    size: str | None
    characters: list[dict[str, str]] | None
    user: str | None
    extra_headers: dict[str, str] | None
    extra_body: dict[str, str] | None


class VideoCreateRequestParams(VideoCreateOptionalRequestParams, total=False):
    """
    TypedDict for request parameters supported by OpenAI's video creation API.

    Params here: https://platform.openai.com/docs/api-reference/videos/create
    """

    prompt: str


class DecodedVideoId(TypedDict, total=False):
    """Structure representing a decoded video ID"""

    custom_llm_provider: str | None
    model_id: str | None
    video_id: str
    served_model: ReadOnly[str | None]
    deployment_id: ReadOnly[str | None]
    requested_model: ReadOnly[str | None]


class CharacterObject(LiteLLMBaseModel):
    """Represents a character created from a video."""

    id: str
    object: Literal["character"] = "character"
    created_at: int
    name: str
    _hidden_params: dict[str, builtins.object] = PrivateAttr(default={})

    @property
    def hidden_params(self) -> dict[str, builtins.object]:  # mutable-ok: API requires mutation
        return self._hidden_params

    @hidden_params.setter
    def hidden_params(self, hidden_params: dict[str, builtins.object]) -> None:  # mutable-ok: API requires mutation
        self._hidden_params = hidden_params

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: builtins.object = None):
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> builtins.object:
        return _ATTRIBUTE_VALUE.validate_python(getattr(self, key))

    def json(self, **kwargs: Unpack[_ModelDumpOptions]):
        try:
            return self.model_dump(**kwargs)
        except Exception:
            return self.dict()


class VideoEditRequestParams(TypedDict, total=False):
    """TypedDict for video edit request parameters."""

    prompt: str
    video: dict[str, str]  # {"id": "video_123"}


class VideoExtensionRequestParams(TypedDict, total=False):
    """TypedDict for video extension request parameters."""

    prompt: str
    seconds: str
    video: dict[str, str]  # {"id": "video_123"}
