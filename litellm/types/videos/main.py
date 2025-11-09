from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel
from litellm.types.utils import FileTypes


class VideoObject(BaseModel):
    """Represents a generated video object."""
    id: str
    object: Literal["video"]
    status: str
    created_at: Optional[int] = None
    completed_at: Optional[int] = None
    expires_at: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    progress: Optional[int] = None
    remixed_from_video_id: Optional[str] = None
    seconds: Optional[str] = None
    size: Optional[str] = None
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    _hidden_params: Dict[str, Any] = {}

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump(**kwargs)
        except Exception:
            # if using pydantic v1
            return self.dict()




class VideoResponse(BaseModel):
    """Response object for video generation requests."""
    data: List[VideoObject]
    hidden_params: Dict[str, Any] = {}

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump(**kwargs)
        except Exception:
            return self.dict()


class VideoCreateOptionalRequestParams(TypedDict, total=False):
    """
    TypedDict for Optional parameters supported by OpenAI's video creation API.

    Params here: https://platform.openai.com/docs/api-reference/videos/create
    """
    input_reference: Optional[FileTypes]  # File reference for input image
    model: Optional[str]
    seconds: Optional[str]
    size: Optional[str]
    user: Optional[str]
    extra_headers: Optional[Dict[str, str]]
    extra_body: Optional[Dict[str, str]]


class VideoCreateRequestParams(VideoCreateOptionalRequestParams, total=False):
    """
    TypedDict for request parameters supported by OpenAI's video creation API.

    Params here: https://platform.openai.com/docs/api-reference/videos/create
    """
    prompt: str

class DecodedVideoId(TypedDict, total=False):
    """Structure representing a decoded video ID"""

    custom_llm_provider: Optional[str]
    model_id: Optional[str]
    video_id: str