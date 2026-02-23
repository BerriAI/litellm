from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

from typing_extensions import Required, TypedDict

from .vertex_ai import (
    GenerationConfig,
    HttpxBlobType,
    HttpxContentType,
    Tools,
    UsageMetadata,
)


class GeminiFilesState(Enum):
    STATE_UNSPECIFIED = "STATE_UNSPECIFIED"
    PROCESSING = "PROCESSING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"


class GeminiFilesSource(Enum):
    SOURCE_UNSPECIFIED = "SOURCE_UNSPECIFIED"
    UPLOADED = "UPLOADED"
    GENERATED = "GENERATED"


class GeminiCreateFilesResponseObject(TypedDict):
    name: str
    displayName: str
    mimeType: str
    sizeBytes: str
    createTime: str
    updateTime: str
    expirationTime: str
    sha256Hash: str
    uri: str
    state: GeminiFilesState
    source: GeminiFilesSource
    error: dict
    metadata: dict


class BidiGenerateContentTranscription(TypedDict):
    text: str
    """Output only. The transcription of the audio."""


class BidiGenerateContentServerContent(TypedDict, total=False):
    generationComplete: bool
    """Output only. If true, indicates that the model is done generating."""

    turnComplete: bool
    """Output only. If true, indicates that the model has completed its turn. Generation will only start in response to additional client messages."""

    interrupted: bool
    """Output only. If true, indicates that a client message has interrupted current model generation. If the client is playing out the content in real time, this is a good signal to stop and empty the current playback queue."""

    groundingMetadata: dict
    """Output only. Grounding metadata for the generated content."""

    inputTranscription: BidiGenerateContentTranscription
    """Output only. Input audio transcription. The transcription is sent independently of the other server messages and there is no guaranteed ordering."""

    outputTranscription: BidiGenerateContentTranscription
    """Output only. Output audio transcription. The transcription is sent independently of the other server messages and there is no guaranteed ordering, in particular not between serverContent and this outputTranscription."""

    modelTurn: HttpxContentType
    """Output only. The content that the model is currently generating."""


class BidiGenerateContentServerMessage(TypedDict, total=False):
    usageMetadata: UsageMetadata
    """Output only. Usage metadata for the generated content."""

    serverContent: BidiGenerateContentServerContent
    """Output only. The content that the model is currently generating."""

    setupComplete: dict
    """Output only. The setup complete message."""


class BidiGenerateContentRealtimeInput(TypedDict, total=False):
    text: str
    """The text to be sent to the model."""

    audio: HttpxBlobType
    """The audio to be sent to the model."""

    video: HttpxBlobType
    """The video to be sent to the model."""

    audioStreamEnd: bool
    """Output only. If true, indicates that the audio stream has ended."""

    activityStart: bool
    """Output only. If true, indicates that the activity has started."""

    activityEnd: bool
    """Output only. If true, indicates that the activity has ended."""


StartOfSpeechSensitivityEnum = Literal[
    "START_SENSITIVITY_UNSPECIFIED", "START_SENSITIVITY_HIGH", "START_SENSITIVITY_LOW"
]
EndOfSpeechSensitivityEnum = Literal[
    "END_SENSITIVITY_UNSPECIFIED", "END_SENSITIVITY_HIGH", "END_SENSITIVITY_LOW"
]


class AutomaticActivityDetection(TypedDict, total=False):
    disabled: bool
    startOfSpeechSensitivity: StartOfSpeechSensitivityEnum
    prefixPaddingMs: int
    endOfSpeechSensitivity: EndOfSpeechSensitivityEnum
    silenceDurationMs: int


class BidiGenerateContentRealtimeInputConfig(TypedDict, total=False):
    automaticActivityDetection: AutomaticActivityDetection


class BidiGenerateContentSetup(TypedDict, total=False):
    model: str
    """The model to be used for the realtime session."""

    generationConfig: GenerationConfig
    """The generation config to be used for the realtime session."""

    systemInstruction: HttpxContentType
    """The system instruction to be used for the realtime session."""

    tools: List[Tools]
    """The tools to be used for the realtime session."""

    realtimeInputConfig: dict
    """The realtime config to be used for the realtime session."""

    sessionResumption: dict
    """The session resumption to be used for the realtime session."""

    sessionResumptionConfig: dict
    """The session resumption config to be used for the realtime session."""

    contextWindowCompression: dict
    """The context window compression to be used for the realtime session."""

    inputAudioTranscription: dict
    """The input audio transcription to be used for the realtime session."""

    outputAudioTranscription: dict
    """The output audio transcription to be used for the realtime session."""


# Image Generation Types
from pydantic import BaseModel


class GeminiImageGenerationInstance(TypedDict):
    """Instance data for Gemini image generation request"""
    prompt: str


class GeminiImageGenerationParameters(BaseModel):
    """Parameters for Gemini image generation request"""
    sampleCount: Optional[int] = None
    """Number of images to generate (maps to OpenAI 'n' parameter)"""
    
    aspectRatio: Optional[str] = None
    """Aspect ratio for generated images (e.g., '1:1', '16:9', '9:16', '4:3', '3:4')"""
    
    personGeneration: Optional[str] = None
    """Controls person generation in images"""
    
    # Additional parameters that might be passed through
    background: Optional[str] = None
    """Background specification"""
    
    input_fidelity: Optional[str] = None
    """Input fidelity specification"""
    
    moderation: Optional[str] = None
    """Moderation settings"""
    
    output_compression: Optional[str] = None
    """Output compression settings"""
    
    output_format: Optional[str] = None
    """Output format specification"""
    
    quality: Optional[str] = None
    """Quality settings"""
    
    response_format: Optional[str] = None
    """Response format specification"""
    
    style: Optional[str] = None
    """Style specification"""
    
    user: Optional[str] = None
    """User specification"""


class GeminiImageGenerationRequest(BaseModel):
    """Complete request body for Gemini image generation"""
    instances: List[GeminiImageGenerationInstance]
    parameters: GeminiImageGenerationParameters


class GeminiGeneratedImage(TypedDict):
    """Individual generated image data from Gemini response"""
    bytesBase64Encoded: str
    """Base64 encoded image data"""


class GeminiImageGenerationPrediction(TypedDict):
    """Prediction object containing generated images"""
    generatedImages: List[GeminiGeneratedImage]


class GeminiImageGenerationResponse(TypedDict):
    """Complete response body from Gemini image generation API"""
    predictions: List[GeminiImageGenerationPrediction]

# Video Generation Types
class GeminiVideoGenerationInstance(TypedDict):
    """Instance data for Gemini video generation request"""
    prompt: str


class GeminiVideoGenerationParameters(BaseModel):
    """
    Parameters for Gemini video generation request.
    
    See: Veo 3/3.1 parameter guide.
    """
    aspectRatio: Optional[str] = None
    """Aspect ratio for generated video (e.g., '16:9', '9:16')."""

    durationSeconds: Optional[int] = None
    """
    Length of the generated video in seconds (e.g., 4, 5, 6, 8).
    Must be 8 when using extension/interpolation or referenceImages.
    """

    resolution: Optional[str] = None
    """
    Video resolution (e.g., '720p', '1080p').
    '1080p' only supports 8s duration; extension only supports '720p'.
    """

    negativePrompt: Optional[str] = None
    """Text describing what not to include in the video."""

    image: Optional[Any] = None
    """
    An initial image to animate (Image object).
    """

    lastFrame: Optional[Any] = None
    """
    The final image for interpolation video to transition.
    Should be used with the 'image' parameter.
    """

    referenceImages: Optional[list] = None
    """
    Up to three images to be used as style/content references.
    Only supported in Veo 3.1 (list of VideoGenerationReferenceImage objects).
    """

    video: Optional[Any] = None
    """
    Video to be used for video extension (Video object).
    Only supported in Veo 3.1 & Veo 3 Fast.
    """

    personGeneration: Optional[str] = None
    """
    Controls the generation of people.
    Text-to-video & Extension: "allow_all" only
    Image-to-video, Interpolation, & Reference images (Veo 3.x): "allow_adult" only
    See documentation for region restrictions & more.
    """


class GeminiVideoGenerationRequest(BaseModel):
    """Complete request body for Gemini video generation"""
    instances: List[GeminiVideoGenerationInstance]
    parameters: Optional[GeminiVideoGenerationParameters] = None


# Video Generation Operation Response Types
class GeminiVideoUri(BaseModel):
    """Video URI in the generated sample"""
    uri: str
    """File URI of the generated video (e.g., 'files/abc123...')"""


class GeminiGeneratedVideoSample(BaseModel):
    """Individual generated video sample"""
    video: GeminiVideoUri
    """Video object containing the URI"""


class GeminiGenerateVideoResponse(BaseModel):
    """Generate video response containing the samples"""
    generatedSamples: List[GeminiGeneratedVideoSample]
    """List of generated video samples"""


class GeminiOperationResponse(BaseModel):
    """Response object in the operation when done"""
    generateVideoResponse: GeminiGenerateVideoResponse
    """Video generation response"""


class GeminiOperationMetadata(BaseModel):
    """Metadata for the operation"""
    createTime: Optional[str] = None
    """Creation timestamp"""
    model: Optional[str] = None
    """Model used for generation"""


class GeminiLongRunningOperationResponse(BaseModel):
    """
    Complete response for a long-running operation.
    
    Used when polling operation status and extracting results.
    """
    name: str
    """Operation name (e.g., 'operations/generate_1234567890')"""
    
    done: bool = False
    """Whether the operation is complete"""
    
    metadata: Optional[GeminiOperationMetadata] = None
    """Operation metadata"""
    
    response: Optional[GeminiOperationResponse] = None
    """Response object when operation is complete"""
    
    error: Optional[Dict[str, Any]] = None
    """Error details if operation failed"""
