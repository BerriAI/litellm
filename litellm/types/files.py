from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Literal, Mapping, Set, Union

from typing_extensions import Required, TypedDict

"""
Base Enums/Consts
"""


class FileType(Enum):
    AAC = "AAC"
    CSV = "CSV"
    DOC = "DOC"
    DOCX = "DOCX"
    FLAC = "FLAC"
    FLV = "FLV"
    GIF = "GIF"
    GOOGLE_DOC = "GOOGLE_DOC"
    GOOGLE_DRAWINGS = "GOOGLE_DRAWINGS"
    GOOGLE_SHEETS = "GOOGLE_SHEETS"
    GOOGLE_SLIDES = "GOOGLE_SLIDES"
    HEIC = "HEIC"
    HEIF = "HEIF"
    HTML = "HTML"
    JPEG = "JPEG"
    JSON = "JSON"
    M4A = "M4A"
    M4V = "M4V"
    MOV = "MOV"
    MP3 = "MP3"
    MP4 = "MP4"
    MPEG = "MPEG"
    MPEGPS = "MPEGPS"
    MPG = "MPG"
    MPA = "MPA"
    MPGA = "MPGA"
    OGG = "OGG"
    OPUS = "OPUS"
    PDF = "PDF"
    PCM = "PCM"
    PNG = "PNG"
    PPT = "PPT"
    PPTX = "PPTX"
    RTF = "RTF"
    THREE_GPP = "3GPP"
    TXT = "TXT"
    WAV = "WAV"
    WEBM = "WEBM"
    WEBP = "WEBP"
    WMV = "WMV"
    XLS = "XLS"
    XLSX = "XLSX"


FILE_EXTENSIONS: Mapping[FileType, List[str]] = MappingProxyType(
    {
        FileType.AAC: ["aac"],
        FileType.CSV: ["csv"],
        FileType.DOC: ["doc"],
        FileType.DOCX: ["docx"],
        FileType.FLAC: ["flac"],
        FileType.FLV: ["flv"],
        FileType.GIF: ["gif"],
        FileType.GOOGLE_DOC: ["gdoc"],
        FileType.GOOGLE_DRAWINGS: ["gdraw"],
        FileType.GOOGLE_SHEETS: ["gsheet"],
        FileType.GOOGLE_SLIDES: ["gslides"],
        FileType.HEIC: ["heic"],
        FileType.HEIF: ["heif"],
        FileType.HTML: ["html", "htm"],
        FileType.JPEG: ["jpeg", "jpg"],
        FileType.JSON: ["json"],
        FileType.M4A: ["m4a"],
        FileType.M4V: ["m4v"],
        FileType.MOV: ["mov"],
        FileType.MP3: ["mp3"],
        FileType.MP4: ["mp4"],
        FileType.MPEG: ["mpeg"],
        FileType.MPEGPS: ["mpegps"],
        FileType.MPG: ["mpg"],
        FileType.MPA: ["mpa"],
        FileType.MPGA: ["mpga"],
        FileType.OGG: ["ogg"],
        FileType.OPUS: ["opus"],
        FileType.PDF: ["pdf"],
        FileType.PCM: ["pcm"],
        FileType.PNG: ["png"],
        FileType.PPT: ["ppt"],
        FileType.PPTX: ["pptx"],
        FileType.RTF: ["rtf"],
        FileType.THREE_GPP: ["3gpp"],
        FileType.TXT: ["txt"],
        FileType.WAV: ["wav"],
        FileType.WEBM: ["webm"],
        FileType.WEBP: ["webp"],
        FileType.WMV: ["wmv"],
        FileType.XLS: ["xls"],
        FileType.XLSX: ["xlsx"],
    }
)

FILE_MIME_TYPES: Mapping[FileType, str] = MappingProxyType(
    {
        FileType.AAC: "audio/aac",
        FileType.CSV: "text/csv",
        FileType.DOC: "application/msword",
        FileType.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        FileType.FLAC: "audio/flac",
        FileType.FLV: "video/x-flv",
        FileType.GIF: "image/gif",
        FileType.GOOGLE_DOC: "application/vnd.google-apps.document",
        FileType.GOOGLE_DRAWINGS: "application/vnd.google-apps.drawing",
        FileType.GOOGLE_SHEETS: "application/vnd.google-apps.spreadsheet",
        FileType.GOOGLE_SLIDES: "application/vnd.google-apps.presentation",
        FileType.HEIC: "image/heic",
        FileType.HEIF: "image/heif",
        FileType.HTML: "text/html",
        FileType.JPEG: "image/jpeg",
        FileType.JSON: "application/json",
        FileType.M4A: "audio/x-m4a",
        FileType.M4V: "video/x-m4v",
        FileType.MOV: "video/quicktime",
        FileType.MP3: "audio/mpeg",
        FileType.MP4: "video/mp4",
        FileType.MPEG: "video/mpeg",
        FileType.MPEGPS: "video/mpegps",
        FileType.MPG: "video/mpg",
        FileType.MPA: "audio/m4a",
        FileType.MPGA: "audio/mpga",
        FileType.OGG: "audio/ogg",
        FileType.OPUS: "audio/opus",
        FileType.PDF: "application/pdf",
        FileType.PCM: "audio/pcm",
        FileType.PNG: "image/png",
        FileType.PPT: "application/vnd.ms-powerpoint",
        FileType.PPTX: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        FileType.RTF: "application/rtf",
        FileType.THREE_GPP: "video/3gpp",
        FileType.TXT: "text/plain",
        FileType.WAV: "audio/wav",
        FileType.WEBM: "video/webm",
        FileType.WEBP: "image/webp",
        FileType.WMV: "video/wmv",
        FileType.XLS: "application/vnd.ms-excel",
        FileType.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
)

"""
Util Functions
"""


def get_file_extension_from_mime_type(mime_type: str) -> str:
    for file_type, mime in FILE_MIME_TYPES.items():
        if mime.lower() == mime_type.lower():
            return FILE_EXTENSIONS[file_type][0]
    raise ValueError(f"Unknown extension for mime type: {mime_type}")


def get_file_type_from_extension(extension: str) -> FileType:
    for file_type, extensions in FILE_EXTENSIONS.items():
        if extension.lower() in extensions:
            return file_type

    raise ValueError(f"Unknown file type for extension: {extension}")


def get_file_extension_for_file_type(file_type: FileType) -> str:
    return FILE_EXTENSIONS[file_type][0]


def get_file_mime_type_for_file_type(file_type: FileType) -> str:
    return FILE_MIME_TYPES[file_type]


def get_file_mime_type_from_extension(extension: str) -> str:
    file_type = get_file_type_from_extension(extension)
    return get_file_mime_type_for_file_type(file_type)


"""
FileType Type Groupings (Videos, Images, etc)
"""

# Images
IMAGE_FILE_TYPES = {
    FileType.PNG,
    FileType.JPEG,
    FileType.GIF,
    FileType.WEBP,
    FileType.HEIC,
    FileType.HEIF,
}


def is_image_file_type(file_type):
    return file_type in IMAGE_FILE_TYPES


# Videos
VIDEO_FILE_TYPES = {
    FileType.MOV,
    FileType.MP4,
    FileType.MPEG,
    FileType.M4V,
    FileType.FLV,
    FileType.MPEGPS,
    FileType.MPG,
    FileType.WEBM,
    FileType.WMV,
    FileType.THREE_GPP,
}


def is_video_file_type(file_type):
    return file_type in VIDEO_FILE_TYPES


# Audio
AUDIO_FILE_TYPES = {
    FileType.AAC,
    FileType.FLAC,
    FileType.MP3,
    FileType.MPA,
    FileType.MPGA,
    FileType.OPUS,
    FileType.PCM,
    FileType.WAV,
}


def is_audio_file_type(file_type):
    return file_type in AUDIO_FILE_TYPES


# Text
TEXT_FILE_TYPES = {FileType.CSV, FileType.HTML, FileType.RTF, FileType.TXT}


def is_text_file_type(file_type):
    return file_type in TEXT_FILE_TYPES


"""
Other FileType Groupings
"""
# Accepted file types for GEMINI 1.5 through Vertex AI
# https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-multimodal-prompts#gemini-send-multimodal-samples-images-nodejs
GEMINI_1_5_ACCEPTED_FILE_TYPES: Set[FileType] = {
    # Image
    FileType.PNG,
    FileType.JPEG,
    FileType.WEBP,
    # Audio
    FileType.AAC,
    FileType.FLAC,
    FileType.MP3,
    FileType.MPA,
    FileType.MPEG,
    FileType.MPGA,
    FileType.OPUS,
    FileType.PCM,
    FileType.WAV,
    FileType.WEBM,
    # Video
    FileType.FLV,
    FileType.MOV,
    FileType.MPEG,
    FileType.MPEGPS,
    FileType.MPG,
    FileType.MP4,
    FileType.WEBM,
    FileType.WMV,
    FileType.THREE_GPP,
    # PDF
    FileType.PDF,
    FileType.TXT,
}


def is_gemini_1_5_accepted_file_type(file_type: FileType) -> bool:
    return file_type in GEMINI_1_5_ACCEPTED_FILE_TYPES


"""
Two-Step File Upload Types
"""


class TwoStepFileUploadRequest(TypedDict):
    """
    Request structure for two-step file upload process.
    
    Step 1: Initial request to get upload URL
    Step 2: Upload file content to the upload URL
    
    Used by providers like Manus and Google Cloud Storage.
    """

    method: Required[str]
    url: Required[str]
    headers: Required[Dict[str, str]]
    data: Required[Union[str, bytes, Dict[str, Any]]]


class TwoStepFileUploadConfig(TypedDict, total=False):
    """
    Configuration for two-step file upload process.
    
    Properties:
        initial_request: Request to create file record and get upload URL
        upload_request: Request to upload actual file content
        upload_url_location: Where to find upload URL ('headers' or 'body')
        upload_url_key: Key name for upload URL in response (default: 'upload_url')
    """

    initial_request: Required[TwoStepFileUploadRequest]
    upload_request: Required[TwoStepFileUploadRequest]
    upload_url_location: Required[Literal["headers", "body"]]
    upload_url_key: str
