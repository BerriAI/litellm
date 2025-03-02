import os
from enum import Enum
from types import MappingProxyType
from typing import List, Set, Mapping, Optional
from urllib.parse import urlparse

"""
Base Enums/Consts
"""


class FileType(Enum):
    AIFF = "AIFF"
    AAC = "AAC"
    AVI = "AVI"
    CSS = "CSS"
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
    JAVA_SCRIPT = "JAVA_SCRIPT"
    JPEG = "JPEG"
    JSON = "JSON"
    M4A = "M4A"
    M4V = "M4V"
    MARKDOWN = "MARKDOWN"
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
    PYTHON = "PYTHON"
    RTF = "RTF"
    THREE_GPP = "3GPP"
    TXT = "TXT"
    WAV = "WAV"
    WEBM = "WEBM"
    WEBP = "WEBP"
    WMV = "WMV"
    XML = "XML"
    XLS = "XLS"
    XLSX = "XLSX"


FILE_EXTENSIONS: Mapping[FileType, List[str]] = MappingProxyType(
    {
        FileType.AIFF: ["aif"],
        FileType.AAC: ["aac"],
        FileType.AVI: ["avi"],
        FileType.CSS: ["css"],
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
        FileType.JAVA_SCRIPT: ["js"],
        FileType.JPEG: ["jpeg", "jpg"],
        FileType.JSON: ["json"],
        FileType.M4A: ["m4a"],
        FileType.M4V: ["m4v"],
        FileType.MARKDOWN: ["md"],
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
        FileType.PYTHON: ["py"],
        FileType.RTF: ["rtf"],
        FileType.THREE_GPP: ["3gpp"],
        FileType.TXT: ["txt"],
        FileType.WAV: ["wav"],
        FileType.WEBM: ["webm"],
        FileType.WEBP: ["webp"],
        FileType.WMV: ["wmv"],
        FileType.XML: ["xml"],
        FileType.XLS: ["xls"],
        FileType.XLSX: ["xlsx"],
    }
)

FILE_MIME_TYPES: Mapping[FileType, str] = MappingProxyType(
    {
        FileType.AIFF: "audio/aiff",
        FileType.AAC: "audio/aac",
        FileType.AVI: "video/avi",
        FileType.CSS: "text/css",
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
        FileType.JAVA_SCRIPT: "text/javascript",
        FileType.JPEG: "image/jpeg",
        FileType.JSON: "application/json",
        FileType.M4A: "audio/x-m4a",
        FileType.M4V: "video/x-m4v",
        FileType.MARKDOWN: "text/md",
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
        FileType.PYTHON: "text/x-python",
        FileType.RTF: "text/rtf",
        FileType.THREE_GPP: "video/3gpp",
        FileType.TXT: "text/plain",
        FileType.WAV: "audio/wav",
        FileType.WEBM: "video/webm",
        FileType.WEBP: "image/webp",
        FileType.WMV: "video/wmv",
        FileType.XML: "text/xml",
        FileType.XLS: "application/vnd.ms-excel",
        FileType.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
)

"""
Util Functions
"""


def get_file_extension_from_mime_type(mime_type: str) -> str:
    lowercase_mime_type = mime_type.lower()
    for file_type, mime in FILE_MIME_TYPES.items():
        if mime.lower() == lowercase_mime_type:
            return FILE_EXTENSIONS[file_type][0]
    raise ValueError(f"Unknown extension for mime type: {mime_type}")


def get_file_type_from_extension(extension: str) -> FileType:
    lowercase_extension = extension.lower()
    for file_type, extensions in FILE_EXTENSIONS.items():
        if lowercase_extension in extensions:
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
    FileType.AVI,
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
    FileType.AIFF,
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
TEXT_FILE_TYPES = {
    FileType.CSS,
    FileType.CSV,
    FileType.HTML,
    FileType.JAVA_SCRIPT,
    FileType.MARKDOWN,
    FileType.PYTHON,
    FileType.RTF,
    FileType.TXT,
    FileType.XML
}


def is_text_file_type(file_type):
    return file_type in TEXT_FILE_TYPES


def get_mime_type_from_url(url: str) -> Optional[str]:
    """
    Get mime type for common URLs, handling query strings and path parameters.

    Example:
        url = https://example.com/image.jpg?width=100
        Returns: image/jpeg
    """
    # Parse the URL and get the path component
    parsed_url = urlparse(url.lower())
    path = parsed_url.path

    # Get extension without the dot
    extension_with_dot = os.path.splitext(path)[-1]  # Ex: ".png"
    if not extension_with_dot:
        raise ValueError(f"URL does not have an extension: {url}")

    extension = extension_with_dot[1:]  # Ex: "png"
    return get_file_mime_type_from_extension(extension)

