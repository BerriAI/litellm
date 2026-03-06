"""
Utils used for litellm.transcription() and litellm.atranscription()
"""

import hashlib
import os
from dataclasses import dataclass
from typing import Optional

from litellm.types.files import get_file_mime_type_from_extension
from litellm.types.utils import FileTypes


@dataclass
class ProcessedAudioFile:
    """
    Processed audio file data.

    Attributes:
        file_content: The binary content of the audio file
        filename: The filename (extracted or generated)
        content_type: The MIME type of the audio file
    """

    file_content: bytes
    filename: str
    content_type: str


def process_audio_file(audio_file: FileTypes) -> ProcessedAudioFile:
    """
    Common utility function to process audio files for audio transcription APIs.

    Handles various input types:
    - File paths (str, os.PathLike)
    - Raw bytes/bytearray
    - Tuples (filename, content, optional content_type)
    - File-like objects with read() method

    Args:
        audio_file: The audio file input in various formats

    Returns:
        ProcessedAudioFile: Structured data with file content, filename, and content type

    Raises:
        ValueError: If audio_file type is unsupported or content cannot be extracted
    """
    file_content = None
    filename = None

    if isinstance(audio_file, (bytes, bytearray)):
        # Raw bytes
        filename = "audio.wav"
        file_content = bytes(audio_file)
    elif isinstance(audio_file, (str, os.PathLike)):
        # File path or PathLike
        file_path = str(audio_file)
        with open(file_path, "rb") as f:
            file_content = f.read()
        filename = file_path.split("/")[-1]
    elif isinstance(audio_file, tuple):
        # Tuple format: (filename, content, content_type) or (filename, content)
        if len(audio_file) >= 2:
            filename = audio_file[0] or "audio.wav"
            content = audio_file[1]
            if isinstance(content, (bytes, bytearray)):
                file_content = bytes(content)
            elif isinstance(content, (str, os.PathLike)):
                # File path or PathLike
                with open(str(content), "rb") as f:
                    file_content = f.read()
            elif hasattr(content, "read"):
                # File-like object
                file_content = content.read()
                if hasattr(content, "seek"):
                    content.seek(0)
            else:
                raise ValueError(f"Unsupported content type in tuple: {type(content)}")
        else:
            raise ValueError("Tuple must have at least 2 elements: (filename, content)")
    elif hasattr(audio_file, "read") and not isinstance(
        audio_file, (str, bytes, bytearray, tuple, os.PathLike)
    ):
        # File-like object (IO) - check this after all other types
        filename = getattr(audio_file, "name", "audio.wav")
        file_content = audio_file.read()  # type: ignore
        # Reset file pointer if possible
        if hasattr(audio_file, "seek"):
            audio_file.seek(0)  # type: ignore
    else:
        raise ValueError(f"Unsupported audio_file type: {type(audio_file)}")

    if file_content is None:
        raise ValueError("Could not extract file content from audio_file")

    # Determine content type using LiteLLM's file type utilities
    content_type = "audio/wav"  # Default fallback
    if filename:
        try:
            # Extract extension from filename
            extension = filename.split(".")[-1].lower() if "." in filename else "wav"
            content_type = get_file_mime_type_from_extension(extension)
        except ValueError:
            # If extension is not recognized, fallback to audio/wav
            content_type = "audio/wav"

    return ProcessedAudioFile(
        file_content=file_content, filename=filename, content_type=content_type
    )


def get_audio_file_name(file_obj: FileTypes) -> str:
    """
    Safely get the name of a file-like object or return its string representation.

    Args:
        file_obj (Any): A file-like object or any other object.

    Returns:
        str: The name of the file if available, otherwise a string representation of the object.
    """
    if hasattr(file_obj, "name"):
        return getattr(file_obj, "name")
    elif hasattr(file_obj, "__str__"):
        return str(file_obj)
    else:
        return repr(file_obj)


def get_audio_file_content_hash(file_obj: FileTypes) -> str:
    """
    Compute SHA-256 hash of audio file content for cache keys.
    Falls back to filename hash if content extraction fails.
    """
    file_content: Optional[bytes] = None
    fallback_filename: Optional[str] = None
    
    if isinstance(file_obj, tuple):
        if len(file_obj) < 2:
            fallback_filename = str(file_obj[0]) if len(file_obj) > 0 else None
        else:
            fallback_filename = str(file_obj[0]) if file_obj[0] is not None else None
            file_content_obj = file_obj[1]
    else:
        file_content_obj = file_obj
        fallback_filename = get_audio_file_name(file_obj)
    
    try:
        if isinstance(file_content_obj, (bytes, bytearray)):
            file_content = bytes(file_content_obj)
        elif isinstance(file_content_obj, (str, os.PathLike)):
            try:
                with open(str(file_content_obj), "rb") as f:
                    file_content = f.read()
                if fallback_filename is None:
                    fallback_filename = str(file_content_obj)
            except (OSError, IOError):
                fallback_filename = str(file_content_obj)
                file_content = None
        elif hasattr(file_content_obj, "read"):
            try:
                current_position = file_content_obj.tell() if hasattr(file_content_obj, "tell") else None
                if hasattr(file_content_obj, "seek"):
                    file_content_obj.seek(0)
                file_content = file_content_obj.read()  # type: ignore
                if current_position is not None and hasattr(file_content_obj, "seek"):
                    file_content_obj.seek(current_position)  # type: ignore
            except (OSError, IOError, AttributeError):
                file_content = None
        else:
            file_content = None
    except Exception:
        file_content = None
    
    if file_content is not None and isinstance(file_content, bytes):
        try:
            hash_object = hashlib.sha256(file_content)
            return hash_object.hexdigest()
        except Exception:
            pass
    
    if fallback_filename:
        hash_object = hashlib.sha256(fallback_filename.encode('utf-8'))
        return hash_object.hexdigest()
    
    file_obj_str = str(file_obj)
    hash_object = hashlib.sha256(file_obj_str.encode('utf-8'))
    return hash_object.hexdigest()


def get_audio_file_for_health_check() -> FileTypes:
    """
    Get an audio file for health check

    Returns the content of `audio_health_check.wav` in the same directory as this file
    """
    pwd = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(pwd, "audio_health_check.wav")
    return open(file_path, "rb")


def calculate_request_duration(file: FileTypes) -> Optional[float]:
    """
    Calculate audio duration from file content.

    Args:
        file: The audio file (can be file path, bytes, or file-like object)

    Returns:
        Duration in seconds, or None if extraction fails or soundfile is not available
    """
    try:
        import soundfile as sf
    except ImportError:
        # soundfile not available, cannot extract duration
        return None

    try:
        import io

        # Handle different file input types
        file_content: Optional[bytes] = None

        if isinstance(file, (bytes, bytearray)):
            # Raw bytes
            file_content = bytes(file)
        elif isinstance(file, (str, os.PathLike)):
            # File path
            with open(str(file), "rb") as f:
                file_content = f.read()
        elif isinstance(file, tuple):
            # Tuple format: (filename, content, optional content_type)
            if len(file) >= 2:
                content = file[1]
                if isinstance(content, bytes):
                    file_content = content
                elif hasattr(content, "read") and not isinstance(
                    content, (str, os.PathLike)
                ):
                    # File-like object in tuple
                    current_pos = getattr(content, "tell", lambda: None)()
                    # Seek to start to ensure we read the entire content
                    if hasattr(content, "seek"):
                        content.seek(0)
                    file_content = content.read()
                    if current_pos is not None and hasattr(content, "seek"):
                        content.seek(current_pos)
        elif hasattr(file, "read") and not isinstance(file, tuple):
            # File-like object (including BytesIO)
            current_position = file.tell() if hasattr(file, "tell") else None
            # Seek to start to ensure we read the entire content
            if hasattr(file, "seek"):
                file.seek(0)
            file_content = file.read()
            # Reset file position if possible
            if current_position is not None and hasattr(file, "seek"):
                file.seek(current_position)

        if file_content is None or not isinstance(file_content, bytes):
            return None

        # Extract duration using soundfile
        file_object = io.BytesIO(file_content)
        with sf.SoundFile(file_object) as audio:
            duration = len(audio) / audio.samplerate
            return duration

    except Exception:
        # Silently fail if duration extraction fails
        return None
