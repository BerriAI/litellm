"""
Utils used for litellm.transcription() and litellm.atranscription()
"""

import os
from dataclasses import dataclass

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
        filename = 'audio.wav'
        file_content = bytes(audio_file)
    elif isinstance(audio_file, (str, os.PathLike)):
        # File path or PathLike
        file_path = str(audio_file)
        with open(file_path, 'rb') as f:
            file_content = f.read()
        filename = file_path.split('/')[-1]
    elif isinstance(audio_file, tuple):
        # Tuple format: (filename, content, content_type) or (filename, content)
        if len(audio_file) >= 2:
            filename = audio_file[0] or 'audio.wav'
            content = audio_file[1]
            if isinstance(content, (bytes, bytearray)):
                file_content = bytes(content)
            elif isinstance(content, (str, os.PathLike)):
                # File path or PathLike
                with open(str(content), 'rb') as f:
                    file_content = f.read()
            elif hasattr(content, 'read'):
                # File-like object
                file_content = content.read()
                if hasattr(content, 'seek'):
                    content.seek(0)
            else:
                raise ValueError(f"Unsupported content type in tuple: {type(content)}")
        else:
            raise ValueError("Tuple must have at least 2 elements: (filename, content)")
    elif hasattr(audio_file, 'read') and not isinstance(audio_file, (str, bytes, bytearray, tuple, os.PathLike)):
        # File-like object (IO) - check this after all other types
        filename = getattr(audio_file, 'name', 'audio.wav')
        file_content = audio_file.read()  # type: ignore
        # Reset file pointer if possible
        if hasattr(audio_file, 'seek'):
            audio_file.seek(0)  # type: ignore
    else:
        raise ValueError(f"Unsupported audio_file type: {type(audio_file)}")

    if file_content is None:
        raise ValueError("Could not extract file content from audio_file")

    # Determine content type using LiteLLM's file type utilities
    content_type = 'audio/wav'  # Default fallback
    if filename:
        try:
            # Extract extension from filename
            extension = filename.split('.')[-1].lower() if '.' in filename else 'wav'
            content_type = get_file_mime_type_from_extension(extension)
        except ValueError:
            # If extension is not recognized, fallback to audio/wav
            content_type = 'audio/wav'
    
    return ProcessedAudioFile(
        file_content=file_content,
        filename=filename,
        content_type=content_type
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


def get_audio_file_for_health_check() -> FileTypes:
    """
    Get an audio file for health check

    Returns the content of `audio_health_check.wav` in the same directory as this file
    """
    pwd = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(pwd, "audio_health_check.wav")
    return open(file_path, "rb")
