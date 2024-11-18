"""
Utils used for litellm.transcription() and litellm.atranscription()
"""

from litellm.types.utils import FileTypes


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
