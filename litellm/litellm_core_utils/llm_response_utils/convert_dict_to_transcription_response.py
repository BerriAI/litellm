from typing import Dict, Optional

from litellm.types.utils import TranscriptionResponse


def convert_dict_to_transcription_response(
    model_response_object: Optional[TranscriptionResponse],
    response_object: Optional[Dict],
    hidden_params: Optional[Dict],
    _response_headers: Optional[Dict],
):
    if response_object is None:
        raise Exception("Error in response object format")

    if model_response_object is None:
        model_response_object = TranscriptionResponse()

    if "text" in response_object:
        model_response_object.text = response_object["text"]

    optional_keys = ["language", "task", "duration", "words", "segments"]
    for key in optional_keys:  # not guaranteed to be in response
        if key in response_object:
            setattr(model_response_object, key, response_object[key])

    if hidden_params is not None:
        model_response_object._hidden_params = hidden_params

    if _response_headers is not None:
        model_response_object._response_headers = _response_headers
