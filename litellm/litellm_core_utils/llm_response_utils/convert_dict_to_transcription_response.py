from typing import Dict, Optional, Union

from litellm.types.utils import TranscriptionResponse, VerboseTranscriptionResponse


def convert_dict_to_transcription_response(
    model_response_object: Optional[TranscriptionResponse],
    response_object: Optional[Dict],
    hidden_params: Optional[Dict],
    _response_headers: Optional[Dict],
) -> Union[TranscriptionResponse, VerboseTranscriptionResponse]:
    if response_object is None:
        raise Exception("Error in response object format")

    verbose_transcription_response = _return_verbose_transcription_response(
        response_object
    )
    if verbose_transcription_response is not None:
        model_response_object = verbose_transcription_response

    if model_response_object is None:
        model_response_object = TranscriptionResponse()

    if "text" in response_object:
        model_response_object.text = response_object["text"]

    if hidden_params is not None:
        model_response_object._hidden_params = hidden_params

    if _response_headers is not None:
        model_response_object._response_headers = _response_headers
    return model_response_object


def _return_verbose_transcription_response(
    response_object: Dict,
) -> Optional[VerboseTranscriptionResponse]:
    """
    If the response object contains any of the keys from a VerboseTranscriptionResponse, then return a VerboseTranscriptionResponse object
    """
    optional_keys = ["language", "task", "duration", "words", "segments"]
    transcription_response: Optional[VerboseTranscriptionResponse] = None
    # if any of these keys are in the response object, then create a verbose transcription response
    if any(key in response_object for key in optional_keys):
        transcription_response = VerboseTranscriptionResponse()
        for key in optional_keys:  #
            if key in response_object:
                setattr(transcription_response, key, response_object[key])
    return transcription_response
