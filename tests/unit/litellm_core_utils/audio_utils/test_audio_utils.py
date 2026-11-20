import datetime
import gc
import io
import weakref

import pytest

from litellm.litellm_core_utils.audio_utils.utils import get_audio_file_name
from litellm.litellm_core_utils.litellm_logging import Logging


def test_get_audio_file_name_tuple_does_not_stringify_bytes():
    """Verify that get_audio_file_name extracts filename from tuple without stringifying binary contents (#50939)."""
    audio_bytes = b"\x00\x01\x02\x03" * 1024
    # (filename, bytes)
    assert get_audio_file_name(("audio.wav", audio_bytes)) == "audio.wav"
    # (filename, bytes, content_type)
    assert get_audio_file_name(("audio.wav", audio_bytes, "audio/wav")) == "audio.wav"
    # (filename, bytes, content_type, headers)
    assert get_audio_file_name(("audio.wav", audio_bytes, "audio/wav", {})) == "audio.wav"

    # Tuple with None filename but inner named buffer
    bio = io.BytesIO(audio_bytes)
    bio.name = "inner.wav"
    assert get_audio_file_name((None, bio)) == "inner.wav"


def test_logging_update_environment_variables_breaks_self_reference_cycle():
    """Verify that update_environment_variables strips self-referential litellm_logging_obj to prevent memory leaks (#50939)."""
    upload = io.BytesIO(b"synthetic audio")
    logging = Logging(
        model="openai/whisper-1",
        messages=[],
        stream=False,
        call_type="transcription",
        start_time=datetime.datetime(2000, 1, 1),
        litellm_call_id="synthetic-call",
        function_id="synthetic-function",
    )
    logging.update_environment_variables(
        litellm_params={"litellm_logging_obj": logging}, optional_params={}
    )
    logging.pre_call(
        input="audio.wav",
        api_key="synthetic",
        additional_args={"complete_input_dict": {"file": upload}},
    )

    upload_ref = weakref.ref(upload)
    logging_ref = weakref.ref(logging)

    del upload
    del logging

    enabled = gc.isenabled()
    gc.disable()
    try:
        # Without cyclic GC running, the upload and logging objects should be freed immediately
        assert upload_ref() is None, "Upload buffer was retained without GC!"
        assert logging_ref() is None, "Logging instance was retained without GC!"
    finally:
        if enabled:
            gc.enable()
