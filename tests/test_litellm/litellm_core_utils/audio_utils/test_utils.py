from litellm.litellm_core_utils.audio_utils.utils import process_audio_file


def test_process_audio_file_labels_webm_as_audio():
    # Regression for https://github.com/BerriAI/litellm/issues/38963: webm is an
    # audio/video container and process_audio_file is audio-only, so a .webm
    # upload must be audio/webm. video/webm makes Vertex Gemini transcription
    # return an empty transcript
    processed = process_audio_file(("speech.webm", b"\x1aE\xdf\xa3"))
    assert processed.content_type == "audio/webm"


def test_process_audio_file_keeps_known_audio_extension():
    processed = process_audio_file(("speech.wav", b"RIFF"))
    assert processed.content_type == "audio/wav"


def test_process_audio_file_unknown_extension_falls_back_to_wav():
    processed = process_audio_file(("recording.unknownext", b"x"))
    assert processed.content_type == "audio/wav"
