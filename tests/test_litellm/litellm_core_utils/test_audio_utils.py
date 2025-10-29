"""
Test the audio utils functionality in litellm_core_utils/audio_utils/utils.py
"""

import io
import os
import tempfile
from unittest.mock import mock_open, patch

import pytest

from litellm.litellm_core_utils.audio_utils.utils import (
    ProcessedAudioFile,
    calculate_request_duration,
    get_audio_file_for_health_check,
    get_audio_file_name,
    process_audio_file,
)


class TestProcessAudioFile:
    """Test the process_audio_file function with various input types"""

    def test_process_bytes_input(self):
        """Test processing raw bytes input"""
        audio_data = b"fake audio data"
        result = process_audio_file(audio_data)

        assert isinstance(result, ProcessedAudioFile)
        assert result.file_content == audio_data
        assert result.filename == "audio.wav"
        assert result.content_type == "audio/wav"

    def test_process_bytearray_input(self):
        """Test processing bytearray input"""
        audio_data = bytearray(b"fake audio data")
        result = process_audio_file(audio_data)

        assert isinstance(result, ProcessedAudioFile)
        assert result.file_content == bytes(audio_data)
        assert result.filename == "audio.wav"
        assert result.content_type == "audio/wav"

    def test_process_file_path_input(self):
        """Test processing file path input"""
        test_content = b"test audio content"

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name

        try:
            result = process_audio_file(temp_file_path)

            assert isinstance(result, ProcessedAudioFile)
            assert result.file_content == test_content
            assert result.filename == os.path.basename(temp_file_path)
            assert result.content_type == "audio/mpeg"  # .mp3 should map to audio/mpeg
        finally:
            os.unlink(temp_file_path)

    def test_process_tuple_input_with_bytes(self):
        """Test processing tuple input with bytes content"""
        filename = "test.wav"
        audio_data = b"fake audio data"
        audio_tuple = (filename, audio_data)

        result = process_audio_file(audio_tuple)

        assert isinstance(result, ProcessedAudioFile)
        assert result.file_content == audio_data
        assert result.filename == filename
        assert result.content_type == "audio/wav"

    def test_process_tuple_input_with_file_path(self):
        """Test processing tuple input with file path content"""
        test_content = b"test audio content"

        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name

        try:
            filename = "custom_name.flac"
            audio_tuple = (filename, temp_file_path)

            result = process_audio_file(audio_tuple)

            assert isinstance(result, ProcessedAudioFile)
            assert result.file_content == test_content
            assert result.filename == filename
            assert result.content_type == "audio/flac"
        finally:
            os.unlink(temp_file_path)

    def test_process_file_like_object(self):
        """Test processing file-like object input"""
        test_content = b"test audio content"
        file_obj = io.BytesIO(test_content)
        file_obj.name = "test_audio.ogg"

        result = process_audio_file(file_obj)

        assert isinstance(result, ProcessedAudioFile)
        assert result.file_content == test_content
        assert result.filename == "test_audio.ogg"
        assert result.content_type == "audio/ogg"

        # Verify file pointer was reset
        assert file_obj.tell() == 0

    def test_process_file_like_object_without_name(self):
        """Test processing file-like object without name attribute"""
        test_content = b"test audio content"
        file_obj = io.BytesIO(test_content)

        result = process_audio_file(file_obj)

        assert isinstance(result, ProcessedAudioFile)
        assert result.file_content == test_content
        assert result.filename == "audio.wav"
        assert result.content_type == "audio/wav"

    def test_process_tuple_with_file_like_object(self):
        """Test processing tuple with file-like object as content"""
        test_content = b"test audio content"
        file_obj = io.BytesIO(test_content)

        filename = "custom.mp3"
        audio_tuple = (filename, file_obj)

        result = process_audio_file(audio_tuple)

        assert isinstance(result, ProcessedAudioFile)
        assert result.file_content == test_content
        assert result.filename == filename
        assert result.content_type == "audio/mpeg"

        # Verify file pointer was reset
        assert file_obj.tell() == 0

    def test_mime_type_detection_various_extensions(self):
        """Test MIME type detection for various audio file extensions"""
        test_cases = [
            ("test.wav", "audio/wav"),
            ("test.mp3", "audio/mpeg"),
            ("test.flac", "audio/flac"),
            ("test.ogg", "audio/ogg"),
            ("test.aac", "audio/aac"),
            ("test.m4a", "audio/x-m4a"),
        ]

        for filename, expected_mime_type in test_cases:
            audio_tuple = (filename, b"fake content")
            result = process_audio_file(audio_tuple)
            assert result.content_type == expected_mime_type, f"Failed for {filename}"

    def test_mime_type_fallback_for_unknown_extension(self):
        """Test MIME type fallback for unknown file extensions"""
        audio_tuple = ("test.unknown", b"fake content")
        result = process_audio_file(audio_tuple)

        assert result.content_type == "audio/wav"  # Should fallback to default

    def test_process_pathlike_object(self):
        """Test processing os.PathLike object"""
        test_content = b"test audio content"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name

        try:
            # Convert to pathlib.Path
            from pathlib import Path

            path_obj = Path(temp_file_path)

            result = process_audio_file(path_obj)

            assert isinstance(result, ProcessedAudioFile)
            assert result.file_content == test_content
            assert result.filename == os.path.basename(temp_file_path)
            assert result.content_type == "audio/wav"
        finally:
            os.unlink(temp_file_path)

    def test_invalid_input_type(self):
        """Test that invalid input types raise ValueError"""
        with pytest.raises(ValueError, match="Unsupported audio_file type"):
            process_audio_file(123)  # Invalid type

    def test_invalid_tuple_length(self):
        """Test that tuple with less than 2 elements raises ValueError"""
        with pytest.raises(ValueError, match="Tuple must have at least 2 elements"):
            process_audio_file(("only_one_element",))

    def test_invalid_tuple_content_type(self):
        """Test that tuple with unsupported content type raises ValueError"""
        with pytest.raises(ValueError, match="Unsupported content type in tuple"):
            process_audio_file(("filename", 123))  # Invalid content type

    def test_tuple_with_none_filename(self):
        """Test tuple with None filename gets default name"""
        audio_tuple = (None, b"fake content")
        result = process_audio_file(audio_tuple)

        assert result.filename == "audio.wav"
        assert result.content_type == "audio/wav"


class TestCalculateRequestDuration:
    """Test the calculate_request_duration function"""

    @pytest.mark.skipif(
        os.environ.get("SKIP_AUDIO_TESTS") == "true",
        reason="Skipping audio tests - soundfile may not be available",
    )
    def test_bytesio_at_end_position(self):
        """
        Test that calculate_request_duration handles BytesIO with file pointer at end.
        This reproduces and verifies the fix for the OGG file bug where BytesIO
        position was at the end after a previous read(), causing "Format not recognised" error.
        """
        # Create a simple WAV file in memory (44 bytes header + some data)
        # This is a minimal valid WAV file
        wav_header = (
            b"RIFF"
            + (36 + 8).to_bytes(4, "little")  # ChunkSize
            + b"WAVE"
            + b"fmt "
            + (16).to_bytes(4, "little")  # Subchunk1Size
            + (1).to_bytes(2, "little")  # AudioFormat (PCM)
            + (1).to_bytes(2, "little")  # NumChannels
            + (16000).to_bytes(4, "little")  # SampleRate
            + (32000).to_bytes(4, "little")  # ByteRate
            + (2).to_bytes(2, "little")  # BlockAlign
            + (16).to_bytes(2, "little")  # BitsPerSample
            + b"data"
            + (8).to_bytes(4, "little")  # Subchunk2Size
            + b"\x00\x00\x00\x00\x00\x00\x00\x00"  # Sample data
        )

        # Create BytesIO object
        file_obj = io.BytesIO(wav_header)
        file_obj.name = "test_audio.wav"

        # Simulate the bug: something reads from the file first, moving position to end
        _ = file_obj.read()
        assert file_obj.tell() == len(wav_header), "File position should be at end"

        # Call calculate_request_duration - this would fail before the fix
        duration = calculate_request_duration(file_obj)

        # Verify it succeeded (returns a duration, not None)
        assert (
            duration is not None
        ), "Duration should be calculated even when BytesIO is at end"
        assert isinstance(duration, float), "Duration should be a float"
        assert duration > 0, "Duration should be positive"

        # Verify the file position was restored
        assert file_obj.tell() == len(
            wav_header
        ), "File position should be restored to original position"
