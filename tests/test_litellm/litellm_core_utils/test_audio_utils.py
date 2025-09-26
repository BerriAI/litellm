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
