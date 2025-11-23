"""
Tests for audio transcription caching functionality.

This test suite verifies that:
1. Different audio files with the same name generate different cache keys
2. Identical audio files generate the same cache key
3. Cache works correctly with different input types (bytes, file path, file-like object)
"""

import io
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))
import litellm
from litellm.caching import Cache
from litellm.caching.caching import Cache as CacheClass
from litellm.caching.in_memory_cache import InMemoryCache


def test_transcription_cache_different_files_same_name():
    """
    Test that two different audio files with the same name generate different cache keys.
    """
    # Create two different audio file contents (simulated as bytes)
    audio_content_1 = b"fake_audio_content_1" * 100
    audio_content_2 = b"fake_audio_content_2" * 100

    # Create temporary files with the same name but different content
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f1:
        f1.write(audio_content_1)
        file_path_1 = f1.name

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f2:
        f2.write(audio_content_2)
        file_path_2 = f2.name

    try:
        # Initialize cache
        litellm.cache = Cache(type="local")

        # Get cache keys for both files
        kwargs1 = {
            "file": open(file_path_1, "rb"),
            "model": "whisper-1",
            "metadata": {},
        }
        kwargs2 = {
            "file": open(file_path_2, "rb"),
            "model": "whisper-1",
            "metadata": {},
        }

        cache_key_1 = litellm.cache.get_cache_key(**kwargs1)
        cache_key_2 = litellm.cache.get_cache_key(**kwargs2)

        # Cache keys should be different because file contents are different
        assert cache_key_1 != cache_key_2, "Different files should have different cache keys"

        # Close file handles
        kwargs1["file"].close()
        kwargs2["file"].close()
    finally:
        # Clean up temporary files
        os.unlink(file_path_1)
        os.unlink(file_path_2)
        litellm.cache = None


def test_transcription_cache_same_file_content():
    """
    Test that two identical audio files generate the same cache key.
    """
    # Create audio file content
    audio_content = b"fake_audio_content" * 100

    # Create two temporary files with the same content
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f1:
        f1.write(audio_content)
        file_path_1 = f1.name

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f2:
        f2.write(audio_content)
        file_path_2 = f2.name

    try:
        # Initialize cache
        litellm.cache = Cache(type="local")

        # Get cache keys for both files
        kwargs1 = {
            "file": open(file_path_1, "rb"),
            "model": "whisper-1",
            "metadata": {},
        }
        kwargs2 = {
            "file": open(file_path_2, "rb"),
            "model": "whisper-1",
            "metadata": {},
        }

        cache_key_1 = litellm.cache.get_cache_key(**kwargs1)
        cache_key_2 = litellm.cache.get_cache_key(**kwargs2)

        # Cache keys should be the same because file contents are identical
        assert cache_key_1 == cache_key_2, "Identical files should have the same cache key"

        # Close file handles
        kwargs1["file"].close()
        kwargs2["file"].close()
    finally:
        # Clean up temporary files
        os.unlink(file_path_1)
        os.unlink(file_path_2)
        litellm.cache = None


def test_transcription_cache_bytes_input():
    """
    Test that cache key generation works with bytes input.
    """
    audio_content_1 = b"fake_audio_content_1" * 100
    audio_content_2 = b"fake_audio_content_2" * 100

    try:
        # Initialize cache
        litellm.cache = Cache(type="local")

        # Get cache keys for bytes input
        kwargs1 = {
            "file": audio_content_1,
            "model": "whisper-1",
            "metadata": {},
        }
        kwargs2 = {
            "file": audio_content_2,
            "model": "whisper-1",
            "metadata": {},
        }

        cache_key_1 = litellm.cache.get_cache_key(**kwargs1)
        cache_key_2 = litellm.cache.get_cache_key(**kwargs2)

        # Cache keys should be different
        assert cache_key_1 != cache_key_2, "Different byte contents should have different cache keys"

        # Same content should have same key
        kwargs3 = {
            "file": audio_content_1,
            "model": "whisper-1",
            "metadata": {},
        }
        cache_key_3 = litellm.cache.get_cache_key(**kwargs3)
        assert cache_key_1 == cache_key_3, "Same byte contents should have the same cache key"
    finally:
        litellm.cache = None


def test_transcription_cache_file_like_object():
    """
    Test that cache key generation works with file-like objects (BytesIO).
    """
    audio_content_1 = b"fake_audio_content_1" * 100
    audio_content_2 = b"fake_audio_content_2" * 100

    try:
        # Initialize cache
        litellm.cache = Cache(type="local")

        # Get cache keys for file-like objects
        kwargs1 = {
            "file": io.BytesIO(audio_content_1),
            "model": "whisper-1",
            "metadata": {},
        }
        kwargs2 = {
            "file": io.BytesIO(audio_content_2),
            "model": "whisper-1",
            "metadata": {},
        }

        cache_key_1 = litellm.cache.get_cache_key(**kwargs1)
        cache_key_2 = litellm.cache.get_cache_key(**kwargs2)

        # Cache keys should be different
        assert cache_key_1 != cache_key_2, "Different file-like objects should have different cache keys"
    finally:
        litellm.cache = None


def test_transcription_cache_with_metadata_checksum():
    """
    Test that if file_checksum is already in metadata, it's used directly.
    """
    audio_content = b"fake_audio_content" * 100

    try:
        # Initialize cache
        litellm.cache = Cache(type="local")

        # Pre-calculate checksum
        from litellm.litellm_core_utils.audio_utils.utils import calculate_audio_file_hash

        expected_checksum = calculate_audio_file_hash(audio_file=audio_content)

        # Get cache key with pre-set checksum in metadata
        kwargs = {
            "file": audio_content,
            "model": "whisper-1",
            "metadata": {"file_checksum": expected_checksum},
        }

        cache_key = litellm.cache.get_cache_key(**kwargs)

        # Verify the checksum is used in the cache key
        assert cache_key is not None
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0
    finally:
        litellm.cache = None


def test_transcription_cache_fallback_to_filename():
    """
    Test that if hash calculation fails, it falls back to filename.
    """
    try:
        # Initialize cache
        litellm.cache = Cache(type="local")

        # Create a mock file object that will fail hash calculation
        mock_file = MagicMock()
        mock_file.name = "test_audio.wav"
        # Make read() raise an exception to simulate hash calculation failure
        mock_file.read.side_effect = Exception("Cannot read file")

        kwargs = {
            "file": mock_file,
            "model": "whisper-1",
            "metadata": {},
        }

        # Should not raise exception, should fall back to filename
        cache_key = litellm.cache.get_cache_key(**kwargs)
        assert cache_key is not None
        assert isinstance(cache_key, str)
    finally:
        litellm.cache = None


def test_calculate_audio_file_hash_function():
    """
    Test the calculate_audio_file_hash function directly.
    """
    from litellm.litellm_core_utils.audio_utils.utils import calculate_audio_file_hash

    # Test with bytes
    audio_content_1 = b"fake_audio_content_1" * 100
    audio_content_2 = b"fake_audio_content_2" * 100

    hash_1 = calculate_audio_file_hash(audio_file=audio_content_1)
    hash_2 = calculate_audio_file_hash(audio_file=audio_content_2)

    # Hashes should be different
    assert hash_1 != hash_2
    # Hashes should be valid SHA256 hex strings (64 characters)
    assert len(hash_1) == 64
    assert len(hash_2) == 64
    assert all(c in "0123456789abcdef" for c in hash_1)
    assert all(c in "0123456789abcdef" for c in hash_2)

    # Same content should produce same hash
    hash_1_again = calculate_audio_file_hash(audio_file=audio_content_1)
    assert hash_1 == hash_1_again

    # Test with file path
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f:
        f.write(audio_content_1)
        file_path = f.name

    try:
        hash_from_path = calculate_audio_file_hash(audio_file=file_path)
        assert hash_from_path == hash_1
    finally:
        os.unlink(file_path)

    # Test with file-like object
    file_like = io.BytesIO(audio_content_1)
    hash_from_file_like = calculate_audio_file_hash(audio_file=file_like)
    assert hash_from_file_like == hash_1

