from litellm.types.files import (
    FILE_EXTENSIONS,
    FILE_MIME_TYPES,
    FileType,
    get_file_extension_from_mime_type,
    get_file_type_from_extension,
    get_file_extension_for_file_type,
    get_file_mime_type_for_file_type,
    get_file_mime_type_from_extension,
)
import pytest


class TestFileConsts:
    def test_all_file_types_have_extensions(self):
        for file_type in FileType:
            assert file_type in FILE_EXTENSIONS.keys()

    def test_all_file_types_have_mime_types(self):
        for file_type in FileType:
            assert file_type in FILE_MIME_TYPES.keys()

    def test_get_file_extension_from_mime_type(self):
        assert get_file_extension_from_mime_type("audio/aac") == "aac"
        assert get_file_extension_from_mime_type("application/pdf") == "pdf"
        with pytest.raises(ValueError):
            get_file_extension_from_mime_type("application/unknown")

    def test_get_file_type_from_extension(self):
        assert get_file_type_from_extension("aac") == FileType.AAC
        assert get_file_type_from_extension("pdf") == FileType.PDF
        with pytest.raises(ValueError):
            get_file_type_from_extension("unknown")

    def test_get_file_extension_for_file_type(self):
        assert get_file_extension_for_file_type(FileType.AAC) == "aac"
        assert get_file_extension_for_file_type(FileType.PDF) == "pdf"

    def test_get_file_mime_type_for_file_type(self):
        assert get_file_mime_type_for_file_type(FileType.AAC) == "audio/aac"
        assert get_file_mime_type_for_file_type(FileType.PDF) == "application/pdf"

    def test_get_file_mime_type_from_extension(self):
        assert get_file_mime_type_from_extension("aac") == "audio/aac"
        assert get_file_mime_type_from_extension("pdf") == "application/pdf"

    def test_uppercase_extensions(self):
        # Test that uppercase extensions return the correct file type
        assert get_file_type_from_extension("AAC") == FileType.AAC
        assert get_file_type_from_extension("PDF") == FileType.PDF

        # Test that uppercase extensions return the correct MIME type
        assert get_file_mime_type_from_extension("AAC") == "audio/aac"
        assert get_file_mime_type_from_extension("PDF") == "application/pdf"
