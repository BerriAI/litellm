import pytest
from litellm.litellm_core_utils.prompt_templates.factory import _parse_content_type

def test_parse_content_type_basic():
    """Test basic content type parsing"""
    assert _parse_content_type("image/jpeg") == "image/jpeg"
    assert _parse_content_type("image/png") == "image/png"
    assert _parse_content_type("application/pdf") == "application/pdf"

def test_parse_content_type_with_parameters():
    """Test content type parsing with additional parameters"""
    assert _parse_content_type("image/jpeg; charset=utf-8") == "image/jpeg"
    assert _parse_content_type("image/png; boundary=something") == "image/png"
    assert _parse_content_type("application/pdf; version=1.7") == "application/pdf"

def test_parse_content_type_with_multiple_parameters():
    """Test content type parsing with multiple parameters"""
    assert _parse_content_type("image/jpeg; charset=utf-8; boundary=something") == "image/jpeg"
    assert _parse_content_type("application/pdf; version=1.7; encoding=binary") == "application/pdf"

def test_parse_content_type_invalid():
    """Test parsing invalid content types"""
    assert _parse_content_type("") == ""
    
    # The current implementation raises TypeError for None
    with pytest.raises(TypeError):
        _parse_content_type(None)

def test_parse_content_type_unusual():
    """Test parsing unusual but valid content types"""
    assert _parse_content_type("x-application/custom") == "x-application/custom"
    assert _parse_content_type("text/plain") == "text/plain"
    assert _parse_content_type("application/octet-stream") == "application/octet-stream" 