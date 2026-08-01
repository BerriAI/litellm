"""
Regression test for https://github.com/BerriAI/litellm/issues/34031
Unbounded upstream error bodies capped at _MAX_ERROR_BODY_BYTES
"""
from litellm.llms.custom_httpx.http_handler import _safe_get_response_text, _MAX_ERROR_BODY_BYTES
from unittest.mock import MagicMock


def test_small_body_passes_through():
    mock_response = MagicMock()
    mock_response.text = "small error"
    result = _safe_get_response_text(mock_response)
    assert result == "small error"
    print("✓ Small body passes through unchanged")


def test_large_body_truncated():
    mock_response = MagicMock()
    mock_response.text = "x" * (200 * 1024)  # 200 KB
    result = _safe_get_response_text(mock_response)
    assert len(result.encode("utf-8")) <= _MAX_ERROR_BODY_BYTES + 200  # allow for truncation message
    assert "truncated" in result
    print(f"✓ Large body truncated to ~{_MAX_ERROR_BODY_BYTES} bytes")


def test_exact_boundary():
    mock_response = MagicMock()
    mock_response.text = "a" * _MAX_ERROR_BODY_BYTES
    result = _safe_get_response_text(mock_response)
    assert result == "a" * _MAX_ERROR_BODY_BYTES
    assert "truncated" not in result
    print("✓ Exact boundary passes through unchanged")


if __name__ == "__main__":
    test_small_body_passes_through()
    test_large_body_truncated()
    test_exact_boundary()
    print("\nAll tests passed ✓")
