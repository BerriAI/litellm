import os
import sys
from unittest.mock import MagicMock, patch
from datetime import datetime

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.duration_parser import duration_in_seconds, _extract_from_regex


@pytest.mark.parametrize("mock_now, duration, expected_seconds", [
    (datetime(2025, 3, 24, 0, 0, 0), "30s", 30),
    (datetime(2025, 3, 24, 0, 0, 0), "1m", 60),  
    (datetime(2025, 3, 24, 0, 0, 0), "1h", 3600),  
    (datetime(2025, 3, 24, 0, 0, 0), "1d", 86400),  
    (datetime(2025, 3, 24, 0, 0, 0), "1w", 604800),
    (datetime(2025, 3, 1, 0, 0, 0), "1mo", 2678400),  
    (datetime(2025, 3, 24, 0, 0, 30), "2m", 60+30),
    (datetime(2025, 3, 24, 0, 30, 0), "2h", 3600+1800),
    (datetime(2025, 3, 24, 12, 0, 0), "2d", 86400+43200),
    (datetime(2025, 3, 25, 12, 0, 0), "2w", 1209600-86400-43200),  # from March 25 12:00:00 UTC to April 8, 2025 00:00:00 UTC
    (datetime(2025, 3, 24, 0, 0, 0), "2mo", 604800+86400+2592000),  # from March 24 to May 1, 2025 00:00:00 UTC
    (datetime(2025, 12, 31, 23, 0, 0), "1h", 3600),
    (datetime(2025, 12, 31, 0, 0, 0), "1d", 86400),
    (datetime(2025, 12, 29, 0, 0, 0), "1w", 604800),
    (datetime(2025, 12, 1, 0, 0, 0), "1mo", 2678400),

])
def test_duration_in_seconds(mock_now, duration, expected_seconds):
    # mock datetime.now() to return a fixed time
    with patch("litellm.litellm_core_utils.duration_parser.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_now
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)  # Allow other datetime functionality
        result = duration_in_seconds(duration)
        assert result == expected_seconds


def test_extract_from_regex():
    assert _extract_from_regex("30s") == (30, "s")
    assert _extract_from_regex("3m") == (3, "m")
    assert _extract_from_regex("4h") == (4, "h")
    assert _extract_from_regex("12d") == (12, "d")
    assert _extract_from_regex("10w") == (10, "w")
    assert _extract_from_regex("2mo") == (2, "mo")
    
