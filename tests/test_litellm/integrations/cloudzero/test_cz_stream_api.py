import os
import sys
import zoneinfo
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import httpx
import polars as pl
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.cloudzero.cz_stream_api import CloudZeroStreamer


class TestCloudZeroStreamer:
    """Test suite for CloudZeroStreamer class."""

    def test_init_with_defaults(self):
        """Test CloudZeroStreamer initialization with default parameters."""
        streamer = CloudZeroStreamer(
            api_key="test-key",
            connection_id="test-connection"
        )
        
        assert streamer.api_key == "test-key"
        assert streamer.connection_id == "test-connection"
        assert streamer.base_url == "https://api.cloudzero.com"
        assert streamer.user_timezone == timezone.utc

    def test_init_with_valid_timezone(self):
        """Test CloudZeroStreamer initialization with valid timezone."""
        streamer = CloudZeroStreamer(
            api_key="test-key",
            connection_id="test-connection",
            user_timezone="America/New_York"
        )
        
        assert streamer.user_timezone == zoneinfo.ZoneInfo("America/New_York")
    
    def test_send_batched_with_valid_data(self):
        """Test send_batched method with valid data."""
        streamer = CloudZeroStreamer("test-key", "test-connection")
        with patch.object(streamer, '_group_by_date') as mock_group, \
             patch.object(streamer, '_send_daily_batch') as mock_send:
            
            mock_group.return_value = {
                '2025-01-19': pl.DataFrame({'test': ['data1']}),
                '2025-01-20': pl.DataFrame({'test': ['data2']})
            }
            
            data = pl.DataFrame({'test': ['data']})
            streamer.send_batched(data, "replace_hourly")
            
            assert mock_send.call_count == 2

    def test_group_by_date_valid_data(self):
        """Test _group_by_date method with valid data."""
        streamer = CloudZeroStreamer("test-key", "test-connection")
        with patch.object(streamer, '_parse_and_convert_timestamp') as mock_parse:
            mock_parse.return_value = datetime(2025, 1, 19, 10, 30, 0, tzinfo=timezone.utc)
            
            data = pl.DataFrame({
                'time/usage_start': ['2025-01-19T10:30:00Z'],
                'cost': [10.0]
            })
            
            result = streamer._group_by_date(data)
            
            assert '2025-01-19' in result
            assert len(result['2025-01-19']) == 1


    def test_parse_and_convert_timestamp_utc(self):
        """Test _parse_and_convert_timestamp method with UTC timestamp."""
        streamer = CloudZeroStreamer("test-key", "test-connection")
        
        result = streamer._parse_and_convert_timestamp('2025-01-19T10:30:00Z')
        
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 19
        assert result.hour == 10
        assert result.minute == 30
        assert result.tzinfo == timezone.utc

    def test_parse_and_convert_timestamp_with_offset(self):
        """Test _parse_and_convert_timestamp method with timezone offset."""
        streamer = CloudZeroStreamer("test-key", "test-connection")
        
        result = streamer._parse_and_convert_timestamp('2025-01-19T10:30:00+05:00')
        
        assert result.tzinfo == timezone.utc
        assert result.hour == 5  # Converted to UTC

    def test_parse_and_convert_timestamp_no_timezone(self):
        """Test _parse_and_convert_timestamp method without timezone info."""
        streamer = CloudZeroStreamer("test-key", "test-connection", user_timezone="America/New_York")
        
        result = streamer._parse_and_convert_timestamp('2025-01-19T10:30:00')
        
        assert result.tzinfo == timezone.utc

    def test_parse_and_convert_timestamp_invalid(self):
        """Test _parse_and_convert_timestamp method with invalid timestamp."""
        streamer = CloudZeroStreamer("test-key", "test-connection")
        
        with pytest.raises(ValueError):
            streamer._parse_and_convert_timestamp('invalid-timestamp')

    def test_prepare_batch_payload(self):
        """Test _prepare_batch_payload method."""
        streamer = CloudZeroStreamer("test-key", "test-connection")
        with patch.object(streamer, '_convert_cbf_to_api_format') as mock_convert:
            mock_convert.return_value = {'test': 'record'}
            
            batch_data = pl.DataFrame({'cost': [10.0]})
            result = streamer._prepare_batch_payload('2025-01-19', batch_data, 'replace_hourly')
            
            assert result['month'] == '2025-01'
            assert result['operation'] == 'replace_hourly'
            assert len(result['data']) == 1



    def test_convert_cbf_to_api_format_valid_data(self):
        """Test _convert_cbf_to_api_format method with valid data."""
        streamer = CloudZeroStreamer("test-key", "test-connection")
        with patch.object(streamer, '_ensure_utc_timestamp') as mock_ensure:
            mock_ensure.return_value = '2025-01-19T10:30:00Z'
            
            row = {
                'time/usage_start': '2025-01-19T10:30:00Z',
                'cost/cost': 10.5,
                'tokens': 100,
                'text_field': 'test'
            }
            
            result = streamer._convert_cbf_to_api_format(row)
            
            assert result['cost/cost'] == '10.5'
            assert result['tokens'] == '100'
            assert result['text_field'] == 'test'

    def test_convert_cbf_to_api_format_float_precision(self):
        """Test _convert_cbf_to_api_format method handles float precision correctly."""
        streamer = CloudZeroStreamer("test-key", "test-connection")
        
        row = {
            'cost': 10.123456789012345,
            'large_float': 1234567890.0
        }
        
        result = streamer._convert_cbf_to_api_format(row)
        
        # Should avoid scientific notation
        assert 'e' not in result['cost'].lower()
        assert 'e' not in result['large_float'].lower()

   