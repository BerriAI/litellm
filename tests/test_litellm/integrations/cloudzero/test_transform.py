import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.cloudzero.transform import CBFTransformer
from litellm.types.integrations.cloudzero import CBFRecord


class TestCBFTransformer:
    """Test suite for CBFTransformer class."""

    def test_init(self):
        """Test CBFTransformer initialization."""
        transformer = CBFTransformer()
        assert hasattr(transformer, 'czrn_generator')
        assert transformer.czrn_generator is not None

    def test_transform_empty_dataframe(self):
        """Test transform method with empty DataFrame."""
        transformer = CBFTransformer()
        empty_df = pl.DataFrame()
        
        result = transformer.transform(empty_df)
        
        assert result.is_empty()
        assert isinstance(result, pl.DataFrame)

    def test_transform_with_zero_successful_requests(self):
        """Test transform method filters out records with zero successful_requests."""
        transformer = CBFTransformer()
        data = pl.DataFrame({
            'date': ['2025-01-19'],
            'successful_requests': [0],
            'spend': [10.0],
            'entity_id': ['test_entity'],
            'model': ['gpt-4']
        })
        
        result = transformer.transform(data)
        
        assert result.is_empty()

    def test_transform_with_valid_data(self):
        """Test transform method with valid data."""
        transformer = CBFTransformer()
        with patch.object(transformer, '_create_cbf_record') as mock_create:
            mock_create.return_value = CBFRecord({'test': 'data'})
            
            data = pl.DataFrame({
                'date': ['2025-01-19'],
                'successful_requests': [5],
                'spend': [10.0],
                'entity_id': ['test_entity'],
                'model': ['gpt-4']
            })
            
            result = transformer.transform(data)
            
            assert len(result) == 1
            mock_create.assert_called_once()

    def test_transform_handles_czrn_generation_failures(self):
        """Test transform method handles CZRN generation failures gracefully."""
        transformer = CBFTransformer()
        with patch.object(transformer, '_create_cbf_record') as mock_create:
            mock_create.side_effect = Exception("CZRN generation failed")
            
            data = pl.DataFrame({
                'date': ['2025-01-19'],
                'successful_requests': [5],
                'spend': [10.0],
                'entity_id': ['test_entity'],
                'model': ['gpt-4']
            })
            
            result = transformer.transform(data)
            
            assert result.is_empty()

    def test_create_cbf_record(self):
        """Test _create_cbf_record method with valid row data."""
        transformer = CBFTransformer()
        with patch.object(transformer.czrn_generator, 'create_from_litellm_data') as mock_czrn, \
             patch.object(transformer.czrn_generator, 'extract_components') as mock_extract:
            
            mock_czrn.return_value = 'test-czrn'
            mock_extract.return_value = ('service', 'provider', 'region', 'account', 'resource', 'local_id')
            
            row = {
                'date': '2025-01-19',
                'spend': 10.5,
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'entity_id': 'test_entity',
                'model': 'gpt-4',
                'entity_type': 'user',
                'model_group': 'openai',
                'custom_llm_provider': 'openai',
                'api_key': 'sk-test123',
                'api_requests': 5,
                'successful_requests': 5,
                'failed_requests': 0
            }
            
            result = transformer._create_cbf_record(row)
            
            assert isinstance(result, CBFRecord)
            assert result['cost/cost'] == 10.5
            assert result['usage/amount'] == 150  # 100 + 50
            assert result['usage/units'] == 'tokens'
            assert result['resource/id'] == 'test-czrn'

    def test_create_cbf_record_minimal_data(self):
        """Test _create_cbf_record method with minimal row data."""
        transformer = CBFTransformer()
        with patch.object(transformer.czrn_generator, 'create_from_litellm_data') as mock_czrn, \
             patch.object(transformer.czrn_generator, 'extract_components') as mock_extract:
            
            mock_czrn.return_value = 'test-czrn'
            mock_extract.return_value = ('service', 'provider', 'region', 'account', 'resource', 'local_id')
            
            row = {
                'date': '2025-01-19',
                'spend': 0.0
            }
            
            result = transformer._create_cbf_record(row)
            
            assert isinstance(result, CBFRecord)
            assert result['cost/cost'] == 0.0
            assert result['usage/amount'] == 0  # no tokens
            assert result['usage/units'] == 'tokens'

    def test_parse_date_with_valid_string(self):
        """Test _parse_date method with valid date string."""
        transformer = CBFTransformer()
        
        result = transformer._parse_date('2025-01-19')
        
        assert isinstance(result, datetime)
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 19

    def test_parse_date_with_datetime_object(self):
        """Test _parse_date method with datetime object."""
        transformer = CBFTransformer()
        dt = datetime(2025, 1, 19)
        
        result = transformer._parse_date(dt)
        
        assert result == dt

    def test_parse_date_with_none(self):
        """Test _parse_date method with None."""
        transformer = CBFTransformer()
        
        result = transformer._parse_date(None)
        
        assert result is None

    def test_parse_date_with_invalid_string(self):
        """Test _parse_date method with invalid date string."""
        transformer = CBFTransformer()
        
        result = transformer._parse_date('invalid-date')
        
        assert result is None

    def test_parse_date_with_iso_format(self):
        """Test _parse_date method with ISO format string."""
        transformer = CBFTransformer()
        
        result = transformer._parse_date('2025-01-19T10:30:00Z')
        
        assert isinstance(result, datetime)
        assert result.year == 2025 