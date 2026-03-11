"""
Test the CloudZero dry run endpoint functionality
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.cloudzero.cloudzero import CloudZeroLogger


class TestCloudZeroDryRunEndpoint:
    """Test suite for CloudZero dry run endpoint functionality."""

    @pytest.mark.asyncio
    async def test_dry_run_export_usage_data_returns_data(self):
        """
        Test that dry_run_export_usage_data returns expected data structure
        instead of just logging to console.
        """
        logger = CloudZeroLogger()
        
        # Mock database data
        mock_usage_data = pl.DataFrame({
            'date': ['2025-01-19', '2025-01-20'],
            'model': ['gpt-4', 'gpt-3.5-turbo'],
            'custom_llm_provider': ['openai', 'openai'],
            'team_id': ['team1', 'team2'],
            'team_alias': ['Team One', 'Team Two'],
            'api_key_alias': ['key1', 'key2'],
            'user_email': ['one@example.com', None],
            'prompt_tokens': [100, 200],
            'completion_tokens': [50, 100],
            'spend': [0.01, 0.02],
            'successful_requests': [1, 2]
        })
        
        # Mock CBF transformed data
        mock_cbf_data = pl.DataFrame({
            'time/usage_start': ['2025-01-19T00:00:00Z', '2025-01-20T00:00:00Z'],
            'cost/cost': [0.01, 0.02],
            'usage/amount': [150, 300],
            'resource/service': ['openai', 'openai'],
            'resource/account': ['litellm', 'litellm'],
            'resource/region': ['us-east-1', 'us-east-1'],
            'resource/id': ['gpt-4', 'gpt-3.5-turbo'],
            'entity_type': ['user', 'user'],
            'entity_id': ['team1', 'team2'],
            'resource/tag:team_id': ['team1', 'team2'],
            'resource/tag:team_alias': ['Team One', 'Team Two'],
            'resource/tag:api_key_alias': ['key1', 'key2'],
            'resource/tag:user_email': ['one@example.com', 'N/A']
        })
        
        with patch('litellm.integrations.cloudzero.database.LiteLLMDatabase') as mock_db_class, \
             patch('litellm.integrations.cloudzero.transform.CBFTransformer') as mock_transformer_class:
            
            # Setup mocks
            mock_db = AsyncMock()
            mock_db.get_usage_data.return_value = mock_usage_data
            mock_db_class.return_value = mock_db
            
            mock_transformer = MagicMock()
            mock_transformer.transform.return_value = mock_cbf_data
            mock_transformer_class.return_value = mock_transformer
            
            # Call the method
            result = await logger.dry_run_export_usage_data(limit=1000)
            
            # Verify the result structure
            assert isinstance(result, dict)
            assert 'usage_data' in result
            assert 'cbf_data' in result
            assert 'summary' in result
            
            # Verify usage_data
            assert isinstance(result['usage_data'], list)
            assert len(result['usage_data']) == 2
            assert result['usage_data'][0]['model'] == 'gpt-4'
            assert result['usage_data'][1]['model'] == 'gpt-3.5-turbo'
            
            # Verify cbf_data
            assert isinstance(result['cbf_data'], list)
            assert len(result['cbf_data']) == 2
            assert result['cbf_data'][0]['cost/cost'] == 0.01
            assert result['cbf_data'][1]['cost/cost'] == 0.02
            assert result['cbf_data'][0]['resource/tag:user_email'] == 'one@example.com'
            
            # Verify summary
            summary = result['summary']
            assert summary['total_records'] == 2
            assert summary['total_cost'] == 0.03
            assert summary['total_tokens'] == 450  # 150 + 300
            assert summary['unique_accounts'] == 1
            assert summary['unique_services'] == 1

    @pytest.mark.asyncio
    async def test_dry_run_export_usage_data_empty_data(self):
        """
        Test that dry_run_export_usage_data handles empty data gracefully.
        """
        logger = CloudZeroLogger()
        
        # Mock empty database data
        mock_empty_data = pl.DataFrame()
        
        with patch('litellm.integrations.cloudzero.database.LiteLLMDatabase') as mock_db_class:
            
            # Setup mocks
            mock_db = AsyncMock()
            mock_db.get_usage_data.return_value = mock_empty_data
            mock_db_class.return_value = mock_db
            
            # Call the method
            result = await logger.dry_run_export_usage_data(limit=1000)
            
            # Verify the result structure for empty data
            assert isinstance(result, dict)
            assert result['usage_data'] == []
            assert result['cbf_data'] == []
            assert result['summary']['total_records'] == 0
            assert result['summary']['total_cost'] == 0
            assert result['summary']['total_tokens'] == 0
