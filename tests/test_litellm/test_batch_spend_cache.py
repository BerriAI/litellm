"""Test the batch spend duplicate fix by testing the cache directly."""
import sys
import os
sys.path.insert(0, os.path.abspath("../.."))

import pytest
from litellm.litellm_core_utils.litellm_logging import _PROCESSED_BATCH_COSTS


def test_processed_batch_costs_cache():
    """Test that the _PROCESSED_BATCH_COSTS cache works correctly."""
    # Clear the cache
    _PROCESSED_BATCH_COSTS.clear()
    
    # Test adding batch IDs
    batch_id_1 = "test_batch_123"
    batch_id_2 = "test_batch_456"
    
    # Initially empty
    assert len(_PROCESSED_BATCH_COSTS) == 0
    assert batch_id_1 not in _PROCESSED_BATCH_COSTS
    assert batch_id_2 not in _PROCESSED_BATCH_COSTS
    
    # Add first batch
    _PROCESSED_BATCH_COSTS.add(batch_id_1)
    assert len(_PROCESSED_BATCH_COSTS) == 1
    assert batch_id_1 in _PROCESSED_BATCH_COSTS
    assert batch_id_2 not in _PROCESSED_BATCH_COSTS
    
    # Add second batch
    _PROCESSED_BATCH_COSTS.add(batch_id_2)
    assert len(_PROCESSED_BATCH_COSTS) == 2
    assert batch_id_1 in _PROCESSED_BATCH_COSTS
    assert batch_id_2 in _PROCESSED_BATCH_COSTS
    
    # Adding same batch again doesn't increase size
    _PROCESSED_BATCH_COSTS.add(batch_id_1)
    assert len(_PROCESSED_BATCH_COSTS) == 2
    assert batch_id_1 in _PROCESSED_BATCH_COSTS
    
    # Clear works
    _PROCESSED_BATCH_COSTS.clear()
    assert len(_PROCESSED_BATCH_COSTS) == 0
    assert batch_id_1 not in _PROCESSED_BATCH_COSTS
    assert batch_id_2 not in _PROCESSED_BATCH_COSTS


def test_batch_cost_caching_logic():
    """Test the logic for checking if batch cost has been processed."""
    _PROCESSED_BATCH_COSTS.clear()
    
    batch_id = "test_batch_789"
    
    # Simulate first processing
    if batch_id in _PROCESSED_BATCH_COSTS:
        # Should not enter here on first call
        assert False, "Should not be in cache on first call"
    else:
        # Process the batch
        _PROCESSED_BATCH_COSTS.add(batch_id)
    
    # Simulate second processing
    if batch_id in _PROCESSED_BATCH_COSTS:
        # Should enter here on second call
        print("Skipping batch cost calculation, already processed")
    else:
        # Should not enter here on second call
        assert False, "Should be in cache on second call"
    
    # Verify it's in the cache
    assert batch_id in _PROCESSED_BATCH_COSTS