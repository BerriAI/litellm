
import asyncio
import pytest
from litellm.scheduler import Scheduler, FlowItem

@pytest.mark.asyncio
async def test_scheduler_remove_request():
    """
    Test that remove_request correctly removes items from the queue.
    Regression test for Issue #20059.
    """
    scheduler = Scheduler(polling_interval=0.001)
    model_name = "test-model"
    
    # Add items
    item1 = FlowItem(priority=0, request_id="req-1", model_name=model_name)
    item2 = FlowItem(priority=1, request_id="req-2", model_name=model_name)
    item3 = FlowItem(priority=2, request_id="req-3", model_name=model_name)
    
    await scheduler.add_request(item1)
    await scheduler.add_request(item2)
    await scheduler.add_request(item3)
    
    # Verify all invalid queue
    queue = await scheduler.get_queue(model_name=model_name)
    assert len(queue) == 3
    
    # Remove item2 (middle)
    await scheduler.remove_request(request_id="req-2", model_name=model_name)
    
    # Verify item2 is gone
    queue = await scheduler.get_queue(model_name=model_name)
    assert len(queue) == 2
    req_ids = [item[1] for item in queue]
    assert "req-2" not in req_ids
    assert "req-1" in req_ids
    assert "req-3" in req_ids
    
    # Remove item1 (top)
    await scheduler.remove_request(request_id="req-1", model_name=model_name)
    queue = await scheduler.get_queue(model_name=model_name)
    assert len(queue) == 1
    assert queue[0][1] == "req-3"

@pytest.mark.asyncio
async def test_scheduler_remove_nonexistent():
    """Test removing a non-existent item doesn't crash."""
    scheduler = Scheduler(polling_interval=0.001)
    model_name = "test-model"
    await scheduler.add_request(FlowItem(priority=0, request_id="req-1", model_name=model_name))
    
    # Try removing non-existent
    await scheduler.remove_request(request_id="non-existent", model_name=model_name)
    
    queue = await scheduler.get_queue(model_name=model_name)
    assert len(queue) == 1
    assert queue[0][1] == "req-1"
