"""
Tests for the optimized database jobs coordinator scheduling logic

This test suite validates:
1. Correct interval pre-calculation at initialization
2. Proper timing logic for job execution
3. Independence of different job schedules
4. Consistency with original implementation behavior
"""
import pytest
from datetime import datetime, timedelta
from litellm.proxy.proxy_server import DatabaseJobsCoordinator


class TestDatabaseJobsCoordinatorInitialization:
    """Test DatabaseJobsCoordinator initialization logic"""
    
    def test_coordinator_initialization(self):
        """Verify coordinator correctly initializes all tasks"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        # Verify all tasks are initialized
        expected_jobs = [
            "update_spend",
            "reset_budget", 
            "add_deployment",
            "get_credentials",
            "spend_log_cleanup",
            "check_batch_cost"
        ]
        
        for job in expected_jobs:
            assert job in coordinator.last_run_times
            assert coordinator.last_run_times[job] is None  # Initially None
    
    def test_precalculated_intervals_in_range(self):
        """Verify pre-calculated intervals are in correct range"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        # update_spend: proxy_batch_write_at ± 3 seconds
        update_interval = coordinator.get_task_interval("update_spend")
        assert update_interval is not None
        assert 57 <= update_interval <= 63
        
        # reset_budget: proxy_budget_rescheduler_min/max_time range
        reset_interval = coordinator.get_task_interval("reset_budget")
        assert reset_interval is not None
        assert 3600 <= reset_interval <= 7200
    
    def test_intervals_remain_constant(self):
        """Verify interval values remain constant throughout lifecycle"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        # Record initial intervals
        initial_update = coordinator.get_task_interval("update_spend")
        initial_reset = coordinator.get_task_interval("reset_budget")
        
        # Simulate multiple calls to should_run and mark_run
        for _ in range(100):
            coordinator.should_run("update_spend", 60)
            coordinator.mark_run("update_spend")
        
        # Verify interval values haven't changed at all
        assert coordinator.get_task_interval("update_spend") == initial_update
        assert coordinator.get_task_interval("reset_budget") == initial_reset
    
    def test_different_instances_have_different_intervals(self):
        """Verify different instances have different random intervals (avoid multi-worker collisions)"""
        coordinators = [
            DatabaseJobsCoordinator(
                proxy_budget_rescheduler_min_time=3600,
                proxy_budget_rescheduler_max_time=7200,
                proxy_batch_write_at=60,
            )
            for _ in range(20)
        ]
        
        update_intervals = [c.get_task_interval("update_spend") for c in coordinators]
        reset_intervals = [c.get_task_interval("reset_budget") for c in coordinators]
        
        # Should have some variation (randomization effect)
        unique_update = len(set(update_intervals))
        unique_reset = len(set(reset_intervals))
        
        # With 20 instances, should have at least 2 different values
        assert unique_update >= 2, f"update_spend interval lacks randomization: {unique_update} unique values"
        assert unique_reset >= 2, f"reset_budget interval lacks randomization: {unique_reset} unique values"


class TestDatabaseJobsCoordinatorTiming:
    """Test task execution timing logic"""
    
    def test_should_run_never_executed(self):
        """Test that never-executed tasks should run immediately"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        # All tasks should return True in initial state
        assert coordinator.should_run("update_spend", 60) is True
        assert coordinator.should_run("reset_budget", 3600) is True
        assert coordinator.should_run("add_deployment", 10) is True
    
    def test_should_run_just_executed(self):
        """Test that just-executed tasks should not run again immediately"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        now = datetime.now()
        
        # Set as just executed
        coordinator.set_last_run_time("update_spend", now)
        
        # Should not meet interval requirement
        assert coordinator.should_run("update_spend", 60) is False
    
    def test_should_run_after_interval(self):
        """Test that tasks should run after interval has passed"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        now = datetime.now()
        
        # Set as executed 61 seconds ago (exceeds 60 second interval)
        coordinator.set_last_run_time("update_spend", now - timedelta(seconds=61))
        assert coordinator.should_run("update_spend", 60) is True
        
        # Set as executed 59 seconds ago (doesn't meet 60 second interval)
        coordinator.set_last_run_time("update_spend", now - timedelta(seconds=59))
        assert coordinator.should_run("update_spend", 60) is False
    
    def test_should_run_exact_interval_boundary(self):
        """Test behavior at exact interval boundary"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        now = datetime.now()
        
        # Exactly 60 seconds ago
        coordinator.set_last_run_time("update_spend", now - timedelta(seconds=60))
        assert coordinator.should_run("update_spend", 60) is True
        
        # Exactly 60.001 seconds ago (just slightly over)
        coordinator.set_last_run_time("update_spend", now - timedelta(seconds=60.001))
        assert coordinator.should_run("update_spend", 60) is True
    
    def test_mark_run_updates_time(self):
        """Test mark_run correctly updates execution time"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        # Initial state
        assert coordinator.get_last_run_time("update_spend") is None
        
        # Mark as run
        before_mark = datetime.now()
        coordinator.mark_run("update_spend")
        after_mark = datetime.now()
        
        # Verify time was updated
        last_run = coordinator.get_last_run_time("update_spend")
        assert last_run is not None
        assert before_mark <= last_run <= after_mark
    
    def test_independent_job_timing(self):
        """Test timing independence of different tasks"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        now = datetime.now()
        
        # Set different execution times for different tasks
        coordinator.set_last_run_time("update_spend", now - timedelta(seconds=70))
        coordinator.set_last_run_time("reset_budget", now - timedelta(seconds=10))
        coordinator.set_last_run_time("add_deployment", now - timedelta(seconds=15))
        
        # Verify each task is judged independently
        assert coordinator.should_run("update_spend", 60) is True  # 70s > 60s
        assert coordinator.should_run("reset_budget", 3600) is False  # 10s < 3600s
        assert coordinator.should_run("add_deployment", 10) is True  # 15s > 10s


class TestDatabaseJobsCoordinatorEdgeCases:
    """Test edge cases and exception handling"""
    
    def test_set_invalid_job_name(self):
        """Test setting invalid job name should raise exception"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        with pytest.raises(ValueError, match="Invalid job name"):
            coordinator.set_last_run_time("invalid_job", datetime.now())
    
    def test_get_invalid_job_name(self):
        """Test getting invalid job name should raise exception"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        with pytest.raises(ValueError, match="Invalid job name"):
            coordinator.get_last_run_time("invalid_job")
    
    def test_set_none_resets_time(self):
        """Test that setting None resets execution time"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        # First set a time
        coordinator.set_last_run_time("update_spend", datetime.now())
        assert coordinator.get_last_run_time("update_spend") is not None
        
        # Reset to None
        coordinator.set_last_run_time("update_spend", None)
        assert coordinator.get_last_run_time("update_spend") is None
        
        # Should behave like never executed
        assert coordinator.should_run("update_spend", 60) is True


class TestDatabaseJobsCoordinatorRealWorldScenarios:
    """Test real-world scenarios"""
    
    def test_typical_update_spend_cycle(self):
        """Simulate typical update_spend execution cycle"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        interval = coordinator.get_task_interval("update_spend")
        
        # First time: should execute
        assert coordinator.should_run("update_spend", interval) is True
        coordinator.mark_run("update_spend")
        
        # Check after 10 seconds: should not execute
        coordinator.set_last_run_time(
            "update_spend", 
            datetime.now() - timedelta(seconds=10)
        )
        assert coordinator.should_run("update_spend", interval) is False
        
        # Check after interval seconds: should execute
        coordinator.set_last_run_time(
            "update_spend", 
            datetime.now() - timedelta(seconds=interval)
        )
        assert coordinator.should_run("update_spend", interval) is True
    
    def test_typical_reset_budget_cycle(self):
        """Simulate typical reset_budget execution cycle"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        interval = coordinator.get_task_interval("reset_budget")
        
        # Verify interval is between 1-2 hours
        assert 3600 <= interval <= 7200
        
        # Simulate execution cycle
        assert coordinator.should_run("reset_budget", interval) is True
        coordinator.mark_run("reset_budget")
        
        # After 2.5 hours: definitely should execute (exceeds max interval of 2h)
        coordinator.set_last_run_time(
            "reset_budget",
            datetime.now() - timedelta(hours=2.5)
        )
        assert coordinator.should_run("reset_budget", interval) is True
    
    def test_multiple_jobs_concurrent_scheduling(self):
        """Test concurrent scheduling of multiple tasks"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        now = datetime.now()
        
        # Set different execution times for different tasks
        coordinator.set_last_run_time("update_spend", now - timedelta(seconds=65))
        coordinator.set_last_run_time("reset_budget", now - timedelta(seconds=1800))
        coordinator.set_last_run_time("add_deployment", now - timedelta(seconds=5))
        coordinator.set_last_run_time("get_credentials", now - timedelta(seconds=12))
        
        # Verify each task is judged independently
        update_interval = coordinator.get_task_interval("update_spend")
        reset_interval = coordinator.get_task_interval("reset_budget")
        
        assert coordinator.should_run("update_spend", update_interval) is True  # 65s > ~60s
        assert coordinator.should_run("reset_budget", reset_interval) is False  # 30min < any random interval (1-2h)
        assert coordinator.should_run("add_deployment", 10) is False  # 5s < 10s
        assert coordinator.should_run("get_credentials", 10) is True  # 12s > 10s
    
    def test_consistent_behavior_across_coordinator_lifecycle(self):
        """Test coordinator behavior consistency throughout lifecycle"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        update_interval = coordinator.get_task_interval("update_spend")
        
        # Simulate multiple execution cycles
        for cycle in range(10):
            # Simulate time passage
            now = datetime.now()
            
            # update_spend should run every interval seconds
            coordinator.set_last_run_time("update_spend", now - timedelta(seconds=update_interval + 1))
            assert coordinator.should_run("update_spend", update_interval) is True
            coordinator.mark_run("update_spend")
            
            # Immediate check should return False
            assert coordinator.should_run("update_spend", update_interval) is False
    
    def test_all_jobs_have_expected_intervals(self):
        """Verify all tasks have expected interval settings"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        # update_spend and reset_budget should have pre-calculated intervals
        assert coordinator.get_task_interval("update_spend") is not None
        assert coordinator.get_task_interval("reset_budget") is not None
        
        # Other tasks should not have pre-calculated intervals (they use fixed intervals)
        assert coordinator.get_task_interval("add_deployment") is None
        assert coordinator.get_task_interval("get_credentials") is None
        assert coordinator.get_task_interval("spend_log_cleanup") is None
        assert coordinator.get_task_interval("check_batch_cost") is None


class TestDatabaseJobsCoordinatorConsistencyWithOriginal:
    """Verify new implementation consistency with original implementation"""
    
    def test_update_spend_interval_matches_original(self):
        """Verify update_spend interval logic matches original implementation"""
        # Original: random.randint(proxy_batch_write_at - 3, proxy_batch_write_at + 3)
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        interval = coordinator.get_task_interval("update_spend")
        
        # Should be in [57, 63] range (60 ± 3)
        assert 57 <= interval <= 63
    
    def test_reset_budget_interval_matches_original(self):
        """Verify reset_budget interval logic matches original implementation"""
        # Original: random.randint(proxy_budget_rescheduler_min_time, proxy_budget_rescheduler_max_time)
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        interval = coordinator.get_task_interval("reset_budget")
        
        # Should be in [3600, 7200] range
        assert 3600 <= interval <= 7200
    
    def test_should_run_logic_matches_original(self):
        """Verify should_run logic matches original implementation"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        # Original logic:
        # if last_run is None: return True
        # elapsed = (datetime.now() - last_run).total_seconds()
        # return elapsed >= interval_seconds
        
        # Test None case
        assert coordinator.should_run("update_spend", 60) is True
        
        # Test just executed
        coordinator.set_last_run_time("update_spend", datetime.now())
        assert coordinator.should_run("update_spend", 60) is False
        
        # Test after interval
        coordinator.set_last_run_time(
            "update_spend",
            datetime.now() - timedelta(seconds=61)
        )
        assert coordinator.should_run("update_spend", 60) is True


class TestDatabaseJobsCoordinatorParameterVariations:
    """Test behavior under different parameter configurations"""
    
    def test_different_batch_write_intervals(self):
        """Test different proxy_batch_write_at values"""
        for batch_write_at in [30, 60, 120, 300]:
            coordinator = DatabaseJobsCoordinator(
                proxy_budget_rescheduler_min_time=3600,
                proxy_budget_rescheduler_max_time=7200,
                proxy_batch_write_at=batch_write_at,
            )
            
            interval = coordinator.get_task_interval("update_spend")
            assert interval is not None
            assert batch_write_at - 3 <= interval <= batch_write_at + 3
    
    def test_different_budget_rescheduler_ranges(self):
        """Test different budget_rescheduler ranges"""
        test_cases = [
            (1800, 3600),   # 30 minutes to 1 hour
            (3600, 7200),   # 1 hour to 2 hours
            (7200, 14400),  # 2 hours to 4 hours
        ]
        
        for min_time, max_time in test_cases:
            coordinator = DatabaseJobsCoordinator(
                proxy_budget_rescheduler_min_time=min_time,
                proxy_budget_rescheduler_max_time=max_time,
                proxy_batch_write_at=60,
            )
            
            interval = coordinator.get_task_interval("reset_budget")
            assert interval is not None
            assert min_time <= interval <= max_time
    
    def test_default_values_from_constants(self):
        """Test using default constant values"""
        # Simulate default values (from litellm.constants)
        # PROXY_BUDGET_RESCHEDULER_MIN_TIME = 3600
        # PROXY_BUDGET_RESCHEDULER_MAX_TIME = 7200
        # PROXY_BATCH_WRITE_AT = 60
        
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        # Verify behavior with default values
        update_interval = coordinator.get_task_interval("update_spend")
        reset_interval = coordinator.get_task_interval("reset_budget")
        
        assert 57 <= update_interval <= 63
        assert 3600 <= reset_interval <= 7200


class TestDatabaseJobsCoordinatorHelperMethods:
    """Test helper method functionality"""
    
    def test_get_task_interval_returns_none_for_non_precalculated(self):
        """Test get_task_interval returns None for non-pre-calculated tasks"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        # These tasks don't have pre-calculated intervals
        assert coordinator.get_task_interval("add_deployment") is None
        assert coordinator.get_task_interval("get_credentials") is None
        assert coordinator.get_task_interval("spend_log_cleanup") is None
        assert coordinator.get_task_interval("check_batch_cost") is None
    
    def test_set_and_get_last_run_time(self):
        """Test set_last_run_time and get_last_run_time round-trip"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        test_time = datetime.now() - timedelta(hours=1)
        
        # Set time
        coordinator.set_last_run_time("update_spend", test_time)
        
        # Get time and verify
        retrieved_time = coordinator.get_last_run_time("update_spend")
        assert retrieved_time == test_time
    
    def test_helper_methods_for_all_jobs(self):
        """Verify helper methods work correctly for all tasks"""
        coordinator = DatabaseJobsCoordinator(
            proxy_budget_rescheduler_min_time=3600,
            proxy_budget_rescheduler_max_time=7200,
            proxy_batch_write_at=60,
        )
        
        all_jobs = [
            "update_spend",
            "reset_budget",
            "add_deployment",
            "get_credentials",
            "spend_log_cleanup",
            "check_batch_cost"
        ]
        
        test_time = datetime.now() - timedelta(minutes=30)
        
        for job in all_jobs:
            # Should be able to set and get time
            coordinator.set_last_run_time(job, test_time)
            assert coordinator.get_last_run_time(job) == test_time
            
            # Should be able to reset to None
            coordinator.set_last_run_time(job, None)
            assert coordinator.get_last_run_time(job) is None


class TestDatabaseJobsCoordinatorRandomizationBehavior:
    """Test randomization behavior (key feature to prevent multi-worker conflicts)"""
    
    def test_randomization_prevents_synchronized_execution(self):
        """Verify randomization prevents synchronized execution across workers"""
        # Create multiple coordinator instances (simulating multiple workers)
        num_workers = 10
        coordinators = [
            DatabaseJobsCoordinator(
                proxy_budget_rescheduler_min_time=3600,
                proxy_budget_rescheduler_max_time=7200,
                proxy_batch_write_at=60,
            )
            for _ in range(num_workers)
        ]
        
        # Collect all update_spend intervals
        update_intervals = [c.get_task_interval("update_spend") for c in coordinators]
        
        # Verify there's at least some variation (not all workers have same interval)
        unique_intervals = set(update_intervals)
        assert len(unique_intervals) > 1, "All workers have same interval, defeats randomization purpose"
    
    def test_randomization_distribution(self):
        """Test whether randomization distribution is reasonable"""
        # Create many instances to verify distribution
        num_samples = 100
        coordinators = [
            DatabaseJobsCoordinator(
                proxy_budget_rescheduler_min_time=3600,
                proxy_budget_rescheduler_max_time=7200,
                proxy_batch_write_at=60,
            )
            for _ in range(num_samples)
        ]
        
        update_intervals = [c.get_task_interval("update_spend") for c in coordinators]
        
        # Verify values are in expected range
        assert all(57 <= i <= 63 for i in update_intervals)
        
        # Verify distribution diversity (should have at least 3 different values)
        unique_count = len(set(update_intervals))
        assert unique_count >= 3, f"Distribution lacks diversity, only {unique_count} unique values"
