#!/bin/bash
# Test performance monitor - tests/monitor_test_performance.sh

echo "=== LiteLLM Test Performance Monitor ==="
echo "Starting at: $(date)"

# Time the test execution
start_time=$(date +%s)

# Run tests with timing
poetry run pytest tests/test_litellm -v --durations=10 --tb=short -n 6

end_time=$(date +%s)
execution_time=$((end_time - start_time))

echo "=== Test Execution Summary ==="
echo "Total execution time: ${execution_time} seconds"
echo "Completed at: $(date)"

# Alert if tests take longer than 20 minutes (1200 seconds)
if [ $execution_time -gt 1200 ]; then
    echo "WARNING: Tests took longer than 20 minutes!"
    echo "Consider optimizing slow tests or increasing timeout."
fi