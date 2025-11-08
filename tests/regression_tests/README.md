# Performance Regression Tests

Automated tests to detect performance regressions in LiteLLM SDK and Router.

## Quick Start

```bash
# Run tests (will skip if no baseline exists)
pytest tests/regression_tests/ -v

# Create/update baselines
pytest tests/regression_tests/ -v --update-baseline
```

## Environment Variables

### Regression Thresholds

Configure acceptable performance degradation thresholds:

- `REGRESSION_THRESHOLD_8_CORES` - Threshold for 8+ CPU cores (default: `0.20` / 20%)
- `REGRESSION_THRESHOLD_4_CORES` - Threshold for 4-7 CPU cores (default: `0.30` / 30%)
- `REGRESSION_THRESHOLD_LOW_CORES` - Threshold for <4 CPU cores (default: `0.60` / 60%)
- `REGRESSION_THRESHOLD_CI_ADJUSTMENT` - Additional threshold for CI environments (default: `0.15` / 15%)

### Test Configuration

- `REGRESSION_NUM_REQUESTS` - Number of requests per test (default: `250`)
- `FAKE_OPENAI_ENDPOINT` - Mock endpoint URL (default: `https://exampleopenaiendpoint-production.up.railway.app`)

## Example

```bash
# Run faster tests with lower request count
export REGRESSION_NUM_REQUESTS=100
pytest tests/regression_tests/ -v

# Adjust threshold for CI
export REGRESSION_THRESHOLD_CI_ADJUSTMENT=0.25
pytest tests/regression_tests/ -v
```

## How It Works

1. Tests measure median and P95 latency for SDK/Router calls
2. Results are compared against saved baselines in `performance_baselines.json`
3. Test fails if performance degrades beyond configured threshold
4. Thresholds auto-adjust based on CPU cores and CI environment
