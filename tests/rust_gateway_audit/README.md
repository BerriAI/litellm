# Rust Gateway 1:1 Mapping Audit Tools

This directory contains automated tools to audit the Rust bridge implementation and ensure 1:1 mapping with Python gateway endpoints.

## Overview

The Rust bridge currently implements SDK-level functions but lacks gateway-level endpoint handlers. These tools trace the Python implementation and compare it with Rust to identify gaps.

## What the Auditors Do

### 1. Component Tracing
- **Traces Python endpoint flow** from HTTP request to SDK call
- **Extracts function signatures** (params, async/sync, module location)
- **Maps call hierarchy** to understand the full request lifecycle

### 2. Rust Comparison
- **Compares with known Rust implementations** from the PR branch
- **Identifies missing components** (endpoint handlers, parsers, processors)
- **Classifies status**: Implemented ✅ | Partial ⚠️ | Missing ❌

### 3. Gap Analysis
- **Generates detailed tables** showing what's missing
- **Counts parameters** for each missing function
- **Provides implementation notes** for each gap

### 4. Implementation Guidance
- **Generates Rust stub code** with proper signatures
- **Proposes implementation phases** with priority
- **Estimates effort** for each component
- **Lists security requirements** that must be preserved

## Available Auditors

### `/chat/completions` Auditor
```bash
poetry run python tests/rust_gateway_audit/test_chat_completions_audit.py
```

**What it audits:**
- ✅ Main endpoint handler (`chat_completion`)
- ✅ Request body parsing (`_read_request_body`)
- ✅ ProxyBaseLLMRequestProcessing initialization
- ✅ Core request processing (`base_process_llm_request`)
- ✅ Response header generation (`get_custom_headers`)
- ✅ Authentication middleware (`user_api_key_auth`)
- ✅ SDK-level function (`chat_completions`)

**Output:**
- Comparison table with 7 components
- Detailed parameter lists (up to 16 params)
- Rust stub implementations
- Coverage: 14.3% (1/7 implemented)

### `/ocr` Auditor
```bash
poetry run python tests/rust_gateway_audit/test_ocr_audit.py
```

**What it audits:**
- ✅ OCR endpoint handler
- ✅ Request parsing (JSON and multipart)
- ✅ Multipart form data handling
- ✅ Document upload utilities
- ✅ Request format resolution (`x-req-format` header)
- ✅ Native response support
- ✅ Shared ProxyBaseLLMRequestProcessing
- ✅ SDK-level functions (`ocr`, `aocr`)

**Output:**
- Comparison table with 9 components
- Security requirements (file type blocking, reducto:// ID blocking)
- Implementation proposal with 4 phases
- Coverage: 0.0% (0/9 implemented, 1 partial)
- Estimated effort: 10-12 days

## Audit Report Structure

Each audit generates a markdown report with:

### 1. Implementation Status
```
Total Components: X
✅ Implemented: X
⚠️  Partial: X
❌ Missing: X
Coverage: X.X%
```

### 2. Component Breakdown Table
| Python Component | Rust Equivalent | Status | Params | Notes |
|------------------|-----------------|--------|--------|-------|
| ... | ... | ... | ... | ... |

### 3. Detailed Missing Parameters
Lists all parameters for each missing function:
```rust
### function_name
  param1
  param2
  param3
```

### 4. Implementation Proposal
- **Current State**: What exists in Rust
- **What's Missing**: Gap summary
- **Phases**: Prioritized implementation plan
- **Security Requirements**: Must-have validations
- **Testing Strategy**: Unit and integration tests needed
- **Estimated Effort**: Time breakdown by phase

### 5. Rust Stub Implementations
Complete Rust code templates with:
- Proper function signatures
- PyO3 decorators
- Parameter mappings
- TODO markers for implementation

## Key Findings

### Common Gaps Across Endpoints

1. **ProxyBaseLLMRequestProcessing** (Shared Component)
   - Missing in all endpoints
   - Required for: authentication, logging, routing, cost tracking
   - 16 parameters in `base_process_llm_request`
   - 13 parameters in `get_custom_headers`

2. **Gateway Endpoint Handlers**
   - Rust has SDK functions but no FastAPI route equivalents
   - Missing: request parsing, metadata injection, error handling
   - Must integrate with ProxyBaseLLMRequestProcessing

3. **Request Parsing**
   - `/chat/completions`: Simple JSON body parsing
   - `/ocr`: Complex - JSON + multipart form data + file uploads

4. **Security Validations**
   - Must be preserved in Rust implementation
   - Examples: file type blocking, provider ID validation

## Usage Examples

### Run Single Audit
```bash
# Chat completions
cd /Users/ishaanjaffer/github/litellm
poetry run python tests/rust_gateway_audit/test_chat_completions_audit.py

# OCR
poetry run python tests/rust_gateway_audit/test_ocr_audit.py
```

### Run All Audits
```bash
poetry run python tests/rust_gateway_audit/run_all_audits.py
```

### View Report
```bash
# Reports are saved to /tmp/
cat /tmp/gateway_endpoint_trace.md
cat /tmp/ocr_endpoint_audit.md
```

## Adding New Endpoint Auditors

To add an auditor for a new endpoint (e.g., `/audio/transcriptions`):

1. **Create auditor file**: `test_audio_transcriptions_audit.py`

2. **Trace Python endpoint**:
```python
from litellm.proxy.audio_endpoints import audio_transcription
endpoint_trace = self._trace_function(audio_transcription)
```

3. **Define expected Rust components**:
```python
expected_rust_components = [
    {
        "python": "audio_transcription (endpoint)",
        "rust": "audio_transcription_endpoint",
        "status": "missing",
        "params": ["request", "fastapi_response", "user_api_key_dict"],
        "notes": "FastAPI endpoint handler equivalent"
    },
    # ... more components
]
```

4. **Generate comparison and stubs**:
```python
auditor.compare_with_rust(endpoint_trace)
table = auditor.generate_comparison_table()
stubs = auditor.generate_rust_stubs()
```

## Report Outputs

### Terminal Output
- Summary with ✅/⚠️/❌ status indicators
- Component counts
- Coverage percentage
- File save location

### Markdown Files (`/tmp/`)
- `gateway_endpoint_trace.md` - Chat completions audit
- `ocr_endpoint_audit.md` - OCR audit
- Full tables, stubs, and proposals

### Structured Data (Future)
Could be extended to output JSON for programmatic consumption:
```json
{
  "endpoint": "/chat/completions",
  "coverage": 14.3,
  "components": [
    {
      "name": "chat_completion",
      "status": "missing",
      "params": 4
    }
  ]
}
```

## Integration with Development Workflow

### 1. Before Starting Rust Implementation
Run audits to understand scope:
```bash
poetry run python tests/rust_gateway_audit/run_all_audits.py
```

### 2. During Implementation
Re-run specific endpoint audit to track progress:
```bash
poetry run python tests/rust_gateway_audit/test_ocr_audit.py
```

### 3. Before PR Review
Verify all components are implemented:
- Coverage should be 100%
- All ❌ should become ✅

### 4. Add to CI/CD
```yaml
- name: Audit Rust Gateway Coverage
  run: poetry run python tests/rust_gateway_audit/run_all_audits.py
  # Fail if coverage < threshold
```

## Limitations

### Current Limitations
1. **Manual status updates**: Requires updating `rust_implementations` dict
2. **No automatic detection**: Doesn't scan Rust codebase automatically
3. **Static analysis only**: Doesn't verify runtime behavior

### Future Enhancements
1. **Parse Rust files**: Automatically detect implemented functions
2. **Integration tests**: Verify behavior matches Python
3. **Performance comparison**: Measure Rust vs Python speed
4. **Memory profiling**: Track memory usage

## Architecture Notes

### Why Separate Auditors?
- Each endpoint has unique request/response handling
- Security validations differ per endpoint
- Implementation priorities vary

### Why Not One Generic Auditor?
- Endpoints have different complexity levels
- `/ocr` has multipart form data, `/chat/completions` doesn't
- Better to have specialized auditors with tailored guidance

### Shared Components
ProxyBaseLLMRequestProcessing is audited in each endpoint auditor because:
- It's the critical shared component
- Every endpoint needs it
- Shows true cost of implementation

## Success Metrics

### Per-Endpoint Goals
- ✅ 100% coverage (all components implemented)
- ✅ All security validations preserved
- ✅ Integration tests passing
- ✅ Performance meets/exceeds Python

### Overall Project Goals
- ✅ ProxyBaseLLMRequestProcessing implemented (unlocks all endpoints)
- ✅ Gateway endpoint pattern established
- ✅ Request/response utilities shared across endpoints
- ✅ Documentation for adding new endpoints

## Contributing

To improve the auditors:

1. **Add more detail**: Trace deeper into call stacks
2. **Add examples**: Include sample requests/responses
3. **Add benchmarks**: Compare Python vs Rust performance
4. **Add validation**: Check parameter types match

## Related Documentation

- Python implementation: `litellm/proxy/proxy_server.py`
- Rust SDK functions: `litellm-rust/crates/python-bridge/src/routes/`
- PR #39031: Rust bridge refactoring
- Issue discussion: Your comment on gateway-level tracing

## Questions?

For questions about:
- **Audit results**: Check the generated markdown reports
- **Rust implementation**: See the stub code in reports
- **Security requirements**: Check Security Requirements sections
- **Missing components**: See Component Breakdown tables
