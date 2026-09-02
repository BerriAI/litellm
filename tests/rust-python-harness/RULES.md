# Rust-Python Harness Structure Rules

## Testing Strategy Organization

Each testing strategy folder follows a **1:1 mapping** with the litellm-rust repository structure:

```
litellm-rust/
├── crates/
│   ├── core/           → tests here map to core/
│   ├── ai-gateway/     → tests here map to ai-gateway/
│   └── python-bridge/  → tests here map to python-bridge/
```

### Structure Rules

1. **Directory Mapping**: Test files mirror the Rust crate structure
   - Example: `litellm-rust/crates/core/src/ocr.rs` → `unit_tests_rust/core/test_ocr.py`
   - Example: `litellm-rust/crates/ai-gateway/src/routes/chat.rs` → `gateway_trace/ai_gateway/test_chat.py`

2. **File Organization**: 
   - USB (unit/sub-method/behavior) tests go under their corresponding crate folder
   - Easy to find: same path structure as Rust source

3. **Strategy Folders**:
   - `e2e_fuzz_tests/` - Public SDK parity over generated inputs
   - `unit_tests_rust/` - Rust-owned behavior tests (mirrors `litellm-rust/crates/`)
   - `validate_sub_methods/` - Isolated transform and helper coverage
   - `gateway_trace/` - Gateway endpoint execution tracing (mirrors `ai-gateway/`)

## Example Mapping

### Rust Source
```
litellm-rust/
└── crates/
    ├── core/
    │   └── src/
    │       ├── ocr.rs
    │       └── messages.rs
    └── ai-gateway/
        └── src/
            └── routes/
                └── chat_completions.rs
```

### Test Structure
```
tests/rust-python-harness/
├── unit_tests_rust/
│   └── core/
│       ├── test_ocr.py          # Tests core/src/ocr.rs
│       └── test_messages.py     # Tests core/src/messages.rs
└── gateway_trace/
    └── ai_gateway/
        └── routes/
            └── test_chat_completions.py  # Tests ai-gateway routes
```

## Finding Tests

To find tests for a Rust file:
1. Identify the crate: `core`, `ai-gateway`, or `python-bridge`
2. Look in the matching strategy folder
3. Follow the same path structure

Example: To test `litellm-rust/crates/core/src/ocr.rs`:
- Strategy: `unit_tests_rust/` (for Rust-owned behavior)
- Path: `core/test_ocr.py`
- Full: `tests/rust-python-harness/unit_tests_rust/core/test_ocr.py`

## Adding New Tests

When adding a new Rust file at `litellm-rust/crates/X/src/Y.rs`:
1. Create test at `tests/rust-python-harness/<strategy>/X/test_Y.py`
2. Update `strategy.json` in the strategy folder
3. Tests automatically appear in the harness matrix
