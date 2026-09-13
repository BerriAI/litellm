# What this is

Validates that unit tests covering traced Python behavior have semantic counterparts among colocated Rust unit tests

# How it works

Trace parity runs representative public API scenarios and records the Python and Rust functions reached, including their source files and lines. The OCR contract selects the behavior-level trace spans that require parity and excludes shared infrastructure such as generic HTTP transport

For Python, those traced functions define the denominator. Static references and explicit includes create a safe pytest discovery universe, then a pytest profiler keeps only tests that actually execute at least one selected function. Static matches do not count by themselves. Parametrized pytest cases are collapsed to one logical test function in the mapping report. Explicit includes and exclusions cover dynamic callers or intentional harness behavior that static discovery cannot express reliably

For Rust, each traced function identifies its source file and module. If that source file has a colocated `#[cfg(test)] mod tests`, the harness inventories that module for the configured Rust target. Rust test names are therefore derived from traced implementation files, not from a hand-maintained list of OCR test modules

The Python-to-Rust mappings remain explicit because equivalent behavior often has different test boundaries and names in each SDK. Host-only exclusions require a reason. The report validates both against the live inventories, then shows mapped, excluded, and unmapped Python tests plus Rust-only tests
