# Boolean decoding compatibility

`parse_env_bool` matches the Python SDK environment decoder: `TypeAdapter(bool).validate_python(value.strip())`. Tokens are ASCII case-insensitive: `1/true/t/yes/y/on` and `0/false/f/no/n/off`. Python whitespace is stripped, including U+001C through U+001F. Empty strings and unknown tokens return `InvalidBoolean`, whose message excludes the input value

Absence and invalid input remain separate caller decisions. Analytics defaults only when a control is absent and fails closed when it is invalid. A present empty control is invalid

`parse_str_bool` preserves the existing secret-manager contract: trimmed `true` or `false` only. `parse_redis_bool` preserves the existing Redis contract: untrimmed `1/true/yes` are true, everything else is false. `core-utils::serde_compat` re-exports both functions so existing callers keep their behavior and import paths

This crate has no tracing, Python, environment access, YAML or network dependency. Hosts own lookup and configuration precedence. `envy` can deserialize typed host settings, but its default boolean parser uses Rust `bool::from_str` rather than the Python token set

Rust and Python tests use the same token fixture at `tests/fixtures/env-booleans.json`
