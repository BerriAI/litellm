- https://github.com/BerriAI/litellm-docs/blob/main/docs/guides/security_settings.md

- Own shared HTTP mechanics, including case-insensitive header lookup, defaults, replacement, and transport
- Do not choose provider credentials, OAuth policy, beta requirements, or model behavior
- Apply provider-supplied header decisions without importing provider implementations
- Preserve existing header precedence, duplicate handling, and unrelated forwarded headers
- Verify observable request behavior rather than the structure of helper functions
- The media SSRF blocklist in `media.rs` follows the IANA special-purpose registries and is a superset of the Python `_is_blocked_ip` policy. `generated/blocked_ips.json` is CPython output from `scripts/generate_blocked_ips.py`, never hand-edited. Regenerate it when the Python policy or probe list changes, and record a Rust-only block in `KNOWN_OVER_BLOCKED` with its reason
