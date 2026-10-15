- https://github.com/BerriAI/litellm-docs/blob/main/docs/guides/security_settings.md

- Own shared HTTP mechanics, including case-insensitive header lookup, defaults, replacement, and transport
- Do not choose provider credentials, OAuth policy, beta requirements, or model behavior
- Apply provider-supplied header decisions without importing provider implementations
- Preserve existing header precedence, duplicate handling, and unrelated forwarded headers
- Verify observable request behavior rather than the structure of helper functions
