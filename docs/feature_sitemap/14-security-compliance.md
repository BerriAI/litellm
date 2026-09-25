# 14 Security and compliance

Cross-cutting security posture rather than a single feature: secret redaction, request validation and size limits, compliance endpoints, the hardened image and supply-chain artifacts, audit trails, headers and CORS, SSRF protections, and credential encryption at rest. Access control itself lives in [04-auth-identity.md](04-auth-identity.md); log redaction overlaps with [08-observability.md](08-observability.md) (`obs.redaction`)

## Security and compliance

### security.secret_redaction: Secret redaction in logs and errors
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/data_security, https://docs.litellm.ai/docs/observability/scrub_data
code: `litellm/litellm_core_utils/secret_redaction.py`, `litellm/litellm_core_utils/redact_messages.py`, `litellm/litellm_core_utils/litellm_logging.py`
tests: `tests/test_litellm/litellm_core_utils/test_redact_messages.py`
registry: none
verify: set `litellm_settings: {turn_off_message_logging: false}` with a spend-logging callback enabled, send a prompt containing a 20+ char fake key like `sk-abcdefghijklmnopqrstuv`, and confirm the spend log and proxy log show the redacted placeholder, not the value

### security.request_validation: Input validation, max sizes, url allowlists (user_url_allowed_hosts, provider_url_destination_allowed_hosts)
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/guides/security_settings
code: `litellm/proxy/litellm_pre_call_utils.py`, `litellm/proxy/auth/auth_utils.py`, `litellm/litellm_core_utils/url_utils.py`, `litellm/proxy/_types.py` (`user_url_allowed_hosts`, `provider_url_destination_allowed_hosts`)
tests: `tests/test_litellm/proxy/` (request validation files), `tests/e2e/access_control/`
registry: llm_conversational.yaml llm.chat_completions.openai.input_validation.*, llm_nonconversational.yaml llm.*.input_validation
verify: set `general_settings: {provider_url_destination_allowed_hosts: ["api.openai.com"]}`, send a request whose `model` is a URL pointing at another host, and confirm it is rejected; for media URLs use `user_url_allowed_hosts` with an image_url on a disallowed host instead

### security.compliance_checks: Compliance checks (/compliance)
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/data_security, https://docs.litellm.ai/docs/shared_responsibility
code: `litellm/proxy/management_endpoints/compliance_endpoints.py` (POST `/compliance/eu-ai-act`, `/compliance/gdpr`)
tests: `tests/test_litellm/proxy/management_endpoints/`
registry: mgmt.yaml mgmt.compliance.*
verify: POST /compliance/gdpr -H "Authorization: Bearer sk-1234" and confirm the report returns current config posture

### security.hardened_image: Hardened docker image, cosign signatures, SBOM
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/docker_image_security
code: `Dockerfile`, `cosign.pub`, `docker-compose.hardened.yml`
tests: none found
registry: none
verify: `cosign verify --key cosign.pub <image>` against the published image per the GitHub release verification block

### security.audit_trail: Audit logs, deleted keys/teams pages
surfaces: ui, api, config | flags: ent, db
docs: https://docs.litellm.ai/docs/proxy/deleted_keys_teams, https://docs.litellm.ai/docs/data_security
code: `enterprise/litellm_enterprise/proxy/audit_logging_endpoints.py` (GET `/audit`, `/audit/{id}`), `litellm/proxy/hooks/key_management_event_hooks.py`, `litellm/proxy/hooks/user_management_event_hooks.py`
tests: `tests/test_litellm/proxy/hooks/`
registry: none
verify: enable store_audit_logs, delete a key, then GET /audit and confirm the delete event is recorded; check the deleted keys page in the UI

### security.headers_and_cors: Security headers, CORS, trusted proxy ranges
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/guides/security_settings
code: `litellm/proxy/proxy_server.py` (CORSMiddleware, trusted proxy handling), `litellm/proxy/_types.py` (`trusted_proxy_ranges`, failed-login fields)
tests: `tests/test_litellm/proxy/`
registry: none
verify: send an OPTIONS preflight and a request with a forged X-Forwarded-For, and confirm CORS scope and real client IP resolution follow config

### security.ssrf_protection: SSRF protections on user-supplied URLs
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/guides/security_settings, https://docs.litellm.ai/docs/data_security
code: `litellm/litellm_core_utils/url_utils.py` (url allowlist checks), `litellm/proxy/litellm_pre_call_utils.py`
tests: `tests/test_litellm/proxy/` (ssrf related files)
registry: none
verify: attempt a request whose media/tool url resolves to a private IP and confirm the proxy refuses to fetch it

### security.encryption: Encryption of stored credentials (salt key)
surfaces: config | flags: db
docs: https://docs.litellm.ai/docs/proxy/security_encryption_faq, https://docs.litellm.ai/docs/proxy/security_best_practices
code: `litellm/proxy/management_endpoints/key_management_endpoints.py` (POST `/credentials/migrate-encryption`), `litellm/proxy/common_utils/` (encrypt/decrypt helpers used by stored credentials)
tests: `tests/test_litellm/proxy/management_endpoints/` (credential migration files)
registry: mgmt.yaml mgmt.credential_migration.*
verify: with `general_settings: {encryption_algorithm: aes-256-gcm}` set and the same `LITELLM_SALT_KEY` (or master key) in place that encrypted the values, store a credential, POST /credentials/migrate-encryption?dry_run=true to scan, then the real run, and confirm the credential still decrypts and works on a completion
