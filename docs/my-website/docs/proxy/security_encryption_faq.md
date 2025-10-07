# LiteLLM Self-Hosted Security & Encryption FAQ

## Data in Transit Encryption

### Does the product encrypt data in transit?

**Yes**, LiteLLM encrypts data in transit using TLS/SSL.

### Available in both OSS and Enterprise?

**Yes**, TLS encryption is available in both Open Source and Enterprise versions.

### In transit between the calling client and the product?

**Yes**, HTTPS/TLS is supported through SSL certificate configuration.

**Configuration:**
```bash
# CLI
litellm --ssl_keyfile_path /path/to/key.pem --ssl_certfile_path /path/to/cert.pem

# Environment Variables
export SSL_KEYFILE_PATH="/path/to/key.pem"
export SSL_CERTFILE_PATH="/path/to/cert.pem"
```

**Documentation Reference:** `docs/my-website/docs/guides/security_settings.md`

### In transit between the product and the LLM providers?

**Yes**, all connections to LLM providers use TLS encryption by default.

**Implementation Details:**
- Uses Python's `ssl.create_default_context()` 
- Leverages HTTPX and aiohttp libraries with SSL/TLS enabled
- Uses certifi CA bundle by default for SSL verification

**Code Reference:** `litellm/llms/custom_httpx/http_handler.py` (lines 43-105)

### Are TCP sessions to the LLM providers shared?

**Yes**, TCP connections are pooled and reused.

**Details:**
- Connection pooling is enabled by default
- Default: 1000 max concurrent connections with keepalive
- Sessions are maintained across requests to the same provider
- Reduces overhead of TLS handshakes

**Code Reference:** `litellm/llms/custom_httpx/http_handler.py` (lines 704-712)

### Or does the product negotiate a new TLS session with the same LLM provider for every sequential call?

**No**, TLS sessions are reused through connection pooling. New TLS handshakes are not performed for every request.

### How is it encrypted?

**TLS 1.2 and TLS 1.3**

Uses Python's default SSL context which supports both TLS 1.2 and TLS 1.3. The specific version negotiated depends on:
- Python version
- System SSL library (typically OpenSSL)
- Server capabilities

**Implementation:** `ssl.create_default_context()` in Python

### How are these added to the product's configuration?

#### x.509 Certificate

**Method 1: CLI Arguments**
```bash
litellm --ssl_certfile_path /path/to/certificate.pem
```

**Method 2: Environment Variable**
```bash
export SSL_CERTFILE_PATH="/path/to/certificate.pem"
```

#### Private Key

**Method 1: CLI Arguments**
```bash
litellm --ssl_keyfile_path /path/to/private_key.pem
```

**Method 2: Environment Variable**
```bash
export SSL_KEYFILE_PATH="/path/to/private_key.pem"
```

#### Certificate Bundle/Chain

**For client-to-proxy connections:**
Use standard SSL certificate setup with intermediate certificates bundled in the certfile.

**For proxy-to-LLM provider connections:**

**Method 1: Config YAML**
```yaml
litellm_settings:
  ssl_verify: "/path/to/ca_bundle.pem"
```

**Method 2: Environment Variable**
```bash
export SSL_CERT_FILE="/path/to/ca_bundle.pem"
```

**Method 3: Client Certificate Authentication**
```yaml
litellm_settings:
  ssl_certificate: "/path/to/client_certificate.pem"
```

or

```bash
export SSL_CERTIFICATE="/path/to/client_certificate.pem"
```

### Documentation Coverage

**Primary Documentation:**
- `docs/my-website/docs/guides/security_settings.md` - SSL/TLS configuration guide

**Additional References:**
- `litellm/proxy/proxy_cli.py` (lines 455-467) - CLI options
- `docs/my-website/docs/completion/http_handler_config.md` - Custom HTTP handler configuration

---

## Data at Rest Encryption

### Does the product encrypt data at rest?

**Partially**. Only specific sensitive data is encrypted at rest.

### What data is stored in encrypted form?

#### Encrypted Data:
1. **LLM API Keys** - Model credentials in `LiteLLM_ProxyModelTable.litellm_params`
2. **Provider Credentials** - Stored in `LiteLLM_CredentialsTable.credential_values`
3. **Configuration Secrets** - Sensitive config values in `LiteLLM_Config` table
4. **Virtual Keys** - When using secret managers (optional feature)

#### NOT Encrypted:
1. **Spend Logs** - Request/response data in `LiteLLM_SpendLogs`
2. **Audit Logs** - Change history in `LiteLLM_AuditLog`
3. **User/Team/Organization Data** - Metadata and configuration
4. **Cached Prompts and Completions** - Cache data is stored in plaintext

### Cached prompts and completions?

**No**, cached prompts and completions are **NOT encrypted**.

Cache backends (Redis, S3, local disk) store data as plaintext JSON.

**Code References:**
- `litellm/caching/redis_cache.py`
- `litellm/caching/s3_cache.py`
- `litellm/caching/caching.py`

### Configuration data?

**Partially encrypted**.

#### What IS Encrypted:
- LLM API keys and credentials in model configurations
- Sensitive values in `LiteLLM_Config` table
- Credential values in `LiteLLM_CredentialsTable`

#### What is NOT Encrypted:
- Model names and aliases
- Rate limits and budget settings
- User/team/organization metadata
- Non-sensitive configuration parameters

**Code Reference:** `litellm/proxy/management_endpoints/model_management_endpoints.py` (lines 275-308)

### Log data?

**No**, log data is **NOT encrypted**.

Log data stored in database tables is in plaintext:
- `LiteLLM_SpendLogs` - Contains request/response data, tokens, spend
- `LiteLLM_ErrorLogs` - Error information
- `LiteLLM_AuditLog` - Audit trail of changes

**Note:** You can disable logging to avoid storing sensitive data:

```yaml
general_settings:
  disable_spend_logs: True   # Disable writing spend logs to DB
  disable_error_logs: True   # Disable writing error logs to DB
```

**Documentation:** `docs/my-website/docs/proxy/db_info.md` (lines 52-60)

### Where is it stored?

#### In the DB?

**Yes**, encrypted data is stored in PostgreSQL database.

**Key Tables with Encrypted Data:**
- `LiteLLM_ProxyModelTable` - Model configurations with encrypted API keys
- `LiteLLM_CredentialsTable` - Credential values
- `LiteLLM_Config` - Configuration secrets

**Schema Reference:** `schema.prisma`

#### In the filesystem?

**No**, encrypted data is not stored in the filesystem by default.

**Note:** If using disk cache (`disk_cache_dir`), cached data is stored unencrypted.

#### Somewhere else?

**Optional:** When using secret managers (AWS Secrets Manager, Azure Key Vault, HashiCorp Vault), encrypted data can be stored externally.

**Configuration:**
```yaml
general_settings:
  key_management_system: "aws_secret_manager"  # or "azure_key_vault", "hashicorp_vault"
```

**Documentation:** `docs/my-website/docs/secret.md`

### How is it encrypted?

**Algorithm:** NaCl SecretBox (XSalsa20-Poly1305 AEAD)

**NOT AES-256** - LiteLLM uses NaCl (Networking and Cryptography Library) which provides:
- XSalsa20 stream cipher
- Poly1305 MAC for authentication
- Equivalent security to AES-256

**Key Derivation:**
1. Takes `LITELLM_SALT_KEY` (or `LITELLM_MASTER_KEY` if salt key not set)
2. Hashes with SHA-256 to derive 256-bit encryption key
3. Uses NaCl SecretBox for authenticated encryption

**Code Reference:** `litellm/proxy/common_utils/encrypt_decrypt_utils.py` (lines 69-112)

**Implementation:**
```python
import hashlib
import nacl.secret

# Derive 256-bit key from salt
hash_object = hashlib.sha256(signing_key.encode())
hash_bytes = hash_object.digest()

# Create SecretBox and encrypt
box = nacl.secret.SecretBox(hash_bytes)
encrypted = box.encrypt(value_bytes)
```

### Setting the Encryption Key

**Required Environment Variable:**
```bash
export LITELLM_SALT_KEY="your-strong-random-key-here"
```

**Important Notes:**
- ⚠️ **Must be set before adding any models**
- ⚠️ **Never change this key** - encrypted data becomes unrecoverable
- ⚠️ Use a strong random key (recommended: https://1password.com/password-generator/)
- If not set, falls back to `LITELLM_MASTER_KEY`

**Documentation:** `docs/my-website/docs/proxy/prod.md` (section 8, lines 184-196)

### Documentation Coverage

**Primary Documentation:**
- `docs/my-website/docs/proxy/prod.md` (section 8) - LITELLM_SALT_KEY setup
- `docs/my-website/docs/secret.md` - Secret management systems
- `docs/my-website/docs/proxy/db_info.md` - Database information

**Additional References:**
- `security.md` - General security measures
- `docs/my-website/docs/data_security.md` - Data privacy overview
- `schema.prisma` - Database schema with encrypted fields

---

## Summary of Security Features

### ✅ Provided Out of the Box

1. **TLS/SSL encryption** for client-to-proxy connections
2. **TLS encryption** for proxy-to-LLM provider connections (with connection pooling)
3. **Encrypted storage** of LLM API keys and credentials
4. **Support for TLS 1.2 and TLS 1.3**
5. **Connection pooling** to reduce TLS handshake overhead

### ⚠️ Important Limitations

1. **Cached data is NOT encrypted** (Redis, S3, disk cache)
2. **Log data is NOT encrypted** (spend logs, audit logs)
3. **Request/response payloads in logs are NOT encrypted**
4. **Uses NaCl SecretBox, NOT AES-256** (equivalent security)
5. **TLS version not explicitly configured** - uses Python/system defaults

### 🔧 Configuration Requirements

**For Production Deployments:**

1. **Set LITELLM_SALT_KEY** before adding any models
2. **Configure SSL certificates** for HTTPS client connections
3. **Consider disabling logs** if they contain sensitive data
4. **Use secret managers** for enhanced security (optional)
5. **Configure CA bundles** if using custom certificates

---

## Quick Start Security Checklist

```bash
# 1. Generate a strong salt key
export LITELLM_SALT_KEY="$(openssl rand -base64 32)"

# 2. Set up SSL certificates (for HTTPS)
export SSL_KEYFILE_PATH="/path/to/private_key.pem"
export SSL_CERTFILE_PATH="/path/to/certificate.pem"

# 3. Configure database
export DATABASE_URL="postgresql://user:password@host:port/dbname"

# 4. (Optional) Disable logs if they contain sensitive data
# Add to config.yaml:
# general_settings:
#   disable_spend_logs: True
#   disable_error_logs: True

# 5. Start LiteLLM Proxy
litellm --config config.yaml
```

---

## Additional Resources

- **LiteLLM Documentation:** https://docs.litellm.ai/
- **Security Settings Guide:** https://docs.litellm.ai/docs/guides/security_settings
- **Production Deployment:** https://docs.litellm.ai/docs/proxy/prod
- **Secret Management:** https://docs.litellm.ai/docs/secret

For security inquiries: support@berri.ai

