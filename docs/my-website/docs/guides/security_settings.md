import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# SSL, HTTP Proxy Security Settings

If you're in an environment using an older TTS bundle, with an older encryption, follow this guide. By default
LiteLLM uses the certifi CA bundle for SSL verification, which is compatible with most modern servers.
 However, if you need to disable SSL verification or use a custom CA bundle, you can do so by following the steps below.

Be aware that environmental variables take precedence over the settings in the SDK.

LiteLLM uses HTTPX for network requests, unless otherwise specified.

## 1. Custom CA Bundle

You can set a custom CA bundle file path using the `SSL_CERT_FILE` environmental variable or passing a string to the the ssl_verify setting.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm
litellm.ssl_verify = "client.pem"
```
</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
litellm_settings:
  ssl_verify: "client.pem"
```

</TabItem>  
<TabItem value="env_var" label="Environment Variables">

```bash
export SSL_CERT_FILE="client.pem"
```
</TabItem>
</Tabs>

## 2. Disable SSL verification


<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm
litellm.ssl_verify = False
```
</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
litellm_settings:
  ssl_verify: false
```

</TabItem>  
<TabItem value="env_var" label="Environment Variables">

```bash
export SSL_VERIFY="False"
```
</TabItem>
</Tabs>

## 3. Lower security settings

The `ssl_security_level` allows setting a lower security level for SSL connections.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm
litellm.ssl_security_level = "DEFAULT@SECLEVEL=1"
```
</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
litellm_settings:
  ssl_security_level: "DEFAULT@SECLEVEL=1"
```
</TabItem>
<TabItem value="env_var" label="Environment Variables">

```bash
export SSL_SECURITY_LEVEL="DEFAULT@SECLEVEL=1"
```
</TabItem>
</Tabs>

## 4. Certificate authentication

The `SSL_CERTIFICATE` environmental variable or `ssl_certificate` attribute allows setting a client side certificate to authenticate the client to the server.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm
litellm.ssl_certificate = "/path/to/certificate.pem"
```
</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
litellm_settings:
  ssl_certificate: "/path/to/certificate.pem"
```
</TabItem>
<TabItem value="env_var" label="Environment Variables">

```bash
export SSL_CERTIFICATE="/path/to/certificate.pem"
```

</TabItem>
</Tabs>

## 5. Configure ECDH Curve for SSL/TLS Performance

The `ssl_ecdh_curve` setting allows you to configure the Elliptic Curve Diffie-Hellman (ECDH) curve used for SSL/TLS key exchange. This is particularly useful for disabling Post-Quantum Cryptography (PQC) to improve performance in environments where PQC is not required.

**Use Case:** Some OpenSSL 3.x systems enable PQC by default, which can slow down TLS handshakes. Setting the ECDH curve to `X25519` disables PQC and can significantly improve connection performance.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm
litellm.ssl_ecdh_curve = "X25519"  # Disables PQC for better performance
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
litellm_settings:
  ssl_ecdh_curve: "X25519"
```

</TabItem>  
<TabItem value="env_var" label="Environment Variables">

```bash
export SSL_ECDH_CURVE="X25519"
```

</TabItem>
</Tabs>

**Common Valid Curves:**

- `X25519` - Modern, fast curve (recommended for disabling PQC)
- `prime256v1` - NIST P-256 curve
- `secp384r1` - NIST P-384 curve
- `secp521r1` - NIST P-521 curve

**Note:** If an invalid curve name is provided or if your Python/OpenSSL version doesn't support this feature, LiteLLM will log a warning and continue with default curves.

## 6. Use HTTP_PROXY environment variable

Both httpx and aiohttp libraries use `urllib.request.getproxies` from environment variables. Before client initialization, you may set proxy (and optional SSL_CERT_FILE) by setting the environment variables:

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm
litellm.aiohttp_trust_env = True
```

```bash
export HTTPS_PROXY='http://username:password@proxy_uri:port'
```
</TabItem>

<TabItem value="proxy" label="PROXY">

```bash
export HTTPS_PROXY='http://username:password@proxy_uri:port'
export AIOHTTP_TRUST_ENV='True'
```
</TabItem>
</Tabs>
## 7. Per-Service SSL Verification

LiteLLM allows you to override SSL verification settings for specific services or provider calls. This is useful when different services (e.g., an internal guardrail vs. a public LLM provider) require different CA certificates.

### Bedrock (SDK)
You can pass `ssl_verify` directly in the `completion` call.

```python
import litellm

response = litellm.completion(
    model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    messages=[{"role": "user", "content": "hi"}],
    ssl_verify="path/to/bedrock_cert.pem" # Or False to disable
)
```

### AIM Guardrail (Proxy)
You can configure `ssl_verify` per guardrail in your `config.yaml`.

```yaml
guardrails:
  - guardrail_name: aim-protected-app
    litellm_params:
      guardrail: aim
      ssl_verify: "/path/to/aim_cert.pem" # Use specific cert for AIM
```

### Priority Logic
LiteLLM resolves `ssl_verify` using the following priority:
1. **Explicit Parameter**: Passed in `completion()` or guardrail config.
2. **Environment Variable**: `SSL_VERIFY` environment variable.
3. **Global Setting**: `litellm.ssl_verify` setting.
4. **System Standard**: `SSL_CERT_FILE` environment variable.
