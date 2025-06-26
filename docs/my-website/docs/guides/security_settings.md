import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# SSL, HTTP Proxy Security Settings

If you're in an environment using an older TTS bundle, with an older encryption, follow this guide. 


LiteLLM uses HTTPX for network requests, unless otherwise specified. 

## 1. Disable SSL verification


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

## 2. Lower security settings

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm
litellm.ssl_security_level = 1
litellm.ssl_certificate = "/path/to/certificate.pem"
```
</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
litellm_settings:
  ssl_security_level: 1
  ssl_certificate: "/path/to/certificate.pem"
```
</TabItem>
<TabItem value="env_var" label="Environment Variables">

```bash
export SSL_SECURITY_LEVEL="1"
export SSL_CERTIFICATE="/path/to/certificate.pem"
```
</TabItem>
</Tabs>

## 3. Use HTTP_PROXY environment variable

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

