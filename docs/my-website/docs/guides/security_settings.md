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

## 5. Use HTTP_PROXY environment variable

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

