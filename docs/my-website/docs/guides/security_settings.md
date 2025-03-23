import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# SSL Security Settings

If you're in an environment using an older TTS bundle, with an older encryption, follow this guide. 


LiteLLM uses HTTPX for network requests, unless otherwise specified. 

1. Disable SSL verification


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

2. Lower security settings

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


