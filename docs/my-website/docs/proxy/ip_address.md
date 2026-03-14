
# IP Address Filtering

:::info

You need a LiteLLM License to unlock this feature. [Grab time](https://calendly.com/d/cx9p-5yf-2nm/litellm-introductions), to get one today!

:::

Restrict which IP's can call the proxy endpoints.

```yaml
general_settings:
  allowed_ips: ["192.168.1.1"]
```

**Expected Response** (if IP not listed)

```bash
{
    "error": {
        "message": "Access forbidden: IP address not allowed.",
        "type": "auth_error",
        "param": "None",
        "code": 403
    }
}
```