
# ✨ IP Address Filtering

:::info

You need a LiteLLM License to unlock this feature. [Grab time](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat), to get one today!

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