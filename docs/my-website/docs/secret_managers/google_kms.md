# Google Key Management Service

:::info

✨ **This is an Enterprise Feature**

[Enterprise Pricing](https://www.litellm.ai/#pricing)

[Contact us here to get a free trial](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

:::

Use encrypted keys from Google KMS on the proxy

Step 1. Add keys to env 
```
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
export GOOGLE_KMS_RESOURCE_NAME="projects/*/locations/*/keyRings/*/cryptoKeys/*"
export PROXY_DATABASE_URL_ENCRYPTED=b'\n$\x00D\xac\xb4/\x8e\xc...'
```

Step 2: Update Config

```yaml
general_settings:
  key_management_system: "google_kms"
  database_url: "os.environ/PROXY_DATABASE_URL_ENCRYPTED"
  master_key: sk-1234
```

Step 3: Start + test proxy

```
$ litellm --config /path/to/config.yaml
```

And in another terminal
```
$ litellm --test 
```

[Quick Test Proxy](../proxy/user_keys)

