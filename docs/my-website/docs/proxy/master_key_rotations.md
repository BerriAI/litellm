# Rotating Master Key

Here are our recommended steps for rotating your master key.


**1. Backup your DB**
In case of any errors during the encryption/de-encryption process, this will allow you to revert back to current state without issues.

**2. Call `/key/regenerate` with the new master key**

```bash
curl -L -X POST 'http://localhost:4000/key/regenerate' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
  "key": "sk-1234",
  "new_master_key": "sk-PIp1h0RekR"
}'
```

This will re-encrypt any models in your Proxy_ModelTable with the new master key.

Expect to start seeing decryption errors in logs, as your old master key is no longer able to decrypt the new values.

```bash
   raise Exception("Unable to decrypt value={}".format(v))
Exception: Unable to decrypt value=<new-encrypted-value>
```

**3. Update LITELLM_MASTER_KEY**

In your environment variables update the value of LITELLM_MASTER_KEY to the new_master_key from Step 2.

This ensures the key used for decryption from db is the new key.

**4. Test it**

Make a test request to a model stored on proxy with a litellm key (new master key or virtual key) and see if it works

```bash
 curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "gpt-4o-mini", # 👈 REPLACE with 'public model name' for any db-model
    "messages": [
        {
            "content": "Hey, how's it going",
            "role": "user"
        }
    ],
}'
```