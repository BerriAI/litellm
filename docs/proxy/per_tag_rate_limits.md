import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Per-Tag Rate Limits

Set independent RPM (requests per minute) limits for each request tag on a single API key. Each tag gets its own counter, so one tag hitting its limit does not affect other tags or untagged requests on the same key.

This is useful when multiple teams, cells, or services share a single key but each needs its own rate budget.

## Pre-Requisites

- A running LiteLLM proxy with a [connected Postgres database](../proxy/virtual_keys.md)
- Virtual keys enabled

## How It Works

When you create or update a key with `tag_rpm_limit`, the proxy tracks a separate RPM counter for each tag. For example, setting `{"cell-1": 5, "cell-2": 10}` means requests tagged `cell-1` are limited to 5 RPM and requests tagged `cell-2` are limited to 10 RPM, independently. Requests with a tag that has no configured limit fall back to the key-level `rpm_limit`.

## Create a Key with Per-Tag Rate Limits

### API

Pass `tag_rpm_limit` when generating a key. Each entry maps a tag name to its RPM cap.

```shell
curl -X POST 'http://0.0.0.0:4000/key/generate' \
     -H 'Authorization: Bearer sk-1234' \
     -H 'Content-Type: application/json' \
     -d '{
            "tag_rpm_limit": {"cell-1": 5, "cell-2": 10}
         }'
```

You can also set a key-level `rpm_limit` alongside `tag_rpm_limit`. Tags without a configured limit fall back to the key-level limit.

```shell
curl -X POST 'http://0.0.0.0:4000/key/generate' \
     -H 'Authorization: Bearer sk-1234' \
     -H 'Content-Type: application/json' \
     -d '{
            "rpm_limit": 100,
            "tag_rpm_limit": {"cell-1": 5, "cell-2": 10}
         }'
```

### UI

In the LiteLLM dashboard, go to Virtual Keys and click **+ Create New Key**. Expand **Optional Settings** and scroll to the **Per-Tag Rate Limits** section. Click **+ Add Tag Limit** to add tag/RPM pairs.

<Image img={require('../img/per_tag_rate_limits_create_key.png')} />

## Update Per-Tag Rate Limits

Use `/key/update` to change tag rate limits on an existing key.

```shell
curl -X POST 'http://0.0.0.0:4000/key/update' \
     -H 'Authorization: Bearer sk-1234' \
     -H 'Content-Type: application/json' \
     -d '{
            "key": "sk-your-key-here",
            "tag_rpm_limit": {"cell-1": 20, "cell-2": 50, "cell-3": 5}
         }'
```

## Sending Tagged Requests

Tag requests using the `x-litellm-tags` header (comma-separated) or include `tags` in the request metadata.

<Tabs>
<TabItem value="header" label="Header">

```shell
curl -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
     -H 'Authorization: Bearer sk-your-key-here' \
     -H 'Content-Type: application/json' \
     -H 'x-litellm-tags: cell-1' \
     -d '{
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}]
         }'
```

</TabItem>
<TabItem value="metadata" label="Request Body Metadata">

```shell
curl -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
     -H 'Authorization: Bearer sk-your-key-here' \
     -H 'Content-Type: application/json' \
     -d '{
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "metadata": {"tags": ["cell-1"]}
         }'
```

</TabItem>
</Tabs>

## Rate Limit Behavior

When a tag exceeds its RPM limit, the proxy returns a `429 Too Many Requests` response for that tag. Other tags on the same key continue working normally.

```json
{
  "error": {
    "message": "Rate limit exceeded for tag 'cell-1' on this key. Limit: 5 RPM",
    "type": "rate_limit_error",
    "code": "429"
  }
}
```

Tags without a configured limit in `tag_rpm_limit` fall back to the key-level `rpm_limit`. If no key-level limit is set either, those requests are not rate-limited by tag.
