---
title: Map baggage to request tags
---

Map values from the HTTP `baggage` header into LiteLLM request tags for spend tracking and tag budgets. Configure the rules in YAML or in **Settings → Router Settings → General → request_tag_mappings**.

## Exact values and variable values

Use `value` and `tag` for individual mappings. Use `tag_prefix` to append any value of a baggage key. Exact mappings take precedence over the any-value fallback for the same key, regardless of rule order.

```yaml
general_settings:
  request_tag_mappings:
    - header: baggage
      key: entrypoint
      value: a
      tag: usecase:support
    - header: baggage
      key: entrypoint
      value: b
      tag: usecase:sales
    - header: baggage
      key: entrypoint
      tag_prefix: "usecase:"
```

`baggage: entrypoint=a` adds `usecase:support`; `entrypoint=b` adds `usecase:sales`; `entrypoint=x` adds `usecase:x`. Omit the final rule to map only the listed values. Mapped tags merge with existing request tags without adding duplicates. Baggage values are percent-decoded according to W3C baggage rules, and baggage properties after a semicolon do not become part of the tag.

In the Admin UI, click **Add mapping**, select **Exact value** or **Any value**, enter the baggage key and target tag or prefix, then click **Update**. Save several exact rules, prefix rules, or both together. YAML-owned settings remain controlled by YAML; use the UI for settings stored in the database.

## Require a mapping or a budget

Both controls default to `false`:

```yaml
general_settings:
  request_tag_mappings:
    - key: entrypoint
      tag_prefix: "usecase:"
      required: true
      require_budgeted_tag: true
```

`required` rejects an inference request with HTTP 400 when its baggage key is missing, empty, repeated, invalid, or has no matching rule. Without it, an unmatched key adds no tag. When a key has several rules, making any of them required requires a match for that key.

`require_budgeted_tag` accepts only tags that already have a maximum budget. Unknown or unbudgeted tags return HTTP 400, and a disconnected database returns HTTP 503. Existing tag-budget checks enforce the mapped tags before the provider call; spending and budgets use the same request metadata. A required mapping does not apply to ordinary admin settings requests.

Baggage is supplied by the client. These mappings classify requests; they do not establish caller identity or permission.

Mappings apply to HTTP inference requests. While any mappings are configured, WebSocket inference is disabled, including Responses WebSocket, Realtime, and live sessions. Use the HTTP API for mapped spend tracking and tag budgets.
