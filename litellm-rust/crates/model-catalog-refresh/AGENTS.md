# Model catalog refresh

`litellm-model-catalog-refresh` fetches a cost map document from an HTTP URL, a `file://` path or an `s3://` object and retries it by main's status rule. It returns bytes and never parses them, so every rule about what a map contains stays in `litellm-model-catalog`

`Fetch` is one source, `SourceFetch` picks it from a location, and `refresh` / `resume` drive the attempts through an injected `Pause`. They return an `Attempt` that is never `Retryable`. S3 clients come from `litellm-storage-s3`
