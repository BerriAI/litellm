# Model catalog refresh

- `fetch` fetches one document over http(s), `s3://` or `gs://` and never parses it
- Only 429, 500, 502, 503, 504 and transport errors are retried. S3 errors are final
- `gs://` reuses the http attempt with a Google application default credentials bearer token, so its statuses retry like http. A token that cannot be acquired in time retries too, since a new GKE pod's metadata server can refuse the first requests. With `STORAGE_EMULATOR_HOST` set it sends no credentials, like Google's SDKs. The token never reaches a reason or a log line
- `Failed.retryable` means only the attempt limit ended the fetch
- A server's Retry-After overrides the backoff, capped at `max_delay`
- `Start::After(failed)` resumes the backoff schedule where another caller's first attempt left off
- Error reasons use the URL with userinfo, query and fragment stripped, since presigned URLs carry credentials
- Bodies over `MAX_BODY_BYTES` are rejected
- `Cache` keeps one source's document between processes: one file per URL, named by a hash of the URL, with the ETag on the first line and the body after it. `store` writes through a temp file and one rename. `load` serves nothing older than `CACHE_MAX_AGE` since the last `store` or `revalidated`, read from the file's modification time. It never parses the document, so the caller checks it like a fetched one
- `if_none_match` is the ETag of the document the caller already holds. A 304 to it is `Fetched::NotModified` on http(s), `s3://` and `gs://`. A 304 to a request that sent no ETag stays a document, and an ETag no header can carry is not sent
- An attempt never sets `Accept-Encoding`. The client does, the same on every request, so a conditional fetch negotiates the encoding of the fetch that recorded its ETag. GitHub's raw host gives each encoding its own ETag and answers 200 when they differ
