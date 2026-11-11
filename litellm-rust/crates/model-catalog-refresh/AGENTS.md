# Model catalog refresh

- `fetch` fetches one document over http(s) or `s3://` and never parses it
- Only 429, 500, 502, 503, 504 and transport errors are retried. S3 errors are final
- `Failed.retryable` means only the attempt limit ended the fetch
- A server's Retry-After overrides the backoff, capped at `max_delay`
- `Start::After(failed)` resumes the backoff schedule where another caller's first attempt left off
- Error reasons use the URL with userinfo, query and fragment stripped, since presigned URLs carry credentials
- Bodies over `MAX_BODY_BYTES` are rejected
