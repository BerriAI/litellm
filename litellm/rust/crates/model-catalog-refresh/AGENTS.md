# Model catalog refresh

- `refresh` is the one way a host gets a newer document: it fetches from a `Source` and reads the body with `CostMap::read`. On `Replaced` the host installs the document, and on `Unavailable` or `Rejected` it keeps what it serves. When to call it and where the live table lives are the host's job
- `Source` is where a document comes from. `RemoteSource` is the one behind a URL: it fetches one document over http(s), `s3://` or `gs://` and never parses it
- A new scheme is a `Location` variant, its attempt in `attempt.rs` and one arm in `RemoteSource::fetch`, plus an `Access` field when a host has to configure it. `Source::fetch` and its callers do not change
- Only 429, 500, 502, 503, 504 and transport errors are retried. S3 errors are final
- `gs://` reuses the http attempt with a Google application default credentials bearer token, so its statuses retry like http. A token that cannot be acquired in time retries too, since a new GKE pod's metadata server can refuse the first requests. With `STORAGE_EMULATOR_HOST` set it sends no credentials, like Google's SDKs. The token never reaches a reason or a log line
- `Failed.retryable` means only the attempt limit ended the fetch
- A server's Retry-After overrides the backoff, capped at `max_delay`
- `RemoteSource::resuming(failed)` resumes the backoff schedule where another caller's first attempt left off
- Error reasons use the URL with userinfo, query and fragment stripped, since presigned URLs carry credentials
- Bodies over `MAX_BODY_BYTES` are rejected
