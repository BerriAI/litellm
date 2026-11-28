# Model catalog refresh

- `refresh` is the one way a host gets a newer document: it fetches from a `Source` and reads the body with `CostMap::read`. A host serves `CostMap::bundled()` first, calls `refresh` in the background, swaps on `Replaced` and keeps what it serves on `Unavailable` or `Rejected`. When to call it and where the live table lives are the host's job
- The rules every host shares live here and take typed values: `Remote::new` decides whether the remote document is off, `RetryPolicy::DEFAULT` is the retry schedule, and `Refreshed::or_kept` is the wording a host reports when nothing was installed. This crate never reads the environment or a config; each host resolves its values and passes them in
- `Source` is where a document comes from. `RemoteSource` is the one behind a URL: it fetches one document over http(s), `s3://` or `gs://` and never parses it
- A new scheme is a `Location` variant, its attempt in `attempt.rs` and one arm in `RemoteSource::fetch`, plus an `Access` field when a host has to configure it. `Source::fetch` and its callers do not change
- Only 429, 500, 502, 503, 504 and transport errors are retried. S3 errors are final
- `gs://` reuses the http attempt with a Google application default credentials bearer token, so its statuses retry like http. A token that cannot be acquired in time retries too, since a new GKE pod's metadata server can refuse the first requests. With `STORAGE_EMULATOR_HOST` set it sends no credentials, like Google's SDKs. The token never reaches a reason or a log line
- `Failed.retryable` means only the attempt limit ended the fetch
- A server's Retry-After overrides the backoff, capped at `max_delay`
- Error reasons use the URL with userinfo, query and fragment stripped, since presigned URLs carry credentials
- Bodies over `MAX_BODY_BYTES` are rejected
