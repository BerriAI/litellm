# Gateway catalog pagination

Tools, prompts, resources, and resource templates retain upstream page boundaries. Follow the gateway's opaque `nextCursor` until it is absent. Each request checks the caller's current access and tool permissions. A cursor can be replayed and can be sent to another replica serving the same configuration

Set the same nonempty `LITELLM_SALT_KEY` on every replica. Pagination state uses authenticated encryption with purpose-separated HKDF keys derived only from this value. The master key is never a fallback. Complete single-page lists and direct tool calls work without a salt key; a listing that needs continuation returns an actionable configuration error instead of truncated results. Some SDKs automatically list tools when validating a tool-call response, so those SDK calls also require a salt when that listing is paginated

Cursors expire ten minutes after the first page. Continuations do not extend that deadline. Rotating the salt invalidates all outstanding cursors; clients must start a new listing. During a rolling key change, replicas with different keys cannot accept each other's cursors. Coordinate the change across the fleet

A changed registry, caller scope, or available upstream revision requires a fresh listing. When an upstream exposes a string or integer `_meta.revision`, subsequent pages must retain it. Otherwise consistency follows that upstream's own cursor guarantees. Repeated upstream cursors and the existing upstream page limit stop traversal with an explicit error

Listing failures retain per-server outcome metadata. An incomplete upstream catalog cannot establish a bare tool-name route. Use the server-prefixed names returned by the gateway
