Own storage-independent trace reads over `TraceStore`: the in-process read cache (identity, single-flight, freshness expiry, weighting), cursor formats, paging, response splitting, spend windows and run batching
Depend on trace domain types, never storage, HTTP or Python
Preserve the full source and authorization scope in every cache key
Keep snapshots immutable and expose borrowed data
Storage adapters implement `TraceStore`. Keep SQL and row encoding there
