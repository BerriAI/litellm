Own the shared page envelope, traversal metadata, opaque signed cursor codec and stable pagination failure codes
Depend on nothing domain specific: never trace types, ClickHouse, HTTP or Python
Keep cursor contents limited to a typed position, revision, publication instant, format version, key id and expiry; never credentials or raw filters
Bind every cursor to its resource, authorization scope and query through the signature, and expose revision mismatches as `traversal_changed`
Test the codec and page bounds through the public API under `tests/`
