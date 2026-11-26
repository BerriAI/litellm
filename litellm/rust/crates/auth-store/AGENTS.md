# Auth store

Database reads for `gateway-auth-core`'s store ports. Each `*Database` reads one auth object through the `repositories` readers and shapes it into the record the auth cache holds for it, so no host shapes a row on its own

## Boundary

- No decision lives here. The core decides who may call what, and this crate only reads. A caller that must refuse on what a read found, such as a writer read whose object permission row is gone, gets the facts back and refuses itself
- No cache is touched here. The value is the family's typed record in `redis-store/src/auth_cache/types/`, and reading, writing and evicting the cache stay with the host
- No query is written here. A row comes from its repository
- It covers only objects the core asks a host for. The object permission is read here because a cached key and a cached team nest it
- Concrete structs, never a trait. A test drives them against real Postgres
- No cache key or byte format is defined here. `AuthObject` and each entry's `IntoValue`/`FromValue` live in `redis-store`, and a read returns the entry itself, so the host caches it with no conversion. `FromValue` is used here only on a JSON column already stored in an entry's shape, such as `members_with_roles`
- A row is refused only where the Python model that dumps its entry would refuse it, as `Error::Row` naming the column. Every other check belongs to the core
- Which reads make up an entry, and how a failed nested read degrades (dropped on a replica read, failed on a writer read), is decided here. Whether the caller may go ahead on the result is not

## The reads

Each object has one entry here, in the same form, and a later object adds its own. Nothing is returned for an id no row has

### Key

- Query: `KeyDatabase::key`, the key's `combined_view` row for a hash, one query
- Entry: `KeyEntry` under `AuthObject::Key`, the `UserAPIKeyAuth` dump
- Nested: the team member, as `Member` dumps it. `KeyDatabase::read` adds the object permission the key points at, for a host with no bound of its own to put between the two reads
  - The Python proxy bounds each read on its own: its lookup gate and deadline go around `RustKeyRepository.find`, which is `KeyDatabase::key`. It then nests the permission through its own `get_object_permission`, under that read's own deadline, and caches the key with `cache_key_object`. A permission table that stalls must leave that key served without the permission
  - A permission row that cannot be read or that its model would refuse leaves a replica read's key without it, as `get_key_object` did, so one bad permission never takes down a key that inference requests still use. A writer read fails on it
- Not read: the fields only a cache entry carries

### Team

- Query: `TeamDatabase::team`, the team's row with its model alias table (`find_with_model_aliases`), then the object permission it points at, two queries
- Query by alias: `TeamDatabase::team_by_alias`, up to two rows with that alias (`find_up_to_two_by_alias`), then the object permission of the team it finds, read from the replica, as `get_team_object` reads it. Every row is shaped before they are counted, as `get_team_object_by_alias` validates both before it refuses a shared alias, so the read is `AliasMatch::Found`, `Absent` or `Ambiguous`. Python caches the team it finds under `AuthObject::TeamAlias` and `AuthObject::Team`, which the host does
- Entry: `TeamEntry` under `AuthObject::Team`, the `LiteLLM_TeamTableCachedObj` dump
- Nested: the roster, each entry as `Member` dumps it, the model alias table as `LiteLLM_ModelTable` dumps it, each `budget_limits` window as `BudgetLimit` dumps it, without the keys the model drops, and the object permission, as `get_team_object` nests it. The permission is read through the key's policy in `object_permission.rs`, so a row that cannot be read or that its model refuses leaves a replica read's team without it and fails a writer read
- Not read: `last_refreshed_at`, which the host that caches the team stamps

## The cached shape

- What the database gives is not what the cache holds. Python's entry is `Model(**row).model_dump(mode="json")`, so each shaper converts its row field by field into the entry, and the entry dumps to the bytes Python writes for the same row
- Each object is pinned in three files that hold the same row and entry literals and change together: `tests/<object>.rs` reads the row from Postgres, `redis-store/tests/auth_cache_<object>.rs` round-trips the entry through Redis, and `tests/unit/redis/stores/test_auth_cache_<object>.py` checks both against Python's own model, reader and writer. An entry literal comes from running Python's model, never from hand. A field Python adds to the model fails there until it is named here
- Nothing the Python model would refuse is returned. A column takes only the shapes its field takes (`metadata` an object, `budget_fallbacks` lists of strings), and a row that breaks one fails the read with the column's name, as the model's validation did
  - A field the model types `X | None` is read with `nullable`, also when the SQL column is NOT NULL, since a JSON column can hold a JSON `null`. `column` is only for a field the model requires
  - Where a model's `mode="before"` validator runs `json.loads` on a string, as the team's `set_model_info` does for `metadata`, `model_max_budget`, `router_settings` and `budget_limits`, the shaper reads that field with `nullable_or_text`, since older writers stored `json.dumps` text in the column
  - A shaper names the column of its own row, and a caller that nests a row adds the field it sits under with `error::within`, so a key whose team member has a role the model does not take is refused as `team_member.role`. A roster entry that is no object or names neither a user id nor an email is refused as the field it sits under, `team_member` or `members_with_roles`, since the model refuses it as a whole

## Other writers

- Python still writes every entry this crate shapes, and pods on an older release fill each from the database themselves, so the entries stay a wire format Python owns. `cache_key_object` writes a key for every key the proxy reads or a management endpoint changes, and `get_team_object` shapes a team from the same repository row on its own and caches it through `cache_team_object`, so no host reads a team through this crate yet. Both write the bytes this crate shapes for the same row
- A writer whose bytes differ from this crate's is listed here with the difference, and is left alone on purpose
