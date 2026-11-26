# Auth store

Database reads for `gateway-auth-core`'s store ports. `KeyDatabase` reads an auth object through the `repositories` readers and shapes it into the record the auth cache holds for it. Every host of the core reads a key's row through this crate, so no host shapes the row on its own

## Boundary

- No decision lives here. The core decides who may call what, and this crate only reads. A caller that must refuse on what a read found, such as a writer read whose object permission row is gone, gets the facts back and refuses itself
- No cache is touched here. The value is the family's typed record in `redis-store/src/auth_cache/types/`, and reading, writing and evicting the cache stay with the host
- No query is written here. A row comes from its repository
- It covers only objects the core asks a host for. The object permission is read here because a cached key nests it
- Concrete structs, never a trait. A test drives them against real Postgres
- No cache key or byte format is defined here. `AuthObject` and each entry's `IntoValue`/`FromValue` live in `redis-store`, and a read returns the entry itself, so the host caches it with no conversion. `FromValue` is used here only on a JSON column already stored in an entry's shape, such as the key's `team_member`
- A row is refused only where the Python model that dumps its entry would refuse it, as `Error::Row` naming the column. Every other check belongs to the core
- Which reads make up an entry, and how a failed nested read degrades (dropped on a replica read, failed on a writer read), is decided here. Whether the caller may go ahead on the result is not

## The read

- `KeyDatabase::key` is the key's one query, shaped into `KeyEntry` with nothing nested. `KeyDatabase::read` adds the object permission it points at, for a host with no bound of its own to put between the two reads
  - The Python proxy bounds each read on its own: its lookup gate and deadline go around `RustKeyRepository.find`, which is `KeyDatabase::key`. It then nests the permission through its own `get_object_permission`, under that read's own deadline, and caches the key with `cache_key_object`. A permission table that stalls must leave that key served without the permission
- Nothing is returned for an id no row has

## The cached shape

- What the database gives is not what the cache holds. A key is cached as Python's merged `UserAPIKeyAuth` dump, so `key.rs` converts the `combined_view` row field by field into `KeyEntry`
- The entry is a wire format Python still owns: `cache_key_object` writes it for every key the proxy reads or a management endpoint changes, and pods on an older release fill it from the database themselves. So the record this crate shapes for a row dumps to the bytes `main`'s `get_key_object` wrote for it. `tests/key.rs`, `redis-store/tests/auth_cache_key.rs` and `tests/unit/redis/stores/test_auth_cache_key.py` hold the same row and entry literals and are changed together, so a field Python adds to `UserAPIKeyAuth` fails there until it is named here
- Nothing the Python model would refuse is returned. A column takes only the shapes its field takes (`metadata` an object, `budget_fallbacks` lists of strings), and a row that breaks one fails the read with the column's name, as the model's validation did
  - A shaper names the column of its own row, and a caller that nests a row adds the field it sits under with `error::within`, so a key whose team member has a role the model does not take is refused as `team_member.role`. A roster entry that is no object or names neither a user id nor an email is refused as the field it sits under, `team_member`, since the model refuses it as a whole
  - The object permission is the exception `get_key_object` made: a permission row that cannot be read or that its model would refuse leaves a replica read's key without it, so one bad permission never takes down a key that inference requests still use. A writer read fails on it
