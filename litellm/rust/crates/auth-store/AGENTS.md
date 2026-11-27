# Auth store

Database reads and writes for `gateway-auth-core`'s store ports. Each `*Database` reads or writes one auth object through its `repositories` repository and shapes the row into the record the auth cache holds for it, so no host shapes a row on its own

## Boundary

- It reads and writes the objects the core asks a host for, and nothing else. A write returns the entry for the row as stored, which is what a host caches after it, and a write that can lose to a concurrent one returns the entry for the row that won
- No decision lives here. The core decides who may call what and whether a write happens, and this crate only carries it out. A caller that must refuse on what a read found, such as a writer read whose object permission row is gone, gets the facts back and refuses itself
- No cache is touched here. The value is the family's typed record in `redis-store/src/auth_cache/types/`, and reading, writing and evicting the cache stay with the host
- No query is written here. A row comes from its repository
- It covers only objects the core asks a host for. The object permission is read here because a cached key, a cached team and a cached end user nest it
- Concrete structs, never a trait. A test drives them against real Postgres
- No cache key or byte format is defined here. `AuthObject` and each entry's `IntoValue`/`FromValue` live in `redis-store`, and a read returns the entry itself, so the host caches it with no conversion. `FromValue` is used here only on a JSON column already stored in an entry's shape, such as `members_with_roles`
- A row is refused only where the Python model that dumps its entry would refuse it, as `Error::Row` naming the column. Every other check belongs to the core
- Which reads make up an entry, and how a failed nested read degrades (dropped on a replica read, failed on a writer read), is decided here. Whether the caller may go ahead on the result is not

## The reads

Each object has one entry here, its writes included, in the same form, and a later object adds its own. Nothing is returned for an id no row has

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

### Project

- Query: `ProjectDatabase::project`, the project's row with its budget (`find_with_budget`), one query
- Entry: `ProjectEntry` under `AuthObject::Project`, the `LiteLLM_ProjectTableCachedObj` dump
- Nested: the budget, as `LiteLLM_BudgetTable` dumps it (`BudgetEntry`). A budget whose `budget_limits` Python finds truthy is refused, as the model's `reject_concurrent_windows` does. An object whose model nests a different budget class, such as a team membership's `LiteLLM_BudgetTableFull`, has its own record
- Not read: the object permission the project points at, and `last_refreshed_at`, which the host that caches the project stamps

### Team membership

- Query: `TeamMembershipDatabase::team_membership`, the membership's row with its budget (`find_with_budget`), one query
- Entry: `TeamMembershipEntry` under `AuthObject::TeamMembership` and `AuthObject::TeamMembershipReservation`, the `LiteLLM_TeamMembership` dump. `AuthObject::TeamMembershipReservation` may instead hold `"__no_team_membership__"` (`NO_TEAM_MEMBERSHIP_SENTINEL`), which `_fetch_team_membership_from_db` caches for a user who is not a member, so a typed read of that key reads the sentinel first
- Nested: the budget, as `LiteLLM_BudgetTableFull` dumps it (`BudgetFullEntry`, the `BudgetEntry` with `budget_reset_at` and `created_at` after it). The model types it `LiteLLM_BudgetTableFull | LiteLLM_BudgetTable`, and pydantic reads every budget row as the full one since none lacks a `created_at`. It is refused for what the project's budget is refused for
- Not read: the user and the team the membership names

### End user

- Query: `EndUserDatabase::end_user`, the end user's row with its budget and object permission (`find_with_budget_and_object_permission`), one query
- Entry: `EndUserEntry` under `AuthObject::EndUser`, the `LiteLLM_EndUserTable` dump
- Nested: the budget, as `LiteLLM_BudgetTable` dumps it (`BudgetEntry`), refused for what the project's budget is refused for, and the object permission, as `LiteLLM_ObjectPermissionTable` dumps it (`ObjectPermissionEntry`). The query nests the permission row with `to_jsonb`, read as the same `ObjectPermission` a key's or a team's permission read returns, so a NULL array column is an empty list on every path and one shaper in `object_permission.rs` serves them all
- Not read: the proxy-wide default end-user budget, which the host merges in for an end user with no budget of its own
- Host: `get_end_user_object` turns every failure into no end user, so a host serving Python's behavior maps `Error::Row` and `Error::Database` to "no end user" rather than failing the request

### Organization

- Query: `OrganizationDatabase::organization`, the organization's row, joined to its budget when the caller asks for it (`find_with_optional_budget`), one query
- Entry: `OrganizationEntry` under `AuthObject::OrgWithBudget` with the budget and `AuthObject::Org` without it, the `LiteLLM_OrganizationTable` dump. `get_org_object_by_alias` caches the same budget-less shape under `AuthObject::OrgAlias` too, and `get_org_object_for_request` copies the with-budget entry to `AuthObject::OrgWithBudgetLastKnown`, which has no read of its own
- Nested: the budget when asked for, as `LiteLLM_BudgetTable` dumps it (`BudgetEntry`), refused for what the project's budget is refused for
- Not read: the organization by alias, which a miss under `AuthObject::OrgAlias` needs, with the 404 and 400 `get_org_object_by_alias` answers for no row and a shared alias, the object permission the organization points at, and `users`, which the readers above never set

### User

- Query: `UserDatabase::user`, the user repository's resolve of a sign-in that carries only the user id (`resolve`), the row with its organization memberships ordered by organization id, one query. With no SSO id or email it has nothing to fall back to or link, so it only reads
- Query by sign-in: `UserDatabase::sign_in`, the resolve of a sign-in that carries the id, the SSO id and the email (`resolve_sign_in`), by id, then by SSO id, then by email, linking the SSO id as `get_user_object` does. It returns `SignedIn`: the entry, `SsoLink::Linked` when the resolve stored the SSO id before returning and `SsoLink::Pending` when it left the link to the background under `links_any_user`, and `owes_email` when the sign-in carries an email the row lacks, which `_backfill_null_user_email` stores. The entry is the row as the resolve read it, which Python caches, so a user that just had its own id linked as its SSO id still holds none, while an email match linked in line holds the new SSO id. How often the host backfills (`email_in_use_checked_at`) stays with the host
- Entry: `UserEntry` under `AuthObject::User`, the `LiteLLM_UserTable` dump. `get_user_object` caches it for the row it read, for the row its upsert stored and for that row with a backfilled email, and the Python auth host recaches it after a role or team change. The password is never dumped
- Nested: each organization membership, as `LiteLLM_OrganizationMembershipTable` dumps it (`OrganizationMembershipEntry`). The model requires `spend` and both times, which the columns let be NULL, so a membership missing one refuses the whole user as `organization_memberships.<column>`. The model types the membership's `user` as any object, so it stays plain data, while an organization's `users` is a list of `UserEntry`
- Writes: `UserDatabase::insert` creates the user under `user_id_upsert` with the `NewUserDefaults` the host resolved from `default_internal_user_params` (`UserRepository::insert`), and returns `Inserted` with the entry for the row that won a concurrent sign-in. `UserDatabase::backfill_email` stores a sign-in's email on a row with none (`backfill_email_returning`), and `UserDatabase::set_role` sets the role the JWT role sync maps (`set_role`). Both return the entry for the row as stored. Python caches the row it already held with the email copied in, so its `updated_at` is the one before the backfill. Joining `default_internal_user_params.teams` after an insert is not written here
- Not read: the object permission the user points at, and each membership's budget and user

## The cached shape

- What the database gives is not what the cache holds. Python's entry is `Model(**row).model_dump(mode="json")`, so each shaper converts its row field by field into the entry, and the entry dumps to the bytes Python writes for the same row
- Each object is pinned in three files that hold the same row and entry literals and change together: `tests/<object>.rs` reads the row from Postgres, `redis-store/tests/auth_cache_<object>.rs` round-trips the entry through Redis, and `tests/unit/redis/stores/test_auth_cache_<object>.py` checks both against Python's own model, reader and writer. An entry literal comes from running Python's model, never from hand. A field Python adds to the model fails there until it is named here
- Nothing the Python model would refuse is returned. A column takes only the shapes its field takes (`metadata` an object, `budget_fallbacks` lists of strings), and a row that breaks one fails the read with the column's name, as the model's validation did
  - A field the model types `X | None` is read with `nullable`, also when the SQL column is NOT NULL, since a JSON column can hold a JSON `null`. `column` is only for a field the model requires
  - Where a model's `mode="before"` validator runs `json.loads` on a string, as the team's `set_model_info` does for `metadata`, `model_max_budget`, `router_settings` and `budget_limits`, the shaper reads that field with `nullable_or_text`, since older writers stored `json.dumps` text in the column
  - A shaper names the column of its own row, and a caller that nests a row adds the field it sits under with `error::within`, so a key whose team member has a role the model does not take is refused as `team_member.role`. A roster entry that is no object or names neither a user id nor an email is refused as the field it sits under, `team_member` or `members_with_roles`, since the model refuses it as a whole

## Other writers

- Python still writes every entry this crate shapes, and pods on an older release fill each from the database themselves, so the entries stay a wire format Python owns. `cache_key_object` writes a key for every key the proxy reads or a management endpoint changes, `get_team_object` shapes a team from the same repository row on its own and caches it through `cache_team_object`, `get_project_object` does the same for a project, the team member budget check and `get_team_membership` do it for a membership, `get_end_user_object` does it for an end user, `get_org_object` and `get_org_object_by_alias` do it for an organization, and `get_user_object` does it for a user. So no host reads a team, a project, a membership, an end user, an organization or a user through this crate yet. All of them write the bytes this crate shapes for the same row
- A writer whose bytes differ from this crate's is listed here with the difference, and is left alone on purpose
  - `get_end_user_object` caches an end user with no budget of its own with the proxy-wide default end-user budget (`litellm.max_end_user_budget_id`) in `litellm_budget_table` and `budget_id` still unset, so the budget it caches is not always the one the row names. `EndUserEntry` carries either budget, and the merge stays with the host
  - The auth prefetch (`auth_object_prefetch.py`) writes `project_id:{id}`, `{team_id}_{user_id}`, `team_membership:{user_id}:{team_id}`, `org_id:{id}`, `org_id:{id}:with_budget` and the bare user id from the key owners row instead, with budget and user times without a `Z`, a user's memberships in no set order and `[]` for a NULL budget `allowed_models`, the project's and the organization's object permission nested, and the organization's budget nested under `org_id:{id}` too, so its bytes differ from this shape for the same row
