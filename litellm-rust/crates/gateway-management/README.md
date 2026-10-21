# Gateway management

This crate scaffolds virtual key creation, retrieval, and revocation with an injected `KeyStore`. `gateway-auth::keys` owns the separate read-only `KeyLookup` interface and credential verification. Neither crate imports a database implementation

The future `db` key repository should implement both interfaces against the existing LiteLLM database. The gateway composition layer can inject the same repository into management and authentication, sharing its connection pool

## Storage contract

`create` inserts the supplied hash and expiration atomically, returning `AlreadyExists` without changing an existing record on collision. Management generates a token from 32 bytes of operating-system randomness and returns its plaintext only after storage succeeds. `KeyHash` uses the existing SHA-256 token hashing function. Storage never receives the plaintext token

`get` returns the stored status, including revoked keys. `revoke` returns `Revoked` for an existing key, including one already revoked, and `NotFound` for a missing key. Once revocation succeeds, every subsequent lookup must return revoked or missing, including lookups from other gateway instances. Database writes and any cache invalidation must preserve this contract

`KeyLookup` returns active status with optional expiration, revoked status, or no record. Backend failures remain errors. `verify_key` checks the current lookup result and expiration on every call, reads its supplied clock after lookup completes, and returns the verified hash. Pass `SystemTime::now` in production or a fixed clock in tests

## Scope

These are backend operations, with no HTTP routes, database adapter, production in-memory store, or gateway CLI wiring. The in-memory repository under `tests` exercises the shared storage contract. Verification establishes only that a credential is valid; it does not grant permissions. Caller identity, management authorization, and inference policy enforcement belong to the gateway auth integration before these operations are exposed over HTTP

The existing master-key authentication path is unchanged. User, team, model, and budget policies are not represented in this scaffold, so existing database rows carrying those policies require that integration before they can authorize traffic
