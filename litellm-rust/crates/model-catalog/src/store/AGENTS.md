# Store

`Store<R: Rows>` is an in-memory store, a small database built for exactly three queries. It knows nothing about the catalog. A source picks its own `Rows::Row` type, how names fold for matching, and the group each key is listed under. The catalog's sources fold to lowercase and group by raw `litellm_provider`

| Query | Method | Real caller |
|---|---|---|
| row by exact key | `Store::get` | reading `litellm.model_cost["gpt-4o"]` |
| actual key for a name that folds to the same form | `Store::find_key` | `get_model_info("GPT-4o")` finding `gpt-4o` |
| keys in a group | `Store::keys_in` | the provider sets, `litellm.open_ai_chat_completion_models` |

## Rules

- nothing in `store/` imports from the rest of the crate. Wire types, case rules and field names such as `litellm_provider` belong to the source that implements `Rows`
- the index in `name_index.rs` is private to the store. Never build a folded-name map or a provider grouping anywhere else in the workspace. A new lookup pattern is a new `Store` method backed by `NameIndex`
- the index holds keys, never rows. Rows are always read from `Rows`, so a price edit is seen on the next read and never rebuilds the index
- `find_key` and `keys_in` never return a key the rows lack. A hit on a removed key rebuilds the index once and asks again
- `keys_in` matches the group exactly as the source wrote it. Provider aliases such as `vertex_ai` over `vertex_ai-language-models` are Python's job
- when two keys fold to the same name the later key in row order wins, like main's `_model_cost_lowercase_map`

## Stamps

`Rows::stamp` is how the store learns that the index may be stale. The store keeps the stamp it built the index at and rebuilds when the source returns a different one. A source must change its stamp whenever a key or a key's group changes, since those are the two things the index holds

| Source | Stamp | Blind spot |
|---|---|---|
| `Catalog`'s `CatalogRows` | `()`, every change goes through `Store::edit`, which drops the index | none |
| python-bridge's bound `litellm.model_cost` | the dict object and its `len` | a delete plus an add between two reads, or a `litellm_provider` rewrite on an existing key |

A source with a blind spot calls `Store::invalidate` when it knows keys changed. Until then a key added in the blind spot is not found, and a key moved to another provider stays listed under the old one. Removed keys are still never returned

## Tests

Tests live in `mod.rs`'s `tests` module and drive the store through `ModelCost`, a fake `litellm.model_cost` with real bundled keys whose row is just its `litellm_provider`. Each Python write is a method (`set`, `del`, `rebind`) and its stamp is what python-bridge can see. Add a scenario as a named test that reads like the Python user code it stands for

The two property tests compare every answer with a linear scan of the same dict. With a stamp that sees every key change, answers must match exactly. With the dict stamp, every answer must be sound (present, same folded name), and `invalidate` must make them match again. A store change that fails either is a bug in the store, not in the test
