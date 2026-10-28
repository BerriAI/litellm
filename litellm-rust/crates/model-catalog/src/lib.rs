//! The model catalog: per-model pricing, context limits and capabilities.
//!
//! This is the Rust form of `model_prices_and_context_window.json`, the map Python reads as
//! `litellm.model_cost` and through `litellm.get_model_info`. It answers "what does this model
//! cost and what can it do"; it knows nothing about configured deployments. The router's view of
//! deployments (groups, aliases, wildcard patterns, team mappings) is the router registry, defined
//! in `litellm-router-core`, which reads this catalog through its dynamic config.
//!
//! # Structure
//! - `Catalog`: the parsed map. `Catalog::parse` takes the JSON body and a `Provenance` (where
//!   the body came from: the bundled copy in the repo or the remote copy, with revision and etag),
//!   and keeps every entry in file order with its raw fields.
//! - `ModelEntry`: one model. `ModelEntry::fields` is the raw JSON object, so unknown fields are
//!   never lost; `ModelEntry::info` is the typed mirror `ModelInfo` (pricing in `pricing`,
//!   modalities and modes in `capabilities`).
//! - Aliases: an entry may list `aliases`; each alias resolves to its canonical key. Problems
//!   found while indexing them (bad lists, collisions) are kept as `AliasIssue`s instead of
//!   failing the parse.
//! - `FallbackGeneralizations` / `FallbackRule`: regex rules that map unknown model ids onto known
//!   families. They are data in the file, not entries.
//! - Validation: `Catalog::validate` checks a catalog against `IntegrityLimits` (a remote copy that
//!   lost most of its models is refused), and `validate_registry` checks every entry against the
//!   checked-in contract.
//!
//! # Lookups
//! `Catalog::lookup` resolves a key exactly first (entry or alias), then case-insensitively, and
//! returns a `ModelMatch` with both the key that matched and the canonical key of the entry. Provider
//! prefixes are not stripped here; callers decide which keys to try.
//!
//! # Readers
//! Cost tracking, the router (through `RouterDynamicConfig`) and the gateway. Nothing in this crate
//! performs I/O; loading and refreshing the file is the caller's job.

mod capabilities;
mod catalog;
mod error;
mod fallback;
mod model_info;
mod pricing;
mod validation;

pub use capabilities::*;
pub use catalog::*;
pub use error::*;
pub use fallback::*;
pub use model_info::*;
pub use pricing::*;
pub use validation::*;

#[cfg(feature = "schema")]
mod schema;
#[cfg(feature = "schema")]
pub use schema::*;
