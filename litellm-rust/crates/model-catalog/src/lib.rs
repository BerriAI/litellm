//! The model catalog: per-model pricing, context limits and capabilities.
//!
//! This is the Rust form of `model_prices_and_context_window.json`, the map Python reads as
//! `litellm.model_cost` and through `litellm.get_model_info`. It answers "what does this model
//! cost and what can it do"; it knows nothing about configured deployments. The router's view of
//! deployments (groups, aliases, wildcard patterns, team mappings) is the router registry, defined
//! in `litellm-router-core`, which reads this catalog through its dynamic config.
//!
//! # Structure
//! - `Catalog`: the parsed map. `Catalog::parse` takes the JSON body and returns `Loaded<Catalog>`,
//!   keeping every entry in file order with its raw fields and collecting row and alias errors.
//! - `ModelEntry`: one model. `ModelEntry::fields` is the raw JSON object, so unknown fields are
//!   never lost; `ModelEntry::model` is the optional typed `Model` with attributes, prices,
//!   adjustments and capabilities. Invalid rows retain their raw fields without a typed model.
//! - Aliases: an entry may list `aliases`; each alias resolves to its canonical key. Problems
//!   found while indexing them (bad lists, collisions) are kept in `Loaded::errors` instead of
//!   failing the document.
//! - `FallbackGeneralizations` / `FallbackRule`: regex rules that map unknown model ids onto known
//!   families. They are data in the file, not entries.
//! - Validation: `Catalog::check_size` checks a catalog against `IntegrityLimits` (a remote copy
//!   that lost most of its models is refused), and `Catalog::row_errors` validates one row.
//!
//! # Lookups
//! `Catalog::lookup` resolves a key exactly first (entry or alias), then case-insensitively, and
//! returns a `ModelMatch` with both the key that matched and the canonical key of the entry. Provider
//! prefixes are not stripped here; callers decide which keys to try.
//!
//! # Readers
//! Cost tracking, the router (through `RouterDynamicConfig`) and the gateway. Nothing in this crate
//! performs I/O; loading and refreshing the file is the caller's job.

macro_rules_attribute::attribute_alias! {
    #[apply(wire_type)] =
        #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))];
}

mod bundled;
mod catalog;
mod error;
mod fallback;
mod layers;
mod lower;
mod model;
mod model_info;
mod register;
mod resolve;
mod store;
mod vocabulary;
pub mod wire;

pub use bundled::*;
pub use catalog::*;
pub use error::*;
pub use fallback::*;
pub use layers::*;
pub use model::*;
pub use model_info::*;
pub use register::*;
pub use resolve::*;
pub use store::*;
pub use vocabulary::*;

#[cfg(feature = "schema")]
mod schema;
#[cfg(feature = "schema")]
pub use schema::*;
