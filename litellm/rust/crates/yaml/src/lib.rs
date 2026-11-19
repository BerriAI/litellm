//! PyYAML's `yaml.safe_load` for LiteLLM config files, so the Rust proxy reads a config the way the
//! Python proxy does: YAML 1.1 booleans (`yes`, `on`), integers (`0755`, `1_000`, `1:30`), floats
//! that need a `.`, merge keys and repeated keys. `AGENTS.md` says what is in scope.
//!
//! saphyr-parser scans and parses. This crate ports PyYAML's composer, resolver and safe
//! constructor on top of it, producing the `serde_yaml_ng::Value` the config types deserialize.

mod compose;
mod construct;
mod error;
mod resolve;

use serde_yaml_ng::Value;

pub use compose::Tag;
pub use error::{Error, Location};

/// `yaml.safe_load(text)`: `Null` for an empty document, an error where PyYAML raises.
pub fn from_str(text: &str) -> Result<Value, Error> {
    compose::compose(text)?.map_or(Ok(Value::Null), construct::construct)
}
