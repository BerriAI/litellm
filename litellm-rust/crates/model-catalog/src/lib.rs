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
mod revalidate;
mod vocabulary;
pub mod wire;

pub use bundled::*;
pub use catalog::*;
pub use error::*;
pub use fallback::*;
pub use layers::*;
pub use lower::{fatal, rejects_row};
pub use model::*;
pub use revalidate::*;
pub use vocabulary::*;

#[cfg(feature = "schema")]
mod schema;
#[cfg(feature = "schema")]
pub use schema::*;
