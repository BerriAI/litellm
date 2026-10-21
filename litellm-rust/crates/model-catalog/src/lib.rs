macro_rules_attribute::attribute_alias! {
    #[apply(wire_type)] =
        #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))];
}

mod bundled;
mod catalog;
mod error;
mod fallback;
mod lower;
mod model;
mod vocabulary;
pub mod wire;

pub use bundled::*;
pub use catalog::*;
pub use error::*;
pub use fallback::*;
pub use model::*;
pub use vocabulary::*;

#[cfg(feature = "schema")]
mod schema;
#[cfg(feature = "schema")]
pub use schema::*;
