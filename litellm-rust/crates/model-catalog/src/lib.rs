macro_rules_attribute::attribute_alias! {
    #[apply(wire_type)] =
        #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))];
}

mod capabilities;
mod catalog;
mod error;
mod fallback;
mod model_info;
mod prices;
mod pricing;
mod rate;
mod served;

pub use capabilities::*;
pub use catalog::*;
pub use error::*;
pub use fallback::*;
pub use model_info::*;
pub use prices::*;
pub use pricing::*;
pub use rate::*;
pub use served::*;

#[cfg(feature = "schema")]
mod schema;
#[cfg(feature = "schema")]
pub use schema::*;
