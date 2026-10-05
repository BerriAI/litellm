macro_rules_attribute::attribute_alias! {
    #[apply(wire_type)] =
        #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))];
}

pub mod formats;
pub mod headers;
pub mod providers;
pub mod recognized;
pub mod serde_compat;
