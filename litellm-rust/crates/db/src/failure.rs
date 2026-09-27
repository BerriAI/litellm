use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(extend("discriminator" = {"propertyName": "kind"})))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DbFailure {
    #[cfg_attr(feature = "schema", schemars(title = "UniqueViolation"))]
    UniqueViolation,
    #[cfg_attr(feature = "schema", schemars(title = "ForeignKeyViolation"))]
    ForeignKeyViolation,
    #[cfg_attr(feature = "schema", schemars(title = "WriteConflict"))]
    WriteConflict,
    #[cfg_attr(feature = "schema", schemars(title = "NotFound"))]
    NotFound,
}
