use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DbFailure {
    UniqueViolation,
    ForeignKeyViolation,
    WriteConflict,
    NotFound,
}
