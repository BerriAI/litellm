#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("migration {version} changed after it was applied")]
pub struct ChangedMigration {
    pub version: u64,
}
