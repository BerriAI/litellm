pub use litellm_migrate_macros::migrate;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Migration {
    pub version: u64,
    pub description: &'static str,
    pub sql: &'static str,
}
