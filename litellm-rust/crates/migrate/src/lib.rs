mod error;

pub use error::ChangedMigration;
pub use litellm_migrate_macros::migrate;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Migration {
    pub version: u64,
    pub description: &'static str,
    pub sql: &'static str,
    pub checksum: &'static str,
}

pub fn pending<'a, 'b>(
    migrations: &'a [Migration],
    applied: impl IntoIterator<Item = (u64, &'b str)>,
) -> Result<Vec<&'a Migration>, ChangedMigration> {
    let applied: Vec<_> = applied.into_iter().collect();
    for (version, checksum) in &applied {
        if migrations
            .iter()
            .any(|migration| migration.version == *version && migration.checksum != *checksum)
        {
            return Err(ChangedMigration { version: *version });
        }
    }
    Ok(migrations
        .iter()
        .filter(|migration| {
            !applied
                .iter()
                .any(|(version, _)| *version == migration.version)
        })
        .collect())
}
