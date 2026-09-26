use std::{fs, path::Path};

use sqlx::{
    AssertSqlSafe, SqlSafeStr,
    migrate::{Migration, MigrationType},
};

use crate::Error;

pub const PRISMA_MIGRATIONS_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../litellm-proxy-extras/litellm_proxy_extras/migrations"
);

const MIGRATION_FILENAME: &str = "migration.sql";

pub fn prisma_migrations(directory: &Path) -> Result<Vec<Migration>, Error> {
    let mut names = migration_names(directory)?;
    names.sort();
    names
        .into_iter()
        .zip(1..)
        .map(|(name, version)| {
            let path = directory.join(&name).join(MIGRATION_FILENAME);
            let sql = fs::read_to_string(&path).map_err(|source| Error::Read { path, source })?;
            Ok(Migration::new(
                version,
                name.into(),
                MigrationType::Simple,
                AssertSqlSafe(sql).into_sql_str(),
                true,
            ))
        })
        .collect()
}

fn migration_names(directory: &Path) -> Result<Vec<String>, Error> {
    let read_error = |source| Error::Read {
        path: directory.to_owned(),
        source,
    };

    fs::read_dir(directory)
        .map_err(read_error)?
        .map(|entry| {
            let entry = entry.map_err(read_error)?;
            let is_dir = entry.file_type().map_err(read_error)?.is_dir();
            Ok(is_dir.then(|| entry.file_name().to_string_lossy().into_owned()))
        })
        .filter_map(Result::transpose)
        .collect()
}
