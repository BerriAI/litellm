use std::fs;

use litellm_db_testing::prisma_migrations;
use rstest::rstest;
use tempfile::TempDir;

fn prisma_folder(migrations: &[(&str, &str)]) -> TempDir {
    let directory = tempfile::tempdir().unwrap();
    for (name, sql) in migrations {
        fs::create_dir(directory.path().join(name)).unwrap();
        fs::write(directory.path().join(name).join("migration.sql"), sql).unwrap();
    }
    fs::write(
        directory.path().join("migration_lock.toml"),
        "provider = \"postgresql\"",
    )
    .unwrap();
    directory
}

#[rstest]
fn folders_apply_in_name_order_numbered_from_one_outside_a_transaction() {
    let directory = prisma_folder(&[
        ("20260305000000_b", "SELECT 2;"),
        ("20260108_short_prefix", "SELECT 3;"),
        ("20260305000000_a", "BEGIN; SELECT 1; COMMIT;"),
        ("20260108000000_first", "SELECT 0;"),
    ]);

    let migrations = prisma_migrations(directory.path()).unwrap();

    let applied: Vec<(i64, &str, &str, bool)> = migrations
        .iter()
        .map(|migration| {
            (
                migration.version,
                migration.description.as_ref(),
                migration.sql.as_str(),
                migration.no_tx,
            )
        })
        .collect();
    assert_eq!(
        applied,
        vec![
            (1, "20260108000000_first", "SELECT 0;", true),
            (2, "20260108_short_prefix", "SELECT 3;", true),
            (3, "20260305000000_a", "BEGIN; SELECT 1; COMMIT;", true),
            (4, "20260305000000_b", "SELECT 2;", true),
        ]
    );
}

#[rstest]
fn a_folder_without_migration_sql_is_an_error() {
    let directory = prisma_folder(&[]);
    fs::create_dir(directory.path().join("20260101000000_empty")).unwrap();

    let error = prisma_migrations(directory.path()).unwrap_err();

    assert!(
        error.to_string().contains("20260101000000_empty"),
        "{error}"
    );
}
