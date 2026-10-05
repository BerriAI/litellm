use litellm_migrate::Migration;
use rstest::rstest;
use sha2::{Digest, Sha256};

const MIGRATIONS: &[Migration] = litellm_migrate::migrate!("tests/fixtures/migrations");

#[rstest]
#[case::first(0, 1, "first", include_str!("fixtures/migrations/1_first.sql"))]
#[case::second(1, 2, "second", include_str!("fixtures/migrations/2_second.sql"))]
#[case::tenth(2, 10, "tenth", include_str!("fixtures/migrations/10_tenth.sql"))]
fn embeds_every_file_sorted_by_numeric_version(
    #[case] index: usize,
    #[case] version: u64,
    #[case] description: &str,
    #[case] sql: &str,
) {
    assert_eq!(MIGRATIONS.len(), 3);
    let migration = &MIGRATIONS[index];
    assert_eq!(migration.version, version);
    assert_eq!(migration.description, description);
    assert_eq!(migration.sql, sql);
    assert_eq!(
        migration.checksum,
        format!("{:x}", Sha256::digest(sql.as_bytes()))
    );
}

#[rstest]
#[case::nothing_applied(&[], &[1, 2, 10])]
#[case::some_applied(&[(1, MIGRATIONS[0].checksum)], &[2, 10])]
#[case::unknown_version_is_ignored(&[(4, "unknown")], &[1, 2, 10])]
#[case::duplicate_rows_are_accepted(
    &[(1, MIGRATIONS[0].checksum), (1, MIGRATIONS[0].checksum)],
    &[2, 10]
)]
fn pending_returns_unapplied_migrations_in_order(
    #[case] applied: &[(u64, &str)],
    #[case] expected_versions: &[u64],
) {
    let migrations = litellm_migrate::pending(MIGRATIONS, applied.iter().copied())
        .expect("applied migration checksums match");
    assert_eq!(
        migrations
            .iter()
            .map(|migration| migration.version)
            .collect::<Vec<_>>(),
        expected_versions
    );
}

#[rstest]
#[case::changed_first_migration(1, "changed")]
#[case::changed_second_migration(2, "changed")]
fn pending_rejects_changed_applied_migrations(#[case] version: u64, #[case] checksum: &str) {
    let error = litellm_migrate::pending(MIGRATIONS, [(version, checksum)]).unwrap_err();
    assert_eq!(error, litellm_migrate::ChangedMigration { version });
}
