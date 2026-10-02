use litellm_migrate::Migration;
use rstest::rstest;

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
}
