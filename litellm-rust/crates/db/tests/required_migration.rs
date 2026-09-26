use std::{env, fs, path::PathBuf};

use litellm_db::REQUIRED_MIGRATION;
use rstest::rstest;

#[rstest]
fn the_required_migration_is_the_newest_in_the_tree() {
    let migrations = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../litellm-proxy-extras/litellm_proxy_extras/migrations");
    let newest = fs::read_dir(migrations)
        .unwrap()
        .map(|entry| entry.unwrap())
        .filter(|entry| entry.file_type().unwrap().is_dir())
        .map(|entry| entry.file_name().into_string().unwrap())
        .max()
        .unwrap();

    assert_eq!(REQUIRED_MIGRATION, newest);
}
