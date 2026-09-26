use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn main() {
    let migrations = migrations_dir();
    println!("cargo::rerun-if-changed={}", migrations.display());
    println!(
        "cargo::rustc-env=LITELLM_REQUIRED_MIGRATION={}",
        newest_migration(&migrations)
    );
}

fn migrations_dir() -> PathBuf {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../../../litellm-proxy-extras/litellm_proxy_extras/migrations")
}

// Prisma prefixes each migration folder with a UTC timestamp, so the lexical max is the newest.
fn newest_migration(migrations: &Path) -> String {
    migration_names(migrations)
        .max()
        .unwrap_or_else(|| panic!("no migrations under {}", migrations.display()))
}

fn migration_names(migrations: &Path) -> impl Iterator<Item = String> {
    fs::read_dir(migrations)
        .unwrap_or_else(|error| panic!("reading {}: {error}", migrations.display()))
        .map(|entry| entry.unwrap())
        .filter(|entry| entry.file_type().unwrap().is_dir())
        .map(|entry| entry.file_name().into_string().unwrap())
}
