//! Enforcement: the workspace release profile keeps LTO, single-codegen-unit,
//! and symbol stripping enabled.
//!
//! These settings shrink and speed up the `litellm-ai-gateway` and
//! `litellm-python-bridge` release artifacts (see #31352). This test fails if
//! `[profile.release]` in the workspace `Cargo.toml` is removed or weakened,
//! so a regression can't land silently.
//!
//! Std-only (no toml crate): same hand-rolled scan style as
//! `workspace_crate_allowlist.rs`.

use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
        .canonicalize()
        .expect("workspace root should resolve")
}

fn release_profile_block(manifest: &str) -> String {
    let after_header = manifest
        .split_once("[profile.release]")
        .map(|(_, rest)| rest)
        .expect("workspace Cargo.toml should declare [profile.release]");
    match after_header.find("\n[") {
        Some(next_section) => after_header[..next_section].to_string(),
        None => after_header.to_string(),
    }
}

#[test]
fn release_profile_enables_lto_codegen_units_and_strip() {
    let root = workspace_root();
    let manifest = fs::read_to_string(root.join("Cargo.toml"))
        .expect("workspace Cargo.toml should be readable");

    let profile = release_profile_block(&manifest);

    assert!(
        profile.contains("lto = true"),
        "[profile.release] must keep `lto = true` (see #31352)"
    );
    assert!(
        profile.contains("codegen-units = 1"),
        "[profile.release] must keep `codegen-units = 1` (see #31352)"
    );
    assert!(
        profile.contains("strip = true"),
        "[profile.release] must keep `strip = true` (see #31352)"
    );
}
