//! Enforcement: the litellm-rust workspace has exactly four crates.
//!
//! `core` (the Rust SDK), `ai-gateway` (the HTTP/WebSocket host),
//! `python-interop` (domain-neutral PyO3 primitives), and `python-bridge` (the
//! PyO3 cdylib). Adding or removing a crate must be a
//! deliberate act: this test fails until the allowlist here is updated, forcing
//! whoever changes the crate set to justify the new crate per the rule that a
//! crate is a layer needing independent compilation / its own deps / a separate
//! artifact — and to keep `litellm-rust/AGENTS.md` in sync.
//!
//! Std-only (no toml crate): we scan the workspace manifest's `members = [...]`
//! block and the `crates/` directory directly.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// The one true crate set. Update BOTH this and `litellm-rust/AGENTS.md` when the
/// workspace legitimately gains or loses a crate.
const EXPECTED_MEMBERS: &[&str] = &[
    "crates/core",
    "crates/ai-gateway",
    "crates/python-interop",
    "crates/python-bridge",
];

/// The crate subdirectory names that must exist under `crates/`.
const EXPECTED_CRATE_DIRS: &[&str] = &["core", "ai-gateway", "python-interop", "python-bridge"];

const MISMATCH: &str = "litellm-rust crate set changed — update this allowlist AND litellm-rust/AGENTS.md, and justify the crate per the rule (crate = layer needing independent compilation / its own deps / a separate artifact).";

/// Absolute path to the workspace root (`litellm-rust/`).
fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is `.../litellm-rust/crates/core`; the workspace root is
    // two levels up.
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
        .canonicalize()
        .expect("workspace root should resolve")
}

/// Parse the `members = [ ... ]` array out of the workspace `[workspace]` table.
///
/// Minimal hand-rolled scan: find `members`, then collect every double-quoted
/// string up to the closing `]`. Good enough for our fixed manifest shape and
/// keeps this test dependency-free.
fn parse_members(manifest: &str) -> BTreeSet<String> {
    let after_members = manifest
        .split_once("members")
        .map(|(_, rest)| rest)
        .expect("workspace manifest should declare members");
    let open = after_members.find('[').expect("members should be an array");
    let close = after_members[open..]
        .find(']')
        .map(|offset| open + offset)
        .expect("members array should be closed");
    let body = &after_members[open + 1..close];

    let mut members = BTreeSet::new();
    let mut rest = body;
    while let Some(start) = rest.find('"') {
        let after_quote = &rest[start + 1..];
        let end = after_quote
            .find('"')
            .expect("opening quote should be matched");
        members.insert(after_quote[..end].to_string());
        rest = &after_quote[end + 1..];
    }
    members
}

/// The crate subdirectory names under `crates/`.
///
/// A directory counts as a crate only when it holds a `Cargo.toml`; non-crate
/// directories (e.g. docs like `CODING_STANDARDS/`) are ignored so they can live
/// under `crates/` without tripping the crate-proliferation guard.
fn crate_dirs(root: &Path) -> BTreeSet<String> {
    fs::read_dir(root.join("crates"))
        .expect("crates/ directory should exist")
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().map(|ty| ty.is_dir()).unwrap_or(false))
        .filter(|entry| entry.path().join("Cargo.toml").is_file())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect()
}

#[test]
fn workspace_members_match_allowlist() {
    let root = workspace_root();
    let manifest = fs::read_to_string(root.join("Cargo.toml"))
        .expect("workspace Cargo.toml should be readable");

    let actual = parse_members(&manifest);
    let expected: BTreeSet<String> = EXPECTED_MEMBERS.iter().map(|s| s.to_string()).collect();
    assert_eq!(actual, expected, "{MISMATCH}");
}

#[test]
fn crates_directory_matches_allowlist() {
    let root = workspace_root();

    let actual = crate_dirs(&root);
    let expected: BTreeSet<String> = EXPECTED_CRATE_DIRS.iter().map(|s| s.to_string()).collect();
    assert_eq!(actual, expected, "{MISMATCH}");
}

/// Parse the dependency names out of a crate manifest's `[dependencies]`
/// table.
///
/// Same hand-rolled approach as [`parse_members`]: take the lines after the
/// `[dependencies]` header up to the next table header (a line starting with
/// `[`), then keep the text before ` = ` on each non-comment line, trimmed of
/// the `.workspace`-style shorthand suffix.
fn parse_dependencies(manifest: &str) -> BTreeSet<String> {
    let Some((_, after_table)) = manifest.split_once("[dependencies]") else {
        return BTreeSet::new();
    };

    after_table
        .lines()
        .map(str::trim)
        .take_while(|line| !line.starts_with('['))
        .filter_map(|line| {
            let (name, _) = line.split_once(" = ")?;
            let name = name.split('.').next().unwrap_or(name);
            (!name.is_empty() && !name.starts_with('#')).then(|| name.to_string())
        })
        .collect()
}

fn crate_manifest(root: &Path, crate_dir: &str) -> String {
    fs::read_to_string(root.join("crates").join(crate_dir).join("Cargo.toml"))
        .unwrap_or_else(|error| panic!("{crate_dir}/Cargo.toml should be readable: {error}"))
}

/// The python bridge depends on the domain layers, never on the gateway: the
/// bridge and the axum server are alternative hosts over `litellm-core`, so a
/// bridge -> gateway edge would drag the server crate into every cdylib build
/// and let provider I/O creep back out of core.
#[test]
fn python_bridge_dependencies_stay_on_core_and_interop() {
    let manifest = crate_manifest(&workspace_root(), "python-bridge");
    let dependencies = parse_dependencies(&manifest);

    assert!(
        dependencies.contains("litellm-core"),
        "python-bridge must depend on litellm-core, got {dependencies:?}"
    );
    assert!(
        dependencies.contains("litellm-python-interop"),
        "python-bridge must depend on litellm-python-interop, got {dependencies:?}"
    );
    assert!(
        !dependencies.contains("litellm-ai-gateway"),
        "python-bridge must not depend on litellm-ai-gateway (it belongs behind the \
         gateway's own host surface, not the cdylib), got {dependencies:?}"
    );
}

/// `litellm-core` stays a pure Rust SDK: Python bindings (pyo3, pythonize,
/// pyo3-async-runtimes) live in `litellm-python-interop` /
/// `litellm-python-bridge`, never in core.
#[test]
fn core_dependencies_stay_python_free() {
    let manifest = crate_manifest(&workspace_root(), "core");
    let dependencies = parse_dependencies(&manifest);

    for banned in ["pyo3", "pyo3-async-runtimes", "pythonize"] {
        assert!(
            !dependencies.contains(banned),
            "litellm-core must not depend on {banned}; Python binding crates own it, \
             got {dependencies:?}"
        );
    }
}
