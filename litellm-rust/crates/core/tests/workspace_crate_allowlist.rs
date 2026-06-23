//! Enforcement test: the litellm-rust workspace crate set is an explicit
//! allowlist. Adding or removing a crate must be a deliberate, reviewed change.
//!
//! This test reads the workspace root `Cargo.toml` `[workspace] members` array
//! AND the on-disk `crates/` directory, and asserts both match exactly:
//!   {"crates/core", "crates/ai-gateway", "crates/python-bridge"}
//!
//! Uses only `std` so it never drags a dependency into `core` (which must stay
//! pure). If you intentionally change the crate set, update this allowlist and
//! the rationale per the crate rule.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const FAILURE_MESSAGE: &str = "litellm-rust crate set changed — update the allowlist in this test AND litellm-rust/AGENTS.md, and justify the new crate per the crate rule (crate = layer that needs independent compilation / its own deps / a separate artifact).";

/// Parse the member path strings out of the `[workspace] members = [...]` array
/// in the root `Cargo.toml`. Hand-rolled (no toml dep) since the shape is fixed:
/// a `members = [` line followed by quoted paths until the closing `]`.
fn parse_workspace_members(manifest: &str) -> BTreeSet<String> {
    let mut members = BTreeSet::new();
    let mut in_members = false;

    for line in manifest.lines() {
        let trimmed = line.trim();

        if !in_members {
            // Match `members = [` possibly with entries on the same line.
            if let Some(rest) = trimmed.strip_prefix("members") {
                let rest = rest.trim_start();
                if let Some(rest) = rest.strip_prefix('=') {
                    in_members = true;
                    collect_quoted(rest, &mut members);
                    if rest.contains(']') {
                        break;
                    }
                }
            }
            continue;
        }

        // Inside the members array.
        collect_quoted(trimmed, &mut members);
        if trimmed.contains(']') {
            break;
        }
    }

    members
}

/// Pull every double-quoted substring out of `segment` into `out`.
fn collect_quoted(segment: &str, out: &mut BTreeSet<String>) {
    let mut chars = segment.chars();
    while let Some(c) = chars.next() {
        if c == '"' {
            let mut value = String::new();
            for inner in chars.by_ref() {
                if inner == '"' {
                    break;
                }
                value.push(inner);
            }
            out.insert(value);
        }
    }
}

#[test]
fn workspace_members_match_allowlist() {
    let manifest_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.toml");
    let manifest = fs::read_to_string(manifest_path)
        .unwrap_or_else(|err| panic!("failed to read {manifest_path}: {err}"));

    let members = parse_workspace_members(&manifest);

    let expected: BTreeSet<String> = [
        "crates/core".to_string(),
        "crates/ai-gateway".to_string(),
        "crates/python-bridge".to_string(),
    ]
    .into_iter()
    .collect();

    assert_eq!(members, expected, "{FAILURE_MESSAGE}");
}

#[test]
fn crates_directory_matches_allowlist() {
    let crates_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let entries = fs::read_dir(Path::new(crates_dir))
        .unwrap_or_else(|err| panic!("failed to read {crates_dir}: {err}"));

    let mut dirs = BTreeSet::new();
    for entry in entries {
        let entry = entry.expect("failed to read crates/ entry");
        if entry
            .file_type()
            .expect("failed to read file type")
            .is_dir()
        {
            dirs.insert(entry.file_name().to_string_lossy().into_owned());
        }
    }

    let expected: BTreeSet<String> = [
        "core".to_string(),
        "ai-gateway".to_string(),
        "python-bridge".to_string(),
    ]
    .into_iter()
    .collect();

    assert_eq!(dirs, expected, "{FAILURE_MESSAGE}");
}
