use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

const EXPECTED_MEMBERS: &[&str] = &[
    "litellm-rust/apps/litellm",
    "litellm-rust/crates/core",
    "litellm-rust/crates/ai-gateway",
    "litellm-rust/crates/python-bridge",
];

#[derive(Deserialize)]
struct Metadata {
    workspace_root: PathBuf,
    packages: Vec<Package>,
}

#[derive(Deserialize)]
struct Package {
    name: String,
    manifest_path: PathBuf,
    dependencies: Vec<Dependency>,
}

#[derive(Deserialize)]
struct Dependency {
    name: String,
    path: Option<PathBuf>,
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("repository root should resolve")
}

fn metadata() -> Metadata {
    let output = Command::new(env!("CARGO"))
        .args([
            "metadata",
            "--no-deps",
            "--format-version=1",
            "--locked",
            "--offline",
        ])
        .current_dir(workspace_root())
        .output()
        .expect("cargo metadata should run");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("cargo metadata should return valid JSON")
}

#[test]
fn workspace_members_match_allowlist() {
    let metadata = metadata();
    assert_eq!(metadata.workspace_root, workspace_root());
    let actual: BTreeSet<_> = metadata
        .packages
        .iter()
        .map(|package| {
            package
                .manifest_path
                .parent()
                .expect("manifest directory")
                .strip_prefix(&metadata.workspace_root)
                .expect("workspace member")
                .to_path_buf()
        })
        .collect();
    let expected: BTreeSet<_> = EXPECTED_MEMBERS.iter().map(PathBuf::from).collect();
    assert_eq!(
        actual, expected,
        "update the allowlist and AGENTS.md when adding a package"
    );
}

#[test]
fn crates_directory_matches_allowlist() {
    let root = workspace_root();
    let actual: BTreeSet<_> = ["litellm-rust/apps", "litellm-rust/crates"]
        .into_iter()
        .flat_map(|directory| fs::read_dir(root.join(directory)).expect("package directory"))
        .map(|entry| entry.expect("package entry").path())
        .filter(|path| path.join("Cargo.toml").is_file())
        .map(|path| {
            path.strip_prefix(&root)
                .expect("workspace package")
                .to_path_buf()
        })
        .collect();
    let expected: BTreeSet<_> = EXPECTED_MEMBERS.iter().map(PathBuf::from).collect();
    assert_eq!(
        actual, expected,
        "every package must be an explicit workspace member"
    );
}

#[test]
fn workspace_dependencies_follow_layers() {
    for package in metadata().packages {
        let expected: BTreeSet<_> = match package.name.as_str() {
            "litellm" => ["litellm-ai-gateway"].into_iter().collect(),
            "litellm-ai-gateway" | "litellm-python-bridge" => {
                ["litellm-core"].into_iter().collect()
            }
            "litellm-core" => BTreeSet::new(),
            name => panic!("unexpected workspace package: {name}"),
        };
        let actual: BTreeSet<_> = package
            .dependencies
            .iter()
            .filter(|dependency| dependency.path.is_some())
            .map(|dependency| dependency.name.as_str())
            .collect();
        assert_eq!(
            actual, expected,
            "{} has incorrect workspace dependencies",
            package.name
        );
    }
}
