mod support;

use std::str::FromStr;

use litellm_testkit::{ClaudeCode, Codex, Error, Installer, Opencode, Target};
use rstest::rstest;
use serde_json::json;
use support::{FakeFetch, script_printing, sha256, tar_gz, zip_archive};
use target_lexicon::Triple;

fn target(triple: &str) -> Target {
    Target::try_from(&Triple::from_str(triple).unwrap()).unwrap()
}

fn linux() -> Target {
    target("x86_64-unknown-linux-gnu")
}
const VERSION: &str = "9.8.7";

fn github_release(asset: &str, download_url: &str, digest: Option<String>) -> Vec<u8> {
    json!({
        "assets": [
            { "name": "unrelated.txt", "digest": "sha256:00", "browser_download_url": "https://example.test/unrelated" },
            { "name": asset, "digest": digest, "browser_download_url": download_url },
        ]
    })
    .to_string()
    .into_bytes()
}

fn claude_routes(binary: &[u8], checksum: &str) -> Vec<(String, Vec<u8>)> {
    let base = "https://downloads.claude.ai/claude-code-releases/9.8.7";
    let manifest = json!({ "platforms": { "linux-x64": { "checksum": checksum } } });
    vec![
        (
            format!("{base}/manifest.json"),
            manifest.to_string().into_bytes(),
        ),
        (format!("{base}/linux-x64/claude"), binary.to_vec()),
    ]
}

fn codex_routes(archive: Vec<u8>, digest: Option<String>) -> Vec<(String, Vec<u8>)> {
    let release = github_release(
        "codex-x86_64-unknown-linux-musl.tar.gz",
        "https://example.test/codex.tar.gz",
        digest,
    );
    vec![
        (
            "https://api.github.com/repos/openai/codex/releases/tags/rust-v9.8.7".to_owned(),
            release,
        ),
        ("https://example.test/codex.tar.gz".to_owned(), archive),
    ]
}

#[tokio::test]
async fn claude_bare_binary_is_installed_and_runnable() {
    let binary = script_printing("9.8.7 (Claude Code)");
    let fetch = FakeFetch::new(claude_routes(&binary, &sha256(&binary)));
    let cache = tempfile::tempdir().unwrap();

    let installed = Installer::new(&fetch, cache.path(), linux())
        .install(&ClaudeCode, VERSION)
        .await
        .unwrap();

    assert_eq!(installed.binary, cache.path().join("claude/9.8.7/claude"));
    assert_eq!(std::fs::read(&installed.binary).unwrap(), binary);
}

#[tokio::test]
async fn codex_binary_is_extracted_from_the_tarball_under_its_own_name() {
    let binary = script_printing("codex-cli 9.8.7");
    let archive = tar_gz("codex-x86_64-unknown-linux-musl", &binary);
    let fetch = FakeFetch::new(codex_routes(
        archive.clone(),
        Some(format!("sha256:{}", sha256(&archive))),
    ));
    let cache = tempfile::tempdir().unwrap();

    let installed = Installer::new(&fetch, cache.path(), linux())
        .install(&Codex, VERSION)
        .await
        .unwrap();

    assert_eq!(std::fs::read(&installed.binary).unwrap(), binary);
    assert_eq!(installed.binary, cache.path().join("codex/9.8.7/codex"));
}

#[tokio::test]
async fn opencode_binary_is_extracted_from_the_darwin_zip() {
    let binary = script_printing("9.8.7");
    let archive = zip_archive("opencode", &binary);
    let release = github_release(
        "opencode-darwin-arm64.zip",
        "https://example.test/opencode.zip",
        Some(format!("sha256:{}", sha256(&archive))),
    );
    let fetch = FakeFetch::new([
        (
            "https://api.github.com/repos/sst/opencode/releases/tags/v9.8.7".to_owned(),
            release,
        ),
        ("https://example.test/opencode.zip".to_owned(), archive),
    ]);
    let cache = tempfile::tempdir().unwrap();

    let installed = Installer::new(&fetch, cache.path(), target("aarch64-apple-darwin"))
        .install(&Opencode, VERSION)
        .await
        .unwrap();

    assert_eq!(std::fs::read(&installed.binary).unwrap(), binary);
}

#[tokio::test]
async fn tampered_download_is_rejected_and_nothing_is_left_behind() {
    let binary = script_printing("9.8.7 (Claude Code)");
    let fetch = FakeFetch::new(claude_routes(&binary, &sha256(b"what the vendor signed")));
    let cache = tempfile::tempdir().unwrap();

    let result = Installer::new(&fetch, cache.path(), linux())
        .install(&ClaudeCode, VERSION)
        .await;

    assert!(matches!(result, Err(Error::ChecksumMismatch { .. })));
    assert!(!cache.path().join("claude/9.8.7").exists());
}

#[tokio::test]
async fn github_asset_without_a_digest_is_refused() {
    let archive = tar_gz(
        "codex-x86_64-unknown-linux-musl",
        &script_printing("codex-cli 9.8.7"),
    );
    let fetch = FakeFetch::new(codex_routes(archive, None));
    let cache = tempfile::tempdir().unwrap();

    let result = Installer::new(&fetch, cache.path(), linux())
        .install(&Codex, VERSION)
        .await;

    assert!(matches!(result, Err(Error::MissingChecksum(_))));
}

#[tokio::test]
async fn binary_reporting_a_different_version_is_removed() {
    let binary = script_printing("1.0.0 (Claude Code)");
    let fetch = FakeFetch::new(claude_routes(&binary, &sha256(&binary)));
    let cache = tempfile::tempdir().unwrap();

    let result = Installer::new(&fetch, cache.path(), linux())
        .install(&ClaudeCode, VERSION)
        .await;

    assert!(matches!(result, Err(Error::VersionMismatch { .. })));
    assert!(!cache.path().join("claude/9.8.7/claude").exists());
}

#[tokio::test]
async fn second_install_reuses_the_cached_binary_without_downloading() {
    let binary = script_printing("9.8.7 (Claude Code)");
    let fetch = FakeFetch::new(claude_routes(&binary, &sha256(&binary)));
    let cache = tempfile::tempdir().unwrap();
    let installer = Installer::new(&fetch, cache.path(), linux());

    let first = installer.install(&ClaudeCode, VERSION).await.unwrap();
    let calls_after_first = fetch.calls();
    let second = installer.install(&ClaudeCode, VERSION).await.unwrap();

    assert_eq!(first, second);
    assert_eq!(fetch.calls(), calls_after_first);
}

#[tokio::test]
async fn corrupted_cache_entry_is_replaced_by_a_fresh_download() {
    let binary = script_printing("9.8.7 (Claude Code)");
    let fetch = FakeFetch::new(claude_routes(&binary, &sha256(&binary)));
    let cache = tempfile::tempdir().unwrap();
    let installer = Installer::new(&fetch, cache.path(), linux());
    let installed = installer.install(&ClaudeCode, VERSION).await.unwrap();
    std::fs::write(&installed.binary, script_printing("0.0.1")).unwrap();

    installer.install(&ClaudeCode, VERSION).await.unwrap();

    assert_eq!(std::fs::read(&installed.binary).unwrap(), binary);
}

#[rstest]
#[case("../9.8.7")]
#[case("9.8")]
#[case("9.8.7-beta")]
#[case("latest")]
#[case("")]
#[tokio::test]
async fn non_release_versions_never_reach_the_network_or_the_filesystem(#[case] version: &str) {
    let fetch = FakeFetch::new([]);
    let cache = tempfile::tempdir().unwrap();

    let result = Installer::new(&fetch, cache.path(), linux())
        .install(&ClaudeCode, version)
        .await;

    assert!(matches!(result, Err(Error::InvalidVersion(_))));
    assert_eq!(fetch.calls(), 0);
    assert_eq!(std::fs::read_dir(cache.path()).unwrap().count(), 0);
}

#[tokio::test]
async fn musl_linux_picks_the_musl_claude_build() {
    let binary = script_printing("9.8.7 (Claude Code)");
    let base = "https://downloads.claude.ai/claude-code-releases/9.8.7";
    let manifest = json!({ "platforms": {
        "linux-x64": { "checksum": sha256(b"glibc build") },
        "linux-x64-musl": { "checksum": sha256(&binary) },
    } });
    let fetch = FakeFetch::new([
        (
            format!("{base}/manifest.json"),
            manifest.to_string().into_bytes(),
        ),
        (format!("{base}/linux-x64-musl/claude"), binary.clone()),
    ]);
    let cache = tempfile::tempdir().unwrap();

    let installed = Installer::new(&fetch, cache.path(), target("x86_64-unknown-linux-musl"))
        .install(&ClaudeCode, VERSION)
        .await
        .unwrap();

    assert_eq!(std::fs::read(&installed.binary).unwrap(), binary);
}

#[rstest]
#[case("x86_64-pc-windows-msvc")]
#[case("riscv64gc-unknown-linux-gnu")]
#[case("wasm32-unknown-unknown")]
fn targets_no_agent_ships_for_are_rejected(#[case] triple: &str) {
    let result = Target::try_from(&Triple::from_str(triple).unwrap());

    assert!(matches!(result, Err(Error::UnsupportedTarget(_))));
}
