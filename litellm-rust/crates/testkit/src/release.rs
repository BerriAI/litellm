use std::collections::BTreeMap;

use serde::Deserialize;

use crate::platform::Os;
use crate::{Client, Error, Fetch, Platform};

const CLAUDE_RELEASES: &str = "https://downloads.claude.ai/claude-code-releases";
const CODEX_RELEASES: &str = "https://api.github.com/repos/openai/codex/releases/tags";
const OPENCODE_RELEASES: &str = "https://api.github.com/repos/sst/opencode/releases/tags";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Packaging {
    Bare,
    TarGz { member: String },
    Zip { member: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Release {
    pub asset: String,
    pub url: String,
    pub sha256: String,
    pub packaging: Packaging,
}

#[derive(Deserialize)]
struct ClaudeManifest {
    platforms: BTreeMap<String, ClaudePlatform>,
}

#[derive(Deserialize)]
struct ClaudePlatform {
    checksum: String,
}

#[derive(Deserialize)]
struct GithubRelease {
    assets: Vec<GithubAsset>,
}

#[derive(Deserialize)]
struct GithubAsset {
    name: String,
    digest: Option<String>,
    browser_download_url: String,
}

pub async fn resolve(
    fetch: &impl Fetch,
    client: Client,
    version: &str,
    platform: Platform,
) -> Result<Release, Error> {
    match client {
        Client::ClaudeCode => resolve_claude(fetch, version, platform).await,
        Client::Codex => resolve_codex(fetch, version, platform).await,
        Client::Opencode => resolve_opencode(fetch, version, platform).await,
    }
}

async fn resolve_claude(
    fetch: &impl Fetch,
    version: &str,
    platform: Platform,
) -> Result<Release, Error> {
    let manifest_url = format!("{CLAUDE_RELEASES}/{version}/manifest.json");
    let manifest: ClaudeManifest = parse(&manifest_url, &fetch.get(&manifest_url).await?)?;
    let key = format!("{}-{}", platform.os_name(), platform.arch_name());
    let entry = manifest
        .platforms
        .get(&key)
        .ok_or_else(|| Error::AssetNotFound(key.clone()))?;
    Ok(Release {
        url: format!("{CLAUDE_RELEASES}/{version}/{key}/claude"),
        asset: key,
        sha256: entry.checksum.clone(),
        packaging: Packaging::Bare,
    })
}

async fn resolve_codex(
    fetch: &impl Fetch,
    version: &str,
    platform: Platform,
) -> Result<Release, Error> {
    let triple = platform.rust_triple();
    let asset = github_asset(
        fetch,
        &format!("{CODEX_RELEASES}/rust-v{version}"),
        &format!("codex-{triple}.tar.gz"),
    )
    .await?;
    Ok(Release {
        packaging: Packaging::TarGz {
            member: format!("codex-{triple}"),
        },
        ..asset
    })
}

async fn resolve_opencode(
    fetch: &impl Fetch,
    version: &str,
    platform: Platform,
) -> Result<Release, Error> {
    let stem = format!("opencode-{}-{}", platform.os_name(), platform.arch_name());
    let member = "opencode".to_owned();
    let (name, packaging) = match platform.os {
        Os::Macos => (format!("{stem}.zip"), Packaging::Zip { member }),
        Os::Linux => (format!("{stem}.tar.gz"), Packaging::TarGz { member }),
    };
    let asset = github_asset(fetch, &format!("{OPENCODE_RELEASES}/v{version}"), &name).await?;
    Ok(Release { packaging, ..asset })
}

async fn github_asset(fetch: &impl Fetch, release_url: &str, name: &str) -> Result<Release, Error> {
    let release: GithubRelease = parse(release_url, &fetch.get(release_url).await?)?;
    let asset = release
        .assets
        .into_iter()
        .find(|asset| asset.name == name)
        .ok_or_else(|| Error::AssetNotFound(name.to_owned()))?;
    let sha256 = asset
        .digest
        .as_deref()
        .and_then(|digest| digest.strip_prefix("sha256:"))
        .ok_or_else(|| Error::MissingChecksum(name.to_owned()))?
        .to_owned();
    Ok(Release {
        asset: asset.name,
        url: asset.browser_download_url,
        sha256,
        packaging: Packaging::Bare,
    })
}

fn parse<T: for<'de> Deserialize<'de>>(url: &str, body: &[u8]) -> Result<T, Error> {
    serde_json::from_slice(body).map_err(|source| Error::Metadata {
        url: url.to_owned(),
        source,
    })
}
