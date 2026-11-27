use serde::Deserialize;

use crate::{Error, Fetch};

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
struct GithubRelease {
    assets: Vec<GithubAsset>,
}

#[derive(Deserialize)]
struct GithubAsset {
    name: String,
    digest: Option<String>,
    browser_download_url: String,
}

pub(crate) async fn github_release(
    fetch: &impl Fetch,
    releases_url: &str,
    tag: &str,
    asset_name: &str,
    packaging: Packaging,
) -> Result<Release, Error> {
    let url = format!("{releases_url}/{tag}");
    let release: GithubRelease = parse(&url, &fetch.get(&url).await?)?;
    let asset = release
        .assets
        .into_iter()
        .find(|asset| asset.name == asset_name)
        .ok_or_else(|| Error::AssetNotFound(asset_name.to_owned()))?;
    let sha256 = asset
        .digest
        .as_deref()
        .and_then(|digest| digest.strip_prefix("sha256:"))
        .ok_or_else(|| Error::MissingChecksum(asset_name.to_owned()))?
        .to_owned();
    Ok(Release {
        asset: asset.name,
        url: asset.browser_download_url,
        sha256,
        packaging,
    })
}

pub(crate) fn parse<T: for<'de> Deserialize<'de>>(url: &str, body: &[u8]) -> Result<T, Error> {
    serde_json::from_slice(body).map_err(|source| Error::Metadata {
        url: url.to_owned(),
        source,
    })
}
