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
    let url: String = litellm_core_utils::url_utils::ApiUrl::parse(releases_url)?
        .append_path(&[tag])?
        .into_url()
        .into();
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

#[cfg(test)]
mod tests {
    use super::*;
    struct Capture(std::sync::Mutex<Vec<String>>);
    impl Fetch for Capture {
        async fn get(&self, url: &str) -> Result<Vec<u8>, Error> {
            self.0.lock().unwrap().push(url.into());
            Ok(br#"{"assets":[{"name":"binary","digest":"sha256:abc","browser_download_url":"https://example.test/binary"}]}"#.to_vec())
        }
    }
    #[rstest::rstest]
    #[tokio::test]
    async fn release_tags_are_segments_before_queries() {
        let fetch = Capture(std::sync::Mutex::new(Vec::new()));
        let release = github_release(
            &fetch,
            "https://example.test/prefix/releases/tags?tenant=a#f",
            "release/a%?#",
            "binary",
            Packaging::Bare,
        )
        .await
        .unwrap();
        assert_eq!(release.sha256, "abc");
        assert_eq!(
            fetch.0.into_inner().unwrap(),
            ["https://example.test/prefix/releases/tags/release%2Fa%25%3F%23?tenant=a#f"]
        );
    }
}
