use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::gateway::{env, path_string};
use crate::install::release::{Packaging, Release, parse};
use crate::{Agent, Error, Fetch, Gateway, LaunchSpec, Target};

const RELEASES: &str = "https://downloads.claude.ai/claude-code-releases";

pub struct ClaudeCode;

#[derive(Deserialize)]
struct Manifest {
    platforms: BTreeMap<String, Platform>,
}

#[derive(Deserialize)]
struct Platform {
    checksum: String,
}

impl Agent for ClaudeCode {
    fn binary(&self) -> &'static str {
        "claude"
    }

    async fn release(
        &self,
        fetch: &impl Fetch,
        version: &str,
        target: Target,
    ) -> Result<Release, Error> {
        let manifest_url = format!("{RELEASES}/{version}/manifest.json");
        let manifest: Manifest = parse(&manifest_url, &fetch.get(&manifest_url).await?)?;
        let key = format!(
            "{}-{}{}",
            target.os_name(),
            target.arch_name(),
            target.musl_suffix()
        );
        let platform = manifest
            .platforms
            .get(&key)
            .ok_or_else(|| Error::AssetNotFound(key.clone()))?;
        Ok(Release {
            url: format!("{RELEASES}/{version}/{key}/claude"),
            asset: key,
            sha256: platform.checksum.clone(),
            packaging: Packaging::Bare,
        })
    }

    fn launch_spec(&self, gateway: &Gateway, home: &Path) -> LaunchSpec {
        LaunchSpec {
            env: env([
                ("HOME", path_string(home)),
                ("CLAUDE_CONFIG_DIR", path_string(&home.join(".claude"))),
                ("ANTHROPIC_BASE_URL", gateway.base_url.clone()),
                ("ANTHROPIC_AUTH_TOKEN", gateway.api_key.clone()),
                ("ANTHROPIC_MODEL", gateway.model.clone()),
                ("DISABLE_AUTOUPDATER", "1".to_owned()),
            ]),
            files: BTreeMap::new(),
        }
    }
}
