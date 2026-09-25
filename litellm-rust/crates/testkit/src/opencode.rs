use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::gateway::{env, path_string, v1};
use crate::release::{Packaging, Release, github_release};
use crate::target::Os;
use crate::{Agent, Error, Fetch, Gateway, LaunchSpec, Target};

const RELEASES: &str = "https://api.github.com/repos/sst/opencode/releases/tags";

pub struct Opencode;

impl Agent for Opencode {
    fn binary(&self) -> &'static str {
        "opencode"
    }

    async fn release(
        &self,
        fetch: &impl Fetch,
        version: &str,
        target: Target,
    ) -> Result<Release, Error> {
        let stem = format!(
            "opencode-{}-{}{}",
            target.os_name(),
            target.arch_name(),
            target.musl_suffix()
        );
        let member = "opencode".to_owned();
        let (asset, packaging) = match target.os {
            Os::Macos => (format!("{stem}.zip"), Packaging::Zip { member }),
            Os::Linux => (format!("{stem}.tar.gz"), Packaging::TarGz { member }),
        };
        github_release(fetch, RELEASES, &format!("v{version}"), &asset, packaging).await
    }

    fn launch_spec(&self, gateway: &Gateway, home: &Path) -> LaunchSpec {
        let config = serde_json::json!({
            "$schema": "https://opencode.ai/config.json",
            "model": format!("litellm/{}", gateway.model),
            "provider": {
                "litellm": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "LiteLLM",
                    "options": { "baseURL": v1(gateway), "apiKey": gateway.api_key },
                    "models": { gateway.model.clone(): { "name": gateway.model } },
                }
            },
        });
        LaunchSpec {
            env: env([
                ("HOME", path_string(home)),
                ("XDG_CONFIG_HOME", path_string(&home.join(".config"))),
                ("XDG_DATA_HOME", path_string(&home.join(".local/share"))),
                ("OPENCODE_DISABLE_AUTOUPDATE", "true".to_owned()),
            ]),
            files: BTreeMap::from([(
                PathBuf::from(".config/opencode/opencode.json"),
                config.to_string(),
            )]),
        }
    }
}
