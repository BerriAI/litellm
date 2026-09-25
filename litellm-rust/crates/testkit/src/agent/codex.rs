use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::gateway::{env, path_string, quoted, v1};
use crate::install::release::{Packaging, Release, github_release};
use crate::target::{Arch, Os};
use crate::{Agent, Error, Fetch, Gateway, LaunchSpec, Target};

const RELEASES: &str = "https://api.github.com/repos/openai/codex/releases/tags";

pub struct Codex;

fn triple(target: Target) -> String {
    let arch = match target.arch {
        Arch::Aarch64 => "aarch64",
        Arch::X86_64 => "x86_64",
    };
    match target.os {
        Os::Macos => format!("{arch}-apple-darwin"),
        Os::Linux => format!("{arch}-unknown-linux-musl"),
    }
}

impl Agent for Codex {
    fn binary(&self) -> &'static str {
        "codex"
    }

    async fn release(
        &self,
        fetch: &impl Fetch,
        version: &str,
        target: Target,
    ) -> Result<Release, Error> {
        let triple = triple(target);
        github_release(
            fetch,
            RELEASES,
            &format!("rust-v{version}"),
            &format!("codex-{triple}.tar.gz"),
            Packaging::TarGz {
                member: format!("codex-{triple}"),
            },
        )
        .await
    }

    fn launch_spec(&self, gateway: &Gateway, home: &Path) -> LaunchSpec {
        let config = format!(
            "model = {model}\nmodel_provider = \"litellm\"\n\n[model_providers.litellm]\nname = \"LiteLLM\"\nbase_url = {base_url}\nenv_key = \"LITELLM_API_KEY\"\nwire_api = \"responses\"\n",
            model = quoted(&gateway.model),
            base_url = quoted(&v1(gateway)),
        );
        LaunchSpec {
            env: env([
                ("HOME", path_string(home)),
                ("CODEX_HOME", path_string(&home.join(".codex"))),
                ("LITELLM_API_KEY", gateway.api_key.clone()),
            ]),
            files: BTreeMap::from([(PathBuf::from(".codex/config.toml"), config)]),
        }
    }
}
