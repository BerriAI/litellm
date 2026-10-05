use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use semver::Version;
use serde::Deserialize;

use super::{
    Configure, Drive, Install, LaunchSpec, Outcome, Prompt, Settings, Usage, Wire, env, json_lines,
    path_string, quoted, v1,
};
use crate::install::release::github_release;
use crate::install::{Packaging, Release};
use crate::target::{Arch, Os};
use crate::{Error, Fetch, Target};

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

impl Install for Codex {
    fn binary(&self) -> &'static str {
        "codex"
    }

    async fn release(
        &self,
        fetch: &impl Fetch,
        version: &Version,
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
}

impl Configure for Codex {
    fn configure(
        &self,
        _version: &Version,
        settings: &Settings,
        home: &Path,
    ) -> Result<LaunchSpec, Error> {
        if settings.wire != Wire::Responses {
            return Err(Error::UnsupportedWire {
                agent: "codex",
                wire: settings.wire,
            });
        }
        let config = format!(
            "model = {model}\nmodel_provider = \"litellm\"\n\n[model_providers.litellm]\nname = \"LiteLLM\"\nbase_url = {base_url}\nenv_key = \"LITELLM_API_KEY\"\nwire_api = \"responses\"\n",
            model = quoted(&settings.model),
            base_url = quoted(&v1(settings)),
        );
        Ok(LaunchSpec {
            env: env([
                ("HOME", path_string(home)),
                ("CODEX_HOME", path_string(&home.join(".codex"))),
                ("LITELLM_API_KEY", settings.api_key.clone()),
            ]),
            files: BTreeMap::from([(PathBuf::from(".codex/config.toml"), config)]),
        })
    }
}

#[derive(Deserialize)]
enum EventKind {
    #[serde(rename = "item.completed")]
    ItemCompleted,
    #[serde(rename = "turn.completed")]
    TurnCompleted,
    #[serde(rename = "turn.failed")]
    TurnFailed,
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct Event {
    #[serde(rename = "type")]
    kind: EventKind,
    item: Option<Item>,
    usage: Option<TokenUsage>,
    error: Option<Failure>,
}

#[derive(Deserialize)]
struct Item {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

#[derive(Deserialize)]
struct TokenUsage {
    input_tokens: u64,
    output_tokens: u64,
}

#[derive(Deserialize)]
struct Failure {
    message: String,
}

const NON_TOOL_ITEMS: [&str; 3] = ["agent_message", "reasoning", "error"];

impl Drive for Codex {
    fn args(&self, _version: &Version, _settings: &Settings, prompt: &Prompt) -> Vec<String> {
        let sandbox = ["--sandbox", "workspace-write"];
        ["exec", "--json", "--skip-git-repo-check"]
            .into_iter()
            .chain(sandbox.into_iter().filter(|_| prompt.allow_tools))
            .chain([prompt.text.as_str()])
            .map(str::to_owned)
            .collect()
    }

    fn parse(&self, _version: &Version, stdout: &str) -> Outcome {
        let events: Vec<Event> = json_lines(stdout).collect();
        let items: Vec<&Item> = events
            .iter()
            .filter(|event| matches!(event.kind, EventKind::ItemCompleted))
            .filter_map(|event| event.item.as_ref())
            .collect();
        Outcome {
            text: items
                .iter()
                .rev()
                .find(|item| item.kind == "agent_message")
                .and_then(|item| item.text.clone())
                .unwrap_or_default(),
            tool_calls: items
                .iter()
                .filter(|item| !NON_TOOL_ITEMS.contains(&item.kind.as_str()))
                .map(|item| item.kind.clone())
                .collect(),
            usage: events
                .iter()
                .filter(|event| matches!(event.kind, EventKind::TurnCompleted))
                .filter_map(|event| event.usage.as_ref())
                .map(|usage| Usage {
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                })
                .fold(Usage::default(), |total, turn| total + turn),
            errors: events
                .iter()
                .filter(|event| matches!(event.kind, EventKind::TurnFailed))
                .filter_map(|event| event.error.as_ref())
                .map(|failure| failure.message.clone())
                .collect(),
            exit_code: None,
        }
    }
}
