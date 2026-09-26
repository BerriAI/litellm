use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use semver::Version;
use serde::Deserialize;

use super::{
    Configure, Drive, Install, LaunchSpec, Outcome, Prompt, Settings, Usage, Wire, env, json_lines,
    path_string, v1,
};
use crate::install::release::github_release;
use crate::install::{Packaging, Release};
use crate::target::Os;
use crate::{Error, Fetch, Target};

const RELEASES: &str = "https://api.github.com/repos/sst/opencode/releases/tags";

pub struct Opencode;

impl Install for Opencode {
    fn binary(&self) -> &'static str {
        "opencode"
    }

    async fn release(
        &self,
        fetch: &impl Fetch,
        version: &Version,
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
}

impl Configure for Opencode {
    fn configure(
        &self,
        _version: &Version,
        settings: &Settings,
        home: &Path,
    ) -> Result<LaunchSpec, Error> {
        let npm = match settings.wire {
            Wire::ChatCompletions => "@ai-sdk/openai-compatible",
            Wire::Responses => "@ai-sdk/openai",
            Wire::Messages => "@ai-sdk/anthropic",
        };
        let config = serde_json::json!({
            "$schema": "https://opencode.ai/config.json",
            "model": format!("litellm/{}", settings.model),
            "provider": {
                "litellm": {
                    "npm": npm,
                    "name": "LiteLLM",
                    "options": { "baseURL": v1(settings), "apiKey": settings.api_key },
                    "models": { settings.model.clone(): { "name": settings.model } },
                }
            },
        });
        Ok(LaunchSpec {
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
        })
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Event {
    Text {
        part: TextPart,
    },
    ToolUse {
        part: ToolPart,
    },
    StepFinish {
        part: StepFinish,
    },
    Error {
        error: Failure,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct TextPart {
    text: String,
}

#[derive(Deserialize)]
struct ToolPart {
    tool: String,
}

#[derive(Deserialize)]
struct StepFinish {
    tokens: Tokens,
}

#[derive(Deserialize)]
struct Tokens {
    input: u64,
    output: u64,
}

#[derive(Deserialize)]
struct Failure {
    name: String,
    data: Option<FailureData>,
}

#[derive(Deserialize)]
struct FailureData {
    message: Option<String>,
}

impl Drive for Opencode {
    fn args(&self, _version: &Version, _settings: &Settings, prompt: &Prompt) -> Vec<String> {
        ["run", "--format", "json", &prompt.text]
            .map(str::to_owned)
            .to_vec()
    }

    fn parse(&self, _version: &Version, stdout: &str) -> Outcome {
        let events: Vec<Event> = json_lines(stdout).collect();
        Outcome {
            text: events
                .iter()
                .rev()
                .find_map(|event| match event {
                    Event::Text { part } => Some(part.text.clone()),
                    _ => None,
                })
                .unwrap_or_default(),
            tool_calls: events
                .iter()
                .filter_map(|event| match event {
                    Event::ToolUse { part } => Some(part.tool.clone()),
                    _ => None,
                })
                .collect(),
            usage: events
                .iter()
                .filter_map(|event| match event {
                    Event::StepFinish { part } => Some(Usage {
                        input_tokens: part.tokens.input,
                        output_tokens: part.tokens.output,
                    }),
                    _ => None,
                })
                .fold(Usage::default(), |total, step| total + step),
            errors: events
                .iter()
                .filter_map(|event| match event {
                    Event::Error { error } => Some(
                        error
                            .data
                            .as_ref()
                            .and_then(|data| data.message.clone())
                            .unwrap_or_else(|| error.name.clone()),
                    ),
                    _ => None,
                })
                .collect(),
            exit_code: None,
        }
    }
}
