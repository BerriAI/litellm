use std::collections::BTreeMap;
use std::path::Path;

use semver::Version;
use serde::Deserialize;

use super::{
    Configure, Drive, Install, LaunchSpec, Outcome, Prompt, Settings, Usage, Wire, env, json_lines,
    path_string,
};
use crate::install::release::parse;
use crate::install::{Packaging, Release};
use crate::{Error, Fetch, Target};

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

impl Install for ClaudeCode {
    fn binary(&self) -> &'static str {
        "claude"
    }

    async fn release(
        &self,
        fetch: &impl Fetch,
        version: &Version,
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
}

impl Configure for ClaudeCode {
    fn configure(
        &self,
        _version: &Version,
        settings: &Settings,
        home: &Path,
    ) -> Result<LaunchSpec, Error> {
        if settings.wire != Wire::Messages {
            return Err(Error::UnsupportedWire {
                agent: "claude",
                wire: settings.wire,
            });
        }
        Ok(LaunchSpec {
            env: env([
                ("HOME", path_string(home)),
                ("CLAUDE_CONFIG_DIR", path_string(&home.join(".claude"))),
                ("ANTHROPIC_BASE_URL", settings.base_url.clone()),
                ("ANTHROPIC_AUTH_TOKEN", settings.api_key.clone()),
                ("ANTHROPIC_MODEL", settings.model.clone()),
                ("DISABLE_AUTOUPDATER", "1".to_owned()),
            ]),
            files: BTreeMap::new(),
        })
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Event {
    Assistant {
        message: AssistantMessage,
    },
    Result(Finished),
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct AssistantMessage {
    content: Vec<Block>,
}

#[derive(Deserialize)]
struct Block {
    #[serde(rename = "type")]
    kind: String,
    name: Option<String>,
}

#[derive(Deserialize)]
struct Finished {
    is_error: bool,
    result: Option<String>,
    usage: Option<TokenUsage>,
}

#[derive(Deserialize)]
struct TokenUsage {
    input_tokens: u64,
    output_tokens: u64,
}

impl Drive for ClaudeCode {
    fn args(&self, _version: &Version, settings: &Settings, prompt: &Prompt) -> Vec<String> {
        let base = [
            "-p",
            &prompt.text,
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            &settings.model,
        ];
        let tools = ["--allowedTools", "Bash,Read,Write,Edit"];
        base.into_iter()
            .chain(tools.into_iter().filter(|_| prompt.allow_tools))
            .map(str::to_owned)
            .collect()
    }

    fn parse(&self, _version: &Version, stdout: &str) -> Outcome {
        let events: Vec<Event> = json_lines(stdout).collect();
        let tool_calls = events
            .iter()
            .filter_map(|event| match event {
                Event::Assistant { message } => Some(&message.content),
                _ => None,
            })
            .flatten()
            .filter(|block| block.kind == "tool_use")
            .filter_map(|block| block.name.clone())
            .collect();
        let finished = events.into_iter().find_map(|event| match event {
            Event::Result(finished) => Some(finished),
            _ => None,
        });
        let Some(finished) = finished else {
            return Outcome {
                tool_calls,
                ..Outcome::default()
            };
        };
        let result = finished.result.unwrap_or_default();
        let (text, errors) = if finished.is_error {
            (String::new(), vec![result])
        } else {
            (result, Vec::new())
        };
        Outcome {
            text,
            tool_calls,
            usage: finished.usage.map_or_else(Usage::default, |usage| Usage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
            }),
            errors,
            exit_code: None,
        }
    }
}
