use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::Client;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gateway {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchSpec {
    pub env: BTreeMap<String, String>,
    pub files: BTreeMap<PathBuf, String>,
}

impl LaunchSpec {
    pub fn write_files(&self, home: &Path) -> std::io::Result<()> {
        self.files.iter().try_for_each(|(relative, contents)| {
            let path = home.join(relative);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, contents)
        })
    }
}

pub fn launch_spec(client: Client, gateway: &Gateway, home: &Path) -> LaunchSpec {
    match client {
        Client::ClaudeCode => claude_code(gateway, home),
        Client::Codex => codex(gateway, home),
        Client::Opencode => opencode(gateway, home),
    }
}

fn env(pairs: impl IntoIterator<Item = (&'static str, String)>) -> BTreeMap<String, String> {
    pairs
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect()
}

fn path_string(path: PathBuf) -> String {
    path.to_string_lossy().into_owned()
}

fn quoted(value: &str) -> String {
    serde_json::Value::from(value).to_string()
}

fn v1(gateway: &Gateway) -> String {
    format!("{}/v1", gateway.base_url.trim_end_matches('/'))
}

fn claude_code(gateway: &Gateway, home: &Path) -> LaunchSpec {
    LaunchSpec {
        env: env([
            ("HOME", path_string(home.to_path_buf())),
            ("CLAUDE_CONFIG_DIR", path_string(home.join(".claude"))),
            ("ANTHROPIC_BASE_URL", gateway.base_url.clone()),
            ("ANTHROPIC_AUTH_TOKEN", gateway.api_key.clone()),
            ("ANTHROPIC_MODEL", gateway.model.clone()),
            ("DISABLE_AUTOUPDATER", "1".to_owned()),
        ]),
        files: BTreeMap::new(),
    }
}

fn codex(gateway: &Gateway, home: &Path) -> LaunchSpec {
    let config = format!(
        "model = {model}\nmodel_provider = \"litellm\"\n\n[model_providers.litellm]\nname = \"LiteLLM\"\nbase_url = {base_url}\nenv_key = \"LITELLM_API_KEY\"\nwire_api = \"responses\"\n",
        model = quoted(&gateway.model),
        base_url = quoted(&v1(gateway)),
    );
    LaunchSpec {
        env: env([
            ("HOME", path_string(home.to_path_buf())),
            ("CODEX_HOME", path_string(home.join(".codex"))),
            ("LITELLM_API_KEY", gateway.api_key.clone()),
        ]),
        files: BTreeMap::from([(PathBuf::from(".codex/config.toml"), config)]),
    }
}

fn opencode(gateway: &Gateway, home: &Path) -> LaunchSpec {
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
            ("HOME", path_string(home.to_path_buf())),
            ("XDG_CONFIG_HOME", path_string(home.join(".config"))),
            ("XDG_DATA_HOME", path_string(home.join(".local/share"))),
            ("OPENCODE_DISABLE_AUTOUPDATE", "true".to_owned()),
        ]),
        files: BTreeMap::from([(
            PathBuf::from(".config/opencode/opencode.json"),
            config.to_string(),
        )]),
    }
}
