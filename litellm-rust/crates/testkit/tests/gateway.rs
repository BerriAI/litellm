use std::path::Path;

use litellm_testkit::{Agent, ClaudeCode, Codex, Gateway, Opencode};
use rstest::rstest;

fn gateway() -> Gateway {
    Gateway {
        base_url: "http://localhost:4000/".to_owned(),
        api_key: "sk-test \"quoted\"".to_owned(),
        model: "some-model".to_owned(),
    }
}

#[rstest]
#[case(&ClaudeCode)]
#[case(&Codex)]
#[case(&Opencode)]
fn every_client_runs_inside_the_given_home(#[case] agent: &impl Agent) {
    let home = Path::new("/scratch/home");

    let spec = agent.launch_spec(&gateway(), home);

    assert_eq!(spec.env["HOME"], "/scratch/home");
    assert!(
        spec.env
            .iter()
            .filter(|(key, _)| key.ends_with("_HOME") || key.as_str() == "CLAUDE_CONFIG_DIR")
            .all(|(_, value)| value.starts_with("/scratch/home"))
    );
    assert!(spec.files.keys().all(|path| path.is_relative()));
}

#[test]
fn claude_code_points_at_the_gateway_root_with_the_key_and_model() {
    let spec = ClaudeCode.launch_spec(&gateway(), Path::new("/h"));

    assert_eq!(spec.env["ANTHROPIC_BASE_URL"], "http://localhost:4000/");
    assert_eq!(spec.env["ANTHROPIC_AUTH_TOKEN"], "sk-test \"quoted\"");
    assert_eq!(spec.env["ANTHROPIC_MODEL"], "some-model");
}

#[test]
fn codex_config_is_valid_toml_routing_the_responses_api_to_the_gateway() {
    let dir = tempfile::tempdir().unwrap();
    let spec = Codex.launch_spec(&gateway(), dir.path());
    spec.write_files(dir.path()).unwrap();

    let config: toml::Table =
        toml::from_str(&std::fs::read_to_string(dir.path().join(".codex/config.toml")).unwrap())
            .unwrap();
    let provider = &config["model_providers"]["litellm"];

    assert_eq!(config["model"].as_str(), Some("some-model"));
    assert_eq!(config["model_provider"].as_str(), Some("litellm"));
    assert_eq!(
        provider["base_url"].as_str(),
        Some("http://localhost:4000/v1")
    );
    assert_eq!(provider["wire_api"].as_str(), Some("responses"));
    let key_var = provider["env_key"].as_str().unwrap();
    assert_eq!(spec.env[key_var], "sk-test \"quoted\"");
}

#[test]
fn opencode_config_is_valid_json_registering_the_gateway_model() {
    let dir = tempfile::tempdir().unwrap();
    let spec = Opencode.launch_spec(&gateway(), dir.path());
    spec.write_files(dir.path()).unwrap();

    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join(".config/opencode/opencode.json")).unwrap(),
    )
    .unwrap();
    let provider = &config["provider"]["litellm"];

    assert_eq!(config["model"], "litellm/some-model");
    assert_eq!(provider["options"]["baseURL"], "http://localhost:4000/v1");
    assert_eq!(provider["options"]["apiKey"], "sk-test \"quoted\"");
    assert!(provider["models"]["some-model"].is_object());
}
