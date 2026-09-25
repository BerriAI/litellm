use std::path::Path;

use litellm_testkit::{ClaudeCode, Codex, Configure, Error, Opencode, Settings, Version, Wire};
use rstest::rstest;

fn settings(wire: Wire) -> Settings {
    Settings {
        base_url: "http://localhost:4000/".to_owned(),
        api_key: "sk-test \"quoted\"".to_owned(),
        model: "some-model".to_owned(),
        wire,
    }
}

fn version() -> Version {
    Version::new(1, 2, 3)
}

#[rstest]
#[case(&ClaudeCode, Wire::Messages)]
#[case(&Codex, Wire::Responses)]
#[case(&Opencode, Wire::ChatCompletions)]
fn every_agent_runs_inside_the_given_home(#[case] agent: &impl Configure, #[case] wire: Wire) {
    let home = Path::new("/scratch/home");

    let spec = agent.configure(&version(), &settings(wire), home).unwrap();

    assert_eq!(spec.env["HOME"], "/scratch/home");
    assert!(
        spec.env
            .iter()
            .filter(|(key, _)| key.ends_with("_HOME") || key.as_str() == "CLAUDE_CONFIG_DIR")
            .all(|(_, value)| value.starts_with("/scratch/home"))
    );
    assert!(spec.files.keys().all(|path| path.is_relative()));
}

#[rstest]
#[case::claude_code(&ClaudeCode, &[Wire::ChatCompletions, Wire::Responses])]
#[case::codex(&Codex, &[Wire::ChatCompletions, Wire::Messages])]
fn wires_an_agent_cannot_speak_are_refused(
    #[case] agent: &impl Configure,
    #[case] refused: &[Wire],
) {
    refused.iter().for_each(|wire| {
        let result = agent.configure(&version(), &settings(*wire), Path::new("/h"));

        assert!(matches!(result, Err(Error::UnsupportedWire { wire: got, .. }) if got == *wire));
    });
}

#[test]
fn claude_code_points_at_the_gateway_root_with_the_key_and_model() {
    let spec = ClaudeCode
        .configure(&version(), &settings(Wire::Messages), Path::new("/h"))
        .unwrap();

    assert_eq!(spec.env["ANTHROPIC_BASE_URL"], "http://localhost:4000/");
    assert_eq!(spec.env["ANTHROPIC_AUTH_TOKEN"], "sk-test \"quoted\"");
    assert_eq!(spec.env["ANTHROPIC_MODEL"], "some-model");
}

#[test]
fn codex_config_is_valid_toml_routing_the_responses_api_to_the_gateway() {
    let dir = tempfile::tempdir().unwrap();
    let spec = Codex
        .configure(&version(), &settings(Wire::Responses), dir.path())
        .unwrap();
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

#[rstest]
#[case(Wire::ChatCompletions)]
#[case(Wire::Responses)]
#[case(Wire::Messages)]
fn opencode_config_is_valid_json_registering_the_gateway_model(#[case] wire: Wire) {
    let dir = tempfile::tempdir().unwrap();
    let spec = Opencode
        .configure(&version(), &settings(wire), dir.path())
        .unwrap();
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

#[test]
fn opencode_uses_a_different_provider_package_for_every_wire() {
    let package = |wire| {
        let dir = tempfile::tempdir().unwrap();
        let spec = Opencode
            .configure(&version(), &settings(wire), dir.path())
            .unwrap();
        let config: serde_json::Value =
            serde_json::from_str(spec.files.values().next().unwrap()).unwrap();
        config["provider"]["litellm"]["npm"]
            .as_str()
            .unwrap()
            .to_owned()
    };
    let packages = [Wire::ChatCompletions, Wire::Responses, Wire::Messages].map(package);

    assert_eq!(
        packages
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        packages.len()
    );
}
