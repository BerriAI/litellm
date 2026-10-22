//! Drives the real agents through a real gateway. Run with `cargo test -p litellm-testkit --test live -- --ignored`
//! after exporting `TESTKIT_GATEWAY_URL`, `TESTKIT_GATEWAY_KEY`, one `TESTKIT_MODEL_<WIRE>` per wire
//! (`MESSAGES`, `RESPONSES`, `CHAT_COMPLETIONS`) and one `TESTKIT_<AGENT>_VERSION` per agent
//! (`CLAUDE`, `CODEX`, `OPENCODE`). `TESTKIT_CACHE_DIR` and `GITHUB_TOKEN` are optional.

use std::path::PathBuf;
use std::time::Duration;

use litellm_testkit::{
    Agent, ClaudeCode, Codex, HttpFetch, Installer, Opencode, Outcome, Prompt, Session, Settings,
    Target, Version, Wire,
};
use rstest::rstest;

const LIMIT: Duration = Duration::from_secs(180);

fn required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set to run the live tests"))
}

fn model_var(wire: Wire) -> &'static str {
    match wire {
        Wire::Messages => "TESTKIT_MODEL_MESSAGES",
        Wire::Responses => "TESTKIT_MODEL_RESPONSES",
        Wire::ChatCompletions => "TESTKIT_MODEL_CHAT_COMPLETIONS",
    }
}

async fn drive(
    agent: &impl Agent,
    version_var: &str,
    wire: Wire,
    model: Option<&str>,
    prompt: Prompt,
) -> Outcome {
    let cache = std::env::var("TESTKIT_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("litellm-testkit-cache"));
    let installer = Installer::new(HttpFetch::from_env(), cache, Target::host().unwrap());
    let installed = installer
        .install(agent, &Version::parse(&required(version_var)).unwrap())
        .await
        .unwrap();
    let settings = Settings {
        base_url: required("TESTKIT_GATEWAY_URL"),
        api_key: required("TESTKIT_GATEWAY_KEY"),
        model: model.map_or_else(|| required(model_var(wire)), str::to_owned),
        wire,
    };
    let home = tempfile::tempdir().unwrap();
    let session = Session::prepare(agent, &installed, settings, home.path()).unwrap();
    session.run(agent, &prompt, LIMIT).await.unwrap()
}

fn text_prompt() -> Prompt {
    Prompt {
        text: "Reply with the single word: pong".to_owned(),
        allow_tools: false,
    }
}

fn tool_prompt() -> Prompt {
    Prompt {
        text: "Run the shell command 'echo tool-ok' and reply with exactly its output.".to_owned(),
        allow_tools: true,
    }
}

#[rstest]
#[case::claude_messages(&ClaudeCode, "TESTKIT_CLAUDE_VERSION", Wire::Messages)]
#[case::codex_responses(&Codex, "TESTKIT_CODEX_VERSION", Wire::Responses)]
#[case::opencode_chat(&Opencode, "TESTKIT_OPENCODE_VERSION", Wire::ChatCompletions)]
#[case::opencode_responses(&Opencode, "TESTKIT_OPENCODE_VERSION", Wire::Responses)]
#[case::opencode_messages(&Opencode, "TESTKIT_OPENCODE_VERSION", Wire::Messages)]
#[ignore = "needs a live gateway, see the module docs"]
#[tokio::test]
async fn plain_prompt_gets_an_answer_and_token_usage(
    #[case] agent: &impl Agent,
    #[case] version_var: &str,
    #[case] wire: Wire,
) {
    let outcome = drive(agent, version_var, wire, None, text_prompt()).await;

    assert!(outcome.succeeded(), "{outcome:?}");
    assert!(outcome.text.to_lowercase().contains("pong"), "{outcome:?}");
    assert!(outcome.usage.output_tokens > 0, "{outcome:?}");
}

#[rstest]
#[case::claude_messages(&ClaudeCode, "TESTKIT_CLAUDE_VERSION", Wire::Messages)]
#[case::codex_responses(&Codex, "TESTKIT_CODEX_VERSION", Wire::Responses)]
#[case::opencode_chat(&Opencode, "TESTKIT_OPENCODE_VERSION", Wire::ChatCompletions)]
#[case::opencode_responses(&Opencode, "TESTKIT_OPENCODE_VERSION", Wire::Responses)]
#[case::opencode_messages(&Opencode, "TESTKIT_OPENCODE_VERSION", Wire::Messages)]
#[ignore = "needs a live gateway, see the module docs"]
#[tokio::test]
async fn tool_use_is_reported_and_its_result_reaches_the_answer(
    #[case] agent: &impl Agent,
    #[case] version_var: &str,
    #[case] wire: Wire,
) {
    let outcome = drive(agent, version_var, wire, None, tool_prompt()).await;

    assert!(outcome.succeeded(), "{outcome:?}");
    assert!(!outcome.tool_calls.is_empty(), "{outcome:?}");
    assert!(outcome.text.contains("tool-ok"), "{outcome:?}");
}

#[rstest]
#[case::claude_messages(&ClaudeCode, "TESTKIT_CLAUDE_VERSION", Wire::Messages)]
#[case::codex_responses(&Codex, "TESTKIT_CODEX_VERSION", Wire::Responses)]
#[case::opencode_chat(&Opencode, "TESTKIT_OPENCODE_VERSION", Wire::ChatCompletions)]
#[case::opencode_responses(&Opencode, "TESTKIT_OPENCODE_VERSION", Wire::Responses)]
#[case::opencode_messages(&Opencode, "TESTKIT_OPENCODE_VERSION", Wire::Messages)]
#[ignore = "needs a live gateway, see the module docs"]
#[tokio::test]
async fn model_the_gateway_rejects_is_reported_as_an_error(
    #[case] agent: &impl Agent,
    #[case] version_var: &str,
    #[case] wire: Wire,
) {
    let outcome = drive(
        agent,
        version_var,
        wire,
        Some("testkit-no-such-model"),
        text_prompt(),
    )
    .await;

    assert!(!outcome.succeeded(), "{outcome:?}");
    assert!(!outcome.errors.is_empty(), "{outcome:?}");
}
