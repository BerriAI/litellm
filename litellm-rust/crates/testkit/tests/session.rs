use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use litellm_testkit::{
    Configure, Drive, Error, Installed, LaunchSpec, Outcome, Prompt, Session, Settings, Version,
    Wire,
};

struct Scripted;

impl Configure for Scripted {
    fn configure(
        &self,
        version: &Version,
        _settings: &Settings,
        home: &Path,
    ) -> Result<LaunchSpec, Error> {
        Ok(LaunchSpec {
            env: [
                ("AGENT_HOME".to_owned(), home.to_string_lossy().into_owned()),
                ("AGENT_SAW_VERSION".to_owned(), version.to_string()),
            ]
            .into(),
            files: [(
                PathBuf::from("conf/agent.toml"),
                "configured = true\n".to_owned(),
            )]
            .into(),
        })
    }
}

impl Drive for Scripted {
    fn args(&self, _version: &Version, _settings: &Settings, prompt: &Prompt) -> Vec<String> {
        vec!["--prompt".to_owned(), prompt.text.clone()]
    }

    fn parse(&self, _version: &Version, stdout: &str) -> Outcome {
        Outcome {
            text: stdout.to_owned(),
            ..Outcome::default()
        }
    }
}

fn settings() -> Settings {
    Settings {
        base_url: "http://gateway.test".to_owned(),
        api_key: "sk-test".to_owned(),
        model: "some-model".to_owned(),
        wire: Wire::Messages,
    }
}

fn prompt(text: &str) -> Prompt {
    Prompt {
        text: text.to_owned(),
        allow_tools: false,
    }
}

fn session(script: &str) -> (Session, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let binary = dir.path().join("agent");
    std::fs::write(&binary, format!("#!/bin/sh\n{script}\n")).unwrap();
    std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755)).unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir(&home).unwrap();
    let installed = Installed {
        version: Version::new(4, 5, 6),
        binary,
    };
    (
        Session::prepare(&Scripted, &installed, settings(), home).unwrap(),
        dir,
    )
}

const LIMIT: Duration = Duration::from_secs(20);

#[tokio::test]
async fn prepare_writes_the_config_files_under_home() {
    let (_session, dir) = session("true");

    let written = std::fs::read_to_string(dir.path().join("home/conf/agent.toml")).unwrap();

    assert_eq!(written, "configured = true\n");
}

#[tokio::test]
async fn configure_and_drive_are_given_the_installed_version() {
    let (session, _dir) = session("echo \"$AGENT_SAW_VERSION\"");

    let outcome = session.run(&Scripted, &prompt("hi"), LIMIT).await.unwrap();

    assert_eq!(outcome.text.trim(), "4.5.6");
}

#[tokio::test]
async fn agent_runs_in_home_with_only_its_own_environment() {
    let (session, dir) = session("pwd -P; env");

    let outcome = session.run(&Scripted, &prompt("hi"), LIMIT).await.unwrap();

    let home = dir.path().join("home").canonicalize().unwrap();
    assert_eq!(outcome.text.lines().next().unwrap(), home.to_string_lossy());
    assert!(outcome.text.contains("AGENT_HOME="));
    assert!(
        !outcome.text.contains("CARGO_"),
        "test runner environment leaked into the agent"
    );
}

#[tokio::test]
async fn prompt_reaches_the_agent_as_one_untouched_argument() {
    let (session, _dir) = session("printf '%s|' \"$@\"");
    let text = "two  spaces; $(echo injected) 'quoted'";

    let outcome = session.run(&Scripted, &prompt(text), LIMIT).await.unwrap();

    assert_eq!(outcome.text, format!("--prompt|{text}|"));
}

#[tokio::test]
async fn clean_exit_is_a_success() {
    let (session, _dir) = session("echo done");

    let outcome = session.run(&Scripted, &prompt("hi"), LIMIT).await.unwrap();

    assert_eq!(outcome.exit_code, Some(0));
    assert!(outcome.succeeded());
}

#[tokio::test]
async fn failing_exit_without_a_parsed_error_reports_stderr() {
    let (session, _dir) = session("echo boom >&2; exit 3");

    let outcome = session.run(&Scripted, &prompt("hi"), LIMIT).await.unwrap();

    assert_eq!(outcome.exit_code, Some(3));
    assert!(!outcome.succeeded());
    assert_eq!(outcome.errors, ["boom\n"]);
}

#[tokio::test]
async fn agent_that_outlives_the_limit_is_stopped() {
    let (session, _dir) = session("sleep 30");

    let result = session
        .run(&Scripted, &prompt("hi"), Duration::from_millis(200))
        .await;

    assert!(matches!(result, Err(Error::Timeout(_))));
}
