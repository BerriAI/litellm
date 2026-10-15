use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use semver::Version;
use tokio::process::Command;
use tokio::time::timeout;

use crate::{Configure, Drive, Error, Installed, Outcome, Prompt, Settings};

const STDERR_LIMIT_CHARS: usize = 2000;

pub struct Session {
    binary: PathBuf,
    home: PathBuf,
    version: Version,
    settings: Settings,
    env: BTreeMap<String, String>,
}

impl Session {
    pub fn prepare(
        agent: &impl Configure,
        installed: &Installed,
        settings: Settings,
        home: impl Into<PathBuf>,
    ) -> Result<Self, Error> {
        let home = home.into();
        let spec = agent.configure(&installed.version, &settings, &home)?;
        spec.write_files(&home)?;
        Ok(Self {
            binary: installed.binary.clone(),
            home,
            version: installed.version.clone(),
            settings,
            env: spec.env,
        })
    }

    pub async fn run(
        &self,
        agent: &impl Drive,
        prompt: &Prompt,
        limit: Duration,
    ) -> Result<Outcome, Error> {
        let child = Command::new(&self.binary)
            .args(agent.args(&self.version, &self.settings, prompt))
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .envs(&self.env)
            .current_dir(&self.home)
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .output();
        let output = timeout(limit, child)
            .await
            .map_err(|_| Error::Timeout(limit))??;
        let parsed = agent.parse(&self.version, &String::from_utf8_lossy(&output.stdout));
        let failed_silently = !output.status.success() && parsed.errors.is_empty();
        Ok(Outcome {
            errors: if failed_silently {
                vec![
                    String::from_utf8_lossy(&output.stderr)
                        .chars()
                        .take(STDERR_LIMIT_CHARS)
                        .collect(),
                ]
            } else {
                parsed.errors
            },
            exit_code: output.status.code(),
            ..parsed
        })
    }
}
