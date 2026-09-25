mod archive;
mod fetch;
pub(crate) mod release;

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;

use tokio::fs;
use tokio::process::Command;

use crate::{Agent, Error, Target};
use archive::{extract_binary, verify_sha256};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Installed {
    pub version: String,
    pub binary: PathBuf,
}

pub struct Installer<F> {
    fetch: F,
    cache_root: PathBuf,
    target: Target,
}

impl<F: Fetch> Installer<F> {
    pub fn new(fetch: F, cache_root: impl Into<PathBuf>, target: Target) -> Self {
        Self {
            fetch,
            cache_root: cache_root.into(),
            target,
        }
    }

    pub async fn install(&self, agent: &impl Agent, version: &str) -> Result<Installed, Error> {
        validate_version(version)?;
        let dir = self.cache_root.join(agent.binary()).join(version);
        let binary = dir.join(agent.binary());
        let installed = Installed {
            version: version.to_owned(),
            binary: binary.clone(),
        };
        if fs::try_exists(&binary).await? && probe_version(&binary, version).await.is_ok() {
            return Ok(installed);
        }

        let release = agent.release(&self.fetch, version, self.target).await?;
        let archive = self.fetch.get(&release.url).await?;
        verify_sha256(&release.asset, &release.sha256, &archive)?;
        let contents = extract_binary(&release.packaging, &archive)?;

        fs::create_dir_all(&dir).await?;
        let staging = dir.join(format!(".{}.partial", agent.binary()));
        fs::write(&staging, contents).await?;
        fs::set_permissions(&staging, std::fs::Permissions::from_mode(0o755)).await?;
        fs::rename(&staging, &binary).await?;

        match probe_version(&binary, version).await {
            Ok(()) => Ok(installed),
            Err(error) => {
                fs::remove_file(&binary).await?;
                Err(error)
            }
        }
    }
}

fn validate_version(version: &str) -> Result<(), Error> {
    let parts: Vec<&str> = version.split('.').collect();
    let plain = parts.len() == 3
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()));
    if plain {
        return Ok(());
    }
    Err(Error::InvalidVersion(version.to_owned()))
}

async fn probe_version(binary: &Path, expected: &str) -> Result<(), Error> {
    let home = std::env::temp_dir();
    let output = Command::new(binary)
        .arg("--version")
        .env_clear()
        .env("HOME", home)
        .env("DISABLE_AUTOUPDATER", "1")
        .stdin(Stdio::null())
        .output()
        .await?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.split_whitespace().any(|token| token == expected) {
        return Ok(());
    }
    Err(Error::VersionMismatch {
        binary: binary.to_owned(),
        expected: expected.to_owned(),
        reported: stdout.trim().to_owned(),
    })
}

pub use fetch::{Fetch, HttpFetch};
pub use release::{Packaging, Release};
