mod archive;
mod fetch;
pub(crate) mod release;

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};

use semver::Version;
use tokio::fs;
use tokio::process::Command;

use crate::{Error, Install, Target};
use archive::{extract_binary, verify_sha256};

static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Installed {
    pub version: Version,
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

    pub async fn install(
        &self,
        agent: &impl Install,
        version: &Version,
    ) -> Result<Installed, Error> {
        validate_release(version)?;
        let dir = self
            .cache_root
            .join(agent.binary())
            .join(version.to_string());
        let binary = dir.join(agent.binary());
        let installed = Installed {
            version: version.clone(),
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
        let staging = dir.join(format!(
            ".{}.{}.{}.partial",
            agent.binary(),
            std::process::id(),
            STAGING_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
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

fn validate_release(version: &Version) -> Result<(), Error> {
    if version.pre.is_empty() && version.build.is_empty() {
        return Ok(());
    }
    Err(Error::InvalidVersion(version.to_string()))
}

async fn probe_version(binary: &Path, expected: &Version) -> Result<(), Error> {
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
    if stdout
        .split_whitespace()
        .filter_map(|token| Version::parse(token).ok())
        .any(|reported| &reported == expected)
    {
        return Ok(());
    }
    Err(Error::VersionMismatch {
        binary: binary.to_owned(),
        expected: expected.to_string(),
        reported: stdout.trim().to_owned(),
    })
}

pub use fetch::{Fetch, HttpFetch};
pub use release::{Packaging, Release};
