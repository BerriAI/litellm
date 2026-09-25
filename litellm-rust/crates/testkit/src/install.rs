use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;

use tokio::fs;
use tokio::process::Command;

use crate::archive::{extract_binary, verify_sha256};
use crate::release::resolve;
use crate::{Client, Error, Fetch, Platform};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Installed {
    pub client: Client,
    pub version: String,
    pub binary: PathBuf,
}

pub struct Installer<F> {
    fetch: F,
    cache_root: PathBuf,
    platform: Platform,
}

impl<F: Fetch> Installer<F> {
    pub fn new(fetch: F, cache_root: impl Into<PathBuf>, platform: Platform) -> Self {
        Self {
            fetch,
            cache_root: cache_root.into(),
            platform,
        }
    }

    pub async fn install(&self, client: Client, version: &str) -> Result<Installed, Error> {
        validate_version(version)?;
        let dir = self.cache_root.join(client.binary()).join(version);
        let binary = dir.join(client.binary());
        let installed = Installed {
            client,
            version: version.to_owned(),
            binary: binary.clone(),
        };
        if fs::try_exists(&binary).await? && probe_version(&binary, version).await.is_ok() {
            return Ok(installed);
        }

        let release = resolve(&self.fetch, client, version, self.platform).await?;
        let archive = self.fetch.get(&release.url).await?;
        verify_sha256(&release.asset, &release.sha256, &archive)?;
        let contents = extract_binary(&release.packaging, &archive)?;

        fs::create_dir_all(&dir).await?;
        let staging = dir.join(format!(".{}.partial", client.binary()));
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
