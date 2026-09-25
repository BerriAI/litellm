use std::io;
use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("unsupported target {0}")]
    UnsupportedTarget(String),
    #[error("'{0}' is not a plain x.y.z release version")]
    InvalidVersion(String),
    #[error("request to {url} failed")]
    Request {
        url: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("{url} answered with status {status}")]
    Status { url: String, status: u16 },
    #[error("release metadata at {url} is malformed")]
    Metadata {
        url: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("release has no asset named {0}")]
    AssetNotFound(String),
    #[error("release publishes no sha256 for {0}")]
    MissingChecksum(String),
    #[error("sha256 mismatch for {asset}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        asset: String,
        expected: String,
        actual: String,
    },
    #[error("archive does not contain {0}")]
    ArchiveMemberNotFound(String),
    #[error("archive is unreadable")]
    Archive(#[source] io::Error),
    #[error("zip archive is unreadable")]
    Zip(#[from] zip::result::ZipError),
    #[error("{binary} reports version '{reported}', expected {expected}")]
    VersionMismatch {
        binary: PathBuf,
        expected: String,
        reported: String,
    },
    #[error("io failure")]
    Io(#[from] io::Error),
}
