use std::path::PathBuf;

use url::Url;

/// Where a fetched catalog comes from
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    Http(Url),
    File(PathBuf),
    S3(S3Object),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S3Object {
    pub bucket: String,
    pub key: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceError {
    UnsupportedScheme { scheme: String },
    Invalid { reason: String },
}
