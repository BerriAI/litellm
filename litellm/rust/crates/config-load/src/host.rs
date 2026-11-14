use std::{
    fmt,
    path::{Path, PathBuf},
};

use litellm_coroutine::{Co, Reply};
use serde_yaml_ng::Value;

use crate::{Error, Location};

pub enum HostCall<E> {
    Exists(PathBuf, Reply<bool>),
    Read(Location, Reply<Result<Value, E>>),
    Warn(Warning, Reply<()>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Warning {
    ReadBesideRoot {
        entry: String,
        declared_in: PathBuf,
        read: PathBuf,
    },
    MatchesTwoFiles {
        entry: String,
        declared_in: PathBuf,
        read: PathBuf,
        skipped: PathBuf,
    },
}

impl fmt::Display for Warning {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadBesideRoot {
                entry,
                declared_in,
                read,
            } => write!(
                formatter,
                "Config include '{entry}' declared in {} was not found next to it, so {} was read instead. \
                 Move the included file next to the config that declares it.",
                declared_in.display(),
                read.display(),
            ),
            Self::MatchesTwoFiles {
                entry,
                declared_in,
                read,
                skipped,
            } => write!(
                formatter,
                "Config include '{entry}' declared in {} matches two files. {} sits next to that config and was \
                 read, so {} was skipped. Rename one of the two to say which one you meant.",
                declared_in.display(),
                read.display(),
                skipped.display(),
            ),
        }
    }
}

pub(crate) struct Host<E> {
    co: Co<HostCall<E>>,
}

impl<E> Host<E> {
    pub(crate) fn new(co: Co<HostCall<E>>) -> Self {
        Self { co }
    }

    pub(crate) async fn exists(&self, path: &Path) -> Result<bool, Error<E>> {
        Ok(self
            .co
            .yield_(|reply| HostCall::Exists(path.to_owned(), reply))
            .await?)
    }

    pub(crate) async fn read(&self, location: &Location) -> Result<Value, Error<E>> {
        self.co
            .yield_(|reply| HostCall::Read(location.clone(), reply))
            .await?
            .map_err(Error::Host)
    }

    pub(crate) async fn warn(&self, warning: Warning) -> Result<(), Error<E>> {
        Ok(self
            .co
            .yield_(|reply| HostCall::Warn(warning, reply))
            .await?)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::beside_root(
        Warning::ReadBesideRoot {
            entry: "more_models.yaml".into(),
            declared_in: "/etc/litellm/shared/models.yaml".into(),
            read: "/etc/litellm/more_models.yaml".into(),
        },
        "Config include 'more_models.yaml' declared in /etc/litellm/shared/models.yaml was not found next to it, \
         so /etc/litellm/more_models.yaml was read instead. Move the included file next to the config that \
         declares it."
    )]
    #[case::two_files(
        Warning::MatchesTwoFiles {
            entry: "more_models.yaml".into(),
            declared_in: "/etc/litellm/shared/models.yaml".into(),
            read: "/etc/litellm/shared/more_models.yaml".into(),
            skipped: "/etc/litellm/more_models.yaml".into(),
        },
        "Config include 'more_models.yaml' declared in /etc/litellm/shared/models.yaml matches two files. \
         /etc/litellm/shared/more_models.yaml sits next to that config and was read, so \
         /etc/litellm/more_models.yaml was skipped. Rename one of the two to say which one you meant."
    )]
    fn a_warning_reads_as_python_logged_it(#[case] warning: Warning, #[case] message: &str) {
        assert_eq!(warning.to_string(), message);
    }
}
