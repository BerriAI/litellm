use std::{
    fmt,
    path::{Path, PathBuf},
};

use crate::Error;

pub trait Store {
    type Key: Ord + Clone + fmt::Debug;

    fn locate(
        &self,
        entry: &str,
        declaring: &Self::Key,
        root: &Self::Key,
    ) -> Result<Self::Key, Error>;

    fn read(&self, key: &Self::Key) -> Result<String, Error>;
}

pub struct Files;

impl Store for Files {
    type Key = PathBuf;

    fn locate(&self, entry: &str, declaring: &PathBuf, root: &PathBuf) -> Result<PathBuf, Error> {
        let declared = sibling(declaring, entry);
        let fallback = sibling(root, entry);
        let location = if declared.exists() || !fallback.exists() {
            declared
        } else {
            fallback
        };
        location
            .canonicalize()
            .map_err(|error| Error::document(&location, error.into()))
    }

    fn read(&self, key: &PathBuf) -> Result<String, Error> {
        Ok(std::fs::read_to_string(key)?)
    }
}

fn sibling(file: &Path, entry: &str) -> PathBuf {
    file.parent().unwrap_or(Path::new(".")).join(entry)
}
