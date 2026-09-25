use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gateway {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchSpec {
    pub env: BTreeMap<String, String>,
    pub files: BTreeMap<PathBuf, String>,
}

impl LaunchSpec {
    pub fn write_files(&self, home: &Path) -> std::io::Result<()> {
        self.files.iter().try_for_each(|(relative, contents)| {
            let path = home.join(relative);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, contents)
        })
    }
}

pub(crate) fn env(
    pairs: impl IntoIterator<Item = (&'static str, String)>,
) -> BTreeMap<String, String> {
    pairs
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect()
}

pub(crate) fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

pub(crate) fn quoted(value: &str) -> String {
    serde_json::Value::from(value).to_string()
}

pub(crate) fn v1(gateway: &Gateway) -> String {
    format!("{}/v1", gateway.base_url.trim_end_matches('/'))
}
