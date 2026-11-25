use std::{
    collections::{BTreeSet, VecDeque},
    path::{Path, PathBuf},
};

use litellm_python_compat::lenient::Truthy;
use serde::Deserialize;
use serde_with::DeserializeAs;
use serde_yaml_ng::{Mapping, Value};

use crate::Error;

const SETTINGS_SECTIONS: [&str; 3] = ["general_settings", "litellm_settings", "router_settings"];

#[derive(Deserialize)]
struct Includes {
    #[serde(default)]
    include: Vec<String>,
}

fn read(path: &Path) -> Result<Mapping, Error> {
    let text = std::fs::read_to_string(path).map_err(|source| Error::Read {
        path: path.to_owned(),
        source,
    })?;
    let document = litellm_yaml::from_str(&text).map_err(|source| Error::LoadFile {
        path: path.to_owned(),
        source,
    })?;
    let config: Mapping =
        serde_yaml_ng::from_value(document).map_err(|source| Error::ParseFile {
            path: path.to_owned(),
            source,
        })?;
    Ok(config
        .into_iter()
        .filter(|(key, value)| !is_empty_settings_section(key, value))
        .collect())
}

fn is_empty_settings_section(key: &Value, value: &Value) -> bool {
    key.as_str()
        .is_some_and(|key| SETTINGS_SECTIONS.contains(&key))
        && !Truthy::deserialize_as(value).unwrap_or(true)
}

fn canonical(path: &Path) -> Result<PathBuf, Error> {
    path.canonicalize().map_err(|source| Error::Read {
        path: path.to_owned(),
        source,
    })
}

fn entries(config: &Mapping, path: &Path) -> Result<Vec<(String, PathBuf)>, Error> {
    let includes: Includes =
        serde_yaml_ng::from_value(Value::Mapping(config.clone())).map_err(|source| {
            Error::ParseFile {
                path: path.to_owned(),
                source,
            }
        })?;
    Ok(includes
        .include
        .into_iter()
        .map(|entry| (entry, path.to_owned()))
        .collect())
}

pub(super) fn load(path: &Path) -> Result<Value, Error> {
    let root = canonical(path)?;
    let mut merged = read(&root)?;
    let mut pending: VecDeque<_> = entries(&merged, &root)?.into();
    let mut loaded = BTreeSet::from([root.clone()]);
    merged.remove(Value::String("include".into()));
    while let Some((entry, declaring)) = pending.pop_front() {
        let declared = declaring.parent().unwrap_or(Path::new(".")).join(&entry);
        let fallback = root.parent().unwrap_or(Path::new(".")).join(&entry);
        let location = canonical(if declared.exists() {
            &declared
        } else {
            &fallback
        })?;
        if !loaded.insert(location.clone()) {
            continue;
        }
        let mut included = read(&location)?;
        pending.extend(entries(&included, &location)?);
        included.remove(Value::String("include".into()));
        for (key, value) in included {
            match (merged.get_mut(&key), value) {
                (Some(Value::Sequence(base)), Value::Sequence(extra)) => base.extend(extra),
                (_, value) => {
                    merged.insert(key, value);
                }
            }
        }
    }
    Ok(Value::Mapping(merged))
}
