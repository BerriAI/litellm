use std::{
    collections::{BTreeSet, VecDeque},
    path::{Path, PathBuf},
};

use serde::Deserialize;
use serde_yaml_ng::{Mapping, Value};

use crate::Error;

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
    serde_yaml_ng::from_str(&text).map_err(|source| Error::ParseFile {
        path: path.to_owned(),
        source,
    })
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
