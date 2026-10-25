use std::collections::{BTreeSet, VecDeque};

use serde_yaml_ng::{Mapping, Value};

use super::Store;
use crate::Error;

#[derive(Debug)]
pub struct Document<K> {
    pub key: K,
    pub body: Mapping,
}

pub fn documents<S: Store>(store: &S, root: S::Key) -> Result<Vec<Document<S::Key>>, Error> {
    let mut pending = VecDeque::from([root.clone()]);
    let mut loaded = BTreeSet::new();
    let mut documents = Vec::new();
    while let Some(key) = pending.pop_front() {
        if !loaded.insert(key.clone()) {
            continue;
        }
        let (body, includes) = read(store, &key).map_err(|error| Error::document(&key, error))?;
        for entry in includes {
            pending.push_back(store.locate(&entry, &key, &root)?);
        }
        documents.push(Document { key, body });
    }
    Ok(documents)
}

fn read<S: Store>(store: &S, key: &S::Key) -> Result<(Mapping, Vec<String>), Error> {
    let mut body = litellm_config::yaml::mapping(&store.read(key)?)?;
    let includes = body
        .remove(Value::from("include"))
        .map(serde_yaml_ng::from_value)
        .transpose()?
        .unwrap_or_default();
    Ok((body, includes))
}
