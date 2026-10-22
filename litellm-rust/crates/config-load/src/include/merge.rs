use std::collections::BTreeMap;

use serde_yaml_ng::{Mapping, Value};

use super::Document;

#[derive(Debug)]
pub struct Merged<K> {
    pub document: Mapping,
    origins: BTreeMap<String, Vec<(K, usize)>>,
}

impl<K> Merged<K> {
    /// Maps item `index` of the merged top-level `field` back to its document and its index there.
    /// A value that is not a list is item 0.
    pub fn locate(&self, field: &str, index: usize) -> Option<(&K, usize)> {
        self.origins
            .get(field)?
            .iter()
            .scan(0, |start, (key, len)| {
                let run = (key, *start, *len);
                *start += len;
                Some(run)
            })
            .find(|(_, start, len)| (*start..start + len).contains(&index))
            .map(|(key, start, _)| (key, index - start))
    }
}

pub fn merge<K: Clone>(documents: &[Document<K>]) -> Merged<K> {
    let fields = documents
        .iter()
        .flat_map(|document| document.body.keys())
        .fold(Vec::new(), |mut fields, field| {
            if !fields.contains(&field) {
                fields.push(field);
            }
            fields
        });
    let (document, origins) = fields
        .into_iter()
        .map(|field| {
            let layers: Vec<_> = documents
                .iter()
                .filter_map(|document| Some((&document.key, document.body.get(field)?)))
                .collect();
            (field, combine(&layers))
        })
        .map(|(field, (value, runs))| {
            (
                (field.clone(), value),
                (field.as_str().map(str::to_owned), runs),
            )
        })
        .unzip::<_, _, Mapping, Vec<_>>();
    Merged {
        document,
        origins: origins
            .into_iter()
            .filter_map(|(field, runs)| Some((field?, runs)))
            .collect(),
    }
}

/// Python's rule (config L16): a list extends a list, anything else replaces. So the merged value is the
/// lists after the last non-list concatenated, or that non-list alone when nothing follows it.
fn combine<K: Clone>(layers: &[(&K, &Value)]) -> (Value, Vec<(K, usize)>) {
    let tail = layers
        .iter()
        .rposition(|(_, value)| !value.is_sequence())
        .map_or(0, |last| last + 1);
    match &layers[tail..] {
        [] => {
            let (key, value) = layers[layers.len() - 1];
            (value.clone(), vec![(key.clone(), 1)])
        }
        lists => (
            Value::Sequence(
                lists
                    .iter()
                    .flat_map(|(_, value)| value.as_sequence().into_iter().flatten().cloned())
                    .collect(),
            ),
            lists
                .iter()
                .map(|(key, value)| ((*key).clone(), value.as_sequence().map_or(0, Vec::len)))
                .collect(),
        ),
    }
}
