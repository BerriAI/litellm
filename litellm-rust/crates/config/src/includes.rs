use std::{
    collections::{BTreeSet, VecDeque},
    convert::Infallible,
    path::{Component, Path, PathBuf},
};

use indexmap::IndexMap;
use serde_yaml_ng::{Mapping, Value};

use crate::{Document, Error};

pub trait IncludeValue: Sized {
    type Failure;

    fn sequence(&self) -> Option<Vec<Self>>;
    fn string(&self) -> Option<String>;
    fn with_sequence(&self, values: Vec<Self>) -> Result<Self, Self::Failure>;
}

impl IncludeValue for Value {
    type Failure = Infallible;

    fn sequence(&self) -> Option<Vec<Self>> {
        self.as_sequence().cloned()
    }

    fn string(&self) -> Option<String> {
        self.as_str().map(str::to_owned)
    }

    fn with_sequence(&self, values: Vec<Self>) -> Result<Self, Self::Failure> {
        Ok(Self::Sequence(values))
    }
}

pub fn include_entries<V: IncludeValue>(value: Option<&V>) -> Result<Vec<String>, Error> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    value
        .sequence()
        .ok_or(Error::Includes)?
        .iter()
        .map(|entry| entry.string().ok_or(Error::Includes))
        .collect()
}

pub fn merge<V: IncludeValue>(
    base: impl IntoIterator<Item = (String, V)>,
    included: impl IntoIterator<Item = (String, V)>,
) -> Result<IndexMap<String, V>, V::Failure> {
    let mut result: IndexMap<_, _> = base
        .into_iter()
        .filter(|(key, _)| key != "include")
        .collect();
    for (key, value) in included.into_iter().filter(|(key, _)| key != "include") {
        let merged = match (result.get(&key).and_then(V::sequence), value.sequence()) {
            (Some(base), Some(extra)) => {
                value.with_sequence(base.into_iter().chain(extra).collect())?
            }
            _ => value,
        };
        result.insert(key, merged);
    }
    Ok(result)
}

pub struct IncludePlan {
    pending: VecDeque<(String, String)>,
    loaded: BTreeSet<String>,
}

impl IncludePlan {
    pub fn new(location: String, entries: Vec<String>) -> Self {
        let mut plan = Self {
            pending: VecDeque::new(),
            loaded: BTreeSet::from([location.clone()]),
        };
        plan.extend(&location, entries);
        plan
    }

    pub fn pop(&mut self) -> Option<(String, String)> {
        self.pending.pop_front()
    }

    pub fn mark_loaded(&mut self, location: String) -> bool {
        self.loaded.insert(location)
    }

    pub fn extend(&mut self, location: &str, entries: Vec<String>) {
        self.pending.extend(
            entries
                .into_iter()
                .map(|entry| (entry, location.to_owned())),
        );
    }
}

pub struct IncludePath {
    pub location: PathBuf,
    pub warning: Option<String>,
}

fn absolute(path: &Path) -> Result<PathBuf, std::io::Error> {
    let joined = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    };
    Ok(joined
        .components()
        .fold(PathBuf::new(), |mut normalized, component| {
            match component {
                Component::CurDir => {}
                Component::ParentDir => {
                    normalized.pop();
                }
                other => normalized.push(other),
            }
            normalized
        }))
}

pub fn resolve_include_file_path(
    entry: &str,
    declared_in: &Path,
    root: &Path,
) -> Result<IncludePath, Error> {
    let declared = absolute(&declared_in.parent().unwrap_or(Path::new("")).join(entry))?;
    let fallback = absolute(&root.parent().unwrap_or(Path::new("")).join(entry))?;
    if fallback == declared || !fallback.exists() {
        return Ok(IncludePath {
            location: declared,
            warning: None,
        });
    }
    if !declared.exists() {
        let warning = format!(
            "Config include '{entry}' declared in {} was not found next to it, so {} was read instead. Move the included file next to the config that declares it.",
            declared_in.display(),
            fallback.display(),
        );
        return Ok(IncludePath {
            location: fallback,
            warning: Some(warning),
        });
    }
    let warning = format!(
        "Config include '{entry}' declared in {} matches two files. {} sits next to that config and was read, so {} was skipped. Rename one of the two to say which one you meant.",
        declared_in.display(),
        declared.display(),
        fallback.display(),
    );
    Ok(IncludePath {
        location: declared,
        warning: Some(warning),
    })
}

fn read(path: &Path, root: bool) -> Result<Mapping, Error> {
    let document = Document::from_yaml(&std::fs::read_to_string(path)?)?;
    match document.value() {
        Value::Mapping(mapping) => Ok(mapping.clone()),
        value if !root && empty(value) => Ok(Mapping::new()),
        _ => Err(Error::Mapping(path.to_owned())),
    }
}

fn empty(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => true,
        Value::Number(number) => number.as_f64() == Some(0.0),
        Value::String(text) => text.is_empty(),
        Value::Sequence(values) => values.is_empty(),
        _ => false,
    }
}

fn string_entries(mapping: Mapping) -> impl Iterator<Item = (String, Value)> {
    mapping.into_iter().map(|(key, value)| {
        (
            key.as_str().expect("parsed keys are strings").to_owned(),
            value,
        )
    })
}

pub(crate) fn load(path: &Path) -> Result<Value, Error> {
    load_with_warnings(path, |warning| tracing::warn!("{warning}"))
}

pub fn load_with_warnings(path: &Path, mut warn: impl FnMut(&str)) -> Result<Value, Error> {
    let root = absolute(path)?;
    let mut merged = read(&root, true)?;
    let mut plan = IncludePlan::new(
        root.to_string_lossy().into_owned(),
        include_entries(merged.get("include"))?,
    );
    merged.shift_remove("include");
    while let Some((entry, declaring)) = plan.pop() {
        let resolved = resolve_include_file_path(&entry, Path::new(&declaring), &root)?;
        if let Some(warning) = resolved.warning {
            warn(&warning);
        }
        let location = resolved.location.to_string_lossy().into_owned();
        if !plan.mark_loaded(location.clone()) {
            continue;
        }
        let included = read(&resolved.location, false)?;
        plan.extend(&location, include_entries(included.get("include"))?);
        let result = merge(string_entries(merged), string_entries(included));
        let Ok(entries) = result;
        merged = entries
            .into_iter()
            .map(|(key, value)| (Value::String(key), value))
            .collect();
    }
    Ok(Value::Mapping(merged))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::{fixture, rstest};
    use tempfile::TempDir;

    #[fixture]
    fn directory() -> TempDir {
        tempfile::tempdir().unwrap()
    }

    #[rstest]
    #[case::fallback(false)]
    #[case::ambiguous(true)]
    fn resolves_paths_lexically_and_reports_legacy_locations(
        directory: TempDir,
        #[case] declared_exists: bool,
    ) {
        let root = directory.path().join("config.yaml");
        let declaring = directory.path().join("nested/first.yaml");
        let fallback = directory.path().join("extra.yaml");
        let declared = directory.path().join("nested/extra.yaml");
        std::fs::create_dir(directory.path().join("nested")).unwrap();
        std::fs::write(&fallback, "{}").unwrap();
        if declared_exists {
            std::fs::write(&declared, "{}").unwrap();
        }
        let resolved = resolve_include_file_path("./extra.yaml", &declaring, &root).unwrap();
        assert_eq!(
            resolved.location,
            if declared_exists {
                declared.clone()
            } else {
                fallback.clone()
            }
        );
        let warning = resolved.warning.unwrap();
        assert!(warning.contains(fallback.to_str().unwrap()));
        assert!(warning.contains(declaring.to_str().unwrap()));
        assert_eq!(warning.contains("matches two files"), declared_exists);
    }

    #[rstest]
    fn merges_lists_but_replaces_other_values_without_reordering_keys(directory: TempDir) {
        let path = directory.path().join("config.yaml");
        std::fs::write(&path, "include: [extra.yaml]\nitems: [first]\nsettings: {discard: true}\nzero: 3\nretained: null").unwrap();
        std::fs::write(
            directory.path().join("extra.yaml"),
            "items: [second]\nsettings: {enabled: false}\nzero: 0\nadded: null",
        )
        .unwrap();
        let document = Document::load(path).unwrap();
        let expected = Document::from_yaml("items: [first, second]\nsettings: {enabled: false}\nzero: 0\nretained: null\nadded: null").unwrap();
        assert_eq!(document.value(), expected.value());
        assert_eq!(
            document
                .value()
                .as_mapping()
                .unwrap()
                .keys()
                .collect::<Vec<_>>(),
            expected
                .value()
                .as_mapping()
                .unwrap()
                .keys()
                .collect::<Vec<_>>()
        );
    }

    #[rstest]
    #[case::empty("")]
    #[case::null("null")]
    #[case::disabled("false")]
    #[case::zero("0")]
    #[case::empty_string("''")]
    #[case::empty_sequence("[]")]
    fn ignores_falsey_included_documents(directory: TempDir, #[case] source: &str) {
        let path = directory.path().join("config.yaml");
        std::fs::write(&path, "include: [extra.yaml]\nretained: true").unwrap();
        std::fs::write(directory.path().join("extra.yaml"), source).unwrap();
        assert_eq!(
            Document::load(path).unwrap().value(),
            Document::from_yaml("retained: true").unwrap().value()
        );
    }

    #[rstest]
    #[case::scalar("include: extra.yaml")]
    #[case::null("include: null")]
    #[case::non_string_entry("include: [1]")]
    fn rejects_invalid_include_directives(directory: TempDir, #[case] source: &str) {
        let path = directory.path().join("config.yaml");
        std::fs::write(&path, source).unwrap();
        assert!(matches!(Document::load(path), Err(Error::Includes)));
    }
}
