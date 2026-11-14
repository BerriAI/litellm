use std::{
    collections::{BTreeSet, VecDeque},
    pin::pin,
    task::{Context, Poll, Waker},
};

use litellm_coroutine::{Coroutine, CoroutineState};
use serde_yaml_ng::{Mapping, Value};

use crate::{
    Error, Location,
    host::{Host, HostCall},
};

const INCLUDE: &str = "include";

pub enum Step<E> {
    Call(HostCall<E>),
    Done(Result<Mapping, Error<E>>),
}

pub struct Includes<E> {
    coroutine: Coroutine<HostCall<E>, Result<Mapping, Error<E>>>,
}

impl<E: Send + 'static> Includes<E> {
    pub fn new(root: Location, document: Value) -> Self {
        Self {
            coroutine: Coroutine::new(|co| resolve(root.normalized(), document, Host::new(co))),
        }
    }

    pub fn step(&mut self) -> Step<E> {
        let mut resume = pin!(self.coroutine.resume());
        match resume
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
        {
            Poll::Ready(Ok(CoroutineState::Yielded(call))) => Step::Call(call),
            Poll::Ready(Ok(CoroutineState::Complete(result))) => Step::Done(result),
            Poll::Ready(Err(_)) | Poll::Pending => Step::Done(Err(Error::Abandoned)),
        }
    }
}

async fn resolve<E>(root: Location, document: Value, host: Host<E>) -> Result<Mapping, Error<E>> {
    let mut merged = mapping::<E>(document, &root)?;
    let mut pending = VecDeque::from(entries::<E>(&merged, &root)?);
    merged.shift_remove(INCLUDE);
    let mut loaded = BTreeSet::from([root.clone()]);
    while let Some((entry, declared_in)) = pending.pop_front() {
        let location = declared_in.included(&entry, &root, &host).await?;
        if !loaded.insert(location.clone()) {
            continue;
        }
        let mut included = mapping::<E>(host.read(&location).await?, &location)?;
        pending.extend(entries::<E>(&included, &location)?);
        included.shift_remove(INCLUDE);
        for (key, value) in included {
            match (merged.get_mut(&key), value) {
                (Some(Value::Sequence(base)), Value::Sequence(extra)) => base.extend(extra),
                (_, value) => {
                    merged.insert(key, value);
                }
            }
        }
    }
    Ok(merged)
}

fn mapping<E>(document: Value, location: &Location) -> Result<Mapping, Error<E>> {
    match document {
        Value::Mapping(mapping) => Ok(mapping),
        _ => Err(Error::NotMapping(location.clone())),
    }
}

fn entries<E>(
    config: &Mapping,
    declared_in: &Location,
) -> Result<Vec<(String, Location)>, Error<E>> {
    let Some(include) = config.get(INCLUDE) else {
        return Ok(Vec::new());
    };
    let Value::Sequence(entries) = include else {
        return Err(Error::IncludeNotPaths);
    };
    entries
        .iter()
        .map(|entry| match entry {
            Value::String(path) => Ok((path.clone(), declared_in.clone())),
            _ => Err(Error::IncludeNotPaths),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rstest::rstest;

    use super::*;
    use crate::Warning;

    #[derive(Debug, PartialEq, thiserror::Error)]
    #[error("no document at {0}")]
    struct Missing(Location);

    struct Loaded {
        result: Result<Mapping, Error<Missing>>,
        reads: Vec<Location>,
        warnings: Vec<Warning>,
    }

    fn file(path: &str) -> Location {
        Location::File(path.into())
    }

    fn object(key: &str) -> Location {
        Location::Object(key.into())
    }

    fn rendered(mapping: &Mapping) -> String {
        serde_yaml_ng::to_string(mapping).unwrap()
    }

    fn yaml(text: &str) -> String {
        rendered(&serde_yaml_ng::from_str(text).unwrap())
    }

    fn load(root: Location, documents: &[(Location, &str)]) -> Loaded {
        let documents: BTreeMap<Location, Value> = documents
            .iter()
            .map(|(location, text)| (location.clone(), serde_yaml_ng::from_str(text).unwrap()))
            .collect();
        let mut includes = Includes::new(root.clone(), documents[&root].clone());
        let mut reads = Vec::new();
        let mut warnings = Vec::new();
        loop {
            match includes.step() {
                Step::Call(HostCall::Exists(path, reply)) => {
                    reply.send(documents.contains_key(&Location::File(path)));
                }
                Step::Call(HostCall::Read(location, reply)) => {
                    reads.push(location.clone());
                    reply.send(documents.get(&location).cloned().ok_or(Missing(location)));
                }
                Step::Call(HostCall::Warn(warning, reply)) => {
                    warnings.push(warning);
                    reply.send(());
                }
                Step::Done(result) => {
                    return Loaded {
                        result,
                        reads,
                        warnings,
                    };
                }
            }
        }
    }

    #[rstest]
    fn a_config_without_includes_asks_the_host_nothing() {
        let loaded = load(
            file("/etc/config.yaml"),
            &[(file("/etc/config.yaml"), "model_list: [root]")],
        );

        assert_eq!(
            rendered(&loaded.result.unwrap()),
            yaml("model_list: [root]")
        );
        assert!(loaded.reads.is_empty());
    }

    #[rstest]
    fn lists_extend_other_values_override_and_keys_keep_their_order() {
        let loaded = load(
            file("/etc/config.yaml"),
            &[
                (
                    file("/etc/config.yaml"),
                    "include: [models.yaml]\nmodel_list: [root]\nlitellm_settings: {drop_params: true}\ngeneral_settings: {master_key: root}",
                ),
                (
                    file("/etc/models.yaml"),
                    "router_settings: {num_retries: 2}\ngeneral_settings: {master_key: included}\nmodel_list: [included]",
                ),
            ],
        );

        assert_eq!(
            rendered(&loaded.result.unwrap()),
            yaml(
                "model_list: [root, included]\nlitellm_settings: {drop_params: true}\ngeneral_settings: {master_key: included}\nrouter_settings: {num_retries: 2}"
            )
        );
        assert!(loaded.warnings.is_empty());
    }

    #[rstest]
    #[case::scalar_over_list("value: [root]", "value: included", "value: included")]
    #[case::list_over_scalar("value: root", "value: [included]", "value: [included]")]
    #[case::null_over_list("value: [root]", "value: null", "value: null")]
    #[case::mapping_replaces_mapping(
        "value: {root: 1}",
        "value: {included: 2}",
        "value: {included: 2}"
    )]
    fn only_a_list_over_a_list_extends(
        #[case] root: &str,
        #[case] included: &str,
        #[case] merged: &str,
    ) {
        let loaded = load(
            file("/etc/config.yaml"),
            &[
                (
                    file("/etc/config.yaml"),
                    &format!("include: [models.yaml]\n{root}"),
                ),
                (file("/etc/models.yaml"), included),
            ],
        );

        assert_eq!(rendered(&loaded.result.unwrap()), yaml(merged));
    }

    #[rstest]
    fn nested_includes_merge_once_in_breadth_first_order() {
        let loaded = load(
            file("/etc/config.yaml"),
            &[
                (
                    file("/etc/config.yaml"),
                    "include: [nested/first.yaml, second.yaml]\nmodel_list: [root]",
                ),
                (
                    file("/etc/nested/first.yaml"),
                    "include: [third.yaml]\nmodel_list: [first]",
                ),
                (
                    file("/etc/second.yaml"),
                    "general_settings: {master_key: second}\nmodel_list: [second]",
                ),
                (
                    file("/etc/nested/third.yaml"),
                    "include: [../config.yaml]\ngeneral_settings: {master_key: third}\nmodel_list: [third]",
                ),
            ],
        );

        assert_eq!(
            rendered(&loaded.result.unwrap()),
            yaml("model_list: [root, first, second, third]\ngeneral_settings: {master_key: third}")
        );
        assert_eq!(
            loaded.reads,
            [
                file("/etc/nested/first.yaml"),
                file("/etc/second.yaml"),
                file("/etc/nested/third.yaml"),
            ]
        );
    }

    #[rstest]
    fn a_file_named_two_ways_is_read_once() {
        let loaded = load(
            file("/etc/config.yaml"),
            &[
                (file("/etc/config.yaml"), "include: [a.yaml, b.yaml]"),
                (file("/etc/a.yaml"), "include: [shared.yaml]"),
                (file("/etc/b.yaml"), "include: [./shared.yaml]"),
                (file("/etc/shared.yaml"), "model_list: [shared]"),
            ],
        );

        assert_eq!(
            rendered(&loaded.result.unwrap()),
            yaml("model_list: [shared]")
        );
        assert_eq!(
            loaded.reads,
            [
                file("/etc/a.yaml"),
                file("/etc/b.yaml"),
                file("/etc/shared.yaml"),
            ]
        );
    }

    #[rstest]
    fn a_root_named_with_dot_segments_is_not_read_again() {
        let loaded = load(
            file("/etc/litellm/../config.yaml"),
            &[
                (
                    file("/etc/litellm/../config.yaml"),
                    "include: [a.yaml]\nmodel_list: [root]",
                ),
                (
                    file("/etc/a.yaml"),
                    "include: [config.yaml]\nmodel_list: [a]",
                ),
            ],
        );

        assert_eq!(
            rendered(&loaded.result.unwrap()),
            yaml("model_list: [root, a]")
        );
        assert_eq!(loaded.reads, [file("/etc/a.yaml")]);
    }

    #[rstest]
    fn a_nested_include_is_read_next_to_the_file_that_declares_it() {
        let loaded = load(
            file("/etc/config.yaml"),
            &[
                (file("/etc/config.yaml"), "include: [shared/models.yaml]"),
                (
                    file("/etc/shared/models.yaml"),
                    "include: [more.yaml]\nmodel_list: [first]",
                ),
                (file("/etc/shared/more.yaml"), "model_list: [second]"),
            ],
        );

        assert_eq!(
            rendered(&loaded.result.unwrap()),
            yaml("model_list: [first, second]")
        );
        assert!(loaded.warnings.is_empty());
    }

    #[rstest]
    fn a_nested_include_left_beside_the_root_is_read_with_a_warning() {
        let loaded = load(
            file("/etc/config.yaml"),
            &[
                (file("/etc/config.yaml"), "include: [shared/models.yaml]"),
                (
                    file("/etc/shared/models.yaml"),
                    "include: [more.yaml]\nmodel_list: [first]",
                ),
                (file("/etc/more.yaml"), "model_list: [second]"),
            ],
        );

        assert_eq!(
            rendered(&loaded.result.unwrap()),
            yaml("model_list: [first, second]")
        );
        assert_eq!(
            loaded.warnings,
            [Warning::ReadBesideRoot {
                entry: "more.yaml".into(),
                declared_in: "/etc/shared/models.yaml".into(),
                read: "/etc/more.yaml".into(),
            }]
        );
    }

    #[rstest]
    fn a_nested_include_matching_two_files_reads_the_one_beside_its_declaring_file() {
        let loaded = load(
            file("/etc/config.yaml"),
            &[
                (file("/etc/config.yaml"), "include: [shared/models.yaml]"),
                (
                    file("/etc/shared/models.yaml"),
                    "include: [more.yaml]\nmodel_list: [first]",
                ),
                (
                    file("/etc/shared/more.yaml"),
                    "model_list: [beside-declaring]",
                ),
                (file("/etc/more.yaml"), "model_list: [beside-root]"),
            ],
        );

        assert_eq!(
            rendered(&loaded.result.unwrap()),
            yaml("model_list: [first, beside-declaring]")
        );
        assert_eq!(
            loaded.warnings,
            [Warning::MatchesTwoFiles {
                entry: "more.yaml".into(),
                declared_in: "/etc/shared/models.yaml".into(),
                read: "/etc/shared/more.yaml".into(),
                skipped: "/etc/more.yaml".into(),
            }]
        );
    }

    #[rstest]
    fn object_includes_resolve_against_the_declaring_objects_prefix() {
        let loaded = load(
            object("configs/config.yaml"),
            &[
                (
                    object("configs/config.yaml"),
                    "include: [shared/models.yaml, /root.yaml]",
                ),
                (
                    object("configs/shared/models.yaml"),
                    "include: [more.yaml]\nmodel_list: [first]",
                ),
                (object("root.yaml"), "model_list: [root]"),
                (object("configs/shared/more.yaml"), "model_list: [second]"),
            ],
        );

        assert_eq!(
            rendered(&loaded.result.unwrap()),
            yaml("model_list: [first, root, second]")
        );
        assert!(loaded.warnings.is_empty());
    }

    #[rstest]
    fn a_cycle_between_objects_terminates() {
        let loaded = load(
            object("configs/config.yaml"),
            &[
                (object("configs/config.yaml"), "include: [a.yaml]"),
                (
                    object("configs/a.yaml"),
                    "include: [b.yaml]\nmodel_list: [a]",
                ),
                (
                    object("configs/b.yaml"),
                    "include: [a.yaml]\nmodel_list: [b]",
                ),
            ],
        );

        assert_eq!(
            rendered(&loaded.result.unwrap()),
            yaml("model_list: [a, b]")
        );
    }

    #[rstest]
    #[case::text("include: models.yaml")]
    #[case::null("include: null")]
    #[case::mapping("include: {path: models.yaml}")]
    #[case::number_entry("include: [models.yaml, 1]")]
    fn an_include_that_is_not_a_list_of_paths_is_refused(#[case] root: &str) {
        let loaded = load(
            file("/etc/config.yaml"),
            &[
                (file("/etc/config.yaml"), root),
                (file("/etc/models.yaml"), "model_list: [included]"),
            ],
        );

        assert!(matches!(loaded.result, Err(Error::IncludeNotPaths)));
        assert!(loaded.reads.is_empty());
    }

    #[rstest]
    fn an_included_config_with_a_bad_include_is_refused() {
        let loaded = load(
            file("/etc/config.yaml"),
            &[
                (file("/etc/config.yaml"), "include: [models.yaml]"),
                (file("/etc/models.yaml"), "include: more.yaml"),
            ],
        );

        assert!(matches!(loaded.result, Err(Error::IncludeNotPaths)));
    }

    #[rstest]
    fn a_read_the_host_fails_ends_the_load_with_the_hosts_error() {
        let loaded = load(
            file("/etc/config.yaml"),
            &[(file("/etc/config.yaml"), "include: [missing.yaml]")],
        );

        assert!(matches!(
            loaded.result,
            Err(Error::Host(Missing(location))) if location == file("/etc/missing.yaml")
        ));
    }

    #[rstest]
    #[case::root("- model_name: root", "model_list: []", "/etc/config.yaml")]
    #[case::included("include: [models.yaml]", "- model_name: included", "/etc/models.yaml")]
    fn a_config_that_is_not_a_mapping_is_named(
        #[case] root: &str,
        #[case] included: &str,
        #[case] named: &str,
    ) {
        let loaded = load(
            file("/etc/config.yaml"),
            &[
                (file("/etc/config.yaml"), root),
                (file("/etc/models.yaml"), included),
            ],
        );

        assert!(matches!(
            loaded.result,
            Err(Error::NotMapping(location)) if location == file(named)
        ));
    }

    #[rstest]
    fn a_dropped_reply_abandons_the_load() {
        let mut includes = Includes::<Missing>::new(
            file("/etc/config.yaml"),
            serde_yaml_ng::from_str("include: [models.yaml]").unwrap(),
        );

        let Step::Call(HostCall::Read(_, reply)) = includes.step() else {
            panic!("the first step reads the include");
        };
        drop(reply);

        assert!(matches!(includes.step(), Step::Done(Err(Error::Abandoned))));
    }
}
