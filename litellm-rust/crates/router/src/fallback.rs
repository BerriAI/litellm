use crate::settings::FallbackEntry;

/// The chain `get_fallback_model_group` resolves for one group: an exact key wins at once, a
/// key the group equals once a provider prefix is stripped is used next, the `"*"` entry last.
/// A bare string entry replaces the chain with itself, as in Python.
pub fn chain_for<'a>(
    fallbacks: &'a [FallbackEntry],
    model_group: &str,
    providers: &[String],
) -> Lookup<'a> {
    let mut stripped = None;
    let mut generic = None;
    let mut chain: Option<Chain<'a>> = None;
    for (index, entry) in fallbacks.iter().enumerate() {
        match entry {
            FallbackEntry::Keyed { key, targets } if key == model_group => {
                chain = Some(Chain::Targets(targets));
                break;
            }
            FallbackEntry::Keyed { key, targets }
                if is_stripped_match(model_group, key, providers) =>
            {
                stripped = Some(Chain::Targets(targets));
            }
            FallbackEntry::Keyed { key, .. } if key == "*" => generic = Some(index),
            FallbackEntry::Keyed { .. } => {}
            FallbackEntry::Bare(group) => chain = Some(Chain::Single(group)),
        }
    }
    let generic_chain = generic.and_then(|index| match &fallbacks[index] {
        FallbackEntry::Keyed { targets, .. } => Some(Chain::Targets(targets)),
        FallbackEntry::Bare(_) => None,
    });
    let specific = chain.is_some() || stripped.is_some();
    Lookup {
        chain: chain.or(stripped).or(generic_chain),
        specific,
    }
}

/// `get_fallback_model_group_for_lookup_groups`: the first group with its own chain wins; the
/// generic chain applies only once every group missed.
pub fn chain_for_groups<'a>(
    fallbacks: &'a [FallbackEntry],
    groups: &[&str],
    providers: &[String],
) -> Option<Chain<'a>> {
    let lookups: Vec<Lookup<'a>> = groups
        .iter()
        .map(|group| chain_for(fallbacks, group, providers))
        .collect();
    lookups
        .iter()
        .find(|lookup| lookup.specific)
        .or_else(|| lookups.iter().find(|lookup| lookup.chain.is_some()))
        .and_then(|lookup| lookup.chain.clone())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Chain<'a> {
    Targets(&'a [String]),
    Single(&'a str),
}

impl Chain<'_> {
    pub fn targets(&self) -> Vec<String> {
        match self {
            Self::Targets(targets) => targets.to_vec(),
            Self::Single(group) => vec![(*group).to_owned()],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lookup<'a> {
    pub chain: Option<Chain<'a>>,
    specific: bool,
}

/// Whether `fallbacks` holds a `"*"` entry, and its targets.
pub fn generic_targets(fallbacks: &[FallbackEntry]) -> Option<&[String]> {
    fallbacks.iter().rev().find_map(|entry| match entry {
        FallbackEntry::Keyed { key, targets } if key == "*" => Some(targets.as_slice()),
        _ => None,
    })
}

fn is_stripped_match(model_group: &str, key: &str, providers: &[String]) -> bool {
    providers.iter().any(|provider| {
        model_group
            .strip_prefix(provider.as_str())
            .and_then(|rest| rest.strip_prefix('/'))
            .is_some_and(|_| model_group.replace(&format!("{provider}/"), "") == key)
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{Chain, chain_for, chain_for_groups};
    use crate::settings::FallbackEntry;

    fn keyed(key: &str, targets: &[&str]) -> FallbackEntry {
        FallbackEntry::Keyed {
            key: key.into(),
            targets: targets.iter().map(|target| (*target).to_owned()).collect(),
        }
    }

    fn providers() -> Vec<String> {
        vec!["openai".into(), "anthropic".into()]
    }

    #[rstest]
    #[case::exact(vec![keyed("*", &["g"]), keyed("a", &["b"])], "a", Some(vec!["b"]))]
    #[case::exact_beats_later_stripped(vec![keyed("a", &["b"]), keyed("x", &["y"])], "a", Some(vec!["b"]))]
    #[case::stripped_provider_prefix(vec![keyed("gpt", &["c"]), keyed("*", &["g"])], "openai/gpt", Some(vec!["c"]))]
    #[case::generic_last(vec![keyed("*", &["g"]), keyed("other", &["o"])], "a", Some(vec!["g"]))]
    #[case::bare_string_sets_the_chain(vec![FallbackEntry::Bare("s".into())], "a", Some(vec!["s"]))]
    #[case::none(vec![keyed("other", &["o"])], "a", None)]
    fn chain_lookup_follows_python_precedence(
        #[case] fallbacks: Vec<FallbackEntry>,
        #[case] group: &str,
        #[case] expected: Option<Vec<&str>>,
    ) {
        assert_eq!(
            chain_for(&fallbacks, group, &providers())
                .chain
                .map(|chain| chain.targets()),
            expected.map(|targets| targets.into_iter().map(String::from).collect())
        );
    }

    #[rstest]
    fn a_later_group_with_its_own_chain_beats_the_generic_chain() {
        let fallbacks = vec![keyed("*", &["g"]), keyed("second", &["s"])];
        assert_eq!(
            chain_for_groups(&fallbacks, &["first", "second"], &providers()),
            Some(Chain::Targets(&["s".to_owned()]))
        );
    }

    #[rstest]
    fn generic_chain_applies_when_every_group_misses() {
        let fallbacks = vec![keyed("*", &["g"])];
        assert_eq!(
            chain_for_groups(&fallbacks, &["first", "second"], &providers()),
            Some(Chain::Targets(&["g".to_owned()]))
        );
    }
}
