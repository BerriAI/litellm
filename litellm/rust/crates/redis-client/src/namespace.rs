#[derive(Clone, Debug, Default, PartialEq)]
pub struct Namespace(Option<String>);

impl Namespace {
    pub fn new(namespace: impl Into<String>) -> Self {
        let namespace = namespace.into();
        Self((!namespace.is_empty()).then_some(namespace))
    }

    pub fn as_str(&self) -> Option<&str> {
        self.0.as_deref()
    }

    pub(crate) fn apply(&self, key: &str) -> String {
        match &self.0 {
            Some(namespace) if !is_prefixed(key, namespace) => {
                format!("{namespace}:{key}")
            }
            _ => key.to_owned(),
        }
    }

    /// A second key in the cluster slot of `key`, so one script may touch both. `None` when no hash
    /// tag can name that slot: `key` is hashed whole and holds a `}`.
    pub(crate) fn sibling(&self, key: &str, label: &str) -> Option<String> {
        let stored = self.apply(key);
        let tag = hash_tag(&stored);
        (!tag.contains('}')).then(|| format!("{{{tag}}}:{label}:{key}"))
    }
}

/// What Redis Cluster hashes: the text of the first non-empty `{...}`, else the whole key.
fn hash_tag(key: &str) -> &str {
    match key
        .split_once('{')
        .and_then(|(_, rest)| rest.split_once('}'))
    {
        Some((tag, _)) if !tag.is_empty() => tag,
        _ => key,
    }
}

fn is_prefixed(key: &str, namespace: &str) -> bool {
    key.strip_prefix(namespace)
        .is_some_and(|rest| rest.starts_with(':'))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::Namespace;

    #[rstest]
    #[case::plain(Namespace::new("litellm"), "spend:key:abc", "litellm:spend:key:abc")]
    #[case::no_namespace(Namespace::default(), "spend:key:abc", "spend:key:abc")]
    #[case::empty_namespace(Namespace::new(""), "spend:key:abc", "spend:key:abc")]
    #[case::already_prefixed(Namespace::new("litellm"), "litellm:abc", "litellm:abc")]
    #[case::shares_prefix_without_delimiter(
        Namespace::new("litellm"),
        "litellm_spend_update_buffer",
        "litellm:litellm_spend_update_buffer"
    )]
    #[case::hash_tagged_key(
        Namespace::new("litellm"),
        "{api_key:sk-1}:requests",
        "litellm:{api_key:sk-1}:requests"
    )]
    fn prefixes_keys_like_python_check_and_fix_namespace(
        #[case] namespace: Namespace,
        #[case] key: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(namespace.apply(key), expected);
    }

    #[rstest]
    #[case::whole_key(Namespace::default(), "team_id:t", Some("{team_id:t}:lease:team_id:t"))]
    #[case::namespaced_whole_key(
        Namespace::new("litellm"),
        "team_id:t",
        Some("{litellm:team_id:t}:lease:team_id:t")
    )]
    #[case::tagged_key(
        Namespace::new("litellm"),
        "{org}:alias",
        Some("{org}:lease:{org}:alias")
    )]
    #[case::unclosed_brace(Namespace::default(), "alias:a{b", Some("{alias:a{b}:lease:alias:a{b"))]
    #[case::empty_tag_with_closing_brace(Namespace::default(), "alias:a{}b", None)]
    #[case::closing_brace_first(Namespace::default(), "alias:a}b{", None)]
    fn a_sibling_key_names_the_slot_of_its_key(
        #[case] namespace: Namespace,
        #[case] key: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(namespace.sibling(key, "lease").as_deref(), expected);
    }
}
