use indexmap::IndexMap;
use std::sync::LazyLock;

static PROVIDERS: LazyLock<IndexMap<String, String>> = LazyLock::new(|| {
    serde_json::from_slice(include_bytes!(
        "../../../../litellm/integrations/otel/model/providers.json"
    ))
    .expect("bundled provider mapping is valid JSON")
});

pub fn canonical_provider(provider: &str) -> &str {
    PROVIDERS
        .get(&provider.to_lowercase())
        .map(String::as_str)
        .unwrap_or(provider)
}

pub(crate) fn provider_names(provider: &str, model: &str) -> Vec<String> {
    if PROVIDERS
        .get(provider)
        .is_some_and(|value| value != provider)
    {
        return vec![provider.to_owned()];
    }
    let candidates: Vec<_> = PROVIDERS
        .iter()
        .filter(|(_, canonical)| canonical.as_str() == provider)
        .map(|(name, _)| name.clone())
        .collect();
    let prefix = model.split('/').next().unwrap_or(model);
    if candidates.iter().any(|candidate| candidate == prefix) {
        vec![prefix.to_owned()]
    } else if candidates.is_empty() {
        vec![provider.to_owned()]
    } else {
        candidates
    }
}

pub(crate) fn known_provider(provider: &str) -> bool {
    PROVIDERS.contains_key(provider)
}
