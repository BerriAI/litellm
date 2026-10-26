pub trait Keys {
    fn contains(&self, key: &str) -> bool;
    fn lowercase(&self, key: &str) -> Option<String>;
    fn provider_of(&self, key: &str) -> Option<String>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Query<'a> {
    pub model: &'a str,
    pub provider: Option<&'a str>,
    pub base_model: Option<&'a str>,
    pub region: Option<&'a str>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::VariantArray)]
pub enum Via {
    Exact,
    BaseModel,
    ProviderPrefix,
    Region,
    RegionStripped,
    ProviderStripped,
    ProviderSuffixStripped,
    SuffixStripped,
    RepeatedProvider,
    ProviderResource,
    Lowercase,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Resolved {
    Found { key: String, via: Via },
    NotFound { tried: Vec<String> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegistrationTarget {
    pub write: String,
    pub inherit_from: Option<String>,
}

pub fn resolve(keys: &impl Keys, query: &Query<'_>) -> Resolved {
    resolve_inner(keys, query, true)
}

pub fn registration_target(keys: &impl Keys, caller_key: &str) -> RegistrationTarget {
    registration_target_with_provider(keys, caller_key, None)
}

pub fn registration_target_with_provider(
    keys: &impl Keys,
    caller_key: &str,
    provider: Option<&str>,
) -> RegistrationTarget {
    let query = Query {
        model: caller_key,
        provider,
        base_model: None,
        region: None,
    };
    let inherit_from = match resolve_inner(keys, &query, false) {
        Resolved::Found { key, .. } => Some(key),
        Resolved::NotFound { .. } => None,
    };
    RegistrationTarget {
        write: caller_key.to_owned(),
        inherit_from,
    }
}

struct Lookup<'a, K> {
    keys: &'a K,
    provider: Option<&'a str>,
    case_insensitive: bool,
    tried: Vec<String>,
}

impl<K: Keys> Lookup<'_, K> {
    fn candidate(&mut self, candidate: &str, via: Via) -> Option<Resolved> {
        if self.tried.iter().any(|key| key == candidate) {
            return None;
        }
        self.tried.push(candidate.to_owned());
        let (key, via) = if self.keys.contains(candidate) {
            (candidate.to_owned(), via)
        } else if self.case_insensitive {
            let key = self.keys.lowercase(candidate)?;
            if !self.keys.contains(&key) {
                return None;
            }
            (key, Via::Lowercase)
        } else {
            return None;
        };
        if let Some(provider) = self.provider
            && !provider_matches(provider, self.keys.provider_of(&key).as_deref())
        {
            return None;
        }
        Some(Resolved::Found { key, via })
    }
}

fn resolve_inner(keys: &impl Keys, query: &Query<'_>, case_insensitive: bool) -> Resolved {
    let model = query.base_model.unwrap_or(query.model);
    let provider = query.provider.filter(|provider| !provider.is_empty());
    let split = provider
        .and_then(|provider| model.strip_prefix(provider)?.strip_prefix('/'))
        .unwrap_or(model);
    let mut lookup = Lookup {
        keys,
        provider,
        case_insensitive,
        tried: Vec::new(),
    };
    if let (Some(provider), Some(region)) = (provider, query.region) {
        let regional = if split.starts_with(&format!("{region}/")) {
            format!("{provider}/{split}")
        } else {
            format!("{provider}/{region}/{split}")
        };
        if let Some(found) = lookup.candidate(&regional, Via::Region) {
            return found;
        }
    }
    if let Some(provider) = provider
        && split == model
        && let Some(found) = lookup.candidate(&format!("{provider}/{model}"), Via::ProviderPrefix)
    {
        return found;
    }
    let exact_via = if query.base_model.is_some() {
        Via::BaseModel
    } else {
        Via::Exact
    };
    if let Some(found) = lookup.candidate(model, exact_via) {
        return found;
    }

    let inferred = model.split_once('/');
    let provider = provider.or_else(|| inferred.map(|(prefix, _)| prefix));
    lookup.provider = provider;
    let split = if query.provider.is_none_or(str::is_empty) {
        inferred.map_or(model, |(_, model)| model)
    } else {
        split
    };
    let routed = if matches!(provider, Some("bedrock" | "bedrock_converse")) {
        bedrock_routing_model(split)
    } else {
        split
    };
    let region_free = strip_region(routed, provider, query.region);
    if region_free != routed
        && provider == Some("bedrock_mantle")
        && let Some(provider) = provider
        && let Some(found) =
            lookup.candidate(&format!("{provider}/{region_free}"), Via::RegionStripped)
    {
        return found;
    }
    let split_candidate = if provider == Some("bedrock_mantle") {
        region_free
    } else {
        routed
    };
    if let Some(found) = lookup.candidate(split_candidate, Via::ProviderStripped) {
        return found;
    }
    let stripped = strip_suffix(region_free, provider);
    if let Some(provider) = provider
        && let Some(found) = lookup.candidate(
            &format!("{provider}/{stripped}"),
            Via::ProviderSuffixStripped,
        )
    {
        return found;
    }
    if let Some(found) = lookup.candidate(stripped, Via::SuffixStripped) {
        return found;
    }
    if provider == Some("fireworks_ai") {
        if let Some(found) = lookup.candidate(&fireworks_resource(split), Via::ProviderResource) {
            return found;
        }
    } else if query.provider.is_some()
        && let Some(provider) = provider
        && let Some(found) = lookup.candidate(&format!("{provider}/{model}"), Via::RepeatedProvider)
    {
        return found;
    }
    Resolved::NotFound {
        tried: lookup.tried,
    }
}

fn provider_matches(requested: &str, actual: Option<&str>) -> bool {
    let Some(actual) = actual else {
        return true;
    };
    requested == actual
        || matches!(requested, "litellm_proxy" | "github" | "nadir")
        || matches!(requested, "vertex_ai" | "fireworks_ai") && actual.starts_with(requested)
        || requested.starts_with("bedrock") && actual.starts_with("bedrock")
        || requested == "azure_ai" && matches!(actual, "azure" | "openai")
}

fn strip_region<'a>(model: &'a str, provider: Option<&str>, region: Option<&str>) -> &'a str {
    let Some((head, tail)) = model.split_once('/') else {
        return model;
    };
    if region == Some(head)
        || matches!(
            provider,
            Some("bedrock" | "bedrock_converse" | "bedrock_mantle")
        ) && is_aws_region(head)
    {
        tail
    } else {
        model
    }
}

fn is_aws_region(region: &str) -> bool {
    let Some((location, number)) = region.rsplit_once('-') else {
        return false;
    };
    matches!(
        location,
        "us-east"
            | "us-west"
            | "us-gov-east"
            | "us-gov-west"
            | "af-south"
            | "ap-east"
            | "ap-south"
            | "ap-southeast"
            | "ap-northeast"
            | "ca-central"
            | "ca-west"
            | "cn-north"
            | "cn-northwest"
            | "eu-central"
            | "eu-west"
            | "eu-north"
            | "eu-south"
            | "il-central"
            | "me-central"
            | "me-south"
            | "mx-central"
            | "sa-east"
    ) && !number.is_empty()
        && number.bytes().all(|c| c.is_ascii_digit())
}

fn bedrock_routing_model(mut model: &str) -> &str {
    for prefix in [
        "bedrock/",
        "chat_completions/",
        "converse/",
        "invoke/",
        "openai/",
        "mantle/",
    ] {
        model = model.strip_prefix(prefix).unwrap_or(model);
    }
    let region_free = strip_region(model, Some("bedrock"), None);
    if region_free.starts_with("arn:") {
        return region_free.rsplit('/').next().unwrap_or(region_free);
    }
    model
}

fn strip_suffix<'a>(model: &'a str, provider: Option<&str>) -> &'a str {
    match provider {
        Some("bedrock" | "bedrock_converse") => {
            if model.starts_with("nova-2/") {
                return "amazon.nova-2-custom";
            }
            if model.starts_with("nova/") {
                return "amazon.nova-custom";
            }
            let model = match model.split_once('.') {
                Some(("global" | "us" | "eu" | "apac" | "jp" | "au" | "us-gov", tail)) => tail,
                _ => model,
            };
            let model = strip_bedrock_throughput(model);
            let model = model
                .rsplit_once('-')
                .filter(|(_, suffix)| suffix.len() == 8 && is_digits(suffix))
                .map_or(model, |(base, _)| base);
            strip_date(model)
        }
        Some("vertex_ai" | "gemini" | "databricks") => model
            .rsplit_once('-')
            .filter(|(_, suffix)| !suffix.is_empty() && suffix.bytes().all(|c| c.is_ascii_digit()))
            .map_or(model, |(base, _)| base),
        _ if model.contains("ft:") => model.rsplitn(4, ':').nth(3).unwrap_or(model),
        _ => strip_date(model),
    }
}

fn fireworks_resource(model: &str) -> String {
    let model = model.strip_prefix("fireworks_ai/").unwrap_or(model);
    let resource =
        if model.starts_with("accounts/") || model.starts_with("FW-") || model.contains('#') {
            model.to_owned()
        } else if model.starts_with("routers/") || model.starts_with("models/") {
            format!("accounts/fireworks/{model}")
        } else if model.ends_with("-fast")
            || matches!(model, "firerouter" | "auto" | "auto-instant")
            || model.starts_with("firerouter/")
        {
            format!("accounts/fireworks/routers/{model}")
        } else {
            format!("accounts/fireworks/models/{model}")
        };
    format!("fireworks_ai/{resource}")
}

fn strip_bedrock_throughput(model: &str) -> &str {
    let model = model
        .rsplit_once(':')
        .and_then(|(base, suffix)| {
            let digits = suffix.strip_suffix('k')?;
            let (_, version) = base.rsplit_once(':')?;
            (is_digits(digits) && is_digits(version)).then_some(base)
        })
        .unwrap_or(model);
    model
        .rsplit_once('[')
        .and_then(|(base, suffix)| {
            let suffix = suffix.strip_suffix(']')?;
            (!suffix.is_empty() && suffix.chars().all(|c| c.is_alphanumeric() || c == '_'))
                .then_some(base)
        })
        .unwrap_or(model)
}

fn is_digits(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|c| c.is_ascii_digit())
}

fn strip_date(model: &str) -> &str {
    let Some((base, suffix)) = model.rsplit_once('-') else {
        return model;
    };
    let Some((base, month)) = base.rsplit_once('-') else {
        return model;
    };
    let Some((base, year)) = base.rsplit_once('-') else {
        return model;
    };
    if year.len() == 4
        && month.len() == 2
        && suffix.len() == 2
        && is_digits(year)
        && is_digits(month)
        && is_digits(suffix)
    {
        base
    } else {
        model
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use rstest::{fixture, rstest};
    use serde_json::{Value, json};

    use super::*;
    use crate::{BUNDLED_REGISTRY, Catalog};

    fn catalog(rows: Value) -> Catalog {
        Catalog::parse(rows.to_string().as_bytes()).unwrap().value
    }

    fn query<'a>(model: &'a str, provider: Option<&'a str>) -> Query<'a> {
        Query {
            model,
            provider,
            base_model: None,
            region: None,
        }
    }

    #[fixture]
    fn models() -> Catalog {
        catalog(json!({
            "o3-pro": {"litellm_provider": "openai"},
            "openai/o3-pro": {"litellm_provider": "openai"},
            "gpt-4o": {"litellm_provider": "openai", "aliases": ["gpt-alias"]},
            "ft:gpt-4o": {"litellm_provider": "openai"},
            "databricks/databricks-claude-opus-4": {"litellm_provider": "databricks"},
            "databricks-claude-opus-4": {"litellm_provider": "databricks"},
            "bedrock/anthropic.claude-sonnet-4-6": {"litellm_provider": "bedrock"},
            "hosted_vllm/Acme-Model": {"litellm_provider": "hosted_vllm"},
            "部署": {"litellm_provider": null}
        }))
    }

    #[rstest]
    #[case::bare("o3-pro", None, "o3-pro", Via::Exact)]
    #[case::provider_override("o3-pro", Some("openai"), "openai/o3-pro", Via::ProviderPrefix)]
    #[case::already_prefixed("openai/o3-pro", Some("openai"), "openai/o3-pro", Via::Exact)]
    #[case::bare_fallback("gpt-4o", Some("openai"), "gpt-4o", Via::Exact)]
    #[case::strip_provider("openai/gpt-4o", Some("openai"), "gpt-4o", Via::ProviderStripped)]
    #[case::infer_provider("openai/gpt-4o", None, "gpt-4o", Via::ProviderStripped)]
    #[case::dated("gpt-4o-2026-01-01", None, "gpt-4o", Via::SuffixStripped)]
    #[case::fine_tune("ft:gpt-4o:org::id", Some("openai"), "ft:gpt-4o", Via::SuffixStripped)]
    #[case::version(
        "databricks-claude-opus-4-99",
        Some("databricks"),
        "databricks/databricks-claude-opus-4",
        Via::ProviderSuffixStripped
    )]
    #[case::bedrock_date(
        "bedrock/anthropic.claude-sonnet-4-6-20260101",
        None,
        "bedrock/anthropic.claude-sonnet-4-6",
        Via::ProviderSuffixStripped
    )]
    #[case::lowercase(
        "hosted_vllm/ACME-MODEL",
        None,
        "hosted_vllm/Acme-Model",
        Via::Lowercase
    )]
    #[case::alias("gpt-alias", None, "gpt-alias", Via::Exact)]
    #[case::alias_provider("openai/gpt-alias", Some("openai"), "gpt-alias", Via::ProviderStripped)]
    #[case::alias_case("GPT-ALIAS", None, "gpt-alias", Via::Lowercase)]
    #[case::unicode("部署", None, "部署", Via::Exact)]
    fn resolves_names(
        models: Catalog,
        #[case] model: &str,
        #[case] provider: Option<&str>,
        #[case] key: &str,
        #[case] via: Via,
    ) {
        assert_eq!(
            resolve(&models, &query(model, provider)),
            Resolved::Found {
                key: key.into(),
                via
            }
        );
    }

    #[rstest]
    #[case::base("deployment", "gpt-4o", None, None, "gpt-4o", Via::BaseModel)]
    #[case::provider_base(
        "gpt-4o",
        "o3-pro",
        Some("openai"),
        None,
        "openai/o3-pro",
        Via::ProviderPrefix
    )]
    #[case::base_beats_registered_deployment(
        "o3-pro",
        "gpt-4o",
        None,
        None,
        "gpt-4o",
        Via::BaseModel
    )]
    fn resolves_base_before_request(
        models: Catalog,
        #[case] model: &str,
        #[case] base: &str,
        #[case] provider: Option<&str>,
        #[case] region: Option<&str>,
        #[case] key: &str,
        #[case] via: Via,
    ) {
        let query = Query {
            model,
            provider,
            region,
            base_model: Some(base),
        };
        assert_eq!(
            resolve(&models, &query),
            Resolved::Found {
                key: key.into(),
                via
            }
        );
    }

    #[rstest]
    fn unknown_base_does_not_price_the_deployment(models: Catalog) {
        let query = Query {
            base_model: Some("missing-base"),
            ..query("gpt-4o", None)
        };
        assert_eq!(
            resolve(&models, &query),
            Resolved::NotFound {
                tried: vec!["missing-base".into()]
            }
        );
    }

    #[rstest]
    #[case::prefixed("openai/o3-pro", Some("o3-pro"))]
    #[case::versioned(
        "databricks/databricks-claude-opus-4-99",
        Some("databricks/databricks-claude-opus-4")
    )]
    #[case::dated_bedrock(
        "bedrock/anthropic.claude-sonnet-4-6-20260101",
        Some("bedrock/anthropic.claude-sonnet-4-6")
    )]
    #[case::exact("gpt-4o", Some("gpt-4o"))]
    #[case::router_id("a042a7be2fb64875abf123ef", None)]
    #[case::azure_deployment("azure/my-deployment", None)]
    #[case::case_variant("hosted_vllm/acme-model", None)]
    #[case::case_variant_with_suffix("hosted_vllm/acme-model-2026-01-01", None)]
    fn registration_keeps_write_separate(#[case] caller: &str, #[case] inherit_from: Option<&str>) {
        let models = catalog(json!({
            "o3-pro": {"litellm_provider":"openai"},
            "gpt-4o": {"litellm_provider":"openai"},
            "databricks/databricks-claude-opus-4": {"litellm_provider":"databricks"},
            "bedrock/anthropic.claude-sonnet-4-6": {"litellm_provider":"bedrock"},
            "hosted_vllm/Acme-Model": {"litellm_provider":"hosted_vllm"}
        }));
        assert_eq!(
            registration_target(&models, caller),
            RegistrationTarget {
                write: caller.into(),
                inherit_from: inherit_from.map(str::to_owned),
            }
        );
        assert_eq!(models.model_count(), 5);
    }

    #[rstest]
    fn repeated_registration_inherits_existing_caller_before_sibling(models: Catalog) {
        assert_eq!(
            registration_target(&models, "openai/o3-pro"),
            RegistrationTarget {
                write: "openai/o3-pro".into(),
                inherit_from: Some("openai/o3-pro".into()),
            }
        );
    }

    #[rstest]
    #[case::folded_provider_beats_bare("MODEL", Some("custom"), "custom/model", Via::Lowercase)]
    #[case::exact_beats_folded("custom/Model", Some("custom"), "custom/Model", Via::Exact)]
    #[case::no_provider_reads_bare("MODEL", None, "MODEL", Via::Exact)]
    fn case_fallback_is_per_candidate(
        #[case] model: &str,
        #[case] provider: Option<&str>,
        #[case] key: &str,
        #[case] via: Via,
    ) {
        let models = catalog(json!({"custom/Model": {}, "custom/model": {}, "MODEL": {}}));
        assert_eq!(
            resolve(&models, &query(model, provider)),
            Resolved::Found {
                key: key.into(),
                via
            }
        );
    }

    #[rstest]
    #[case::regional_first(
        "model",
        Some("us-east-1"),
        "bedrock_mantle/us-east-1/model",
        Via::Region
    )]
    #[case::embedded_region(
        "bedrock_mantle/us-east-1/model",
        None,
        "bedrock_mantle/us-east-1/model",
        Via::Exact
    )]
    #[case::already_regional(
        "bedrock_mantle/us-east-1/model",
        Some("us-east-1"),
        "bedrock_mantle/us-east-1/model",
        Via::Region
    )]
    #[case::region_fallback(
        "bedrock_mantle/us-west-2/model",
        None,
        "bedrock_mantle/model",
        Via::RegionStripped
    )]
    #[case::query_region_fallback(
        "model",
        Some("eu-west-1"),
        "bedrock_mantle/model",
        Via::ProviderPrefix
    )]
    fn region_precedence(
        #[case] model: &str,
        #[case] region: Option<&str>,
        #[case] key: &str,
        #[case] via: Via,
    ) {
        let models = catalog(json!({
            "bedrock_mantle/us-east-1/model": {"litellm_provider":"bedrock_mantle"},
            "bedrock_mantle/model": {"litellm_provider":"bedrock_mantle"},
            "model": {"litellm_provider":"bedrock_mantle"}
        }));
        let query = Query {
            region,
            ..query(model, Some("bedrock_mantle"))
        };
        assert_eq!(
            resolve(&models, &query),
            Resolved::Found {
                key: key.into(),
                via
            }
        );
    }

    #[rstest]
    #[case::bare_cross_region(
        "us.anthropic.model-v1:0",
        "anthropic.model-v1:0",
        Via::SuffixStripped
    )]
    #[case::routed(
        "bedrock/converse/anthropic.model-v1:0",
        "anthropic.model-v1:0",
        Via::ProviderStripped
    )]
    #[case::region(
        "bedrock/us-east-1/anthropic.model-v1:0",
        "bedrock/anthropic.model-v1:0",
        Via::ProviderSuffixStripped
    )]
    #[case::throughput(
        "anthropic.model-v1:0:51k",
        "anthropic.model-v1:0",
        Via::SuffixStripped
    )]
    #[case::context(
        "anthropic.model-v1:0[1m]",
        "anthropic.model-v1:0",
        Via::SuffixStripped
    )]
    #[case::arn(
        "bedrock/arn:aws:bedrock:us-east-1:123:inference-profile/us.anthropic.model-v1:0",
        "anthropic.model-v1:0",
        Via::SuffixStripped
    )]
    #[case::nova(
        "bedrock/nova/arn:aws:bedrock:us-east-1:123:custom-model/id",
        "amazon.nova-custom",
        Via::SuffixStripped
    )]
    #[case::nova2(
        "bedrock/converse/nova-2/arn:aws:bedrock:us-east-1:123:custom-model/id",
        "amazon.nova-2-custom",
        Via::SuffixStripped
    )]
    fn bedrock_normalization(#[case] model: &str, #[case] key: &str, #[case] via: Via) {
        let models = catalog(json!({key: {"litellm_provider":"bedrock"}}));
        assert_eq!(
            resolve(&models, &query(model, Some("bedrock"))),
            Resolved::Found {
                key: key.into(),
                via
            }
        );
    }

    #[rstest]
    #[case::gemini("model-001", "gemini", "gemini/model", Via::ProviderSuffixStripped)]
    #[case::vertex("model-002", "vertex_ai", "model", Via::SuffixStripped)]
    #[case::bare_first("databricks/model-99", "databricks", "model-99", Via::ProviderStripped)]
    #[case::double_prefix(
        "openai/gpt-model",
        "openai",
        "openai/openai/gpt-model",
        Via::RepeatedProvider
    )]
    #[case::fireworks(
        "model",
        "fireworks_ai",
        "fireworks_ai/accounts/fireworks/models/model",
        Via::ProviderResource
    )]
    #[case::fireworks_models(
        "models/model",
        "fireworks_ai",
        "fireworks_ai/accounts/fireworks/models/model",
        Via::ProviderResource
    )]
    #[case::fireworks_router(
        "model-fast",
        "fireworks_ai",
        "fireworks_ai/accounts/fireworks/routers/model-fast",
        Via::ProviderResource
    )]
    fn provider_specific_candidates(
        #[case] model: &str,
        #[case] provider: &str,
        #[case] key: &str,
        #[case] via: Via,
    ) {
        let models = catalog(json!({key: {"litellm_provider": provider}}));
        assert_eq!(
            resolve(&models, &query(model, Some(provider))),
            Resolved::Found {
                key: key.into(),
                via
            }
        );
    }

    #[rstest]
    #[case::same_provider("openai", Some("openai"), true)]
    #[case::missing_provider("openai", None, true)]
    #[case::mismatch("openai", Some("anthropic"), false)]
    #[case::vertex_family("vertex_ai", Some("vertex_ai-language-models"), true)]
    #[case::fireworks_family("fireworks_ai", Some("fireworks_ai-embedding-models"), true)]
    #[case::bedrock_family("bedrock_converse", Some("bedrock"), true)]
    #[case::azure_ai("azure_ai", Some("openai"), true)]
    #[case::proxy("litellm_proxy", Some("anthropic"), true)]
    #[case::github("github", Some("openai"), true)]
    #[case::nadir("nadir", Some("anthropic"), true)]
    fn provider_metadata_filters_matches(
        #[case] provider: &str,
        #[case] actual: Option<&str>,
        #[case] matches: bool,
    ) {
        let models = catalog(json!({"model": {"litellm_provider": actual}}));
        let result = resolve(&models, &query("model", Some(provider)));
        assert_eq!(matches!(result, Resolved::Found { .. }), matches);
    }

    #[rstest]
    fn mismatched_candidate_does_not_hide_valid_fallback() {
        let models = catalog(json!({
            "openai/model": {"litellm_provider":"anthropic"},
            "model": {"litellm_provider":"openai"}
        }));
        assert_eq!(
            resolve(&models, &query("model", Some("openai"))),
            Resolved::Found {
                key: "model".into(),
                via: Via::Exact,
            }
        );
    }

    #[rstest]
    fn miss_reports_unique_candidates_in_order() {
        let models = catalog(json!({"unrelated": {}}));
        assert_eq!(
            resolve(&models, &query("databricks/model-99", Some("databricks"))),
            Resolved::NotFound {
                tried: vec![
                    "databricks/model-99".into(),
                    "model-99".into(),
                    "databricks/model".into(),
                    "model".into(),
                    "databricks/databricks/model-99".into()
                ],
            }
        );
    }

    #[rstest]
    #[case::empty("")]
    #[case::unknown("missing")]
    #[case::not_a_date("model-2026-01")]
    #[case::numeric_id("model-20260101")]
    #[case::incomplete_finetune("ft:model:org")]
    #[case::unicode("未知-版本")]
    fn leaves_unrecognized_names_intact(#[case] model: &str) {
        assert_eq!(
            resolve(&catalog(json!({"model": {}})), &query(model, None)),
            Resolved::NotFound {
                tried: vec![model.into()],
            }
        );
    }

    struct CountingKeys {
        existing: Option<&'static str>,
        folded: Option<&'static str>,
        exact_calls: Cell<usize>,
        lowercase_calls: Cell<usize>,
        provider_calls: Cell<usize>,
    }

    impl Keys for CountingKeys {
        fn contains(&self, key: &str) -> bool {
            self.exact_calls.set(self.exact_calls.get() + 1);
            self.existing == Some(key)
        }
        fn lowercase(&self, _: &str) -> Option<String> {
            self.lowercase_calls.set(self.lowercase_calls.get() + 1);
            self.folded.map(str::to_owned)
        }
        fn provider_of(&self, _: &str) -> Option<String> {
            self.provider_calls.set(self.provider_calls.get() + 1);
            None
        }
    }

    #[fixture]
    fn counted() -> CountingKeys {
        CountingKeys {
            existing: Some("model"),
            folded: None,
            exact_calls: Cell::new(0),
            lowercase_calls: Cell::new(0),
            provider_calls: Cell::new(0),
        }
    }

    #[rstest]
    fn exact_hit_uses_one_lookup(counted: CountingKeys) {
        assert_eq!(
            resolve(&counted, &query("model", None)),
            Resolved::Found {
                key: "model".into(),
                via: Via::Exact
            }
        );
        assert_eq!(counted.exact_calls.get(), 1);
        assert_eq!(counted.lowercase_calls.get(), 0);
        assert_eq!(counted.provider_calls.get(), 0);
    }

    #[rstest]
    fn reads_keys_again_after_the_view_changes(mut counted: CountingKeys) {
        assert_eq!(
            resolve(&counted, &query("model", None)),
            Resolved::Found {
                key: "model".into(),
                via: Via::Exact,
            }
        );
        counted.existing = Some("replacement");
        assert_eq!(
            resolve(&counted, &query("model", None)),
            Resolved::NotFound {
                tried: vec!["model".into()],
            }
        );
        assert_eq!(
            resolve(&counted, &query("replacement", None)),
            Resolved::Found {
                key: "replacement".into(),
                via: Via::Exact,
            }
        );
    }

    #[rstest]
    fn unstripped_bare_key_beats_provider_stripped_suffix() {
        let models = catalog(json!({
            "model-99": {"litellm_provider": "databricks"},
            "databricks/model": {"litellm_provider": "databricks"},
            "model": {"litellm_provider": "databricks"}
        }));
        assert_eq!(
            resolve(&models, &query("databricks/model-99", Some("databricks"))),
            Resolved::Found {
                key: "model-99".into(),
                via: Via::ProviderStripped,
            }
        );
    }

    #[rstest]
    fn provider_hit_uses_one_exact_lookup(counted: CountingKeys) {
        let counted = CountingKeys {
            existing: Some("custom/model"),
            ..counted
        };
        assert_eq!(
            resolve(&counted, &query("model", Some("custom"))),
            Resolved::Found {
                key: "custom/model".into(),
                via: Via::ProviderPrefix,
            }
        );
        assert_eq!(counted.exact_calls.get(), 1);
        assert_eq!(counted.lowercase_calls.get(), 0);
    }

    #[rstest]
    fn deleted_lowercase_target_is_not_returned(counted: CountingKeys) {
        let counted = CountingKeys {
            folded: Some("deleted"),
            ..counted
        };
        assert_eq!(
            resolve(&counted, &query("DELETED", None)),
            Resolved::NotFound {
                tried: vec!["DELETED".into()]
            }
        );
        assert_eq!(counted.provider_calls.get(), 0);
    }

    #[rstest]
    #[case::unscoped(None)]
    #[case::scoped(Some("custom"))]
    fn registration_never_uses_the_lowercase_index(
        counted: CountingKeys,
        #[case] provider: Option<&str>,
    ) {
        let counted = CountingKeys {
            folded: Some("model"),
            ..counted
        };
        assert_eq!(
            registration_target_with_provider(&counted, "MODEL", provider),
            RegistrationTarget {
                write: "MODEL".into(),
                inherit_from: None
            }
        );
        assert_eq!(counted.lowercase_calls.get(), 0);
    }

    #[rstest]
    #[case::provider_mismatch("same", Some("azure"), None)]
    #[case::no_provider("same", None, Some("same"))]
    #[case::empty_provider("same", Some(""), Some("same"))]
    #[case::qualified_preference("model", Some("openai"), Some("openai/model"))]
    #[case::already_qualified("openai/model", Some("openai"), Some("openai/model"))]
    #[case::bare_after_folded_candidate("bare", Some("openai"), Some("bare"))]
    #[case::no_lowercase("hosted_vllm/example-model", Some("hosted_vllm"), None)]
    #[case::upper_case(
        "hosted_vllm/Example-Model",
        Some("hosted_vllm"),
        Some("hosted_vllm/Example-Model")
    )]
    #[case::independent_upper("Case", Some("openai"), Some("Case"))]
    #[case::independent_lower("case", Some("openai"), Some("case"))]
    #[case::suffix("version-99", Some("databricks"), Some("databricks/version"))]
    fn registration_uses_provider_without_case_folding(
        #[case] key: &str,
        #[case] provider: Option<&str>,
        #[case] inherited: Option<&str>,
    ) {
        let models = catalog(json!({
            "same": {"litellm_provider": "openai"},
            "model": {"litellm_provider": "openai"},
            "openai/model": {"litellm_provider": "openai"},
            "openai/Bare": {"litellm_provider": "openai"},
            "bare": {"litellm_provider": "openai"},
            "hosted_vllm/Example-Model": {"litellm_provider": "hosted_vllm"},
            "Case": {"litellm_provider": "openai"},
            "case": {"litellm_provider": "openai"},
            "databricks/version": {"litellm_provider": "databricks"}
        }));
        assert_eq!(
            registration_target_with_provider(&models, key, provider),
            RegistrationTarget {
                write: key.into(),
                inherit_from: inherited.map(str::to_owned),
            }
        );
    }

    #[rstest]
    fn every_bundled_exact_key_is_preserved() {
        let catalog = Catalog::parse(BUNDLED_REGISTRY).unwrap().value;
        for key in catalog
            .model_names()
            .chain(catalog.aliases().keys().map(String::as_str))
        {
            assert_eq!(
                resolve(&catalog, &query(key, None)),
                Resolved::Found {
                    key: key.into(),
                    via: Via::Exact
                }
            );
        }
    }
}
