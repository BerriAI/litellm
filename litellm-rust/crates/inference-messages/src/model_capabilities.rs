use std::sync::LazyLock;

use fancy_regex::Regex;
use litellm_auth_aws::is_bedrock_region;
use litellm_core_utils::get_llm_provider_logic::get_custom_llm_provider;
use litellm_llms::base_llm::messages::context::{MessagesModelCapabilities, SupportedEffortTiers};
use litellm_llms::bedrock::common_utils::{bedrock_base_model, without_routing_prefixes};
use litellm_model_catalog::{Catalog, wire::Fields};

const SAMPLING_RESTRICTED_NAMES: [&str; 9] = [
    "fable", "opus-4-7", "opus_4_7", "opus-4.7", "opus_4.7", "opus-4-8", "opus_4_8", "opus-4.8",
    "opus_4.8",
];
const BACKSTOP_PREFIXES: [&str; 4] = [
    "bedrock/converse/",
    "bedrock/invoke/",
    "bedrock/",
    "vertex_ai/",
];

static VERSION_SUFFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"-v\d+(?::\d+)?$").expect("valid regex"));
static PROFILE_MINOR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r":\d+$").expect("valid regex"));
static DATED_RELEASE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"-\d{8}$").expect("valid regex"));
static DOTTED_VERSION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\d)\.(\d)").expect("valid regex"));
static TRAILING_NUMBER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"-\d+$").expect("valid regex"));
static TRAILING_DATE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"-\d{4}-\d{2}-\d{2}$").expect("valid regex"));

pub fn anthropic_model_capabilities(
    catalog: &Catalog,
    model: &str,
    custom_llm_provider: Option<&str>,
) -> MessagesModelCapabilities {
    let (name, provider) = get_custom_llm_provider(model, custom_llm_provider)
        .map_or((model, "anthropic"), |resolved| {
            (resolved.model, resolved.custom_llm_provider)
        });
    let probe = Probe {
        catalog,
        model,
        info: catalog.model_info(
            &candidate_names(name, provider)
                .each_ref()
                .map(String::as_str),
            provider,
        ),
        bare: catalog.lookup(name).map(|found| found.entry.fields()),
    };
    MessagesModelCapabilities {
        supports_reasoning: probe.supports("supports_reasoning"),
        supports_adaptive_thinking: probe.supports("supports_adaptive_thinking"),
        thinking_always_on: probe.supports("thinking_always_on"),
        supports_legacy_thinking: probe.supports("supports_legacy_thinking"),
        supports_output_config: probe.supports("supports_output_config"),
        supports_sampling_params: backstop(catalog, name, "supports_sampling_params")
            .unwrap_or_else(|| {
                let lowered = name.to_lowercase();
                !SAMPLING_RESTRICTED_NAMES
                    .iter()
                    .any(|restricted| lowered.contains(restricted))
            }),
        supports_speed: provider == "anthropic"
            && flag(exact(catalog, name), "supports_speed") == Some(true),
        supports_mid_conversation_system: probe.supports("supports_mid_conversation_system"),
        effort_tiers: SupportedEffortTiers {
            minimal: probe.supports("supports_minimal_reasoning_effort"),
            low: probe.supports("supports_low_reasoning_effort"),
            medium: probe.supports("supports_medium_reasoning_effort"),
            high: probe.supports("supports_high_reasoning_effort"),
            xhigh: probe.supports("supports_xhigh_reasoning_effort"),
            max: probe.supports("supports_max_reasoning_effort"),
        },
    }
}

struct Probe<'a> {
    catalog: &'a Catalog,
    model: &'a str,
    info: Option<Fields>,
    bare: Option<&'a Fields>,
}

impl Probe<'_> {
    fn supports(&self, capability: &str) -> bool {
        if let Some(supported) = flag(self.info.as_ref(), capability) {
            return supported;
        }
        if self.info.is_some() && flag(self.bare, capability) == Some(true) {
            return true;
        }
        backstop(self.catalog, self.model, capability) == Some(true)
    }
}

fn flag(fields: Option<&Fields>, capability: &str) -> Option<bool> {
    fields?.get(capability)?.as_bool()
}

fn exact<'a>(catalog: &'a Catalog, name: &str) -> Option<&'a Fields> {
    catalog
        .lookup(name)
        .filter(|found| found.matched_key == name)
        .map(|found| found.entry.fields())
}

fn is_bedrock(provider: &str) -> bool {
    matches!(provider, "bedrock" | "bedrock_converse")
}

fn candidate_names(model: &str, provider: &str) -> [String; 5] {
    let stripped = stripped_model(model, provider);
    [
        format!("{provider}/{model}"),
        model.to_owned(),
        split_model(model, provider).to_owned(),
        format!("{provider}/{stripped}"),
        stripped,
    ]
}

fn split_model<'a>(model: &'a str, provider: &str) -> &'a str {
    if !is_bedrock(provider) {
        return model;
    }
    let routed = without_routing_prefixes(model);
    let region_free = match routed.split_once('/') {
        Some((region, rest)) if is_bedrock_region(region) => rest,
        _ => routed,
    };
    if !region_free.starts_with("arn:") {
        return routed;
    }
    region_free.rsplit('/').next().unwrap_or(region_free)
}

fn stripped_model(model: &str, provider: &str) -> String {
    match provider {
        provider if is_bedrock(provider) => bedrock_base_model(model).to_owned(),
        "vertex_ai" | "gemini" | "databricks" => TRAILING_NUMBER.replace(model, "").into_owned(),
        _ => TRAILING_DATE.replace(model, "").into_owned(),
    }
}

fn backstop(catalog: &Catalog, model: &str, capability: &str) -> Option<bool> {
    backstop_candidates(model)
        .iter()
        .find_map(|name| flag(exact(catalog, name), capability))
}

fn backstop_candidates(model: &str) -> Vec<String> {
    let base = bedrock_base_model(model);
    let primary: Vec<String> = std::iter::once(model)
        .chain(
            BACKSTOP_PREFIXES
                .iter()
                .filter_map(|prefix| model.strip_prefix(prefix)),
        )
        .map(str::to_owned)
        .chain([base.to_owned(), format!("bedrock/{base}")])
        .collect();
    let normalized: Vec<String> = primary
        .iter()
        .flat_map(|name| normalized_names(name))
        .collect();
    primary.into_iter().chain(normalized).collect()
}

fn normalized_names(name: &str) -> [String; 6] {
    let undotted = DOTTED_VERSION.replace_all(name, "$1-$2");
    let unversioned = VERSION_SUFFIX.replace(&undotted, "").into_owned();
    [
        VERSION_SUFFIX.replace(name, "").into_owned(),
        PROFILE_MINOR.replace(name, "").into_owned(),
        DATED_RELEASE.replace(name, "").into_owned(),
        undotted.into_owned(),
        DATED_RELEASE.replace(&unversioned, "").into_owned(),
        name.rfind('@')
            .filter(|at| *at > 0)
            .map_or(name, |at| &name[..at])
            .to_owned(),
    ]
}

#[cfg(test)]
mod tests {
    use litellm_model_catalog::{BUNDLED_REGISTRY, Catalog};
    use rstest::{fixture, rstest};
    use serde_json::Value;

    use super::anthropic_model_capabilities;

    const CATALOG: &str = r#"{
        "claude-test-5": {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "supports_reasoning": true,
            "supports_speed": true,
            "supports_xhigh_reasoning_effort": true,
            "aliases": ["claude-test-5-latest"]
        },
        "claude-slow-5": {"litellm_provider": "anthropic", "mode": "chat", "supports_reasoning": false},
        "anthropic.claude-test-5-v1:0": {
            "litellm_provider": "bedrock_converse",
            "mode": "chat",
            "supports_reasoning": true,
            "supports_speed": true
        },
        "vertex_ai/claude-test-5": {
            "litellm_provider": "vertex_ai-anthropic_models",
            "mode": "chat",
            "supports_reasoning": true,
            "supports_speed": true
        },
        "fallback_generalizations": {
            "rules": [
                {"name": "routing", "pattern": "claude", "model_info": {"litellm_provider": "anthropic", "supports_speed": true}},
                {"name": "new-family", "pattern": "claude-(?=newfamily)", "model_info": {"supports_reasoning": true}},
                {
                    "name": "fill",
                    "pattern": "claude",
                    "model_info": {"supports_adaptive_thinking": true},
                    "fill_missing_for_providers": ["anthropic"]
                }
            ]
        }
    }"#;

    #[fixture]
    fn catalog() -> Catalog {
        Catalog::parse(CATALOG.as_bytes()).unwrap().value
    }

    #[rstest]
    #[case::row("anthropic/claude-test-5", None, (true, true, true, true))]
    #[case::alias("anthropic/claude-test-5-latest", None, (true, true, true, true))]
    #[case::bare_name("claude-test-5", None, (true, true, true, true))]
    #[case::configured_provider("claude-test-5", Some("anthropic"), (true, true, true, true))]
    #[case::other_casing("anthropic/Claude-Test-5", None, (true, true, false, true))]
    #[case::filled_row("anthropic/claude-slow-5", None, (false, true, false, true))]
    #[case::unknown_family("anthropic/claude-newfamily-6", None, (true, true, false, true))]
    #[case::cross_region_bedrock("bedrock/us.anthropic.claude-test-5-v1:0", None, (true, false, false, true))]
    #[case::region_path_bedrock("bedrock/us-east-1/anthropic.claude-test-5-v1:0", None, (true, false, false, true))]
    #[case::converse_bedrock("bedrock/converse/anthropic.claude-test-5-v1:0", None, (true, false, false, true))]
    #[case::vertex_version("vertex_ai/claude-test-5@20261001", None, (true, true, false, true))]
    #[case::sampling_restricted_name("anthropic/claude-fable-test", None, (false, true, false, false))]
    fn reads_capabilities_from_the_catalog(
        catalog: Catalog,
        #[case] model: &str,
        #[case] custom_llm_provider: Option<&str>,
        #[case] expected: (bool, bool, bool, bool),
    ) {
        let capabilities = anthropic_model_capabilities(&catalog, model, custom_llm_provider);

        assert_eq!(
            (
                capabilities.supports_reasoning,
                capabilities.supports_adaptive_thinking,
                capabilities.supports_speed,
                capabilities.supports_sampling_params,
            ),
            expected
        );
    }

    #[rstest]
    fn effort_tiers_follow_the_row(catalog: Catalog) {
        let tiers =
            anthropic_model_capabilities(&catalog, "anthropic/claude-test-5", None).effort_tiers;

        assert_eq!((tiers.xhigh, tiers.low), (true, false));
    }

    #[test]
    fn every_bundled_anthropic_row_shapes_with_its_own_flags() {
        let catalog = Catalog::parse(BUNDLED_REGISTRY).unwrap().value;
        let mismatches: Vec<String> = catalog
            .served()
            .filter(|(name, fields)| {
                !name.contains('/')
                    && fields.get("litellm_provider").and_then(Value::as_str) == Some("anthropic")
            })
            .flat_map(|(name, fields)| {
                let capabilities =
                    anthropic_model_capabilities(&catalog, &format!("anthropic/{name}"), None);
                [
                    ("supports_reasoning", capabilities.supports_reasoning),
                    (
                        "supports_adaptive_thinking",
                        capabilities.supports_adaptive_thinking,
                    ),
                    (
                        "supports_output_config",
                        capabilities.supports_output_config,
                    ),
                    (
                        "supports_sampling_params",
                        capabilities.supports_sampling_params,
                    ),
                    ("supports_speed", capabilities.supports_speed),
                    (
                        "supports_mid_conversation_system",
                        capabilities.supports_mid_conversation_system,
                    ),
                    (
                        "supports_xhigh_reasoning_effort",
                        capabilities.effort_tiers.xhigh,
                    ),
                    (
                        "supports_max_reasoning_effort",
                        capabilities.effort_tiers.max,
                    ),
                ]
                .into_iter()
                .filter(|(flag, shaped)| {
                    fields
                        .get(*flag)
                        .and_then(Value::as_bool)
                        .is_some_and(|row| row != *shaped)
                })
                .map(move |(flag, _)| format!("{name} {flag}"))
            })
            .collect();

        assert_eq!(mismatches, Vec::<String>::new());
    }
}
