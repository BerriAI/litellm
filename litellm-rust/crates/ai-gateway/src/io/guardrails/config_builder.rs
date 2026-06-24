//! Build a fully-resolved [`ProviderConfig`] from the raw guardrail params the
//! proxy already holds. This is the host edge: it parses the params dict into
//! typed structs, resolves `os.environ/` secrets and env fallbacks, and reads
//! any referenced files. When a config uses a feature the Rust engine does not
//! support, it returns [`Unsupported`] so the caller falls back to Python.

use std::collections::HashMap;

use litellm_core::guardrails::{
    AzurePromptShieldConfig, AzureTextModerationConfig, BedrockConfig, GenericApiConfig,
    LakeraV2Config, LocalPiiConfig, OnFlagged, OpenaiModerationConfig, PiiAction, PresidioConfig,
    ProviderConfig, UnreachableFallback,
};
use serde::Deserialize;
use serde_json::{Map, Value};

/// The Rust engine cannot handle this config; the caller must use the Python
/// implementation. The message is for logs only.
#[derive(Debug)]
pub struct Unsupported(pub String);

fn unsupported(reason: impl Into<String>) -> Unsupported {
    Unsupported(reason.into())
}

/// Resolve a config value: `os.environ/NAME` reads the env var (absent => None);
/// any other non-empty value passes through; empty/whitespace is treated as
/// absent.
fn resolve_secret(value: Option<&str>) -> Option<String> {
    let value = value?;
    if let Some(name) = value.strip_prefix("os.environ/") {
        return std::env::var(name).ok().filter(|v| !v.trim().is_empty());
    }
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| value.to_owned())
}

fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn parse_params<T: for<'de> Deserialize<'de>>(params: &Value) -> Result<T, Unsupported> {
    serde_json::from_value(params.clone())
        .map_err(|e| unsupported(format!("could not parse guardrail params: {e}")))
}

/// Treat an explicit JSON `null` like an absent field. The proxy serializes
/// guardrail params from a Pydantic `model_dump()`, which emits `null` for every
/// unset field; plain `#[serde(default)]` only covers *missing* keys, so a
/// non-Option collection or bool would otherwise fail to deserialize from `null`.
fn null_to_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    T: Default + Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

pub fn build_config(guardrail_type: &str, params: &Value) -> Result<ProviderConfig, Unsupported> {
    match guardrail_type {
        "generic_guardrail_api" => Ok(ProviderConfig::GenericGuardrailApi(generic(params)?)),
        "openai_moderation" => Ok(ProviderConfig::OpenaiModeration(openai_moderation(params)?)),
        "azure/prompt_shield" => Ok(ProviderConfig::AzurePromptShield(azure_prompt_shield(
            params,
        )?)),
        "azure/text_moderations" => Ok(ProviderConfig::AzureTextModeration(azure_text_moderation(
            params,
        )?)),
        "presidio" => Ok(ProviderConfig::Presidio(presidio(params)?)),
        "lakera_v2" => Ok(ProviderConfig::LakeraV2(lakera_v2(params)?)),
        "bedrock" => Ok(ProviderConfig::Bedrock(bedrock(params)?)),
        "local_pii" => Ok(ProviderConfig::LocalPii(local_pii(params)?)),
        other => Err(unsupported(format!(
            "guardrail type not supported by the Rust engine: {other}"
        ))),
    }
}

#[derive(Deserialize, Default)]
struct GenericParams {
    #[serde(default)]
    api_base: Option<String>,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    headers: Option<HashMap<String, String>>,
    #[serde(default, deserialize_with = "null_to_default")]
    additional_provider_specific_params: Map<String, Value>,
    #[serde(default)]
    unreachable_fallback: Option<UnreachableFallback>,
}

fn generic(params: &Value) -> Result<GenericApiConfig, Unsupported> {
    let raw: GenericParams = parse_params(params)?;
    let api_base = resolve_secret(raw.api_base.as_deref())
        .ok_or_else(|| unsupported("generic_guardrail_api requires api_base"))?;
    Ok(GenericApiConfig {
        api_base,
        api_key: resolve_secret(raw.api_key.as_deref()),
        headers: raw.headers,
        additional_provider_specific_params: raw.additional_provider_specific_params,
        unreachable_fallback: raw.unreachable_fallback,
    })
}

#[derive(Deserialize, Default)]
struct OpenAiParams {
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    api_base: Option<String>,
    #[serde(default)]
    model: Option<String>,
}

fn openai_moderation(params: &Value) -> Result<OpenaiModerationConfig, Unsupported> {
    let raw: OpenAiParams = parse_params(params)?;
    let api_key = resolve_secret(raw.api_key.as_deref()).or_else(|| {
        std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|v| !v.trim().is_empty())
    });
    Ok(OpenaiModerationConfig {
        api_key,
        api_base: resolve_secret(raw.api_base.as_deref()),
        model: raw.model,
    })
}

fn local_pii(params: &Value) -> Result<LocalPiiConfig, Unsupported> {
    parse_params(params)
}

#[derive(Deserialize, Default)]
struct BedrockParams {
    #[serde(default, rename = "guardrailIdentifier")]
    guardrail_identifier: Option<String>,
    #[serde(default, rename = "guardrailVersion")]
    guardrail_version: Option<String>,
    #[serde(default)]
    disable_exception_on_block: Option<bool>,
    #[serde(default)]
    aws_role_name: Option<String>,
    #[serde(default)]
    aws_profile_name: Option<String>,
    #[serde(default)]
    aws_web_identity_token: Option<String>,
    #[serde(default)]
    aws_region_name: Option<String>,
    #[serde(default)]
    aws_access_key_id: Option<String>,
    #[serde(default)]
    aws_secret_access_key: Option<String>,
    #[serde(default)]
    aws_session_token: Option<String>,
    #[serde(default)]
    aws_bedrock_runtime_endpoint: Option<String>,
}

fn bedrock(params: &Value) -> Result<BedrockConfig, Unsupported> {
    let raw: BedrockParams = parse_params(params)?;

    // Role/profile/web-identity credential flows and the exception-suppression
    // option are Python-only; fall back so those paths keep working.
    if raw.aws_role_name.is_some()
        || raw.aws_profile_name.is_some()
        || raw.aws_web_identity_token.is_some()
        || raw.disable_exception_on_block.unwrap_or(false)
    {
        return Err(unsupported(
            "bedrock role/profile/web-identity/disable_exception",
        ));
    }

    let guardrail_identifier = raw
        .guardrail_identifier
        .filter(|v| !v.trim().is_empty())
        .ok_or_else(|| unsupported("bedrock requires guardrailIdentifier"))?;
    let guardrail_version = raw
        .guardrail_version
        .filter(|v| !v.trim().is_empty())
        .ok_or_else(|| unsupported("bedrock requires guardrailVersion"))?;

    // Only static credentials are supported on the Rust path; without them fall
    // back to Python's credential chain (instance roles, SSO, etc.).
    let aws_access_key_id = resolve_secret(raw.aws_access_key_id.as_deref())
        .or_else(|| env_var("AWS_ACCESS_KEY_ID"))
        .ok_or_else(|| unsupported("bedrock requires static aws_access_key_id"))?;
    let aws_secret_access_key = resolve_secret(raw.aws_secret_access_key.as_deref())
        .or_else(|| env_var("AWS_SECRET_ACCESS_KEY"))
        .ok_or_else(|| unsupported("bedrock requires static aws_secret_access_key"))?;
    let aws_session_token =
        resolve_secret(raw.aws_session_token.as_deref()).or_else(|| env_var("AWS_SESSION_TOKEN"));

    let aws_region_name = raw
        .aws_region_name
        .filter(|v| !v.trim().is_empty())
        .or_else(|| env_var("AWS_REGION"))
        .or_else(|| env_var("AWS_DEFAULT_REGION"))
        .ok_or_else(|| unsupported("bedrock requires aws_region_name or AWS_REGION"))?;

    Ok(BedrockConfig {
        guardrail_identifier,
        guardrail_version,
        disable_exception_on_block: false,
        aws_region_name: Some(aws_region_name),
        aws_access_key_id: Some(aws_access_key_id),
        aws_secret_access_key: Some(aws_secret_access_key),
        aws_session_token,
        aws_bedrock_runtime_endpoint: raw.aws_bedrock_runtime_endpoint,
    })
}

#[derive(Deserialize, Default)]
struct LakeraParams {
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    api_base: Option<String>,
    #[serde(default)]
    project_id: Option<String>,
    #[serde(default)]
    payload: Option<bool>,
    #[serde(default)]
    breakdown: Option<bool>,
    #[serde(default)]
    metadata: Value,
    #[serde(default)]
    dev_info: Option<bool>,
    #[serde(default)]
    on_flagged: Option<OnFlagged>,
}

fn lakera_v2(params: &Value) -> Result<LakeraV2Config, Unsupported> {
    let raw: LakeraParams = parse_params(params)?;
    let api_key = resolve_secret(raw.api_key.as_deref()).or_else(|| env_var("LAKERA_API_KEY"));
    Ok(LakeraV2Config {
        api_key,
        api_base: resolve_secret(raw.api_base.as_deref()),
        project_id: raw.project_id,
        payload: raw.payload,
        breakdown: raw.breakdown,
        metadata: raw.metadata,
        dev_info: raw.dev_info,
        on_flagged: raw.on_flagged,
    })
}

#[derive(Deserialize, Default)]
struct AzurePromptShieldParams {
    #[serde(default)]
    api_base: Option<String>,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    api_version: Option<String>,
}

fn azure_prompt_shield(params: &Value) -> Result<AzurePromptShieldConfig, Unsupported> {
    let raw: AzurePromptShieldParams = parse_params(params)?;
    let api_base = resolve_secret(raw.api_base.as_deref())
        .ok_or_else(|| unsupported("azure/prompt_shield requires api_base"))?;
    Ok(AzurePromptShieldConfig {
        api_base,
        api_key: resolve_secret(raw.api_key.as_deref()),
        api_version: raw.api_version,
    })
}

#[derive(Deserialize, Default)]
struct AzureTextModerationParams {
    #[serde(default)]
    api_base: Option<String>,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    api_version: Option<String>,
    #[serde(default)]
    severity_threshold: Option<u8>,
    #[serde(default, deserialize_with = "null_to_default")]
    severity_threshold_by_category: HashMap<String, u8>,
    #[serde(default)]
    categories: Option<Vec<String>>,
    #[serde(
        default,
        rename = "blocklistNames",
        deserialize_with = "null_to_default"
    )]
    blocklist_names: Vec<String>,
    #[serde(
        default,
        rename = "haltOnBlocklistHit",
        deserialize_with = "null_to_default"
    )]
    halt_on_blocklist_hit: bool,
    #[serde(default, rename = "outputType")]
    output_type: Option<String>,
}

fn azure_text_moderation(params: &Value) -> Result<AzureTextModerationConfig, Unsupported> {
    let raw: AzureTextModerationParams = parse_params(params)?;
    let api_base = resolve_secret(raw.api_base.as_deref())
        .ok_or_else(|| unsupported("azure/text_moderations requires api_base"))?;
    Ok(AzureTextModerationConfig {
        api_base,
        api_key: resolve_secret(raw.api_key.as_deref()),
        api_version: raw.api_version,
        severity_threshold: raw.severity_threshold,
        severity_threshold_by_category: raw.severity_threshold_by_category,
        categories: raw.categories,
        blocklist_names: raw.blocklist_names,
        halt_on_blocklist_hit: raw.halt_on_blocklist_hit,
        output_type: raw.output_type,
    })
}

#[derive(Deserialize, Default)]
struct PresidioParams {
    #[serde(default)]
    output_parse_pii: Option<bool>,
    #[serde(default)]
    mock_redacted_text: Option<Value>,
    #[serde(default)]
    presidio_analyzer_api_base: Option<String>,
    #[serde(default)]
    presidio_anonymizer_api_base: Option<String>,
    #[serde(default, deserialize_with = "null_to_default")]
    pii_entities_config: HashMap<String, PiiAction>,
    #[serde(default)]
    presidio_language: Option<String>,
    #[serde(default, deserialize_with = "null_to_default")]
    presidio_score_thresholds: HashMap<String, f64>,
    #[serde(default, deserialize_with = "null_to_default")]
    presidio_entities_deny_list: Vec<String>,
    #[serde(default)]
    presidio_ad_hoc_recognizers: Option<String>,
}

fn presidio(params: &Value) -> Result<PresidioConfig, Unsupported> {
    let raw: PresidioParams = parse_params(params)?;

    // The Rust path does not implement output_parse_pii (un-masking the
    // response) or the mock_redacted_text test hook; fall back to Python.
    if raw.output_parse_pii.unwrap_or(false) || raw.mock_redacted_text.is_some() {
        return Err(unsupported("presidio output_parse_pii/mock_redacted_text"));
    }

    let analyzer = resolve_secret(raw.presidio_analyzer_api_base.as_deref())
        .or_else(|| env_var("PRESIDIO_ANALYZER_API_BASE"))
        .ok_or_else(|| unsupported("presidio requires presidio_analyzer_api_base"))?;
    let anonymizer = resolve_secret(raw.presidio_anonymizer_api_base.as_deref())
        .or_else(|| env_var("PRESIDIO_ANONYMIZER_API_BASE"));

    let ad_hoc_recognizers = match raw.presidio_ad_hoc_recognizers.as_deref() {
        Some(path) if !path.trim().is_empty() => read_ad_hoc_recognizers(path)?,
        _ => Value::Null,
    };

    Ok(PresidioConfig {
        presidio_analyzer_api_base: analyzer,
        presidio_anonymizer_api_base: anonymizer,
        pii_entities_config: raw.pii_entities_config,
        presidio_language: raw.presidio_language,
        presidio_score_thresholds: raw.presidio_score_thresholds,
        presidio_entities_deny_list: raw.presidio_entities_deny_list,
        presidio_ad_hoc_recognizers: ad_hoc_recognizers,
    })
}

fn read_ad_hoc_recognizers(path: &str) -> Result<Value, Unsupported> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| unsupported(format!("could not read presidio recognizers file: {e}")))?;
    let parsed: Value = serde_json::from_str(&contents)
        .map_err(|e| unsupported(format!("invalid presidio recognizers JSON: {e}")))?;
    Ok(parsed.get("recognizers").cloned().unwrap_or(Value::Null))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_falls_back_to_env_key() {
        // SAFETY: single-threaded test, restores nothing it didn't set.
        std::env::set_var("OPENAI_API_KEY", "sk-from-env");
        let cfg = openai_moderation(&serde_json::json!({})).unwrap();
        assert_eq!(cfg.api_key.as_deref(), Some("sk-from-env"));
        std::env::remove_var("OPENAI_API_KEY");
    }

    #[test]
    fn explicit_key_wins_over_env() {
        let cfg = openai_moderation(&serde_json::json!({"api_key": "sk-explicit"})).unwrap();
        assert_eq!(cfg.api_key.as_deref(), Some("sk-explicit"));
    }

    #[test]
    fn os_environ_prefix_is_resolved() {
        std::env::set_var("MY_MOD_KEY", "sk-resolved");
        let cfg =
            openai_moderation(&serde_json::json!({"api_key": "os.environ/MY_MOD_KEY"})).unwrap();
        assert_eq!(cfg.api_key.as_deref(), Some("sk-resolved"));
        std::env::remove_var("MY_MOD_KEY");
    }

    #[test]
    fn unknown_type_is_unsupported() {
        assert!(build_config("definitely_not_a_guardrail", &serde_json::json!({})).is_err());
    }

    #[test]
    fn null_collection_fields_are_treated_as_absent() {
        // The proxy's Pydantic model_dump() emits null for every unset field.
        // Non-Option collections/bools must tolerate null, not fail to parse.
        let presidio = serde_json::json!({
            "presidio_analyzer_api_base": "http://localhost:5002",
            "presidio_anonymizer_api_base": "http://localhost:5001",
            "pii_entities_config": null,
            "presidio_score_thresholds": null,
            "presidio_entities_deny_list": null,
            "presidio_language": null,
            "output_parse_pii": null,
            "mock_redacted_text": null,
            "presidio_ad_hoc_recognizers": null,
        });
        assert!(build_config("presidio", &presidio).is_ok());

        let azure = serde_json::json!({
            "api_base": "https://contoso.cognitiveservices.azure.com",
            "severity_threshold_by_category": null,
            "blocklistNames": null,
            "haltOnBlocklistHit": null,
            "categories": null,
        });
        assert!(build_config("azure/text_moderations", &azure).is_ok());

        let generic = serde_json::json!({
            "api_base": "http://localhost:8080",
            "additional_provider_specific_params": null,
            "headers": null,
        });
        assert!(build_config("generic_guardrail_api", &generic).is_ok());
    }
}
