use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "guardrail", rename_all = "snake_case")]
pub enum ProviderConfig {
    #[serde(rename = "generic_guardrail_api")]
    GenericGuardrailApi(GenericApiConfig),
    #[serde(rename = "openai_moderation")]
    OpenaiModeration(OpenaiModerationConfig),
    #[serde(rename = "azure/prompt_shield")]
    AzurePromptShield(AzurePromptShieldConfig),
    #[serde(rename = "azure/text_moderations")]
    AzureTextModeration(AzureTextModerationConfig),
    Presidio(PresidioConfig),
    #[serde(rename = "lakera_v2")]
    LakeraV2(LakeraV2Config),
    Bedrock(BedrockConfig),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericApiConfig {
    pub api_base: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub headers: Option<HashMap<String, String>>,
    #[serde(default)]
    pub additional_provider_specific_params: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub unreachable_fallback: Option<UnreachableFallback>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenaiModerationConfig {
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub api_base: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzurePromptShieldConfig {
    #[serde(default)]
    pub api_key: Option<String>,
    pub api_base: String,
    #[serde(default)]
    pub api_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureTextModerationConfig {
    #[serde(default)]
    pub api_key: Option<String>,
    pub api_base: String,
    #[serde(default)]
    pub api_version: Option<String>,
    #[serde(default)]
    pub severity_threshold: Option<u8>,
    #[serde(default)]
    pub severity_threshold_by_category: HashMap<String, u8>,
    #[serde(default)]
    pub categories: Option<Vec<String>>,
    #[serde(default)]
    pub blocklist_names: Vec<String>,
    #[serde(default)]
    pub halt_on_blocklist_hit: bool,
    #[serde(default)]
    pub output_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresidioConfig {
    pub presidio_analyzer_api_base: String,
    #[serde(default)]
    pub presidio_anonymizer_api_base: Option<String>,
    #[serde(default)]
    pub pii_entities_config: HashMap<String, PiiAction>,
    #[serde(default)]
    pub presidio_language: Option<String>,
    #[serde(default)]
    pub presidio_score_thresholds: HashMap<String, f64>,
    #[serde(default)]
    pub presidio_entities_deny_list: Vec<String>,
    #[serde(default)]
    pub presidio_ad_hoc_recognizers: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum PiiAction {
    Mask,
    Block,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LakeraV2Config {
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub api_base: Option<String>,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub payload: Option<bool>,
    #[serde(default)]
    pub breakdown: Option<bool>,
    #[serde(default)]
    pub metadata: serde_json::Value,
    #[serde(default)]
    pub dev_info: Option<bool>,
    #[serde(default)]
    pub on_flagged: Option<OnFlagged>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnFlagged {
    Block,
    Monitor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockConfig {
    #[serde(rename = "guardrailIdentifier")]
    pub guardrail_identifier: String,
    #[serde(rename = "guardrailVersion")]
    pub guardrail_version: String,
    #[serde(default)]
    pub disable_exception_on_block: bool,
    #[serde(default)]
    pub aws_region_name: Option<String>,
    #[serde(default)]
    pub aws_access_key_id: Option<String>,
    #[serde(default)]
    pub aws_secret_access_key: Option<String>,
    #[serde(default)]
    pub aws_session_token: Option<String>,
    #[serde(default)]
    pub aws_bedrock_runtime_endpoint: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnreachableFallback {
    FailClosed,
    FailOpen,
}
