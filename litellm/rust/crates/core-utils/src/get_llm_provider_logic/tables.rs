use std::collections::{HashMap, HashSet};

use fancy_regex::Regex;
use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
pub struct ModelLists {
    pub open_ai_chat_completion_models: HashSet<String>,
    pub openai_video_generation_models: HashSet<String>,
    pub open_ai_text_completion_models: HashSet<String>,
    pub anthropic_models: HashSet<String>,
    pub cohere_models: HashSet<String>,
    pub cohere_embedding_models: HashSet<String>,
    pub cohere_chat_models: HashSet<String>,
    pub replicate_models: HashSet<String>,
    pub openrouter_models: HashSet<String>,
    pub maritalk_models: HashSet<String>,
    pub vertex_chat_models: HashSet<String>,
    pub vertex_code_chat_models: HashSet<String>,
    pub vertex_text_models: HashSet<String>,
    pub vertex_code_text_models: HashSet<String>,
    pub vertex_language_models: HashSet<String>,
    pub vertex_embedding_models: HashSet<String>,
    pub vertex_vision_models: HashSet<String>,
    pub vertex_ai_image_models: HashSet<String>,
    pub vertex_ai_video_models: HashSet<String>,
    pub ai21_chat_models: HashSet<String>,
    pub ai21_models: HashSet<String>,
    pub aleph_alpha_models: HashSet<String>,
    pub baseten_models: HashSet<String>,
    pub nlp_cloud_models: HashSet<String>,
    pub petals_models: HashSet<String>,
    pub bedrock_models: HashSet<String>,
    pub bedrock_embedding_models: HashSet<String>,
    pub bedrock_converse_models: HashSet<String>,
    pub watsonx_models: HashSet<String>,
    pub open_ai_embedding_models: HashSet<String>,
    pub empower_models: HashSet<String>,
    pub gradient_ai_models: HashSet<String>,
    pub mistral_chat_models: HashSet<String>,
    pub provider_list: HashSet<String>,
    pub model_list_set: HashSet<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JsonProvider {
    pub slug: String,
    pub base_url: String,
    pub api_key_env: String,
    pub api_base_env: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LatestAlias {
    pub key: String,
    pub target: String,
    pub provider: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RoutingRule {
    pub pattern: String,
    pub provider: String,
}

#[derive(Debug, Deserialize)]
pub struct ProviderSnapshot {
    pub model_lists: ModelLists,
    pub openai_compatible_endpoints: Vec<String>,
    pub json_providers: Vec<JsonProvider>,
    pub custom_providers: Option<HashSet<String>>,
    pub latest_aliases: Vec<LatestAlias>,
    pub routing_rules: Vec<RoutingRule>,
    pub use_litellm_proxy: bool,
    pub suppress_debug_info: bool,
    pub secret_manager: bool,
    pub replicate_model_name_with_id_length: usize,
    pub nadir_default_api_base: String,
    pub nscale_api_base_url: String,
}

#[derive(Debug)]
pub enum RoutingRules {
    Compiled(Vec<(Regex, String)>),
    Uncompilable,
}

#[derive(Debug)]
pub struct ProviderTables {
    pub lists: ModelLists,
    pub openai_compatible_endpoints: Vec<String>,
    pub json_providers: Vec<JsonProvider>,
    json_slugs: HashSet<String>,
    pub custom_providers: Option<HashSet<String>>,
    latest_aliases: HashMap<String, LatestAlias>,
    pub routing_rules: RoutingRules,
    pub use_litellm_proxy: bool,
    pub suppress_debug_info: bool,
    pub secret_manager: bool,
    pub replicate_model_name_with_id_length: usize,
    pub nadir_default_api_base: String,
    pub nscale_api_base_url: String,
}

impl From<ProviderSnapshot> for ProviderTables {
    fn from(snapshot: ProviderSnapshot) -> Self {
        let routing_rules = snapshot
            .routing_rules
            .into_iter()
            .map(|rule| {
                Regex::new(&format!("(?i){}", rule.pattern))
                    .map(|pattern| (pattern, rule.provider))
                    .ok()
            })
            .collect::<Option<Vec<_>>>()
            .map_or(RoutingRules::Uncompilable, RoutingRules::Compiled);
        Self {
            json_slugs: snapshot
                .json_providers
                .iter()
                .map(|provider| provider.slug.clone())
                .collect(),
            latest_aliases: snapshot
                .latest_aliases
                .into_iter()
                .map(|alias| (alias.key.clone(), alias))
                .collect(),
            lists: snapshot.model_lists,
            openai_compatible_endpoints: snapshot.openai_compatible_endpoints,
            json_providers: snapshot.json_providers,
            custom_providers: snapshot.custom_providers,
            routing_rules,
            use_litellm_proxy: snapshot.use_litellm_proxy,
            suppress_debug_info: snapshot.suppress_debug_info,
            secret_manager: snapshot.secret_manager,
            replicate_model_name_with_id_length: snapshot.replicate_model_name_with_id_length,
            nadir_default_api_base: snapshot.nadir_default_api_base,
            nscale_api_base_url: snapshot.nscale_api_base_url,
        }
    }
}

impl ProviderTables {
    pub fn is_json_provider(&self, slug: &str) -> bool {
        self.json_slugs.contains(slug)
    }

    pub fn json_provider(&self, slug: &str) -> Option<&JsonProvider> {
        self.json_providers
            .iter()
            .find(|provider| provider.slug == slug)
    }

    pub fn json_provider_by_base_url(&self, base_url: &str) -> Option<&JsonProvider> {
        self.json_providers
            .iter()
            .find(|provider| provider.base_url == base_url)
    }

    fn alias_target(&self, key: &str, provider: Option<&str>) -> Option<String> {
        let alias = self.latest_aliases.get(key)?;
        let alias_provider = match key.split_once('/') {
            Some((prefix, _)) if self.lists.provider_list.contains(prefix) => Some(prefix),
            _ => alias.provider.as_deref(),
        };
        if alias_provider != provider {
            return None;
        }
        let prefix = format!("{}/", provider.unwrap_or("None"));
        Some(
            alias
                .target
                .strip_prefix(&prefix)
                .unwrap_or(&alias.target)
                .to_owned(),
        )
        .filter(|target| !target.is_empty())
    }

    pub fn resolve_latest_alias(&self, model: String, provider: Option<&str>) -> String {
        self.alias_target(&format!("{}/{model}", provider.unwrap_or("None")), provider)
            .or_else(|| self.alias_target(&model, provider))
            .unwrap_or(model)
    }
}
