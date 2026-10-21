mod error;
mod includes;
mod mcp;
mod model;
mod settings;
mod value;

use std::{fmt, path::Path};

use serde::Deserialize;

pub use error::Error;
pub use mcp::{McpAuth, McpServer, McpTransport};
pub use model::{LiteLlmParams, Model};
pub use settings::{
    ClickHouseStoreSettings, GeneralSettings, LiteLlmSettings, RouterSettings, TracingSettings,
    TracingStoreSettings,
};
pub use value::{AdditionalFields, Flag, NumberOrString, Object, OneOrMany, Value};

#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model_list: Box<[Model]>,
    pub general_settings: GeneralSettings,
    pub router_settings: RouterSettings,
    pub litellm_settings: LiteLlmSettings,
    pub environment_variables: Object,
    pub callback_settings: Object,
    pub assistant_settings: Object,
    pub default_vertex_config: Object,
    pub mcp_servers: std::collections::BTreeMap<String, McpServer>,
    pub credential_list: Box<[Object]>,
    pub guardrails: Box<[Object]>,
    pub prompts: Box<[Object]>,
    pub sandbox_tools: Box<[Object]>,
    pub search_tools: Box<[Object]>,
    pub files_settings: Box<[Object]>,
    pub finetune_settings: Box<[Object]>,
    pub mcp_tools: Box<[Object]>,
    pub vector_store_registry: Box<[Object]>,
    pub worker_registry: Box<[Object]>,
    pub agents: Box<[Object]>,
    pub agent_list: Box<[Object]>,
    pub policies: Object,
    pub policy_attachments: Box<[Object]>,
    pub include: Box<[String]>,
    #[serde(flatten)]
    pub additional_fields: AdditionalFields,
}

impl fmt::Debug for Config {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Config")
            .field("model_list", &self.model_list)
            .field("general_settings", &self.general_settings)
            .field("router_settings", &self.router_settings)
            .field("litellm_settings", &self.litellm_settings)
            .field("environment_variables", &self.environment_variables)
            .field("callback_settings", &self.callback_settings)
            .field("assistant_settings", &self.assistant_settings)
            .field("default_vertex_config", &self.default_vertex_config)
            .field("mcp_servers", &self.mcp_servers)
            .field("credential_list", &self.credential_list)
            .field("guardrails", &self.guardrails)
            .field("prompts", &self.prompts)
            .field("sandbox_tools", &self.sandbox_tools)
            .field("search_tools", &self.search_tools)
            .field("files_settings", &self.files_settings)
            .field("finetune_settings", &self.finetune_settings)
            .field("mcp_tools", &self.mcp_tools)
            .field("vector_store_registry", &self.vector_store_registry)
            .field("worker_registry", &self.worker_registry)
            .field("agents", &self.agents)
            .field("agent_list", &self.agent_list)
            .field("policies", &self.policies)
            .field("policy_attachments", &self.policy_attachments)
            .field("include", &self.include)
            .field("additional_fields", &self.additional_fields.keys())
            .finish()
    }
}

impl Config {
    pub fn from_yaml(yaml: &str) -> Result<Self, Error> {
        Ok(serde_yaml_ng::from_str(yaml)?)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        Ok(serde_yaml_ng::from_value(includes::load(path.as_ref())?)?)
    }
}
