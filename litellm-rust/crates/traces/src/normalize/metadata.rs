use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};

use super::{SpanContext, attr};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentType {
    Root,
    Subagent,
    Middleware,
    Compaction,
}

#[derive(
    Clone, Debug, Eq, PartialEq, Serialize, Deserialize, strum::EnumString, strum::Display,
)]
#[serde(from = "String", into = "String")]
#[strum(serialize_all = "kebab-case")]
pub enum Integration {
    ClaudeCode,
    ClaudeAgentSdk,
    OpenaiCodex,
    DeepagentsCode,
    Cursor,
    Pi,
    Opencode,
    Copilot,
    Langchain,
    Langgraph,
    Deepagents,
    Autogen,
    Crewai,
    GoogleAdk,
    LlamaIndex,
    Mastra,
    MicrosoftAgentFramework,
    OpenaiAgents,
    PydanticAi,
    SemanticKernel,
    Strands,
    VercelAiSdk,
    Instructor,
    N8n,
    Temporal,
    #[strum(default)]
    Other(String),
}

impl From<String> for Integration {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
    }
}

impl From<Integration> for String {
    fn from(value: Integration) -> Self {
        value.to_string()
    }
}

#[derive(Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct AgentMetadata {
    #[serde(deserialize_with = "optional")]
    pub lc_agent_name: Option<String>,
    #[serde(deserialize_with = "optional")]
    pub ls_integration: Option<Integration>,
    #[serde(deserialize_with = "optional")]
    pub ls_agent_type: Option<AgentType>,
    #[serde(deserialize_with = "optional")]
    pub ls_agent_purpose: Option<String>,
    #[serde(deserialize_with = "optional")]
    pub ls_agent_runtime: Option<String>,
    #[serde(deserialize_with = "optional")]
    pub ls_agent_version: Option<String>,
    #[serde(deserialize_with = "optional")]
    pub ls_trace_schema_version: Option<String>,
    #[serde(deserialize_with = "optional")]
    pub thread_id: Option<String>,
    #[serde(deserialize_with = "optional")]
    pub ls_subagent_id: Option<String>,
    #[serde(deserialize_with = "optional")]
    pub ls_subagent_type: Option<String>,
    #[serde(deserialize_with = "optional")]
    pub ls_tool_name: Option<String>,
    #[serde(deserialize_with = "optional")]
    pub ls_model_name: Option<String>,
    #[serde(deserialize_with = "optional")]
    pub ls_provider: Option<String>,
    #[serde(deserialize_with = "optional")]
    pub git_branch: Option<String>,
    #[serde(deserialize_with = "optional")]
    pub git_commit_sha: Option<String>,
    #[serde(deserialize_with = "optional")]
    pub git_repo_url: Option<String>,
    #[serde(deserialize_with = "optional")]
    pub working_directory: Option<String>,
}

impl AgentMetadata {
    pub(crate) fn byte_len(&self) -> usize {
        let strings = [
            &self.lc_agent_name,
            &self.ls_agent_purpose,
            &self.ls_agent_runtime,
            &self.ls_agent_version,
            &self.ls_trace_schema_version,
            &self.thread_id,
            &self.ls_subagent_id,
            &self.ls_subagent_type,
            &self.ls_tool_name,
            &self.ls_model_name,
            &self.ls_provider,
            &self.git_branch,
            &self.git_commit_sha,
            &self.git_repo_url,
            &self.working_directory,
        ];
        strings
            .into_iter()
            .filter_map(Option::as_ref)
            .map(String::len)
            .sum::<usize>()
            + self
                .ls_integration
                .as_ref()
                .map_or(0, |integration| integration.to_string().len())
    }
}

#[derive(strum::EnumString, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum MetadataField {
    LcAgentName,
    LsIntegration,
    LsAgentType,
    LsAgentPurpose,
    LsAgentRuntime,
    #[strum(serialize = "ls_agent_runtime_version", to_string = "ls_agent_version")]
    LsAgentVersion,
    LsTraceSchemaVersion,
    ThreadId,
    LsSubagentId,
    LsSubagentType,
    LsToolName,
    LsModelName,
    LsProvider,
    GitBranch,
    GitCommitSha,
    #[strum(serialize = "repository_url", to_string = "git_repo_url")]
    GitRepoUrl,
    #[strum(serialize = "cwd", to_string = "working_directory")]
    WorkingDirectory,
}

fn optional<'de, D: Deserializer<'de>, T: DeserializeOwned>(
    deserializer: D,
) -> Result<Option<T>, D::Error> {
    let value = Value::deserialize(deserializer)?;
    Ok(serde_json::from_value(value).ok())
}

fn field(key: &str, value: impl FnOnce() -> Value) -> Option<(String, Value)> {
    let canonical: &'static str = MetadataField::try_from(key).ok()?.into();
    let value = value();
    if value.is_null() || value.as_str().is_some_and(str::is_empty) {
        return None;
    }
    Some((canonical.to_owned(), value))
}

pub(super) fn extract(context: &SpanContext<'_>) -> AgentMetadata {
    let nested = serde_json::from_str::<Map<String, Value>>(attr(context.attributes, "metadata"))
        .unwrap_or_default();
    let values: BTreeMap<String, Value> = nested
        .into_iter()
        .filter_map(|(key, value)| field(&key, || value))
        .chain(
            context
                .attributes
                .iter()
                .filter_map(|(key, value)| field(key, || Value::String(value.clone()))),
        )
        .chain(context.attributes.iter().filter_map(|(key, value)| {
            field(key.strip_prefix("langsmith.metadata.")?, || {
                Value::String(value.clone())
            })
        }))
        .collect();
    serde_json::from_value(Value::Object(values.into_iter().collect())).unwrap_or_default()
}
