use serde::{Deserialize, Serialize};

use crate::base_llm::auth::Headers;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct BedrockMessagesConnection {
    pub api_base: Option<String>,
    pub region: Option<String>,
    pub model_id: Option<String>,
    pub workspace_id: Option<String>,
}

impl BedrockMessagesConnection {
    pub(crate) fn headers(&self, headers: Headers) -> Headers {
        let headers: Headers = headers
            .into_iter()
            .filter(|(name, _)| {
                self.workspace_id.is_none() || !name.eq_ignore_ascii_case("anthropic-workspace-id")
            })
            .collect();
        headers
            .into_iter()
            .chain(
                self.workspace_id
                    .as_ref()
                    .filter(|value| !value.is_empty())
                    .map(|value| ("anthropic-workspace-id".into(), value.clone())),
            )
            .collect()
    }
}
