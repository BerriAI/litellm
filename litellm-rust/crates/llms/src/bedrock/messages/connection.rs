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
        let Some(trusted) = self.workspace_id.as_ref().filter(|value| !value.is_empty()) else {
            return headers;
        };
        headers
            .into_iter()
            .filter(|(name, _)| !name.eq_ignore_ascii_case("anthropic-workspace-id"))
            .chain([("anthropic-workspace-id".into(), trusted.clone())])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn connection(workspace_id: Option<&str>) -> BedrockMessagesConnection {
        BedrockMessagesConnection {
            workspace_id: workspace_id.map(str::to_string),
            ..BedrockMessagesConnection::default()
        }
    }

    #[rstest]
    #[case::replaces("Anthropic-Workspace-Id")]
    #[case::replaces_lowercase("anthropic-workspace-id")]
    fn trusted_workspace_replaces_the_caller_header(#[case] name: &str) {
        assert_eq!(
            connection(Some("trusted")).headers(vec![
                ("x-other".into(), "keep".into()),
                (name.into(), "caller".into()),
            ]),
            vec![
                ("x-other".into(), "keep".into()),
                ("anthropic-workspace-id".into(), "trusted".into()),
            ]
        );
    }

    #[rstest]
    #[case::empty(Some(""))]
    #[case::absent(None)]
    fn untrusted_workspace_keeps_the_caller_header(#[case] workspace_id: Option<&str>) {
        let headers = vec![("Anthropic-Workspace-Id".into(), "caller".into())];
        assert_eq!(connection(workspace_id).headers(headers.clone()), headers);
    }

    #[rstest]
    fn other_headers_keep_their_order() {
        let headers = vec![
            ("x-first".into(), "1".into()),
            ("x-second".into(), "2".into()),
        ];
        assert_eq!(
            connection(Some("trusted")).headers(headers.clone()),
            headers
                .into_iter()
                .chain([("anthropic-workspace-id".into(), "trusted".into())])
                .collect::<Vec<_>>()
        );
    }
}
