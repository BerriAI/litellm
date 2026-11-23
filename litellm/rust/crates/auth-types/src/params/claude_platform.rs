use serde::Deserialize;

use super::{ParamKind, ParamSpec};

/// The Claude Platform on AWS settings a `bedrock/claude_platform/` call carries: the workspace,
/// under any of the four spellings Python's config reads, and the body params the endpoint rejects.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct ClaudePlatformParams {
    pub workspace_id: Option<WorkspaceId>,
    pub aws_workspace_id: Option<WorkspaceId>,
    #[serde(rename = "anthropic-workspace-id")]
    pub anthropic_workspace_header: Option<WorkspaceId>,
    pub anthropic_workspace_id: Option<WorkspaceId>,
    pub claude_platform_unsupported_params: Option<UnsupportedParams>,
}

impl ClaudePlatformParams {
    /// The workspace Python's config requires before it builds the call, raising `AuthenticationError`.
    pub const WORKSPACE: ParamSpec = ParamSpec {
        setting: "workspace_id",
        kind: ParamKind::Credential,
        wire: &[
            "workspace_id",
            "aws_workspace_id",
            "anthropic-workspace-id",
            "anthropic_workspace_id",
        ],
        module_global: None,
        env: &["ANTHROPIC_AWS_WORKSPACE_ID", "ANTHROPIC_WORKSPACE_ID"],
    };

    const UNSUPPORTED_PARAMS: &'static str = "claude_platform_unsupported_params";

    /// The wire names a host projects out of a caller's kwargs, derived from [`Self::WORKSPACE`].
    pub fn fields() -> impl Iterator<Item = &'static str> {
        Self::WORKSPACE
            .wire
            .iter()
            .copied()
            .chain([Self::UNSUPPORTED_PARAMS])
    }

    /// The workspace in Python's lookup order, skipping empty spellings.
    pub fn workspace(&self) -> Option<String> {
        [
            &self.workspace_id,
            &self.aws_workspace_id,
            &self.anthropic_workspace_header,
            &self.anthropic_workspace_id,
        ]
        .into_iter()
        .find_map(|workspace| workspace.as_ref()?.text())
    }

    /// The body params to drop instead of the endpoint's default set, when a list names them.
    pub fn unsupported_params(&self) -> Option<&[String]> {
        match self.claude_platform_unsupported_params.as_ref()? {
            UnsupportedParams::Names(names) => Some(names),
            _ => None,
        }
    }

    /// The Python type name of an override that is not a list, which Python warns it ignores.
    pub fn ignored_override_type(&self) -> Option<&'static str> {
        match self.claude_platform_unsupported_params.as_ref()? {
            UnsupportedParams::Names(_) | UnsupportedParams::MixedList(_) => None,
            UnsupportedParams::Flag(_) => Some("bool"),
            UnsupportedParams::Integer(_) => Some("int"),
            UnsupportedParams::Float(_) => Some("float"),
            UnsupportedParams::Text(_) => Some("str"),
            UnsupportedParams::Mapping(_) => Some("dict"),
        }
    }
}

/// A workspace id as a config writes it: text, or a bare number YAML reads as one.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum WorkspaceId {
    Text(String),
    Number(serde_json::Number),
}

impl WorkspaceId {
    fn text(&self) -> Option<String> {
        match self {
            Self::Text(text) if text.is_empty() => None,
            Self::Text(text) => Some(text.clone()),
            Self::Number(number) => Some(number.to_string()),
        }
    }
}

/// `claude_platform_unsupported_params`: a list of names, or any other value, which Python
/// ignores in favour of the default set.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum UnsupportedParams {
    Names(Vec<String>),
    MixedList(Vec<serde::de::IgnoredAny>),
    Flag(bool),
    Integer(i64),
    Float(f64),
    Text(String),
    Mapping(serde::de::IgnoredAny),
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Map, Value, json};

    use super::*;

    fn platform(fields: Value) -> ClaudePlatformParams {
        serde_json::from_value(fields).unwrap()
    }

    #[rstest]
    #[case::none(json!({}), None)]
    #[case::text(json!({"workspace_id": "w"}), Some("w"))]
    #[case::number(json!({"aws_workspace_id": 42}), Some("42"))]
    #[case::header_spelling(json!({"anthropic-workspace-id": "h"}), Some("h"))]
    #[case::empty_skipped(json!({"workspace_id": "", "anthropic_workspace_id": "u"}), Some("u"))]
    fn the_workspace_is_read_from_any_spelling(
        #[case] fields: Value,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(platform(fields).workspace().as_deref(), expected);
    }

    #[rstest]
    #[case::workspace_id("workspace_id")]
    #[case::aws_workspace_id("aws_workspace_id")]
    #[case::header("anthropic-workspace-id")]
    #[case::anthropic_workspace_id("anthropic_workspace_id")]
    fn each_spelling_python_reads_is_projected_and_read(#[case] spelling: &str) {
        assert!(ClaudePlatformParams::fields().any(|name| name == spelling));
        assert_eq!(
            platform(json!({spelling: "w"})).workspace().as_deref(),
            Some("w")
        );
    }

    #[rstest]
    #[case::all_four(0)]
    #[case::from_aws_workspace_id(1)]
    #[case::from_header(2)]
    #[case::last_only(3)]
    fn the_first_spelling_set_in_the_specs_order_wins(#[case] first: usize) {
        let wire = ClaudePlatformParams::WORKSPACE.wire;
        let fields: Map<String, Value> = wire[first..]
            .iter()
            .map(|spelling| (spelling.to_string(), json!(spelling)))
            .collect();
        assert_eq!(
            platform(Value::Object(fields)).workspace().as_deref(),
            Some(wire[first])
        );
    }

    #[rstest]
    #[case::absent(json!({}), None)]
    #[case::names(json!({"claude_platform_unsupported_params": ["a", "b"]}), Some(vec!["a", "b"]))]
    #[case::empty_list(json!({"claude_platform_unsupported_params": []}), Some(vec![]))]
    #[case::text_ignored(json!({"claude_platform_unsupported_params": "a"}), None)]
    #[case::mixed_list_ignored(json!({"claude_platform_unsupported_params": ["a", 1]}), None)]
    fn only_a_list_of_names_overrides_the_rejected_params(
        #[case] fields: Value,
        #[case] expected: Option<Vec<&str>>,
    ) {
        let params = platform(fields);
        assert_eq!(
            params
                .unsupported_params()
                .map(|names| names.iter().map(String::as_str).collect::<Vec<_>>()),
            expected
        );
    }

    #[rstest]
    #[case::list(json!({"claude_platform_unsupported_params": ["a"]}), None)]
    #[case::mixed_list(json!({"claude_platform_unsupported_params": ["a", 1]}), None)]
    #[case::text(json!({"claude_platform_unsupported_params": "a"}), Some("str"))]
    #[case::integer(json!({"claude_platform_unsupported_params": 1}), Some("int"))]
    #[case::float(json!({"claude_platform_unsupported_params": 1.5}), Some("float"))]
    #[case::flag(json!({"claude_platform_unsupported_params": true}), Some("bool"))]
    #[case::mapping(json!({"claude_platform_unsupported_params": {"a": 1}}), Some("dict"))]
    fn an_ignored_override_is_named_by_its_python_type(
        #[case] fields: Value,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(platform(fields).ignored_override_type(), expected);
    }
}
