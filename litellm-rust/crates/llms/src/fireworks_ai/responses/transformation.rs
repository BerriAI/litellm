//! Fireworks AI Responses, matching `litellm/llms/fireworks_ai/responses/transformation.py`

use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_llms_types::formats::responses::ResponsesApiResponse;
use serde_json::{Map, Value, json};

use crate::{
    Error,
    base_llm::{
        auth::{AuthScheme, ValidatedEnvironment},
        responses::transformation::BaseResponsesApiConfig,
    },
    openai::responses::{
        request::ResponsesHost,
        transformation::{responses_request, responses_response},
    },
};

const API_KEY_ENVS: [&str; 4] = [
    "FIREWORKS_API_KEY",
    "FIREWORKS_AI_API_KEY",
    "FIREWORKSAI_API_KEY",
    "FIREWORKS_AI_TOKEN",
];
const API_BASE_ENV: &str = "FIREWORKS_API_BASE";
const DEFAULT_BASE: &str = "https://api.fireworks.ai/inference/v1";
const INSTRUCTION_ROLES: [&str; 2] = ["system", "developer"];
const AZURE_FOUNDRY_MODEL_ID_PREFIX: &str = "FW-";
const FIREROUTER: &str = "firerouter";
const ROUTER_SHORT_NAMES: [&str; 3] = [FIREROUTER, "auto", "auto-instant"];

pub struct FireworksAiResponsesApiConfig;
pub const FIREWORKS_AI_RESPONSES_CONFIG: FireworksAiResponsesApiConfig =
    FireworksAiResponsesApiConfig;

impl BaseResponsesApiConfig for FireworksAiResponsesApiConfig {
    fn secret_names(
        &self,
        api_key: Option<&str>,
        api_base: Option<&str>,
    ) -> &'static [&'static str] {
        match (
            api_key.is_some_and(|key| !key.is_empty()),
            api_base.is_some_and(|base| !base.is_empty()),
        ) {
            (true, true) => &[],
            (true, false) => &[API_BASE_ENV],
            (false, true) => &API_KEY_ENVS,
            (false, false) => &[
                "FIREWORKS_API_KEY",
                "FIREWORKS_AI_API_KEY",
                "FIREWORKSAI_API_KEY",
                "FIREWORKS_AI_TOKEN",
                API_BASE_ENV,
            ],
        }
    }

    fn validate_environment(
        &self,
        headers: Vec<(String, String)>,
        api_key: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let key = api_key
            .filter(|key| !key.is_empty())
            .map(str::to_owned)
            .or_else(|| {
                API_KEY_ENVS
                    .iter()
                    .find_map(|name| lookup(name).filter(|key| !key.is_empty()))
            })
            .ok_or(litellm_auth::Error::MissingApiKey {
                provider: "Fireworks AI",
                environment_variable: API_KEY_ENVS[0],
            })?;
        Ok(ValidatedEnvironment {
            headers: litellm_http::request::with_default_headers(
                headers,
                &[("content-type", "application/json")],
            ),
            auth: AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                secret: SecretValue::new(key),
            },
        })
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _api_version: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let api_base = api_base
            .filter(|base| !base.is_empty())
            .map(str::to_owned)
            .or_else(|| lookup(API_BASE_ENV).filter(|base| !base.is_empty()))
            .unwrap_or_else(|| DEFAULT_BASE.into());
        Ok(format!("{}/responses", api_base.trim_end_matches('/')))
    }

    fn transform_responses_api_request(
        &self,
        model: &str,
        input: Value,
        mut params: Map<String, Value>,
    ) -> Result<Value, Error> {
        let input = match params.remove("instructions") {
            None | Some(Value::Null) => fold_instruction_items(input, None, &mut params),
            Some(Value::String(instructions)) => {
                fold_instruction_items(input, Some(instructions), &mut params)
            }
            Some(other) => {
                params.insert("instructions".into(), other);
                developer_items_as_system(input)
            }
        };
        responses_request(
            &resource_name(model),
            input,
            params,
            ResponsesHost::Compatible,
        )
    }

    fn transform_response_api_response(&self, body: Value) -> Result<ResponsesApiResponse, Error> {
        responses_response(body)
    }
}

/// Python's `resolve_fireworks_resource_name`: a bare model id becomes its account path
fn resource_name(model: &str) -> String {
    let stripped = model.strip_prefix("fireworks_ai/").unwrap_or(model);
    if stripped.starts_with("accounts/")
        || stripped.starts_with(AZURE_FOUNDRY_MODEL_ID_PREFIX)
        || stripped.contains('#')
    {
        stripped.into()
    } else if stripped.starts_with("routers/") || stripped.starts_with("models/") {
        format!("accounts/fireworks/{stripped}")
    } else if stripped.ends_with("-fast")
        || ROUTER_SHORT_NAMES.contains(&stripped)
        || stripped.starts_with(&format!("{FIREROUTER}/"))
    {
        format!("accounts/fireworks/routers/{stripped}")
    } else {
        format!("accounts/fireworks/models/{stripped}")
    }
}

fn role(item: &Value) -> Option<&str> {
    item.get("role").and_then(Value::as_str)
}

fn is_instruction_role(role: Option<&str>) -> bool {
    role.is_some_and(|role| INSTRUCTION_ROLES.contains(&role))
}

/// Fireworks takes no `developer` role, so those items replay as plain `system` messages
fn developer_item_as_system(item: Value) -> Value {
    match role(&item) {
        Some("developer") => json!({
            "role": "system",
            "content": item.get("content").cloned().unwrap_or(Value::Null),
            "type": "message",
        }),
        _ => item,
    }
}

fn developer_items_as_system(input: Value) -> Value {
    match input {
        Value::Array(items) => {
            Value::Array(items.into_iter().map(developer_item_as_system).collect())
        }
        other => other,
    }
}

/// The text of a text-only system or developer item, with its parts joined by blank lines
fn text_only_content(item: &Value) -> Option<String> {
    if !is_instruction_role(role(item)) {
        return None;
    }
    match item.get("content")? {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => {
            let texts = parts
                .iter()
                .map(|part| match part.get("type").and_then(Value::as_str) {
                    Some("input_text") => part.get("text").and_then(Value::as_str),
                    _ => None,
                })
                .collect::<Option<Vec<&str>>>()?;
            Some(
                texts
                    .into_iter()
                    .filter(|text| !text.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n\n"),
            )
        }
        _ => None,
    }
}

/// Python's `_hoisted_indices`: the leading instruction block, plus every developer item before
/// a closing instruction block that follows the last assistant turn
fn hoisted_indices(roles: &[Option<&str>]) -> Vec<usize> {
    let leading = roles
        .iter()
        .position(|role| !is_instruction_role(*role))
        .unwrap_or(roles.len());
    let closing = match (leading..roles.len())
        .rev()
        .find(|&index| !is_instruction_role(roles[index]))
    {
        Some(index) if roles[index] == Some("assistant") => index + 1,
        _ => roles.len(),
    };
    (0..closing)
        .filter(|&index| index < leading || roles[index] == Some("developer"))
        .collect()
}

/// Python's `_with_instruction_items_folded`: hoisted text-only instruction items join
/// `instructions`, and the remaining developer items become system messages
fn fold_instruction_items(
    input: Value,
    instructions: Option<String>,
    params: &mut Map<String, Value>,
) -> Value {
    let (instructions, input) = match input {
        Value::Array(items) => {
            let roles: Vec<Option<&str>> = items.iter().map(role).collect();
            let folded: Vec<(usize, String)> = hoisted_indices(&roles)
                .into_iter()
                .filter_map(|index| Some((index, text_only_content(&items[index])?)))
                .collect();
            let instructions = match folded.is_empty() {
                true => instructions,
                false => {
                    let joined = instructions
                        .iter()
                        .map(String::as_str)
                        .chain(folded.iter().map(|(_, text)| text.as_str()))
                        .filter(|chunk| !chunk.is_empty())
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    (!joined.is_empty()).then_some(joined)
                }
            };
            let kept = items
                .into_iter()
                .enumerate()
                .filter(|(index, _)| !folded.iter().any(|(folded, _)| folded == index))
                .map(|(_, item)| developer_item_as_system(item))
                .collect();
            (instructions, Value::Array(kept))
        }
        other => (instructions, other),
    };
    if let Some(instructions) = instructions {
        params.insert("instructions".into(), Value::String(instructions));
    }
    input
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned())
        }
    }

    #[rstest]
    #[case::default(None, &[], "https://api.fireworks.ai/inference/v1/responses")]
    #[case::explicit(Some("https://proxy.test/v1/"), &[], "https://proxy.test/v1/responses")]
    #[case::secret(None, &[("FIREWORKS_API_BASE", "https://secret.test/v1")], "https://secret.test/v1/responses")]
    fn url_is_the_base_plus_responses(
        #[case] api_base: Option<&str>,
        #[case] secrets: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        assert_eq!(
            FIREWORKS_AI_RESPONSES_CONFIG
                .get_complete_url(api_base, None, &env(secrets))
                .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::explicit(Some("explicit"), &[("FIREWORKS_API_KEY", "managed")], "explicit")]
    #[case::first_secret(None, &[("FIREWORKS_AI_API_KEY", "b"), ("FIREWORKS_API_KEY", "a")], "a")]
    #[case::token(None, &[("FIREWORKS_AI_TOKEN", "token")], "token")]
    fn bearer_comes_from_the_call_then_the_fireworks_secrets(
        #[case] api_key: Option<&str>,
        #[case] secrets: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        let environment = FIREWORKS_AI_RESPONSES_CONFIG
            .validate_environment(vec![], api_key, &env(secrets))
            .unwrap();
        let AuthScheme::Credential { placement, secret } = environment.auth else {
            panic!("expected a credential");
        };
        assert_eq!(placement, CredentialPlacement::Bearer);
        assert_eq!(secret.expose(), expected);
    }

    #[rstest]
    #[case::bare(
        "llama-v3p1-8b-instruct",
        "accounts/fireworks/models/llama-v3p1-8b-instruct"
    )]
    #[case::prefixed("fireworks_ai/qwen3", "accounts/fireworks/models/qwen3")]
    #[case::account("accounts/me/models/x", "accounts/me/models/x")]
    #[case::azure_foundry("FW-x", "FW-x")]
    #[case::deployment("x#accounts/me/deployments/d", "x#accounts/me/deployments/d")]
    #[case::models_path("models/x", "accounts/fireworks/models/x")]
    #[case::router_path("routers/x", "accounts/fireworks/routers/x")]
    #[case::fast("glm-fast", "accounts/fireworks/routers/glm-fast")]
    #[case::short_router("auto", "accounts/fireworks/routers/auto")]
    #[case::firerouter("firerouter/code", "accounts/fireworks/routers/firerouter/code")]
    fn model_is_the_fireworks_resource_name(#[case] model: &str, #[case] expected: &str) {
        assert_eq!(resource_name(model), expected);
    }

    fn request(input: Value, params: Value) -> Value {
        let Value::Object(params) = params else {
            unreachable!()
        };
        FIREWORKS_AI_RESPONSES_CONFIG
            .transform_responses_api_request("qwen3", input, params)
            .unwrap()
    }

    #[test]
    fn leading_instruction_items_fold_into_instructions() {
        let body = request(
            json!([
                {"role": "system", "content": "Be brief"},
                {"role": "developer", "content": [{"type": "input_text", "text": "Use French"}]},
                {"role": "user", "content": "hi"},
            ]),
            json!({"instructions": "Top"}),
        );
        assert_eq!(body["instructions"], json!("Top\n\nBe brief\n\nUse French"));
        assert_eq!(body["input"], json!([{"role": "user", "content": "hi"}]));
        assert_eq!(body["model"], json!("accounts/fireworks/models/qwen3"));
    }

    #[test]
    fn developer_items_after_the_last_assistant_turn_stay_as_system_messages() {
        let body = request(
            json!([
                {"role": "user", "content": "hi"},
                {"role": "developer", "content": "mid"},
                {"role": "assistant", "content": "hello"},
                {"role": "developer", "content": "late", "id": "x"},
            ]),
            json!({}),
        );
        assert_eq!(body["instructions"], json!("mid"));
        assert_eq!(
            body["input"],
            json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "system", "content": "late", "type": "message"},
            ])
        );
    }

    #[test]
    fn non_text_instruction_items_are_kept_and_non_string_instructions_only_rename_roles() {
        let image =
            json!({"role": "system", "content": [{"type": "input_image", "image_url": "u"}]});
        let body = request(
            json!([image.clone(), {"role": "user", "content": "hi"}]),
            json!({}),
        );
        assert!(body.get("instructions").is_none());
        assert_eq!(body["input"][0], image);
        let body = request(
            json!([{"role": "developer", "content": "d"}]),
            json!({"instructions": ["x"]}),
        );
        assert_eq!(body["instructions"], json!(["x"]));
        assert_eq!(
            body["input"],
            json!([{"role": "system", "content": "d", "type": "message"}])
        );
    }
}
