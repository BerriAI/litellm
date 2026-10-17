//! Azure OpenAI chat completions through the OpenAI passthrough: `litellm/llms/azure/azure.py`
//! with the client `BaseAzureLLM.get_azure_openai_client` builds. The deployment's settings
//! and `os.environ/` references resolve through the secret source, and an Entra ID token is
//! acquired through `litellm_auth_azure` when the deployment has no key.

use std::collections::BTreeMap;

use litellm_auth::azure::{AzureAuthInputs, SECRET_NAMES as AZURE_AUTH_SECRET_NAMES};
use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_llms::{
    Error as LlmError,
    azure::{
        chat::transformation::{AzureChatBody, AzureChatRequest},
        common_utils::{self, AZURE_API_KEY_ENV, AzureCredential},
    },
    base_llm::auth::resolve_auth,
};
use serde_json::{Map, Value};

use crate::{
    ChatCompletionsRoute, Error,
    common_utils::string_headers,
    openai::{OpenAiChatCall, PreparedCall},
};

const ENV_REFERENCE: &str = "os.environ/";

/// The `litellm_params` an `os.environ/` reference may stand in for.
const REFERABLE_PARAMS: &[&str] = &[
    "api_version",
    "base_model",
    "azure_ad_token",
    "tenant_id",
    "client_id",
    "client_secret",
    "azure_scope",
    "azure_authority_host",
    "azure_credential",
    "azure_federated_token_file",
];

impl ChatCompletionsRoute {
    pub(crate) async fn prepare_azure(
        &self,
        deployment: String,
        call: OpenAiChatCall,
    ) -> Result<PreparedCall, Error> {
        let OpenAiChatCall {
            body,
            api_key,
            api_base,
            extra_headers,
            litellm_params,
            drop_params,
            model_info,
            ..
        } = call;
        let references: Vec<String> = [api_key.as_deref(), api_base.as_deref()]
            .into_iter()
            .flatten()
            .chain(
                REFERABLE_PARAMS
                    .iter()
                    .filter_map(|name| litellm_params.get(*name).and_then(Value::as_str)),
            )
            .filter_map(|value| value.strip_prefix(ENV_REFERENCE))
            .map(str::to_owned)
            .collect();
        let names: Vec<&str> = common_utils::SECRET_NAMES
            .iter()
            .chain(AZURE_AUTH_SECRET_NAMES)
            .copied()
            .chain(references.iter().map(String::as_str))
            .collect();
        let secrets = self.secrets.resolve(&names).await?;
        let lookup = |key: &str| secrets.get(key);
        // Python's router reads every `os.environ/` setting from the environment when it loads
        // the deployment. A reference to an unset variable counts as unset.
        let resolve = |value: &str| match value.strip_prefix(ENV_REFERENCE) {
            Some(name) => lookup(name),
            None => Some(value.to_owned()),
        };
        let params: Map<String, Value> = litellm_params
            .into_iter()
            .map(|(name, value)| match value {
                Value::String(text) if REFERABLE_PARAMS.contains(&name.as_str()) => {
                    let value = resolve(&text).map_or(Value::Null, Value::String);
                    (name, value)
                }
                other => (name, other),
            })
            .collect();
        let configured = |name: &str| params.get(name).and_then(Value::as_str);

        let api_version = common_utils::resolve_api_version(configured("api_version"), &lookup);
        let api_base = common_utils::resolve_api_base(
            api_base.as_deref().and_then(resolve).as_deref(),
            &lookup,
        )?;
        let api_key =
            common_utils::resolve_api_key(api_key.as_deref().and_then(resolve).as_deref(), &lookup);
        let model_info = model_info.as_deref();
        let AzureChatBody {
            deployment: url_deployment,
            body,
        } = AzureChatRequest {
            deployment: &deployment,
            base_model: configured("base_model"),
            api_version: &api_version,
            drop_params,
            model_info,
        }
        .transform(body)?;
        let url = common_utils::complete_url(&api_base, &url_deployment, &api_version)?;

        let credential = self
            .azure_credential(api_key.clone(), &params, &lookup)
            .await?;
        let environment = common_utils::environment(
            string_headers(extra_headers)?,
            credential,
            common_utils::is_v1_api_version(&api_version),
        );
        let authenticated = resolve_auth(&self.auth, environment, &lookup).await?;
        Ok(PreparedCall {
            model: url_deployment,
            provider: <&str>::from(LlmProviders::Azure),
            url,
            headers: authenticated.headers,
            body,
            api_key,
            adapter: None,
        })
    }

    /// The SDK sends a configured `azure_ad_token` ahead of a key. Without either, LiteLLM
    /// gets a token from Entra ID: client credentials, an OIDC exchange, or managed identity
    /// when `azure_credential` or `enable_azure_ad_token_refresh` asks for it.
    async fn azure_credential(
        &self,
        api_key: Option<String>,
        params: &Map<String, Value>,
        lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<AzureCredential, Error> {
        let configured_token = params
            .get("azure_ad_token")
            .and_then(Value::as_str)
            .is_some_and(|token| !token.is_empty());
        if let Some(key) = &api_key
            && !configured_token
        {
            return Ok(AzureCredential::ApiKey(litellm_auth::SecretValue::new(
                key.clone(),
            )));
        }
        let inputs = AzureAuthInputs::from_sourced_optional_params(params, &BTreeMap::new())?;
        let token = self.auth.azure.get_azure_ad_token(&inputs, lookup).await?;
        match (token, api_key) {
            (Some(token), _) => Ok(AzureCredential::AdToken(
                token.into_value().secret().clone(),
            )),
            (None, Some(key)) => Ok(AzureCredential::ApiKey(litellm_auth::SecretValue::new(key))),
            (None, None) => Err(Error::from(LlmError::Auth(
                litellm_auth::Error::MissingApiKey {
                    provider: "Azure",
                    environment_variable: AZURE_API_KEY_ENV,
                },
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use bytes::Bytes;
    use futures_util::TryStreamExt;
    use litellm_host::call::CallOutput;
    use litellm_http::transport;
    use litellm_inference_testing::RecordingSecrets;
    use rstest::rstest;
    use serde_json::json;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_partial_json, header, method, path, query_param},
    };

    use super::*;
    use crate::test_support::{Recording, route, run};

    const COMPLETION: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-4o-2024-08-06","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}"#;
    const STREAM: &str = "data: {\"id\":\"c\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\ndata: {\"id\":\"c\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\ndata: [DONE]\n\n";
    const DEPLOYMENT_PATH: &str = "/openai/deployments/gpt-4o/chat/completions";

    fn call(api_base: &str, body: Value) -> OpenAiChatCall {
        let Value::Object(body) = body else {
            panic!("the body is an object");
        };
        OpenAiChatCall {
            model: "azure/gpt-4o".into(),
            body,
            api_key: Some("sk-azure".into()),
            api_base: Some(format!("{api_base}/")),
            litellm_params: Map::from_iter([("api_version".into(), json!("2024-10-21"))]),
            timeout: Some(Duration::from_secs(5)),
            ..OpenAiChatCall::default()
        }
    }

    fn hi() -> Value {
        json!({"model": "public", "messages": [{"role": "user", "content": "hi"}]})
    }

    fn with_secrets(route: ChatCompletionsRoute, values: &[(&str, &str)]) -> ChatCompletionsRoute {
        ChatCompletionsRoute {
            secrets: Arc::new(RecordingSecrets::new(values.iter().copied())),
            ..route
        }
    }

    async fn complete(output: crate::openai::OpenAiChatOutput) -> Bytes {
        match output {
            CallOutput::Complete(body) => body,
            CallOutput::Stream { .. } => panic!("expected a complete body"),
        }
    }

    #[rstest]
    #[tokio::test]
    async fn a_completion_goes_to_the_deployment_with_its_key(route: ChatCompletionsRoute) {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(DEPLOYMENT_PATH))
            .and(query_param("api-version", "2024-10-21"))
            .and(header("api-key", "sk-azure"))
            .and(body_partial_json(json!({"model": "gpt-4o"})))
            .respond_with(ResponseTemplate::new(200).set_body_raw(COMPLETION, "application/json"))
            .expect(1)
            .mount(&upstream)
            .await;
        let interceptors = Recording::default();

        let output = run(&route, call(&upstream.uri(), hi()), &interceptors)
            .await
            .unwrap();

        assert_eq!(complete(output).await, COMPLETION.as_bytes());
        let requests = upstream.received_requests().await.unwrap();
        assert!(!requests[0].headers.contains_key("authorization"));
        let contexts = interceptors.contexts.lock().unwrap();
        assert_eq!(
            (
                contexts[0].model.as_str(),
                contexts[0].custom_llm_provider.as_str()
            ),
            ("gpt-4o", "azure")
        );
        assert_eq!(*interceptors.raw.lock().unwrap(), [COMPLETION]);
        assert_eq!(
            interceptors.results.lock().unwrap()[0].provider.provider,
            "azure"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_stream_relays_every_byte(route: ChatCompletionsRoute) {
        let upstream = MockServer::start().await;
        Mock::given(path(DEPLOYMENT_PATH))
            .and(body_partial_json(
                json!({"stream": true, "stream_options": {"include_usage": true}}),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_raw(STREAM, "text/event-stream"))
            .expect(1)
            .mount(&upstream)
            .await;
        let mut body = hi();
        body["stream"] = json!(true);
        body["stream_options"] = json!({"include_usage": true});

        let output = run(&route, call(&upstream.uri(), body), &Recording::default())
            .await
            .unwrap();

        let CallOutput::Stream { chunks, .. } = output else {
            panic!("a streaming call streams");
        };
        let relayed = chunks.try_collect::<Vec<_>>().await.unwrap().concat();
        assert_eq!(relayed, STREAM.as_bytes());
    }

    #[rstest]
    #[tokio::test]
    async fn the_v1_surface_takes_the_key_as_a_bearer(route: ChatCompletionsRoute) {
        let upstream = MockServer::start().await;
        Mock::given(path("/openai/v1/chat/completions"))
            .and(header("authorization", "Bearer sk-azure"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(COMPLETION, "application/json"))
            .expect(1)
            .mount(&upstream)
            .await;
        let mut v1 = call(&upstream.uri(), hi());
        v1.litellm_params
            .insert("api_version".into(), json!("latest"));

        run(&route, v1, &Recording::default()).await.unwrap();

        let requests = upstream.received_requests().await.unwrap();
        assert_eq!(requests[0].url.query(), None);
        assert!(!requests[0].headers.contains_key("api-key"));
    }

    #[rstest]
    #[tokio::test]
    async fn base_key_and_version_fall_back_to_the_environment(route: ChatCompletionsRoute) {
        let upstream = MockServer::start().await;
        Mock::given(path(DEPLOYMENT_PATH))
            .and(query_param("api-version", "2024-06-01"))
            .and(header("api-key", "sk-env"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(COMPLETION, "application/json"))
            .expect(1)
            .mount(&upstream)
            .await;
        let base = upstream.uri();
        let route = with_secrets(
            route,
            &[
                ("AZURE_API_BASE", &base),
                ("AZURE_API_KEY", "sk-env"),
                ("AZURE_API_VERSION", "2024-06-01"),
            ],
        );
        let bare = OpenAiChatCall {
            api_key: None,
            api_base: None,
            litellm_params: Map::new(),
            ..call(&base, hi())
        };

        run(&route, bare, &Recording::default()).await.unwrap();
    }

    #[rstest]
    #[tokio::test]
    async fn the_default_api_version_is_pythons(route: ChatCompletionsRoute) {
        let upstream = MockServer::start().await;
        Mock::given(query_param(
            "api-version",
            common_utils::AZURE_DEFAULT_API_VERSION,
        ))
        .respond_with(ResponseTemplate::new(200).set_body_raw(COMPLETION, "application/json"))
        .expect(1)
        .mount(&upstream)
        .await;
        let mut unversioned = call(&upstream.uri(), hi());
        unversioned.litellm_params.clear();

        run(&route, unversioned, &Recording::default())
            .await
            .unwrap();
    }

    #[rstest]
    #[tokio::test]
    async fn environment_references_resolve(route: ChatCompletionsRoute) {
        let upstream = MockServer::start().await;
        Mock::given(query_param("api-version", "2025-01-01-preview"))
            .and(header("api-key", "sk-referenced"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(COMPLETION, "application/json"))
            .expect(1)
            .mount(&upstream)
            .await;
        let route = with_secrets(
            route,
            &[
                ("MY_AZURE_KEY", "sk-referenced"),
                ("MY_AZURE_VERSION", "2025-01-01-preview"),
            ],
        );
        let mut referenced = call(&upstream.uri(), hi());
        referenced.api_key = Some("os.environ/MY_AZURE_KEY".into());
        referenced
            .litellm_params
            .insert("api_version".into(), json!("os.environ/MY_AZURE_VERSION"));

        run(&route, referenced, &Recording::default())
            .await
            .unwrap();
    }

    #[rstest]
    #[case::configured_token_outranks_the_key(Some("sk-azure"), &[("azure_ad_token", "configured-token")], &[])]
    #[case::environment_token(None, &[], &[("AZURE_AD_TOKEN", "configured-token")])]
    #[tokio::test]
    async fn an_ad_token_goes_as_a_bearer(
        route: ChatCompletionsRoute,
        #[case] api_key: Option<&str>,
        #[case] params: &[(&str, &str)],
        #[case] environment: &[(&str, &str)],
    ) {
        let upstream = MockServer::start().await;
        Mock::given(header("authorization", "Bearer configured-token"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(COMPLETION, "application/json"))
            .expect(1)
            .mount(&upstream)
            .await;
        let route = with_secrets(route, environment);
        let mut tokened = call(&upstream.uri(), hi());
        tokened.api_key = api_key.map(str::to_owned);
        tokened.litellm_params.extend(
            params
                .iter()
                .map(|(name, value)| ((*name).to_owned(), json!(value))),
        );

        run(&route, tokened, &Recording::default()).await.unwrap();

        let requests = upstream.received_requests().await.unwrap();
        assert!(!requests[0].headers.contains_key("api-key"));
    }

    #[rstest]
    #[tokio::test]
    async fn a_missing_credential_fails_before_any_request(route: ChatCompletionsRoute) {
        let mut keyless = call("http://127.0.0.1:1", hi());
        keyless.api_key = None;
        assert_eq!(
            run(&route, keyless, &Recording::default()).await.err(),
            Some(Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "Azure",
                environment_variable: "AZURE_API_KEY",
            }))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn an_unsupported_param_fails_before_any_request(route: ChatCompletionsRoute) {
        let mut body = hi();
        body["reasoning_effort"] = json!("high");
        assert!(matches!(
            run(
                &route,
                call("http://127.0.0.1:1", body),
                &Recording::default()
            )
            .await,
            Err(Error::InvalidRequest(_))
        ));
    }

    #[rstest]
    #[tokio::test]
    async fn a_content_filter_error_keeps_azures_body(route: ChatCompletionsRoute) {
        let upstream = MockServer::start().await;
        let body = r#"{"error":{"message":"The response was filtered","type":null,"param":"prompt","code":"content_filter","status":400,"innererror":{"code":"ResponsibleAIPolicyViolation","content_filter_result":{"hate":{"filtered":true,"severity":"high"}}}}}"#;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400).set_body_raw(body, "application/json"))
            .mount(&upstream)
            .await;

        let error = run(&route, call(&upstream.uri(), hi()), &Recording::default())
            .await
            .err()
            .expect("the upstream failure fails the call");

        assert_eq!(
            error,
            Error::Transport(transport::Error::Http {
                status: 400,
                body: body.into()
            })
        );
    }
}
