use std::{collections::BTreeMap, time::Duration};

use litellm_auth::azure::{AzureAuthInputs, SECRET_NAMES as AZURE_AUTH_SECRET_NAMES};
use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_host::interceptors::{
    Cost, ExecutionFacts, Interceptors, ProviderIdentity, RequestContext, ResultSource, WireRequest,
};
use litellm_http::{
    outbound::OutboundRequest,
    request::{string_headers, truncate_error_body},
    transport,
};
use litellm_inference::{context::CallContext, provider::resolve_llm_provider};
use litellm_llms::{
    ErrorDetail,
    azure::{
        common_utils::{self as azure_utils, AZURE_API_KEY_ENV, AzureCredential},
        decisions::transformation as azure,
    },
    base_llm::{
        auth::{Authenticated, ValidatedEnvironment, resolve_auth},
        decisions::{Target, caller_response, provider_request},
        model_info::{field, supports},
    },
    openai::decisions::transformation as openai,
};
use litellm_llms_types::formats::decisions::DecisionsResponse;
use serde_json::{Map, Value};

use crate::{DecisionsCall, DecisionsRoute, Error};

const TIMEOUT: Duration = Duration::from_secs(600);
const ENV_REFERENCE: &str = "os.environ/";
const HEADER_CONTEXT: &str = "decisions";
const AZURE_REFERABLE_PARAMS: &[&str] = &[
    "azure_ad_token",
    "tenant_id",
    "client_id",
    "client_secret",
    "azure_scope",
    "azure_authority_host",
    "azure_credential",
    "azure_federated_token_file",
];
const UNPORTED_PROVIDERS: &[&str] = &[
    "azure_ai",
    "bespoke",
    "cloudflare",
    "databricks",
    "empiriolabs",
    "eu_jev",
    "fireworks_ai",
    "hosted_vllm",
    "laya",
    "lm_studio",
    "nace",
    "ollama",
    "openrouter",
    "parallel_ai",
    "perplexity",
    "strands_decider",
    "typesafe",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Provider {
    OpenAi,
    Azure,
}

impl Provider {
    fn name(self) -> &'static str {
        match self {
            Self::OpenAi => <&str>::from(LlmProviders::Openai),
            Self::Azure => <&str>::from(LlmProviders::Azure),
        }
    }
}

pub fn serves(model: &str, custom_llm_provider: Option<&str>) -> bool {
    provider(model, custom_llm_provider).is_ok()
}

fn provider<'a>(
    model: &'a str,
    custom_llm_provider: Option<&'a str>,
) -> Result<(Provider, &'a str), Error> {
    let named = custom_llm_provider.or_else(|| model.split_once('/').map(|(prefix, _)| prefix));
    if named.is_some_and(|name| UNPORTED_PROVIDERS.contains(&name)) {
        return Err(Error::Unsupported("Decisions through this provider"));
    }
    let resolved = resolve_llm_provider(model, custom_llm_provider, "decisions")?;
    match resolved.provider {
        LlmProviders::Openai => Ok((Provider::OpenAi, resolved.model)),
        LlmProviders::Azure => Ok((Provider::Azure, resolved.model)),
        other if UNPORTED_PROVIDERS.contains(&<&str>::from(other)) => {
            Err(Error::Unsupported("Decisions through this provider"))
        }
        other => Err(Error::InvalidProvider(format!(
            "Unknown Decisions provider '{}'. Supported providers: openai, azure",
            <&str>::from(other)
        ))),
    }
}

struct Connection {
    url: String,
    headers: Vec<(String, String)>,
    api_key: Option<String>,
}

pub(crate) async fn execute(
    route: &DecisionsRoute,
    call: DecisionsCall,
    context: CallContext<'_, impl Interceptors<Error>>,
) -> Result<DecisionsResponse, Error> {
    let (provider, model) = provider(&call.model, call.custom_llm_provider.as_deref())?;
    let provider_name = provider.name();
    litellm_inference::diagnostic::provider(model, provider_name);
    let prepared = provider_request(
        &call.request,
        Target {
            provider: provider_name,
            model,
            requested_model: &call.model,
            supports_box_questions: supports_box_questions(&call, provider_name, model),
            drop_params: call.drop_params,
        },
    )?;
    let headers = string_headers(HEADER_CONTEXT, call.extra_headers.clone())?;
    let connection = match provider {
        Provider::OpenAi => connect_openai(route, &call, headers).await?,
        Provider::Azure => connect_azure(route, &call, headers).await?,
    };
    let body = serde_json::to_value(&prepared.body).map_err(|error| {
        Error::InvalidRequest(ErrorDetail::invalid("Decisions provider request", error))
    })?;
    let wire = context
        .interceptors
        .before_provider_request(
            WireRequest {
                url: connection.url,
                headers: connection.headers,
                body,
            },
            RequestContext {
                model: model.to_owned(),
                custom_llm_provider: provider_name.to_owned(),
                optional_params: Value::Object(Map::new()),
                secret_fields: Vec::new(),
                api_key: connection.api_key.map(litellm_auth::SecretValue::new),
            },
        )
        .await?;
    let payload = send(route, wire, call.timeout.unwrap_or(TIMEOUT)).await?;
    context
        .response_received(&String::from_utf8_lossy(&payload))
        .await?;
    let response = caller_response(
        &payload,
        &call.request,
        &prepared.sent,
        &call.model,
        provider_name,
    )?;
    context
        .result_ready(ExecutionFacts {
            provider: ProviderIdentity {
                model: model.to_owned(),
                provider: provider_name.to_owned(),
            },
            source: ResultSource::Provider,
            cost: Cost::Deferred,
        })
        .await?;
    Ok(response)
}

fn supports_box_questions(call: &DecisionsCall, provider: &str, model: &str) -> bool {
    call.supports_box_questions.unwrap_or_else(|| {
        let info = call.model_info.as_deref();
        [model.to_owned(), format!("{provider}/{model}")]
            .iter()
            .any(|name| {
                field(info, name, "mode")
                    .is_some_and(|mode| matches!(mode.as_str(), Some("decisions" | "evaluation")))
                    && supports(info, name, "supports_box_questions")
            })
    })
}

async fn send(
    route: &DecisionsRoute,
    wire: WireRequest,
    timeout: Duration,
) -> Result<bytes::Bytes, Error> {
    let request = OutboundRequest::json(wire.url, wire.headers, &wire.body, Some(timeout))?;
    let response = litellm_inference::outbound::send(request, &route.http)
        .await
        .map_err(|error| Error::Transport(transport::Error::from_reqwest_before_dispatch(error)))?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.map_err(network)?;
        return Err(Error::Transport(transport::Error::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        }));
    }
    response.bytes().await.map_err(network)
}

fn network(error: reqwest::Error) -> Error {
    Error::Transport(transport::Error::from(error))
}

fn references<'a>(values: impl Iterator<Item = &'a str>) -> Vec<&'a str> {
    values
        .filter_map(|value| value.strip_prefix(ENV_REFERENCE))
        .collect()
}

fn resolved(value: &str, lookup: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    match value.strip_prefix(ENV_REFERENCE) {
        Some(name) => lookup(name),
        None => Some(value.to_owned()),
    }
}

async fn authenticate(
    route: &DecisionsRoute,
    environment: ValidatedEnvironment,
    lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Vec<(String, String)>, Error> {
    let Authenticated { headers, .. } = resolve_auth(&route.auth, environment, lookup).await?;
    Ok(headers)
}

async fn connect_openai(
    route: &DecisionsRoute,
    call: &DecisionsCall,
    headers: Vec<(String, String)>,
) -> Result<Connection, Error> {
    let connection = [call.api_key.as_deref(), call.api_base.as_deref()];
    let names: Vec<&str> = openai::secret_names(connection[0], connection[1])
        .iter()
        .copied()
        .chain(references(connection.into_iter().flatten()))
        .collect();
    let secrets = route.secrets.resolve(&names).await?;
    let lookup = |name: &str| secrets.get(name);
    let api_key = call
        .api_key
        .as_deref()
        .and_then(|key| resolved(key, &lookup));
    let api_base = call
        .api_base
        .as_deref()
        .and_then(|base| resolved(base, &lookup));
    let environment = openai::environment(headers, api_key.as_deref(), &lookup)?;
    Ok(Connection {
        url: openai::complete_url(api_base.as_deref(), &lookup),
        headers: authenticate(route, environment, &lookup).await?,
        api_key,
    })
}

async fn connect_azure(
    route: &DecisionsRoute,
    call: &DecisionsCall,
    headers: Vec<(String, String)>,
) -> Result<Connection, Error> {
    let params = &call.litellm_params;
    let names: Vec<&str> = azure_utils::SECRET_NAMES
        .iter()
        .chain(AZURE_AUTH_SECRET_NAMES)
        .copied()
        .chain(references(
            [call.api_key.as_deref(), call.api_base.as_deref()]
                .into_iter()
                .flatten()
                .chain(
                    AZURE_REFERABLE_PARAMS
                        .iter()
                        .filter_map(|name| params.get(*name).and_then(Value::as_str)),
                ),
        ))
        .collect();
    let secrets = route.secrets.resolve(&names).await?;
    let lookup = |name: &str| secrets.get(name);
    let params: Map<String, Value> = params
        .iter()
        .map(|(name, value)| match value {
            Value::String(text) if AZURE_REFERABLE_PARAMS.contains(&name.as_str()) => (
                name.clone(),
                resolved(text, &lookup).map_or(Value::Null, Value::String),
            ),
            other => (name.clone(), other.clone()),
        })
        .collect();
    let api_base = azure_utils::resolve_api_base(
        call.api_base
            .as_deref()
            .and_then(|base| resolved(base, &lookup))
            .as_deref(),
        &lookup,
    )?;
    let api_key = azure_utils::resolve_api_key(
        call.api_key
            .as_deref()
            .and_then(|key| resolved(key, &lookup))
            .as_deref(),
        &lookup,
    );
    let credential = match &api_key {
        Some(key) => AzureCredential::ApiKey(litellm_auth::SecretValue::new(key.clone())),
        None => {
            let inputs = AzureAuthInputs::from_sourced_optional_params(&params, &BTreeMap::new())?;
            let token = route
                .auth
                .azure
                .get_azure_ad_token(&inputs, &lookup)
                .await?
                .ok_or(litellm_auth::Error::MissingApiKey {
                    provider: "Azure",
                    environment_variable: AZURE_API_KEY_ENV,
                })?;
            AzureCredential::AdToken(token.into_value().secret().clone())
        }
    };
    Ok(Connection {
        url: azure::complete_url(&api_base)?,
        headers: authenticate(route, azure::environment(headers, credential), &lookup).await?,
        api_key,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use litellm_host::interceptors::RawResponse;
    use litellm_inference::{CallOptions, context::CallContext};
    use litellm_inference_testing::RecordingSecrets;
    use litellm_llms::base_llm::decisions::parse_request;
    use rstest::{fixture, rstest};
    use serde_json::json;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_json, header, method, path},
    };

    use super::*;

    const ANSWER: &str = r#"{"model":"gpt-6-luna-2026","answers":[{"type":"predicate","name":"refund","probability":0.25}],"usage":{"input_tokens":2000,"output_tokens":20,"total_tokens":2020}}"#;

    #[derive(Default)]
    struct Recording {
        contexts: Mutex<Vec<RequestContext>>,
        raw: Mutex<Vec<String>>,
    }

    impl Interceptors<Error> for Recording {
        async fn before_provider_request(
            &self,
            wire: WireRequest,
            context: RequestContext,
        ) -> Result<WireRequest, Error> {
            self.contexts.lock().unwrap().push(context);
            Ok(wire)
        }

        async fn after_provider_response(&self, raw: RawResponse) -> Result<(), Error> {
            self.raw.lock().unwrap().push(raw.body);
            Ok(())
        }
    }

    fn route_with(secrets: &[(&str, &str)]) -> DecisionsRoute {
        DecisionsRoute::new(
            litellm_http::Client::plain_for_test(),
            Arc::new(litellm_auth::AuthServices::default()),
            Arc::new(RecordingSecrets::new(secrets.iter().copied())),
        )
    }

    #[fixture]
    fn route() -> DecisionsRoute {
        route_with(&[])
    }

    fn call(model: &str, api_base: &str) -> DecisionsCall {
        DecisionsCall {
            request: Arc::new(
                parse_request(&json!({
                    "model": "public",
                    "input": "Customer wrote: refund order 4411",
                    "questions": [{"type": "predicate", "name": "refund", "instructions": "Refund?"}],
                    "metadata": {"ignored": true},
                }))
                .unwrap(),
            ),
            model: model.into(),
            api_key: Some("sk-deployment".into()),
            api_base: Some(api_base.into()),
            custom_llm_provider: None,
            litellm_params: Map::new(),
            extra_headers: None,
            timeout: Some(Duration::from_secs(5)),
            drop_params: false,
            supports_box_questions: None,
            model_info: None,
        }
    }

    async fn run(
        route: &DecisionsRoute,
        call: DecisionsCall,
        interceptors: &Recording,
    ) -> Result<DecisionsResponse, Error> {
        execute(
            route,
            call,
            CallContext::new(interceptors, CallOptions::default()),
        )
        .await
    }

    fn sent(model: &str) -> Value {
        json!({
            "model": model,
            "input": "Customer wrote: refund order 4411",
            "questions": [{"type": "predicate", "name": "refund", "instructions": "Refund?"}],
        })
    }

    #[rstest]
    #[tokio::test]
    async fn openai_gets_the_rebuilt_body_with_the_deployment_key(route: DecisionsRoute) {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/d/0/v1/decisions"))
            .and(header("authorization", "Bearer sk-deployment"))
            .and(body_json(sent("gpt-6-luna")))
            .respond_with(ResponseTemplate::new(200).set_body_raw(ANSWER, "application/json"))
            .expect(1)
            .mount(&upstream)
            .await;
        let interceptors = Recording::default();

        let response = run(
            &route,
            call("openai/gpt-6-luna", &format!("{}/d/0/v1", upstream.uri())),
            &interceptors,
        )
        .await
        .unwrap();

        assert_eq!(response.model, "gpt-6-luna-2026");
        assert_eq!(response.usage.total_tokens, 2020);
        assert_eq!(*interceptors.raw.lock().unwrap(), [ANSWER]);
        let contexts = interceptors.contexts.lock().unwrap();
        assert_eq!(
            (
                contexts[0].model.as_str(),
                contexts[0].custom_llm_provider.as_str()
            ),
            ("gpt-6-luna", "openai")
        );
    }

    #[rstest]
    #[tokio::test]
    async fn azure_sends_its_key_in_api_key_to_the_v1_route(route: DecisionsRoute) {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/v1/decisions"))
            .and(header("api-key", "sk-deployment"))
            .and(body_json(sent("luna-deployment")))
            .respond_with(ResponseTemplate::new(200).set_body_raw(ANSWER, "application/json"))
            .expect(1)
            .mount(&upstream)
            .await;

        let response = run(
            &route,
            call("azure/luna-deployment", &upstream.uri()),
            &Recording::default(),
        )
        .await
        .unwrap();

        assert_eq!(response.answers.len(), 1);
        let requests = upstream.received_requests().await.unwrap();
        assert!(!requests[0].headers.contains_key("authorization"));
    }

    #[rstest]
    #[tokio::test]
    async fn an_environment_reference_resolves_through_the_secret_source() {
        let upstream = MockServer::start().await;
        Mock::given(header("authorization", "Bearer sk-from-env"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(ANSWER, "application/json"))
            .expect(1)
            .mount(&upstream)
            .await;
        let route = route_with(&[("LUNA_KEY", "sk-from-env")]);

        run(
            &route,
            DecisionsCall {
                api_key: Some("os.environ/LUNA_KEY".into()),
                ..call("openai/gpt-6-luna", &upstream.uri())
            },
            &Recording::default(),
        )
        .await
        .unwrap();
    }

    #[rstest]
    #[tokio::test]
    async fn an_upstream_error_keeps_its_status_and_body(route: DecisionsRoute) {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(429).set_body_string(r#"{"error":{"message":"slow down"}}"#),
            )
            .mount(&upstream)
            .await;
        let interceptors = Recording::default();

        let error = run(
            &route,
            call("openai/gpt-6-luna", &upstream.uri()),
            &interceptors,
        )
        .await
        .unwrap_err();

        assert!(matches!(
            error,
            Error::Transport(transport::Error::Http { status: 429, ref body }) if body.contains("slow down")
        ));
        assert!(interceptors.raw.lock().unwrap().is_empty());
    }

    #[rstest]
    #[tokio::test]
    async fn an_answer_python_rejects_is_an_invalid_response(route: DecisionsRoute) {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(r#"{"answers":[]}"#, "application/json"),
            )
            .mount(&upstream)
            .await;

        let error = run(
            &route,
            call("openai/gpt-6-luna", &upstream.uri()),
            &Recording::default(),
        )
        .await
        .unwrap_err();

        assert!(matches!(
            error,
            Error::Decisions(litellm_llms::base_llm::decisions::Error::InvalidResponse {
                provider: "openai",
                ..
            })
        ));
        assert!(!error.is_request());
    }

    #[rstest]
    #[case::openai("openai/gpt-6-luna", None, true)]
    #[case::azure("azure/luna", None, true)]
    #[case::custom_provider("luna", Some("azure"), true)]
    #[case::unported("ollama/llama", None, false)]
    #[case::anthropic("anthropic/claude", None, false)]
    fn only_the_openai_wire_providers_are_served(
        #[case] model: &str,
        #[case] custom_llm_provider: Option<&str>,
        #[case] served: bool,
    ) {
        assert_eq!(serves(model, custom_llm_provider), served);
    }

    #[rstest]
    #[case::unported("ollama/llama", true)]
    #[case::not_a_decisions_provider("anthropic/claude", false)]
    fn an_unported_provider_is_not_implemented_rather_than_unknown(
        #[case] model: &str,
        #[case] not_implemented: bool,
    ) {
        let error = provider(model, None).unwrap_err();

        assert_eq!(matches!(error, Error::Unsupported(_)), not_implemented);
        assert!(error.is_request());
    }
}
