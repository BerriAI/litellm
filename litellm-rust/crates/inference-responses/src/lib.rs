use litellm_host::observation::ObservationSender;
pub use litellm_inference::RouteError as Error;

pub mod route;
pub mod types;
pub mod websocket;

mod handler;

use std::sync::Arc;

use litellm_auth::AuthServices;
use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_host::interceptors::Interceptors;
use litellm_inference::provider::{ResolvedProvider, resolve_llm_provider};
use litellm_llms::{
    base_llm::responses::transformation::BaseResponsesApiConfig,
    openai::responses::transformation::OpenAiResponsesApiConfig,
};
use litellm_secrets::source::SecretSource;
use types::{ResponsesCall, ResponsesOutput};

#[derive(Clone)]
pub struct ResponsesRoute {
    http: litellm_http::Client,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn SecretSource>,
    cache: Option<litellm_cache_response::ScopedCache>,
}

impl ResponsesRoute {
    pub fn new(
        http: litellm_http::Client,
        auth: Arc<AuthServices>,
        secrets: Arc<dyn SecretSource>,
    ) -> Self {
        Self {
            http,
            auth,
            secrets,
            cache: None,
        }
    }

    pub fn with_cache(self, cache: litellm_cache_response::ScopedCache) -> Self {
        Self {
            cache: Some(cache),
            ..self
        }
    }

    pub async fn execute(
        &self,
        call: ResponsesCall,
        interceptors: &impl Interceptors<Error>,
        options: impl Into<litellm_inference::CallOptions>,
    ) -> Result<ResponsesOutput, Error> {
        let litellm_inference::CallOptions {
            cache: cache_options,
            observers,
        } = options.into();
        litellm_host::lifecycle::observe_call(
            observers.clone(),
            self.run(call, cache_options, interceptors, observers.as_ref()),
        )
        .await
    }

    #[tracing::instrument(name = "litellm.route", skip_all, fields(
        route = "responses",
        model = %call.model,
        provider,
        resolved_model,
        stream = call.optional_params.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false),
        outcome
    ))]
    async fn run(
        &self,
        call: ResponsesCall,
        cache_options: Option<litellm_cache_response::CachePolicy>,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<ResponsesOutput, Error> {
        litellm_inference::diagnostic::call(async {
            let model = call.model.clone();
            let custom_llm_provider = call.custom_llm_provider.clone();
            let (provider, config) =
                resolve_provider_config(&model, custom_llm_provider.as_deref())?;
            litellm_inference::diagnostic::provider(
                provider.model,
                <&'static str>::from(provider.provider),
            );
            let secrets = self
                .secrets
                .resolve(config.secret_names(call.api_key.as_deref(), call.api_base.as_deref()))
                .await?;
            let execute: futures_util::future::BoxFuture<'_, Result<ResponsesOutput, Error>> =
                Box::pin(handler::execute(
                    &self.http,
                    &self.auth,
                    config,
                    provider,
                    call,
                    secrets,
                    self.cache.clone(),
                    cache_options,
                    interceptors,
                    observers,
                ));
            execute.await
        })
        .await
    }
}

fn resolve_provider_config<'a>(
    model: &'a str,
    custom_llm_provider: Option<&'a str>,
) -> Result<(ResolvedProvider<'a>, &'static dyn BaseResponsesApiConfig), Error> {
    if custom_llm_provider.is_some_and(|provider| provider != "openai") {
        return Err(Error::Unsupported("native HTTP responses provider"));
    }
    let resolved = resolve_llm_provider(model, custom_llm_provider, "responses");
    let provider = match resolved {
        Ok(_) if custom_llm_provider.is_none() && !model.starts_with("openai/") => {
            ResolvedProvider {
                model: model.strip_prefix("openai/").unwrap_or(model),
                provider: LlmProviders::Openai,
            }
        }
        Ok(provider) => provider,
        Err(_) if custom_llm_provider.is_none() => ResolvedProvider {
            model: model.strip_prefix("openai/").unwrap_or(model),
            provider: LlmProviders::Openai,
        },
        Err(error) => return Err(error),
    };
    if provider.provider != LlmProviders::Openai {
        return Err(Error::Unsupported("native HTTP responses provider"));
    }
    if provider.model.is_empty() || provider.model.contains('/') {
        return Err(Error::InvalidProvider(model.into()));
    }
    Ok((provider, &OpenAiResponsesApiConfig))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[derive(Debug)]
    enum ExpectedResolution {
        OpenAi(&'static str),
        InvalidProvider(&'static str),
        Unsupported,
    }

    #[rstest]
    #[case::unprefixed("test", None, ExpectedResolution::OpenAi("test"))]
    #[case::openai_prefix("openai/gpt-4o", None, ExpectedResolution::OpenAi("gpt-4o"))]
    #[case::explicit_openai("gpt-4o", Some("openai"), ExpectedResolution::OpenAi("gpt-4o"))]
    #[case::other_prefix(
        "anthropic/x",
        None,
        ExpectedResolution::InvalidProvider("anthropic/x")
    )]
    #[case::other_explicit_provider("x", Some("anthropic"), ExpectedResolution::Unsupported)]
    #[case::nested_openai_model(
        "openai/a/b",
        None,
        ExpectedResolution::InvalidProvider("openai/a/b")
    )]
    #[case::empty_openai_model("openai/", None, ExpectedResolution::InvalidProvider("openai/"))]
    fn provider_resolution_preserves_the_previous_responses_contract(
        #[case] model: &str,
        #[case] custom_llm_provider: Option<&str>,
        #[case] expected: ExpectedResolution,
    ) {
        match (
            resolve_provider_config(model, custom_llm_provider),
            expected,
        ) {
            (Ok((provider, _)), ExpectedResolution::OpenAi(expected_model)) => {
                assert_eq!(provider.provider, LlmProviders::Openai);
                assert_eq!(provider.model, expected_model);
            }
            (
                Err(Error::InvalidProvider(actual)),
                ExpectedResolution::InvalidProvider(expected),
            ) => {
                assert_eq!(actual, expected);
            }
            (Err(Error::Unsupported(actual)), ExpectedResolution::Unsupported) => {
                assert_eq!(actual, "native HTTP responses provider");
            }
            (_, expected) => panic!("unexpected resolution, expected {expected:?}"),
        }
    }
}
