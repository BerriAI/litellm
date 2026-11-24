//! Builds the request a provider receives for one Messages call, in Python's handler order.
//!
//! The config's four parts:
//! - Endpoint: the endpoint a call selects, its URL and the header edits that URL needs
//! - Auth: the secrets a call reads, its credential and how the request is signed
//! - Dialect: every body and header edit, and what the call logs
//! - Reply: the response, stream, cost and retries, read in `handler.rs`
//!
//! The request is built in this order:
//! 1. Dialect: the effective request, the caller's params shaped, trimmed and gated
//! 2. Auth: deferred secrets read, the environment resolved and the scheme prepared
//! 3. Endpoint: the endpoint selected from the login and the model map
//! 4. Dialect: the body transformed
//! 5. Endpoint: the URL
//! 6. Dialect: the body sent to that URL, then the headers and finalization that build the wire
//!    request
//! 7. Auth: the credential applied and the request signed
//!
//! The retry without native output rebuilds its body through steps 4 and 6, for the same URL.

use std::time::Duration;

use litellm_auth::AuthServices;
use litellm_core_utils::{
    dot_notation_indexing::delete_nested_value,
    get_provider_specific_headers::get_provider_specific_headers,
};
use litellm_env::Lookup;
use litellm_host::interceptors::{Interceptors, ProviderIdentity, RequestContext, WireRequest};
use litellm_inference::context::CallContext;
use litellm_llms::anthropic::messages::transformation::is_claude_model;
use litellm_llms::base_llm::{
    auth::{
        AzureAdTokenHosts, Headers, LoggedIn, ValidatedEnvironment, prepare_auth, resolve_auth,
    },
    messages::{
        context::MessagesTransformContext,
        tools::ToolCustomField,
        transformation::{BaseMessagesConfig, FederatedEnvironment},
    },
};
use litellm_llms_types::{
    formats::messages::{MessagesOptionalParams, MessagesRequest},
    headers::ProviderSpecificHeaders,
};
use litellm_router_types::LitellmParams;
use litellm_secret_value::SecretValue;
use litellm_secrets::source::{SecretSource, Secrets};
use serde_json::{Map, Value};

use super::{
    Error, MessagesCall, MessagesShaping, WorkloadIdentity,
    common_utils::{MessagesProvider, merge_messages_header_sources, string_headers},
    types::invalid_request,
};

/// A provider request ready to send, with what sending, retrying and decoding it need.
pub(super) struct ProviderCall {
    pub(super) identity: ProviderIdentity,
    pub(super) wire: WireRequest,
    pub(super) provider: MessagesProvider,
    pub(super) signer: Option<litellm_auth_aws::SigV4Signer>,
    pub(super) timeout: Option<Duration>,
    pub(super) stream: bool,
    pub(super) whole_sse_frames: bool,
    pub(super) joined_reads: bool,
    pub(super) detached_drain: Option<litellm_host::protocol::DetachedDrain>,
    pub(super) transform_context: MessagesTransformContext,
    pub(super) native_output_fallback: Option<NativeOutputFallback>,
    pub(super) ascii_response_headers: bool,
    pub(super) unencodable_request_header: Option<Vec<u8>>,
    pub(super) ascii_request_headers: bool,
    pub(super) connect_retry: bool,
}

impl ProviderCall {
    pub(super) fn cacheable(&self) -> bool {
        self.signer.is_none()
    }
}

/// What a provider call may authenticate with beyond its config: the shared token caches, a
/// workload identity federation host and the Entra ID token sources of the caller and host.
pub(super) struct CallerCredentials<'a> {
    pub(super) auth: &'a AuthServices,
    pub(super) workload_identity: Option<&'a dyn WorkloadIdentity>,
    pub(super) azure_ad_token_hosts: AzureAdTokenHosts,
    pub(super) deferred_secrets: Option<&'a dyn SecretSource>,
}

/// The request as it was before its first transform, kept to rebuild the body without native
/// output when the provider rejects it.
pub(super) struct NativeOutputFallback {
    input: MessagesRequest,
    context: MessagesTransformContext,
    headers: Headers,
    url: String,
    transform: BodyTransform,
}

impl NativeOutputFallback {
    pub(super) fn body(&self, config: &dyn BaseMessagesConfig) -> Result<Value, Error> {
        let transformed = self
            .transform
            .transformed(config, self.input.clone(), &self.context)?;
        let body = self
            .transform
            .sent_to(config, transformed, &self.headers, &self.url)?;
        Ok(finalized(config, body, self.headers.clone(), &self.context)?.wire_body)
    }
}

#[derive(Clone, Copy)]
struct BodyTransform {
    tool_custom_field: ToolCustomField,
    drops_unsupported_params: bool,
    drop_params: bool,
}

impl BodyTransform {
    fn transformed(
        self,
        config: &dyn BaseMessagesConfig,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        Ok(self.tool_custom_field.apply(
            config.transform_anthropic_messages_request(request, context)?,
            self.drops_unsupported_params,
        ))
    }

    fn sent_to(
        self,
        config: &dyn BaseMessagesConfig,
        transformed: MessagesRequest,
        headers: &Headers,
        url: &str,
    ) -> Result<MessagesRequest, Error> {
        Ok(config.request_body(transformed, headers, url, self.drop_params)?)
    }
}

struct FinalizedRequest {
    body: MessagesRequest,
    headers: Headers,
    wire_body: Value,
    stream_moved_to_url: bool,
}

pub(super) async fn provider_call(
    credentials: CallerCredentials<'_>,
    provider: MessagesProvider,
    model: String,
    call: MessagesCall,
    secrets: &(dyn Lookup + Send + Sync),
    host_transport: bool,
    context: &CallContext<'_, impl Interceptors<Error>>,
) -> Result<ProviderCall, Error> {
    let MessagesCall {
        body,
        api_key,
        api_base,
        litellm_params,
        extra_headers,
        provider_specific_header,
        timeout,
        shaping,
        ..
    } = call;

    let connection_params = LitellmParams {
        api_base: api_base.clone().or(litellm_params.api_base.clone()),
        ..litellm_params
    };
    let config = provider.config();
    let transform_context = transform_context(&shaping);
    // The SDK credential chains `resolve_auth` mirrors (azure-identity, boto3) read the process
    // environment, not the call's secrets.
    let process_env = |key: &str| std::env::var(key).ok();

    let effective = effective_request(
        provider,
        MessagesRequest { model, ..body },
        &shaping,
        &transform_context,
    )?;
    let logged_params = logged_params(config, &effective, &transform_context)?;
    let trimmed =
        without_additional_drop_params(effective, &shaping.settings.additional_drop_params)?;

    let auth_services = credentials.auth;
    let AuthenticatedCall {
        validated,
        logged_in,
        deferred,
    } = authenticated_call(
        config,
        credentials,
        CallAuthInput {
            request: &trimmed,
            forwarded: forwarded_headers(
                provider,
                extra_headers,
                provider_specific_header.as_ref(),
            )?,
            configured_key: configured_key(
                api_key.as_deref(),
                shaping.settings.module_api_key.as_ref(),
            ),
            api_base: api_base.as_deref(),
            connection_params: &connection_params,
        },
        secrets,
    )
    .await?;
    let env_lookup = secret_lookup(secrets, deferred.as_ref());

    let connection_params = config.select_endpoint(
        connection_params,
        &trimmed.model,
        logged_in,
        &shaping.vertex.supported_regions,
        &env_lookup,
    );

    let transform = BodyTransform {
        tool_custom_field: config.tool_custom_field(&trimmed.model),
        drops_unsupported_params: shaping.settings.drops_unsupported_params,
        drop_params: shaping.settings.drop_params,
    };
    let requested_output_format = trimmed.params.output_format.clone();
    let fallback = config
        .native_output_fallback(&trimmed, &transform_context)
        .map(|fallback_context| (trimmed.clone(), fallback_context));
    let transformed = transform.transformed(
        config,
        config.without_rejected_params(trimmed, &connection_params)?,
        &transform_context,
    )?;

    let url = config.get_complete_url(
        api_base.as_deref(),
        &transformed.model,
        &connection_params,
        transformed.params.streams(),
        &env_lookup,
    )?;

    let body = transform.sent_to(config, transformed, &validated.headers, &url)?;
    let headers = outbound_headers(
        config,
        validated.headers,
        &body,
        requested_output_format,
        &url,
    );
    let native_output_fallback = fallback.map(|(input, context)| NativeOutputFallback {
        input,
        context,
        headers: headers.clone(),
        url: url.clone(),
        transform,
    });

    let finalized = finalized(config, body, headers, &transform_context)?;

    let request_context = RequestContext {
        model: config.logged_model(&finalized.body.model),
        custom_llm_provider: provider.as_str().to_string(),
        optional_params: logged_params,
        secret_fields: Vec::new(),
        api_key: api_key.map(SecretValue::new),
        host_transport,
    };
    let authenticated = resolve_auth(
        auth_services,
        ValidatedEnvironment {
            headers: finalized.headers,
            auth: validated.auth,
        },
        &process_env,
    )
    .await?;

    let identity = ProviderIdentity {
        model: request_context.model.clone(),
        provider: request_context.custom_llm_provider.clone(),
    };
    let wire = context
        .interceptors
        .before_provider_request(
            WireRequest {
                url,
                headers: joined_header_values(authenticated.headers),
                body: finalized.wire_body,
            },
            request_context,
        )
        .await?;
    let stream = wire.body.get("stream").map_or(
        finalized.stream_moved_to_url,
        litellm_python_compat::truthy::json_truthy,
    );

    Ok(ProviderCall {
        identity,
        wire,
        provider,
        signer: authenticated.signer,
        timeout: timeout.or_else(|| shaping.settings.request_timeout()),
        stream,
        whole_sse_frames: shaping.settings.whole_sse_frames,
        joined_reads: shaping.settings.joined_reads,
        detached_drain: shaping.settings.detached_drain(),
        transform_context,
        native_output_fallback,
        ascii_response_headers: shaping.settings.ascii_response_headers,
        unencodable_request_header: shaping.settings.unencodable_request_header,
        ascii_request_headers: shaping.settings.ascii_request_headers,
        connect_retry: shaping.settings.connect_retry,
    })
}

/// The request as it takes effect, before auth or any provider rewrite, and what the call logs:
/// - Shape: the config cleans up the messages, validates `metadata` and sets the thinking
///   auto-summary
/// - Trim: the top-level `additional_drop_params` are dropped; nested paths go after logging
/// - Gate: effort is clamped, and params the model or provider does not support are dropped or
///   rejected
fn effective_request(
    provider: MessagesProvider,
    request: MessagesRequest,
    shaping: &MessagesShaping,
    context: &MessagesTransformContext,
) -> Result<MessagesRequest, Error> {
    let shaped = provider
        .config()
        .shape_request(request, shaping.settings.shape_options())?;
    requested_optional_params(
        without_additional_drop_params(
            shaped,
            &top_level_paths(&shaping.settings.additional_drop_params),
        )?,
        provider,
        context,
    )
}

struct CallAuthInput<'a> {
    request: &'a MessagesRequest,
    forwarded: Headers,
    configured_key: Option<&'a str>,
    api_base: Option<&'a str>,
    connection_params: &'a LitellmParams,
}

struct AuthenticatedCall {
    validated: ValidatedEnvironment,
    logged_in: LoggedIn,
    deferred: Option<Secrets>,
}

async fn authenticated_call(
    config: &dyn BaseMessagesConfig,
    credentials: CallerCredentials<'_>,
    input: CallAuthInput<'_>,
    secrets: &(dyn Lookup + Send + Sync),
) -> Result<AuthenticatedCall, Error> {
    let deferred = read_deferred_secrets(
        credentials.deferred_secrets,
        config.deferred_secret_names(
            credentials.azure_ad_token_hosts.host.is_some(),
            &input.forwarded,
            input.configured_key,
            input.api_base,
            input.connection_params,
            &|key: &str| secrets.get(key),
        ),
    )
    .await?;
    let (validated, logged_in) = {
        let env_lookup = secret_lookup(secrets, deferred.as_ref());
        let validated =
            resolve_environment(config, input, credentials.workload_identity, &env_lookup).await?;
        prepare_auth(
            credentials.auth,
            validated,
            credentials.azure_ad_token_hosts,
            &env_lookup,
        )
        .await?
    };
    Ok(AuthenticatedCall {
        validated,
        logged_in,
        deferred,
    })
}

fn secret_lookup<'a>(
    secrets: &'a (dyn Lookup + Send + Sync),
    deferred: Option<&'a Secrets>,
) -> impl Fn(&str) -> Option<String> + 'a {
    move |key: &str| {
        secrets
            .get(key)
            .or_else(|| deferred.and_then(|deferred| deferred.get(key)))
    }
}

fn transform_context(shaping: &MessagesShaping) -> MessagesTransformContext {
    MessagesTransformContext {
        drops_unsupported_params: shaping.settings.drops_unsupported_params,
        global_drop_params: shaping.settings.global_drop_params,
        claude_base_model: shaping
            .settings
            .base_model
            .as_deref()
            .is_some_and(is_claude_model),
        ..MessagesTransformContext::new(shaping.capabilities, shaping.settings.drop_params)
    }
}

fn top_level_paths(paths: &[String]) -> Vec<String> {
    paths
        .iter()
        .filter(|path| !path.contains(['.', '[']))
        .cloned()
        .collect()
}

fn without_additional_drop_params(
    request: MessagesRequest,
    paths: &[String],
) -> Result<MessagesRequest, Error> {
    if paths.is_empty() {
        return Ok(request);
    }
    let params = serde_json::to_value(request.params).map_err(invalid_request)?;
    let trimmed = paths
        .iter()
        .fold(params, |params, path| delete_nested_value(params, path));
    Ok(MessagesRequest {
        params: serde_json::from_value(trimmed).map_err(invalid_request)?,
        ..request
    })
}

/// Python's `get_requested_anthropic_messages_optional_param`, which gates speed, service tier
/// and sampling params before `additional_drop_params` and the environment check run.
fn requested_optional_params(
    request: MessagesRequest,
    provider: MessagesProvider,
    context: &MessagesTransformContext,
) -> Result<MessagesRequest, Error> {
    let requested_effort = request.params.reasoning_effort.clone();
    let clamped = provider
        .config()
        .clamp_requested_effort(request, &context.thinking.capabilities)?;
    let gated = litellm_llms::anthropic::messages::transformation::drop_unsupported_params(
        clamped, context,
    )?;
    let request = MessagesRequest {
        params: MessagesOptionalParams {
            reasoning_effort: gated.params.reasoning_effort.and(requested_effort),
            ..gated.params
        },
        ..gated
    };
    if provider.accepts_service_tier() {
        return Ok(request);
    }
    if request.params.service_tier.is_some() {
        tracing::warn!(
            "Dropping `service_tier` for model={} on provider={}. Only the Anthropic API routes accept it.",
            request.model,
            provider.as_str()
        );
    }
    Ok(MessagesRequest {
        params: MessagesOptionalParams {
            service_tier: None,
            ..request.params
        },
        ..request
    })
}

fn logged_params(
    config: &dyn BaseMessagesConfig,
    requested: &MessagesRequest,
    context: &MessagesTransformContext,
) -> Result<Value, Error> {
    serde_json::to_value(config.logged_params(requested.params.clone(), context)?)
        .map_err(serialize_failure)
}

fn forwarded_headers(
    provider: MessagesProvider,
    extra_headers: Option<Map<String, Value>>,
    provider_specific_header: Option<&ProviderSpecificHeaders>,
) -> Result<Headers, Error> {
    string_headers(Some(merge_messages_header_sources([
        extra_headers.unwrap_or_default(),
        get_provider_specific_headers(provider_specific_header, provider.as_str()),
    ])))
}

fn configured_key<'a>(
    api_key: Option<&'a str>,
    module_api_key: Option<&'a SecretValue>,
) -> Option<&'a str> {
    match module_api_key {
        Some(module_key) => api_key
            .filter(|key| !key.is_empty())
            .or(Some(module_key.expose())),
        None => api_key,
    }
}

async fn read_deferred_secrets(
    source: Option<&dyn SecretSource>,
    names: &[&str],
) -> Result<Option<Secrets>, Error> {
    match source {
        Some(source) if !names.is_empty() => Ok(Some(source.resolve(names).await?)),
        _ => Ok(None),
    }
}

/// The config's validated environment, or, when it finds no credential, one federated from the
/// workload identity's token; without a token the call fails as missing its credential.
async fn resolve_environment(
    config: &dyn BaseMessagesConfig,
    input: CallAuthInput<'_>,
    workload_identity: Option<&dyn WorkloadIdentity>,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<ValidatedEnvironment, Error> {
    let environment = config.validate_federated_environment(
        input.forwarded,
        input.configured_key,
        &input.request.model,
        &input.request.params,
        input.connection_params,
        env_lookup,
    )?;
    let (headers, missing, federate) = match environment {
        FederatedEnvironment::Validated(validated) => return Ok(validated),
        FederatedEnvironment::Unauthenticated {
            headers,
            missing,
            federate,
        } => (headers, missing, federate),
    };
    let Some(identity) = workload_identity else {
        return Err(missing.into());
    };
    match identity
        .token(input.connection_params, input.api_base)
        .await?
    {
        Some(token) => Ok(federate(headers, token)),
        None => Err(missing.into()),
    }
}

fn outbound_headers(
    config: &dyn BaseMessagesConfig,
    validated: Headers,
    body: &MessagesRequest,
    requested_output_format: Option<Value>,
    url: &str,
) -> Headers {
    config.headers_for_url(
        config.request_headers(
            with_python_default_headers(validated, config.default_headers()),
            &with_requested_output_format(body, requested_output_format),
        ),
        url,
    )
}

/// Python's `finalize_messages_headers` adds a default only when the exact lowercase name is
/// absent, so a caller's `Anthropic-Version` is sent alongside the default, not instead of it.
fn with_python_default_headers(headers: Headers, defaults: &[(&str, &str)]) -> Headers {
    let missing: Headers = defaults
        .iter()
        .filter(|(name, _)| !headers.iter().any(|(present, _)| present == name))
        .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
        .collect();
    headers.into_iter().chain(missing).collect()
}

fn with_requested_output_format(
    body: &MessagesRequest,
    requested: Option<Value>,
) -> std::borrow::Cow<'_, MessagesRequest> {
    match requested {
        Some(output_format) if body.params.output_format.is_none() => {
            std::borrow::Cow::Owned(MessagesRequest {
                params: MessagesOptionalParams {
                    output_format: Some(output_format),
                    ..body.params.clone()
                },
                ..body.clone()
            })
        }
        _ => std::borrow::Cow::Borrowed(body),
    }
}

fn finalized(
    config: &dyn BaseMessagesConfig,
    body: MessagesRequest,
    headers: Headers,
    context: &MessagesTransformContext,
) -> Result<FinalizedRequest, Error> {
    let (body, headers) = config.finalize_request(body, headers, context)?;
    let serialized = serde_json::to_value(&body).map_err(serialize_failure)?;
    let serialized_stream = serialized.get("stream").is_some();
    let wire_body = config.wire_body(serialized);
    Ok(FinalizedRequest {
        stream_moved_to_url: body.params.streams()
            && serialized_stream
            && wire_body.get("stream").is_none(),
        body,
        headers,
        wire_body,
    })
}

/// Python sends its header dict through `httpx.Headers`, which folds names that differ only in
/// case into one lowercase header whose values are joined with `", "`.
fn joined_header_values(headers: Headers) -> Headers {
    let mut joined: Headers = Vec::with_capacity(headers.len());
    let mut positions: std::collections::HashMap<String, usize> =
        std::collections::HashMap::with_capacity(headers.len());
    for (name, value) in headers {
        match positions.entry(name.to_ascii_lowercase()) {
            std::collections::hash_map::Entry::Occupied(position) => {
                let present = &mut joined[*position.get()].1;
                present.push_str(", ");
                present.push_str(&value);
            }
            std::collections::hash_map::Entry::Vacant(position) => {
                position.insert(joined.len());
                joined.push((name, value));
            }
        }
    }
    joined
}

pub(super) fn serialize_failure(err: serde_json::Error) -> Error {
    Error::invalid_request(litellm_llms::ErrorDetail::failed(
        "Anthropic messages request serialization",
        err,
    ))
}

#[cfg(test)]
mod tests {
    use litellm_llms_types::headers::ProviderSpecificHeaders;
    use rstest::{fixture, rstest};
    use serde_json::{Map, Value, json};

    use super::*;
    use crate::{MessagesSettings, MessagesShaping, common_utils::messages_provider};
    use litellm_inference::outbound::{raw_leaf_body, serialized_outbound_request};
    use litellm_inference::provider::resolve_llm_provider;
    use litellm_llms::base_llm::auth::Authenticated;

    #[fixture]
    fn shaping() -> MessagesShaping {
        MessagesShaping::default()
    }

    fn opus_4_6_on_bedrock() -> MessagesTransformContext {
        use litellm_llms::base_llm::messages::context::{
            BedrockCapabilities, MessagesModelCapabilities, SupportedEffortTiers,
        };
        MessagesTransformContext::new(
            MessagesModelCapabilities {
                supports_reasoning: true,
                supports_adaptive_thinking: true,
                supports_output_config: true,
                effort_tiers: SupportedEffortTiers {
                    low: true,
                    medium: true,
                    high: true,
                    max: true,
                    ..SupportedEffortTiers::default()
                },
                bedrock: BedrockCapabilities {
                    supports_adaptive_thinking: true,
                    output_config_effort_ceiling: Some(
                        litellm_llms_types::formats::messages::EffortLevel::Max,
                    ),
                    ..BedrockCapabilities::default()
                },
                ..MessagesModelCapabilities::default()
            },
            true,
        )
    }

    #[rstest]
    #[case::bedrock_clamps_before_the_gate("bedrock/anthropic.claude-opus-4-6-v1", Some(json!("xhigh")))]
    #[case::anthropic_has_no_clamp("anthropic/claude-opus-4-6", None)]
    fn drop_params_keeps_an_effort_the_bedrock_ceiling_clamps(
        #[case] model: &str,
        #[case] kept: Option<Value>,
    ) {
        let resolved = resolve_llm_provider(model, None, "messages").unwrap();
        let provider = messages_provider(resolved.provider, resolved.model).unwrap();
        let requested = requested_optional_params(
            body(json!({"model": resolved.model, "max_tokens": 4096, "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "xhigh"})),
            provider,
            &opus_4_6_on_bedrock(),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(&requested.params)
                .unwrap()
                .get("reasoning_effort")
                .cloned(),
            kept
        );
    }

    fn body(value: Value) -> MessagesRequest {
        serde_json::from_value(value).unwrap()
    }

    async fn provider_call_for(call: MessagesCall) -> Result<ProviderCall, Error> {
        provider_call_with_secrets(call, &|_: &str| None).await
    }

    use litellm_auth_gcp::{google::OnGce, vertex::VertexLoginService};

    async fn provider_call_with_secrets(
        call: MessagesCall,
        secrets: &(dyn Lookup + Send + Sync),
    ) -> Result<ProviderCall, Error> {
        provider_call_federating(call, secrets, None).await
    }

    async fn provider_call_federating(
        call: MessagesCall,
        secrets: &(dyn Lookup + Send + Sync),
        workload_identity: Option<&dyn WorkloadIdentity>,
    ) -> Result<ProviderCall, Error> {
        let provider = resolve_llm_provider(
            &call.body.model,
            call.custom_llm_provider.as_deref(),
            "messages",
        )?;
        let config = messages_provider(provider.provider, provider.model)
            .ok_or_else(|| Error::InvalidProvider(<&str>::from(provider.provider).to_string()))?;
        let model = provider.model.to_owned();
        let interceptors = ();
        let context = CallContext::new(&interceptors, litellm_inference::CallOptions::default());
        let auth = AuthServices {
            vertex_login: VertexLoginService::new(std::sync::Arc::new(OnGce {
                project: "token-project",
                token: "google-token",
            })),
            ..AuthServices::default()
        };
        provider_call(
            CallerCredentials {
                auth: &auth,
                workload_identity,
                azure_ad_token_hosts: AzureAdTokenHosts::default(),
                deferred_secrets: None,
            },
            config,
            model,
            call,
            secrets,
            false,
            &context,
        )
        .await
    }

    async fn wire_body(fields: Value, shaping: MessagesShaping) -> Result<Value, Error> {
        provider_call_for(MessagesCall {
            body: body(fields),
            api_key: Some("sk-test".into()),
            api_base: Some("https://anthropic.test".into()),
            custom_llm_provider: Some("anthropic".into()),
            litellm_params: Default::default(),
            extra_headers: None,
            provider_specific_header: None,
            timeout: None,
            shaping,
        })
        .await
        .map(|call| call.wire.body)
    }

    #[rstest]
    #[case::api_key(
        &[("ANTHROPIC_API_KEY", "sk-secret")],
        &[("x-api-key", "sk-secret")],
        "https://api.anthropic.com/v1/messages"
    )]
    #[case::auth_token(
        &[("ANTHROPIC_AUTH_TOKEN", "token")],
        &[("authorization", "Bearer token")],
        "https://api.anthropic.com/v1/messages"
    )]
    #[case::api_base(
        &[("ANTHROPIC_API_KEY", "sk-secret"), ("ANTHROPIC_API_BASE", "https://gateway.test")],
        &[("x-api-key", "sk-secret")],
        "https://gateway.test/v1/messages"
    )]
    #[case::sdk_base_url(
        &[("ANTHROPIC_API_KEY", "sk-secret"), ("ANTHROPIC_BASE_URL", "https://sdk.test")],
        &[("x-api-key", "sk-secret")],
        "https://sdk.test/v1/messages"
    )]
    #[tokio::test]
    async fn credentials_and_base_come_from_the_resolved_secrets(
        shaping: MessagesShaping,
        #[case] secrets: &[(&str, &str)],
        #[case] expected_auth: &[(&str, &str)],
        #[case] expected_url: &str,
    ) {
        let lookup = |name: &str| {
            secrets
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        };
        let call = provider_call_with_secrets(
            MessagesCall {
                body: body(
                    json!({"model": "claude-test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}),
                ),
                api_key: None,
                api_base: None,
                custom_llm_provider: Some("anthropic".into()),
                litellm_params: Default::default(),
                extra_headers: None,
                provider_specific_header: None,
                timeout: None,
                shaping,
            },
            &lookup,
        )
        .await
        .unwrap();
        let auth: Vec<(&str, &str)> = call
            .wire
            .headers
            .iter()
            .filter(|(name, _)| matches!(name.as_str(), "x-api-key" | "authorization"))
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        assert_eq!(
            (auth.as_slice(), call.wire.url.as_str()),
            (expected_auth, expected_url)
        );
    }

    fn vertex_call_requesting(
        shaping: MessagesShaping,
        requested: &str,
        drop_params: bool,
    ) -> MessagesCall {
        MessagesCall {
            body: body(
                json!({"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}),
            ),
            api_key: None,
            api_base: Some("https://vertex.test".into()),
            custom_llm_provider: Some("vertex_ai".into()),
            litellm_params: Default::default(),
            extra_headers: Some(Map::from_iter([
                ("authorization".to_string(), json!("Bearer vertex-token")),
                ("anthropic-version".to_string(), json!(requested)),
            ])),
            provider_specific_header: None,
            timeout: None,
            shaping: MessagesShaping {
                settings: MessagesSettings {
                    drop_params,
                    ..shaping.settings
                },
                ..shaping
            },
        }
    }

    fn vertex_project(name: &str) -> Option<String> {
        (name == "VERTEXAI_PROJECT").then(|| "version-project".to_string())
    }

    #[rstest]
    #[case::sdk_2("2026-10-21", false, "2026-10-21")]
    #[case::sdk_1("2023-06-01", false, "vertex-2023-10-16")]
    #[case::unknown_under_drop_params("latest", true, "vertex-2023-10-16")]
    #[tokio::test]
    async fn vertex_gets_the_requested_version_in_the_body_alone(
        shaping: MessagesShaping,
        #[case] requested: &str,
        #[case] drop_params: bool,
        #[case] body_version: &str,
    ) {
        let call = provider_call_with_secrets(
            vertex_call_requesting(shaping, requested, drop_params),
            &vertex_project,
        )
        .await
        .unwrap();
        let version_headers: Vec<&str> = call
            .wire
            .headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-version"))
            .map(|(_, value)| value.as_str())
            .collect();
        assert_eq!(
            (version_headers, &call.wire.body["anthropic_version"]),
            (Vec::<&str>::new(), &json!(body_version))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn vertex_sends_the_google_token_beside_a_caller_bearer_and_only_vertex_betas(
        shaping: MessagesShaping,
    ) {
        let call = vertex_call_requesting(shaping, "2023-06-01", false);
        let call = MessagesCall {
            extra_headers: call.extra_headers.map(|headers| {
                headers
                    .into_iter()
                    .chain([(
                        "anthropic-beta".to_string(),
                        json!("files-api-2025-04-14,context-1m-2025-08-07"),
                    )])
                    .collect()
            }),
            ..call
        };
        let call = provider_call_with_secrets(call, &vertex_project)
            .await
            .unwrap();
        let sent = |name: &str| {
            call.wire
                .headers
                .iter()
                .find(|(present, _)| present.eq_ignore_ascii_case(name))
                .map(|(_, value)| value.as_str())
        };
        assert_eq!(
            (
                sent("authorization"),
                sent("anthropic-beta"),
                sent("content-type")
            ),
            (
                Some("Bearer vertex-token, Bearer google-token"),
                Some("context-1m-2025-08-07"),
                Some("application/json")
            )
        );
    }

    #[rstest]
    #[tokio::test]
    async fn vertex_refuses_an_unsupported_version_without_drop_params(shaping: MessagesShaping) {
        assert!(matches!(
            provider_call_with_secrets(
                vertex_call_requesting(shaping, "latest", false),
                &vertex_project
            )
            .await,
            Err(Error::Llm(litellm_llms::Error::InvalidRequest(
                litellm_llms::ErrorDetail::UnsupportedVertexAnthropicVersion { .. }
            )))
        ));
    }

    fn call_requesting_version(
        shaping: MessagesShaping,
        provider: &str,
        api_base: Option<&str>,
        version: &str,
        drop_params: bool,
    ) -> MessagesCall {
        MessagesCall {
            body: body(
                json!({"model": "claude-test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16, "temperature": 1}),
            ),
            api_key: Some("sk-test".into()),
            api_base: api_base.map(Into::into),
            custom_llm_provider: Some(provider.into()),
            litellm_params: Default::default(),
            extra_headers: Some(Map::from_iter([(
                "anthropic-version".to_string(),
                json!(version),
            )])),
            provider_specific_header: None,
            timeout: None,
            shaping: MessagesShaping {
                settings: MessagesSettings {
                    drop_params,
                    ..shaping.settings
                },
                ..shaping
            },
        }
    }

    #[rstest]
    #[case::anthropic_under_sdk_2(
        "anthropic",
        Some("https://api.anthropic.com"),
        "2026-10-21",
        None
    )]
    #[case::anthropic_default_base_under_sdk_2("anthropic", None, "2026-10-21", None)]
    #[case::anthropic_under_sdk_1(
        "anthropic",
        Some("https://api.anthropic.com"),
        "2023-06-01",
        Some(json!(1))
    )]
    #[case::non_anthropic_api_under_sdk_2(
        "deepseek",
        Some("https://upstream.test"),
        "2026-10-21",
        Some(json!(1))
    )]
    #[tokio::test]
    async fn temperature_reaches_the_wire_only_under_an_api_version_that_has_it(
        shaping: MessagesShaping,
        #[case] provider: &str,
        #[case] api_base: Option<&str>,
        #[case] version: &str,
        #[case] temperature: Option<Value>,
    ) {
        let call = provider_call_for(call_requesting_version(
            shaping, provider, api_base, version, true,
        ))
        .await
        .unwrap();
        assert_eq!(call.wire.body.get("temperature").cloned(), temperature);
    }

    #[rstest]
    #[case::deployment_api_base(Some("https://api.deepseek.com/anthropic"), &[])]
    #[case::api_base_env(None, &[("ANTHROPIC_API_BASE", "https://api.deepseek.com/anthropic")])]
    #[case::base_url_env(None, &[("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")])]
    #[tokio::test]
    async fn a_third_party_anthropic_api_receives_temperature_under_2026_10_21(
        shaping: MessagesShaping,
        #[case] api_base: Option<&str>,
        #[case] secrets: &[(&str, &str)],
        #[values(false, true)] drop_params: bool,
    ) {
        let lookup = |name: &str| {
            secrets
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        };
        let call = provider_call_with_secrets(
            call_requesting_version(shaping, "anthropic", api_base, "2026-10-21", drop_params),
            &lookup,
        )
        .await
        .unwrap();
        assert_eq!(
            (call.wire.url.as_str(), call.wire.body.get("temperature")),
            (
                "https://api.deepseek.com/anthropic/v1/messages",
                Some(&json!(1))
            )
        );
    }

    #[rstest]
    #[tokio::test]
    async fn an_api_version_without_sampling_params_refuses_them_before_the_provider_call(
        shaping: MessagesShaping,
    ) {
        let Err(Error::Llm(litellm_llms::Error::InvalidRequest(detail))) =
            provider_call_for(call_requesting_version(
                shaping,
                "anthropic",
                Some("https://api.anthropic.com"),
                "2026-10-21",
                false,
            ))
            .await
        else {
            panic!("expected an invalid request");
        };
        assert_eq!(
            detail.to_string(),
            "claude-test does not support temperature=1. anthropic-version 2026-10-21 removed \
             temperature, top_p and top_k. To drop unsupported params, set \
             `litellm.drop_params = True`."
        );
    }

    #[rstest]
    #[tokio::test]
    async fn missing_credentials_are_reported_before_the_request_is_transformed(
        shaping: MessagesShaping,
    ) {
        let result = provider_call_for(MessagesCall {
            body: body(
                json!({"model": "claude-test", "messages": [{"role": "user", "content": "hi"}]}),
            ),
            api_key: None,
            api_base: None,
            custom_llm_provider: Some("anthropic".into()),
            litellm_params: Default::default(),
            extra_headers: None,
            provider_specific_header: None,
            timeout: None,
            shaping,
        })
        .await;
        let Err(error) = result else {
            panic!("a call without credentials fails");
        };
        assert!(error.is_missing_credential(), "{error:?}");
    }

    struct ScriptedIdentity {
        minted: Result<Option<&'static str>, ()>,
        asked: std::sync::Mutex<Vec<(Option<Value>, Option<String>)>>,
    }

    impl ScriptedIdentity {
        fn new(minted: Result<Option<&'static str>, ()>) -> Self {
            Self {
                minted,
                asked: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn asked(&self) -> Vec<(Option<Value>, Option<String>)> {
            self.asked.lock().unwrap().clone()
        }
    }

    impl WorkloadIdentity for ScriptedIdentity {
        fn token<'a>(
            &'a self,
            litellm_params: &'a litellm_router_types::LitellmParams,
            api_base: Option<&'a str>,
        ) -> futures_util::future::BoxFuture<'a, Result<Option<SecretValue>, litellm_auth::Error>>
        {
            self.asked.lock().unwrap().push((
                litellm_params
                    .extra
                    .get("anthropic_federation_rule_id")
                    .cloned(),
                api_base.map(str::to_owned),
            ));
            let minted = self.minted;
            Box::pin(async move {
                minted
                    .map(|token| token.map(SecretValue::new))
                    .map_err(|()| litellm_auth::Error::InvalidHeader)
            })
        }
    }

    fn federated_call(
        shaping: MessagesShaping,
        api_key: Option<&str>,
        extra_headers: Option<Value>,
    ) -> MessagesCall {
        MessagesCall {
            body: body(
                json!({"model": "claude-test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}),
            ),
            api_key: api_key.map(str::to_owned),
            api_base: Some("https://anthropic.test".into()),
            custom_llm_provider: Some("anthropic".into()),
            litellm_params: crate::litellm_params(
                serde_json::from_value(
                    json!({"model": "claude-test", "anthropic_federation_rule_id": "fdrl_1"}),
                )
                .unwrap(),
            )
            .unwrap(),
            extra_headers: extra_headers.map(|headers| serde_json::from_value(headers).unwrap()),
            provider_specific_header: None,
            timeout: None,
            shaping,
        }
    }

    fn sent_headers<'a>(call: &'a ProviderCall, names: &[&str]) -> Vec<(&'a str, &'a str)> {
        call.wire
            .headers
            .iter()
            .filter(|(name, _)| names.contains(&name.as_str()))
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect()
    }

    #[rstest]
    #[case::alone(None, &[("anthropic-beta", "oauth-2025-04-20"), ("authorization", "Bearer sk-ant-oat01-minted")])]
    #[case::beside_a_caller_beta(
        Some(json!({"anthropic-beta": "files-api-2025-04-14"})),
        &[("anthropic-beta", "files-api-2025-04-14,oauth-2025-04-20"), ("authorization", "Bearer sk-ant-oat01-minted")]
    )]
    #[tokio::test]
    async fn a_minted_federation_token_is_sent_as_an_oauth_bearer(
        shaping: MessagesShaping,
        #[case] extra_headers: Option<Value>,
        #[case] expected: &[(&str, &str)],
    ) {
        let identity = ScriptedIdentity::new(Ok(Some("sk-ant-oat01-minted")));
        let call = provider_call_federating(
            federated_call(shaping, None, extra_headers),
            &|_: &str| None,
            Some(&identity),
        )
        .await
        .unwrap();
        let mut sent = sent_headers(
            &call,
            &[
                "anthropic-beta",
                "authorization",
                "x-api-key",
                "anthropic-dangerous-direct-browser-access",
            ],
        );
        sent.sort_unstable();
        assert_eq!(
            (sent.as_slice(), identity.asked()),
            (
                expected,
                vec![(Some(json!("fdrl_1")), Some("https://anthropic.test".into()))]
            )
        );
    }

    #[rstest]
    #[case::static_key(Some("sk-static"), None, &[("x-api-key", "sk-static")])]
    #[case::caller_credential(
        None,
        Some(json!({"x-api-key": "sk-caller"})),
        &[("x-api-key", "sk-caller")]
    )]
    #[tokio::test]
    async fn a_configured_credential_is_never_federated(
        shaping: MessagesShaping,
        #[case] api_key: Option<&str>,
        #[case] extra_headers: Option<Value>,
        #[case] expected: &[(&str, &str)],
    ) {
        let identity = ScriptedIdentity::new(Ok(Some("sk-ant-oat01-minted")));
        let call = provider_call_federating(
            federated_call(shaping, api_key, extra_headers),
            &|_: &str| None,
            Some(&identity),
        )
        .await
        .unwrap();
        assert_eq!(
            (
                sent_headers(&call, &["authorization", "x-api-key"]),
                identity.asked()
            ),
            (expected.to_vec(), Vec::new())
        );
    }

    #[rstest]
    #[case::nothing_minted(Ok(None), true)]
    #[case::minting_failed(Err(()), false)]
    #[tokio::test]
    async fn an_unminted_federation_token_fails_the_call(
        shaping: MessagesShaping,
        #[case] minted: Result<Option<&'static str>, ()>,
        #[case] missing_credential: bool,
    ) {
        let identity = ScriptedIdentity::new(minted);
        let Err(error) = provider_call_federating(
            federated_call(shaping, None, None),
            &|_: &str| None,
            Some(&identity),
        )
        .await
        else {
            panic!("a call without a minted token fails");
        };
        assert_eq!(
            error.is_missing_credential(),
            missing_credential,
            "{error:?}"
        );
    }

    #[rstest]
    #[case::nothing_configured(None, None, None)]
    #[case::configured_setting(None, Some(1800.0), Some(Duration::from_secs(1800)))]
    #[case::call_timeout_wins(
        Some(Duration::from_secs(30)),
        Some(1800.0),
        Some(Duration::from_secs(30))
    )]
    #[case::call_timeout_only(Some(Duration::from_secs(30)), None, Some(Duration::from_secs(30)))]
    #[case::unusable_setting_ignored(None, Some(-1.0), None)]
    #[case::zero_setting_ignored(None, Some(0.0), None)]
    #[tokio::test]
    async fn timeout_falls_back_to_the_configured_request_timeout(
        shaping: MessagesShaping,
        #[case] call_timeout: Option<Duration>,
        #[case] request_timeout: Option<f64>,
        #[case] expected: Option<Duration>,
    ) {
        let call = provider_call_for(MessagesCall {
            body: body(json!({"model": "claude-test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16})),
            api_key: Some("sk-test".into()),
            api_base: Some("https://anthropic.test".into()),
            custom_llm_provider: Some("anthropic".into()),
            litellm_params: Default::default(),
            extra_headers: None,
            provider_specific_header: None,
            timeout: call_timeout,
            shaping: MessagesShaping {
                settings: MessagesSettings { request_timeout, ..shaping.settings },
                ..shaping
            },
        })
        .await
        .unwrap();
        assert_eq!(call.timeout, expected);
    }

    #[rstest]
    #[case::top_level_and_nested_paths(
        json!({
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "context_management": {"edits": [{"type": "clear_thinking_20251015"}]},
            "metadata": {"user_id": "u1"},
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}, "input_examples": [{"q": "x"}]}]
        }),
        &["thinking", "context_management", "tools[*].input_examples"],
        json!({
            "max_tokens": 1024,
            "metadata": {"user_id": "u1"},
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}]
        }),
    )]
    #[case::no_paths(
        json!({"max_tokens": 16, "safeguards": [{"type": "dangerous_tool_use"}]}),
        &[],
        json!({"max_tokens": 16, "safeguards": [{"type": "dangerous_tool_use"}]}),
    )]
    #[case::model_and_messages_are_never_dropped(
        json!({"max_tokens": 16}),
        &["model", "messages", "messages[0].content"],
        json!({"max_tokens": 16}),
    )]
    #[tokio::test]
    async fn wire_body_drops_configured_paths(
        shaping: MessagesShaping,
        #[case] fields: Value,
        #[case] additional_drop_params: &[&str],
        #[case] expected_fields: Value,
    ) {
        let with_messages = |fields: Value| -> Value {
            let Value::Object(fields) = fields else {
                unreachable!()
            };
            Value::Object(
                [
                    ("model".to_string(), json!("claude-test")),
                    (
                        "messages".to_string(),
                        json!([{"role": "user", "content": "hi"}]),
                    ),
                ]
                .into_iter()
                .chain(fields)
                .collect(),
            )
        };
        let shaping = MessagesShaping {
            settings: MessagesSettings {
                additional_drop_params: additional_drop_params
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
                ..shaping.settings
            },
            ..shaping
        };
        assert_eq!(
            wire_body(with_messages(fields), shaping).await,
            Ok(with_messages(expected_fields))
        );
    }

    #[rstest]
    #[case::model_prefix_picks_the_provider(
        "azure_ai/claude-test",
        None,
        &[("x-priority", "extra"), ("x-scoped", "azure_ai")]
    )]
    #[case::explicit_provider(
        "claude-test",
        Some("anthropic"),
        &[("x-priority", "scoped"), ("x-scoped", "anthropic")]
    )]
    #[case::provider_prefix_on_an_anthropic_model(
        "anthropic/claude-test",
        None,
        &[("x-priority", "scoped"), ("x-scoped", "anthropic")]
    )]
    #[tokio::test]
    async fn provider_specific_headers_follow_the_resolved_provider(
        shaping: MessagesShaping,
        #[case] model: &str,
        #[case] custom_llm_provider: Option<&str>,
        #[case] expected: &[(&str, &str)],
    ) {
        let configured: ProviderSpecificHeaders = serde_json::from_value(json!([
            {"custom_llm_provider": "azure_ai", "extra_headers": {"x-scoped": "azure_ai"}},
            {"custom_llm_provider": "anthropic", "extra_headers": {"x-scoped": "anthropic", "x-priority": "scoped"}}
        ]))
        .unwrap();
        let call = provider_call_for(MessagesCall {
            body: body(
                json!({"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}),
            ),
            api_key: Some("sk-test".into()),
            api_base: Some("https://resource.services.ai.azure.com".into()),
            custom_llm_provider: custom_llm_provider.map(Into::into),
            litellm_params: Default::default(),
            extra_headers: Some(Map::from_iter([("x-priority".into(), json!("extra"))])),
            provider_specific_header: Some(configured),
            timeout: None,
            shaping,
        })
        .await
        .unwrap();
        let caller_headers: Vec<(&str, &str)> = call
            .wire
            .headers
            .iter()
            .filter(|(name, _)| matches!(name.as_str(), "x-priority" | "x-scoped"))
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        assert_eq!(caller_headers, expected);
    }

    #[rstest]
    #[tokio::test]
    async fn wire_body_carries_the_provider_stripped_model(shaping: MessagesShaping) {
        assert_eq!(
            wire_body(
                json!({
                    "model": "anthropic/claude-test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16
                }),
                shaping,
            )
            .await,
            Ok(json!({
                "model": "claude-test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16
            }))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn dropped_thinking_display_is_not_restored_by_auto_summary(shaping: MessagesShaping) {
        let shaping = MessagesShaping {
            settings: MessagesSettings {
                reasoning_auto_summary: true,
                additional_drop_params: vec!["thinking.display".to_string()],
                ..shaping.settings
            },
            ..shaping
        };
        assert_eq!(
            wire_body(
                json!({
                    "model": "claude-test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 4096,
                    "thinking": {"type": "enabled", "budget_tokens": 2048}
                }),
                shaping,
            )
            .await,
            Ok(json!({
                "model": "claude-test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4096,
                "thinking": {"type": "enabled", "budget_tokens": 2048}
            }))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn dropping_an_invalid_metadata_user_id_does_not_skip_its_validation(
        shaping: MessagesShaping,
    ) {
        let shaping = MessagesShaping {
            settings: MessagesSettings {
                additional_drop_params: vec!["metadata.user_id".to_string()],
                ..shaping.settings
            },
            ..shaping
        };
        let error = wire_body(
            json!({
                "model": "claude-test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
                "metadata": {"user_id": 123}
            }),
            shaping,
        )
        .await
        .unwrap_err();
        assert!(error.is_request(), "{error:?}");
    }

    #[rstest]
    #[tokio::test]
    async fn wire_body_rejects_invalid_metadata_before_the_call(shaping: MessagesShaping) {
        assert_eq!(
            wire_body(
                json!({
                    "model": "claude-test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                    "metadata": {"user_id": 123}
                }),
                shaping,
            )
            .await,
            Err(Error::invalid_request(
                litellm_llms::ErrorDetail::InvalidValue {
                    field: "metadata.user_id",
                    expected: "a string",
                    actual: json!(123),
                }
            ))
        );
    }

    #[rstest]
    #[case::mantle(
        "bedrock_mantle/claude-test",
        "https://bedrock-mantle.us-west-2.api.aws",
        "/anthropic/v1/messages",
        "bedrock"
    )]
    #[case::platform(
        "bedrock/claude_platform/claude-test",
        "https://aws-external-anthropic.us-west-2.api.aws",
        "/v1/messages",
        "aws-external-anthropic"
    )]
    #[tokio::test]
    async fn explicit_endpoint_region_is_used_by_the_signer(
        shaping: MessagesShaping,
        #[case] model: &str,
        #[case] base: &str,
        #[case] path: &str,
        #[case] service: &str,
    ) {
        let call = provider_call_with_secrets(MessagesCall {
            body: body(json!({"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]})),
            api_key: None,
            api_base: Some(base.into()),
            custom_llm_provider: None,
            litellm_params: crate::litellm_params(Map::from_iter([
                ("model".into(), json!(model)),
                ("aws_access_key_id".into(), json!("AKIDEXAMPLE")),
                ("aws_secret_access_key".into(), json!("secret")),
            ]))
            .unwrap(),
            extra_headers: None,
            provider_specific_header: None,
            timeout: None,
            shaping,
        }, &|name: &str| match name {
            "AWS_REGION_NAME" => Some("us-east-1".into()),
            "ANTHROPIC_AWS_WORKSPACE_ID" => Some("workspace".into()),
            _ => None,
        })
        .await
        .unwrap();
        assert_eq!(call.wire.url, format!("{base}{path}"));
        let signed = serialized_outbound_request(
            Authenticated {
                headers: call.wire.headers,
                signer: call.signer,
            },
            &call.wire.url,
            raw_leaf_body(&call.wire.body).unwrap(),
            None,
        )
        .unwrap();
        let authorization = signed.headers()["authorization"].to_str().unwrap();
        assert!(
            authorization.contains(&format!("/us-west-2/{service}/aws4_request")),
            "{authorization}"
        );
    }

    async fn mantle_call(model: &str, fields: Value, api_key: Option<&str>) -> ProviderCall {
        provider_call_with_secrets(
            MessagesCall {
                body: body(Value::Object(
                    [
                        ("model".to_string(), json!(model)),
                        ("max_tokens".to_string(), json!(16)),
                        (
                            "messages".to_string(),
                            json!([{"role": "user", "content": "hi"}]),
                        ),
                    ]
                    .into_iter()
                    .chain(fields.as_object().cloned().unwrap_or_default())
                    .collect(),
                )),
                api_key: api_key.map(str::to_string),
                api_base: Some("https://mantle.test".into()),
                custom_llm_provider: None,
                litellm_params: crate::litellm_params(Map::from_iter([
                    ("model".into(), json!(model)),
                    ("aws_access_key_id".into(), json!("AKIDEXAMPLE")),
                    ("aws_secret_access_key".into(), json!("secret")),
                ]))
                .unwrap(),
                extra_headers: None,
                provider_specific_header: None,
                timeout: None,
                shaping: MessagesShaping::default(),
            },
            &|name: &str| (name == "AWS_BEARER_TOKEN_BEDROCK").then(|| "env-token".into()),
        )
        .await
        .unwrap()
    }

    fn authorization(call: &ProviderCall) -> Option<&str> {
        call.wire
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
            .map(|(_, value)| value.as_str())
    }

    #[rstest]
    #[case::bedrock_mantle_route_signs("bedrock/mantle/anthropic.claude-sonnet-5-5", None)]
    #[case::bedrock_mantle_provider_reads_the_env(
        "bedrock_mantle/anthropic.claude-sonnet-5-5",
        Some("Bearer env-token")
    )]
    #[case::bedrock_invoke_signs("bedrock/anthropic.claude-sonnet-5-5", None)]
    #[tokio::test]
    async fn an_empty_api_key_reads_the_env_only_on_the_bedrock_mantle_provider(
        #[case] model: &str,
        #[case] expected: Option<&str>,
    ) {
        let call = mantle_call(model, json!({}), Some("")).await;
        assert_eq!(
            (authorization(&call), call.signer.is_some()),
            (expected, expected.is_none())
        );
    }

    #[rstest]
    #[case::listed_region_is_routing_only(
        "bedrock_mantle/us-west-2/anthropic.claude-sonnet-5-5",
        "anthropic.claude-sonnet-5-5"
    )]
    #[case::unlisted_region_stays_in_the_model(
        "bedrock_mantle/ca-west-1/anthropic.claude-sonnet-5-5",
        "ca-west-1/anthropic.claude-sonnet-5-5"
    )]
    #[case::bedrock_route_logs_its_mantle_prefix(
        "bedrock/mantle/us-west-2/anthropic.claude-sonnet-5-5",
        "mantle/us-west-2/anthropic.claude-sonnet-5-5"
    )]
    #[tokio::test]
    async fn mantle_calls_are_logged_under_the_model_python_resolves(
        #[case] model: &str,
        #[case] logged: &str,
    ) {
        let call = mantle_call(model, json!({}), Some("key")).await;
        assert_eq!(call.identity.model, logged);
    }

    #[rstest]
    #[case::bedrock_mantle("bedrock_mantle/anthropic.claude-sonnet-5-5")]
    #[case::bedrock_route("bedrock/mantle/anthropic.claude-sonnet-5-5")]
    #[tokio::test]
    async fn a_string_stream_flag_neither_streams_nor_sends_a_stream_key_to_mantle(
        #[case] model: &str,
    ) {
        let call = mantle_call(model, json!({"stream": "true"}), Some("key")).await;
        assert_eq!((call.wire.body.get("stream"), call.stream), (None, false));
    }
}
