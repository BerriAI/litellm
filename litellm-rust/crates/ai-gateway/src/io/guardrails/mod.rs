//! End-to-end guardrail execution in Rust.
//!
//! `run_guardrail` is the single entry point the Python bridge calls: it builds
//! the provider config from the raw params (resolving secrets and files), runs
//! the provider, and returns the verdict. Python passes params and inputs in and
//! gets a verdict out; it never receives a config to round-trip back for the
//! HTTP call.

mod config_builder;
mod http;
mod provider;
mod providers;

use std::time::Instant;

use litellm_core::guardrails::{
    GuardrailInput, GuardrailOutcome, InputType, ProviderError, RequestContext, Verdict,
};

pub use config_builder::Unsupported;

/// Build and run a guardrail end to end.
///
/// `Err(Unsupported)` means the Rust engine cannot handle this config and the
/// caller should fall back to the Python implementation. Provider failures
/// (network, timeout, bad upstream) are folded into a fail-closed [`Verdict::Block`]
/// outcome rather than surfaced as `Unsupported`, so a reachable-but-erroring
/// provider does not silently disable the guardrail.
pub fn run_guardrail(
    guardrail_type: &str,
    params: &serde_json::Value,
    input: &GuardrailInput,
    input_type: InputType,
    ctx: &RequestContext,
) -> Result<GuardrailOutcome, Unsupported> {
    let config = config_builder::build_config(guardrail_type, params)?;
    let provider = providers::build(config)
        .map_err(|e| Unsupported(format!("could not build provider: {e}")))?;

    let start = Instant::now();
    match provider.apply(input, input_type, ctx) {
        Ok(outcome) => Ok(outcome),
        Err(err) => Ok(fail_closed(err, start)),
    }
}

fn fail_closed(err: ProviderError, start: Instant) -> GuardrailOutcome {
    GuardrailOutcome {
        verdict: Verdict::Block {
            violation_message: format!("Guardrail unavailable: {err}"),
            detections: vec![],
        },
        provider_response: serde_json::Value::Null,
        duration_ms: start.elapsed().as_millis() as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn one_text(text: &str) -> GuardrailInput {
        GuardrailInput {
            texts: vec![text.to_owned()],
            ..Default::default()
        }
    }

    async fn moderation_server(flagged: bool) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{
                    "flagged": flagged,
                    "categories": {"violence": flagged},
                    "category_scores": {"violence": 0.97},
                }]
            })))
            .mount(&server)
            .await;
        server
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn openai_moderation_blocks_flagged_content() {
        let server = moderation_server(true).await;
        let params = serde_json::json!({"api_key": "sk-test", "api_base": server.uri()});
        let outcome = tokio::task::spawn_blocking(move || {
            run_guardrail(
                "openai_moderation",
                &params,
                &one_text("bad"),
                InputType::Request,
                &RequestContext::default(),
            )
        })
        .await
        .unwrap()
        .unwrap();
        match outcome.verdict {
            Verdict::Block { detections, .. } => {
                assert!(detections.iter().any(|d| d.category == "violence"));
            }
            other => panic!("expected block, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn openai_moderation_passes_clean_content() {
        let server = moderation_server(false).await;
        let params = serde_json::json!({"api_key": "sk-test", "api_base": server.uri()});
        let outcome = tokio::task::spawn_blocking(move || {
            run_guardrail(
                "openai_moderation",
                &params,
                &one_text("hello"),
                InputType::Request,
                &RequestContext::default(),
            )
        })
        .await
        .unwrap()
        .unwrap();
        assert_eq!(outcome.verdict, Verdict::Pass);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn upstream_error_fails_closed() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderations"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let params = serde_json::json!({"api_key": "sk-test", "api_base": server.uri()});
        let outcome = tokio::task::spawn_blocking(move || {
            run_guardrail(
                "openai_moderation",
                &params,
                &one_text("hi"),
                InputType::Request,
                &RequestContext::default(),
            )
        })
        .await
        .unwrap()
        .unwrap();
        assert!(matches!(outcome.verdict, Verdict::Block { .. }));
    }

    #[test]
    fn unsupported_type_signals_python_fallback() {
        let result = run_guardrail(
            "lakera_v2",
            &serde_json::json!({}),
            &one_text("hi"),
            InputType::Request,
            &RequestContext::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn local_pii_masks_without_network() {
        let outcome = run_guardrail(
            "local_pii",
            &serde_json::json!({}),
            &one_text("email jane.doe@example.com"),
            InputType::Request,
            &RequestContext::default(),
        )
        .unwrap();
        match outcome.verdict {
            Verdict::Mask { texts, .. } => assert_eq!(texts[0], "email <EMAIL_ADDRESS>"),
            other => panic!("expected mask, got {other:?}"),
        }
    }
}
