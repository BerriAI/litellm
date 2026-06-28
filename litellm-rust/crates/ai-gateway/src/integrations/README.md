# LiteLLM Rust integrations

This directory contains Rust-native equivalents of LiteLLM integration hooks.
The first supported surfaces are terminal custom loggers and pre/during-call
custom guardrails.

## File layout

Every integration is a folder:

- `mod.rs` contains the implementation, trait, runner, or adapter
- `types.rs` contains the integration-local request, response, error, and future
  types

Do not add new flat integration files such as `custom_logger.rs`. Shared wire
contracts that are used by multiple integrations can stay in
`integrations/types.rs`.

Call ordering and lifecycle timing live in `litellm-core/src/call_lifecycle`.
Call-type modules, such as OCR, adapt their request and response shapes into
that generic lifecycle runner.

## CustomLogger

Implement `CustomLogger` when Rust code needs to observe terminal success or
failure events. Method names intentionally match Python `CustomLogger` names.

```rust
use litellm_ai_gateway::integrations::custom_logger::{
    CallbackTiming, CallbackValue, CustomLogger, LogFuture, ModelCallDetails,
};

struct RecordingLogger;

impl CustomLogger for RecordingLogger {
    fn async_log_success_event<'a>(
        &'a self,
        model_call_details: &'a ModelCallDetails,
        response_obj: &'a CallbackValue,
        timing: CallbackTiming,
    ) -> LogFuture<'a> {
        Box::pin(async move {
            let model = &model_call_details.model;
            let provider = &model_call_details.custom_llm_provider;
            let call_type = model_call_details.call_type.to_string();
            let request_id = model_call_details.request_id.as_deref();
            let response_object = &response_obj.object;
            let duration = timing.end_time - timing.start_time;
            let standard_payload = model_call_details.standard_logging_payload.as_ref();

            Ok(())
        })
    }

    fn async_log_failure_event<'a>(
        &'a self,
        model_call_details: &'a ModelCallDetails,
        response_obj: Option<&'a CallbackValue>,
        timing: CallbackTiming,
    ) -> LogFuture<'a> {
        Box::pin(async move {
            let error = model_call_details.failure_error.as_ref();
            let response_object = response_obj.map(|value| value.object.as_str());
            let duration = timing.end_time - timing.start_time;

            Ok(())
        })
    }
}
```

Use `CustomLoggerRunner` to fan out terminal events to configured loggers. The
runner is a no-op when no loggers are configured, which is the expected fast
path for requests without callbacks.

## CustomGuardrail

Implement `CustomGuardrail` when Rust code needs to run pre-call or native
during-call checks. Method names intentionally match Python `CustomGuardrail`
entrypoints inherited from Python `CustomLogger`.

```rust
use litellm_ai_gateway::integrations::custom_guardrail::{
    CustomGuardrail, GuardrailContext, GuardrailDecision, GuardrailEventHook,
    GuardrailFuture, GuardrailRequest,
};

struct BlocklistedPromptGuardrail;

impl CustomGuardrail for BlocklistedPromptGuardrail {
    fn guardrail_name(&self) -> &str {
        "blocklisted-prompt"
    }

    fn supported_event_hooks(&self) -> &[GuardrailEventHook] {
        &[GuardrailEventHook::PreCall]
    }

    fn async_pre_call_hook<'a>(
        &'a self,
        _context: &'a GuardrailContext,
        request: GuardrailRequest,
    ) -> GuardrailFuture<'a> {
        Box::pin(async move {
            if request.data.to_string().contains("blocked phrase") {
                return Ok(GuardrailDecision::Block(
                    litellm_ai_gateway::integrations::custom_guardrail::GuardrailError::blocked(
                        "blocked phrase detected",
                    ),
                ));
            }
            Ok(GuardrailDecision::Allow(request))
        })
    }
}
```

Use `CustomGuardrailRunner::run_pre_call` for `pre_call` guardrails and
`CustomGuardrailRunner::run_during_call` for `during_call` guardrails. A
`GuardrailDecision::Mask` continues with modified request data.
`GuardrailDecision::Block` short-circuits the provider call.

## Current boundary

These are Rust-only primitives. Python callback and guardrail adapters are a
separate layer that should implement these Rust traits instead of changing the
runner interfaces.
