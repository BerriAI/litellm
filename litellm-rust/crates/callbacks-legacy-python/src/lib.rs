//! The legacy `@client` wrapper as the native call sees it: litellm's `Logging` object, the
//! sync and async callback registries it fans out to, the deployment hooks, the deferred
//! proxy release and the Python handler body the route would have run. All of it sits
//! behind [`hooks`], so the driver, the routes and core never learn which Python object is
//! on the other end.
//!
//! Legacy callbacks receive the caller's own objects and may mutate them. [`PublicCall`]
//! is where those objects live.

mod adapter;
mod call;
mod callbacks;
mod deferred;
mod logger;
mod mapping;
mod operation;
mod preflight;
mod python;
mod resolve;
use litellm_host::call::Operation;
use litellm_host_python::{HookLayer, Hooks};
use pyo3::Python;

pub(crate) use adapter::LegacyLogging;
pub use call::PublicCall;
pub(crate) use callbacks::{LegacyCallbacks, is_internal_call};
pub(crate) use logger::{DeploymentHooks, PythonLogger, finalize, setup};
pub use mapping::{CallbackMapping, Dispatch, Operations, callback_mappings};
use preflight::SdkPolicy;
use resolve::Resolve;

/// The legacy hooks a native call runs, outermost first: the `@client` wrapper's `Logging`,
/// the SDK policy the wrapper applies to the keyword dict, then the edge that resolves the
/// keyword dict over the signature base (running the Python handler body's own hooks where
/// the handler has any). The keyword dict never crosses that edge: every hook inside it
/// sees one resolved dict.
pub struct LegacyLayer {
    operation: Operation,
    asynchronous: bool,
}

impl LegacyLayer {
    pub fn new(operation: Operation, asynchronous: bool) -> Self {
        Self {
            operation,
            asynchronous,
        }
    }
}

impl HookLayer<PublicCall> for LegacyLayer {
    fn layer(&self, py: Python<'_>, call: PublicCall) -> impl IntoIterator<Item = Hooks> {
        let resolve = Resolve::new(py, self.operation, &call, self.asynchronous);
        [
            Hooks::new(LegacyLogging::new(
                py,
                self.operation,
                call,
                self.asynchronous,
            )),
            Hooks::new(SdkPolicy),
            Hooks::new(resolve),
        ]
    }
}

#[cfg(test)]
mod test_support;

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use litellm_auth::SecretValue;
    use litellm_host::call::Operation;
    use litellm_host::hooks::CallHooks;
    use litellm_host::interceptors::{RequestContext, WireRequest};
    use litellm_host_python::{HookChain, HookStep};
    use pyo3::{
        prelude::*,
        types::{PyDict, PyTuple},
    };
    use rstest::rstest;
    use serde_json::json;

    use crate::test_support::{local, local_dict, namespace, preflight_lock, preflight_stubs, run};
    use crate::{LegacyLayer, PublicCall};

    /// A logger that keeps what `update_logging` and `pre_call` hand it, a Messages body
    /// whose fan-out replaces `pages`, and one listed credential the call names.
    const CALL: &CStr = c"
import asyncio

class RecordingLogger(StubLogger):
    def update_from_kwargs(self, **update):
        self.update = update

    def pre_call(self, input, api_key, additional_args):
        self.pre = additional_args

    def post_call(self, original_response, api_key, additional_args):
        pass

class Body:
    async def prepare(self, request):
        await asyncio.sleep(0)
        assert request['api_key'] == 'inherited', request
        return {**request, 'pages': replacement}

class Credential:
    credential_name = 'listed'
    credential_values = {'api_key': 'inherited'}

preflight.credential_list = lambda: [Credential()]
preflight.check_limits = lambda kwargs: preflight.checked.append(dict(kwargs))
logger = RecordingLogger()
logger.hooks = {'pre': lambda kwargs: kwargs}
original = [0]
replacement = [1]
bound = {'model': 'anthropic/claude', 'max_tokens': 8, 'api_key': None, 'body': Body()}
kwargs = {'logger': logger, 'model': 'anthropic/claude', 'litellm_credential_name': 'listed', 'pages': original}
";

    /// Drives a chain step to its value: a coroutine is run to completion, any other
    /// awaitable (the deployment hook's answer) is handed back as is.
    fn finish<T>(py: Python<'_>, chain: &mut HookChain, step: HookStep<HookChain, T>) -> T {
        match step {
            HookStep::Ready(value) => value,
            HookStep::Await(awaitable, resume) => {
                let value = if awaitable.bind(py).hasattr("__await__").unwrap() {
                    py.import("asyncio")
                        .unwrap()
                        .call_method1("run", (awaitable,))
                        .map(Bound::unbind)
                } else {
                    Ok(awaitable)
                };
                let next = resume(chain, py, value).unwrap();
                finish(py, chain, next)
            }
        }
    }

    fn prepared<'py>(
        py: Python<'py>,
        locals: &Bound<'py, PyDict>,
        operation: Operation,
        asynchronous: bool,
    ) -> (HookChain, Bound<'py, PyDict>) {
        let call = PublicCall::capture(
            &local_dict(locals, "bound"),
            &PyTuple::empty(py),
            &local_dict(locals, "kwargs"),
        )
        .unwrap();
        let arguments = call.arguments(py);
        let mut chain =
            HookChain::new().layer(py, &LegacyLayer::new(operation, asynchronous), call);
        let step = chain.prepare_arguments(py, arguments, 0.0).unwrap();
        let arguments = finish(py, &mut chain, step);
        let step = chain.prepare_request(py, arguments).unwrap();
        let resolved = finish(py, &mut chain, step);
        chain.arguments_prepared(py, &resolved).unwrap();
        (chain, resolved.into_bound(py))
    }

    fn send(py: Python<'_>, chain: &mut HookChain) {
        let context = RequestContext {
            model: "model".into(),
            custom_llm_provider: "provider".into(),
            optional_params: json!({}),
            secret_fields: vec![],
            api_key: Some(SecretValue::new("route-key")),
        };
        let wire = WireRequest {
            url: "https://provider.invalid/messages".into(),
            headers: vec![],
            body: json!({"pages": [1]}),
        };
        let step = chain
            .before_provider_request(py, Box::new(wire), &context)
            .unwrap();
        assert!(matches!(step, HookStep::Ready(_)));
    }

    #[rstest]
    fn the_resolved_call_leaves_the_layer_and_every_keyword_rewrite_reaches_the_logger() {
        let _guard = preflight_lock();
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, c"");
            preflight_stubs(py, &locals, CALL);
            let (mut chain, resolved) = prepared(py, &locals, Operation::Messages, true);
            locals.set_item("resolved", &resolved).unwrap();
            run(
                py,
                &locals,
                c"
assert resolved['max_tokens'] == 8
assert resolved['api_key'] == 'inherited'
assert resolved['pages'] is replacement
assert resolved['litellm_logging_obj'] is logger
assert preflight.checked[0]['litellm_credential_name'] == 'listed'
assert 'max_tokens' not in preflight.checked[0]
assert kwargs == {'logger': logger, 'model': 'anthropic/claude', 'litellm_credential_name': 'listed', 'pages': original}
",
            );
            send(py, &mut chain);
            run(
                py,
                &locals,
                c"
assert logger.update['kwargs']['api_key'] == 'inherited'
assert logger.update['kwargs']['pages'] is replacement
assert 'max_tokens' not in logger.update['kwargs']
assert 'body' not in logger.update['kwargs']
assert logger.pre['complete_input_dict']['pages'] is replacement
assert original == [0]
",
            );
        });
    }

    #[rstest]
    #[case::sync_ocr(Operation::Ocr, false)]
    #[case::async_completion(Operation::Completion, true)]
    fn the_logger_keeps_the_keyword_layer_while_the_route_reads_the_resolved_call(
        #[case] operation: Operation,
        #[case] asynchronous: bool,
    ) {
        let _guard = preflight_lock();
        Python::initialize();
        Python::attach(|py| {
            let locals = namespace(py, c"");
            preflight_stubs(py, &locals, CALL);
            let (mut chain, resolved) = prepared(py, &locals, operation, asynchronous);
            locals.set_item("resolved", &resolved).unwrap();
            send(py, &mut chain);
            run(
                py,
                &locals,
                c"
assert resolved['max_tokens'] == 8
assert resolved['api_key'] == 'inherited'
assert resolved['pages'] is original
assert 'max_tokens' not in logger.update['kwargs']
assert logger.update['kwargs']['api_key'] == 'inherited'
assert logger.pre['complete_input_dict']['pages'] == [1]
",
            );
            assert!(!local(&locals, "resolved").is(local_dict(&locals, "kwargs")));
        });
    }
}
