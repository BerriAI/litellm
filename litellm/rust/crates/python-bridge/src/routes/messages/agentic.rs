use std::sync::{Arc, Mutex, PoisonError};

use litellm_host::{
    hooks::CallHooks,
    interceptors::{RequestContext, WireRequest},
    lifecycle::Timing,
};
use litellm_host_python::{HookStep, PythonOwned, PythonRuntime, json_to_py, to_py};
use litellm_llms::bedrock_mantle::messages::transformation::mantle_capability_lookup;
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};
use serde_json::Value;

use super::host::ROUTE_HOST_MODULE;

/// What the provider was sent, which Python's agentic hooks read back as the handler's own
/// model, provider and request params.
struct Sent {
    request: Py<PyDict>,
    hooked: bool,
    body: Py<PyAny>,
    model: String,
    provider: String,
    requested: Py<PyAny>,
}

/// Python's Messages handler hands the provider's answer to the callbacks overriding
/// `async_should_run_agentic_loop` before `@client` sees it, and wraps a stream so they run
/// once it ends. This hook asks the route host to do both at the same points.
pub(super) type HookedRequest = Arc<Mutex<Option<Py<PyDict>>>>;

pub(super) struct AgenticLoop {
    asynchronous: bool,
    request: Option<Py<PyDict>>,
    hooked: HookedRequest,
    decided: bool,
    sent: Option<Sent>,
}

impl AgenticLoop {
    pub(super) fn new(asynchronous: bool, hooked: HookedRequest) -> Self {
        Self {
            asynchronous,
            request: None,
            hooked,
            decided: false,
            sent: None,
        }
    }

    fn hooked_request(&self, py: Python<'_>) -> Option<Py<PyDict>> {
        self.hooked
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .map(|request| request.clone_ref(py))
    }

    fn host_call<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        subject: &Bound<'py, PyAny>,
        sent: &Sent,
    ) -> PyResult<Bound<'py, PyAny>> {
        py.import(ROUTE_HOST_MODULE)?.getattr(name)?.call1((
            subject,
            sent.request.bind(py),
            sent.body.bind(py),
            &sent.model,
            &sent.provider,
            sent.requested.bind(py),
            sent.hooked,
        ))
    }

    fn resume_response(
        &mut self,
        _: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<HookStep<Self, Py<PyAny>>> {
        result.map(HookStep::Ready)
    }
}

impl CallHooks<PythonRuntime> for AgenticLoop {
    fn arguments_prepared(&mut self, py: Python<'_>, arguments: &Py<PyDict>) -> PyResult<()> {
        if self.asynchronous {
            self.request = Some(arguments.clone_ref(py));
        }
        Ok(())
    }

    fn before_provider_request(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<HookStep<Self, Box<WireRequest>>> {
        if self.decided {
            return Ok(HookStep::Ready(wire));
        }
        let Some(caller) = &self.request else {
            return Ok(HookStep::Ready(wire));
        };
        self.decided = true;
        let hooked = self.hooked_request(py);
        let hooked_flag = hooked.is_some();
        let request = hooked.unwrap_or_else(|| caller.clone_ref(py));
        let module = py.import(ROUTE_HOST_MODULE)?;
        if module
            .getattr("runs_agentic_hooks")?
            .call1((request.bind(py),))?
            .extract::<bool>()?
        {
            module
                .getattr("record_agentic_loop")?
                .call1((request.bind(py), &context.custom_llm_provider))?;
            self.sent = Some(Sent {
                request,
                hooked: hooked_flag,
                body: to_py(py, &hooked_body(&wire.body, context))?,
                model: context.model.clone(),
                provider: context.custom_llm_provider.clone(),
                requested: json_to_py(py, &context.optional_params)?.unbind(),
            });
        }
        Ok(HookStep::Ready(wire))
    }

    fn transform_response(
        &mut self,
        py: Python<'_>,
        response: Py<PyAny>,
        _: Timing,
    ) -> PyResult<HookStep<Self, Py<PyAny>>> {
        let Some(sent) = &self.sent else {
            return Ok(HookStep::Ready(response));
        };
        let awaitable = self.host_call(py, "agentic_response", response.bind(py), sent)?;
        Ok(HookStep::Await(awaitable.unbind(), Self::resume_response))
    }

    fn on_stream_open(&mut self, py: Python<'_>, head: &Py<PyAny>) -> PyResult<()> {
        if let Some(sent) = &self.sent {
            self.host_call(py, "agentic_stream", head.bind(py), sent)?;
        }
        Ok(())
    }
}

/// The body Python's agentic hooks read as the call's params. Python hands them the params from
/// before its Mantle transform, so they keep the `stream: false` the Mantle wire body leaves out.
fn hooked_body(body: &Value, context: &RequestContext) -> Value {
    let requested_stream = context.optional_params.get("stream");
    match (body, requested_stream) {
        (Value::Object(fields), Some(stream))
            if !fields.contains_key("stream")
                && mantle_capability_lookup(&context.custom_llm_provider, &context.model)
                    .is_some() =>
        {
            let mut fields = fields.clone();
            fields.insert("stream".to_string(), stream.clone());
            Value::Object(fields)
        }
        _ => body.clone(),
    }
}

impl PythonOwned for AgenticLoop {
    fn close(&mut self, _: Python<'_>) {
        self.request = None;
        self.sent = None;
        self.hooked
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.request)?;
        if let Some(sent) = &self.sent {
            visit.call(&sent.request)?;
            visit.call(&sent.body)?;
            visit.call(&sent.requested)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn context(provider: &str, model: &str) -> RequestContext {
        RequestContext {
            model: model.to_string(),
            custom_llm_provider: provider.to_string(),
            optional_params: json!({"max_tokens": 8, "stream": false}),
            secret_fields: Vec::new(),
            api_key: None,
        }
    }

    #[rstest]
    #[case::bedrock_mantle("bedrock_mantle", "anthropic.claude-sonnet-5-5", Some(json!(false)))]
    #[case::bedrock_mantle_route("bedrock", "mantle/anthropic.claude-sonnet-5-5", Some(json!(false)))]
    #[case::bedrock_invoke("bedrock", "anthropic.claude-sonnet-5-5", None)]
    fn agentic_hooks_read_the_requested_stream_flag_on_mantle_routes(
        #[case] provider: &str,
        #[case] model: &str,
        #[case] expected: Option<Value>,
    ) {
        let body = hooked_body(&json!({"max_tokens": 8}), &context(provider, model));
        assert_eq!(body.get("stream").cloned(), expected);
    }
}
