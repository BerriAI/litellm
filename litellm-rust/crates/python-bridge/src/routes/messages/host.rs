use crate::cache::{CacheCall, PythonCache, Selection};
use litellm_host_python::{PythonHostCalls, PythonOwned};

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use bytes::Bytes;
use litellm_host::protocol::Reply;
use litellm_host_python::{InvokeError, PythonBinding, effective_py_args, from_py, to_py};
use litellm_inference_messages::{
    Error, MessagesCall, MessagesSettings, MessagesShaping, litellm_params, messages_body,
    route::{Messages, MessagesStreamHead},
};
use litellm_llms::base_llm::messages::context::MessagesModelCapabilities;
use litellm_llms_types::headers::ProviderSpecificHeaders;
use litellm_router_types::LitellmParams;
use pyo3::{
    exceptions::PyException,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    sync::PyOnceLock,
    types::{PyBytes, PyDict},
};
use serde_json::{Map, Value};

use super::agentic::HookedRequest;
use super::safeguards::{SafeguardsCall, SafeguardsHost};
use super::transport::{HostedTransport, TransportCall, python_failure};
use crate::{
    errors::{RequestMarker, route_failure},
    marshal::{
        ArgumentError, optional_timeout, project_optional_fields, public_response,
        python_timeout_seconds,
    },
    python_settings::PythonSettings,
    routes::RequestView,
};

pub(super) const ROUTE_HOST_MODULE: &str = "litellm.rust_bridge.host.routes.messages";
const MODEL_CAPABILITIES_MODULE: &str = "litellm.rust_bridge.host.model_capabilities";

fn merge_headers(
    forwarded: Option<Map<String, Value>>,
    extra_headers: Option<Map<String, Value>>,
) -> Option<Map<String, Value>> {
    let merged = litellm_inference_messages::merge_messages_header_sources(
        forwarded.into_iter().chain(extra_headers),
    );
    (!merged.is_empty()).then_some(merged)
}

const FEDERATING_PROVIDER: &str = "anthropic";

fn workload_identity_fields(py: Python<'_>) -> PyResult<&'static [String]> {
    static FIELDS: PyOnceLock<Vec<String>> = PyOnceLock::new();
    FIELDS
        .get_or_try_init(py, || {
            py.import(ROUTE_HOST_MODULE)?
                .getattr("WORKLOAD_IDENTITY_FIELDS")?
                .extract()
        })
        .map(Vec::as_slice)
}

fn module_global<'py>(py: Python<'py>, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    PythonSettings::ProviderDefaults
        .read_or_unset(py)?
        .map_or(Ok(None), |defaults| defaults.attribute(name))
}

/// The caller's litellm params, read from the kwargs by the names the typed params declare,
/// so a key the configs do not read is never converted; a spec whose spellings the call all
/// leaves out falls back to the module global it names, which stays the host's concern.
fn project_litellm_params<'a, 'py>(
    names: impl IntoIterator<Item = &'a str>,
    argument: impl Fn(&str) -> PyResult<Option<Bound<'py, PyAny>>>,
    global: impl Fn(&str) -> PyResult<Option<Bound<'py, PyAny>>>,
) -> PyResult<Result<LitellmParams, Error>> {
    let fields = project_optional_fields(
        LitellmParams::fields()
            .map(|name| -> &'a str { name })
            .chain(names),
        &argument,
    )?;
    let globals = LitellmParams::specs()
        .filter(|spec| !spec.wire.iter().any(|name| fields.contains_key(*name)))
        .filter_map(|spec| spec.module_global);
    let folded = project_optional_fields(globals, &global)?;
    Ok(litellm_params(fields.into_iter().chain(folded).collect()))
}

pub(super) struct MessagesHosted;

impl litellm_host::protocol::Protocol for MessagesHosted {
    type Request = (MessagesCall, Selection, bool);
    type Response = <Messages as litellm_host::protocol::Protocol>::Response;
    type Error = Error;
    type HostCall = MessagesHostCall;
    type Chunk = Bytes;
    type StreamHead = MessagesStreamHead;
}

pub(super) enum MessagesHostCall {
    Answer { stream: bool, reply: Reply<()> },
    Cache(CacheCall),
    PreRequest(Reply<Result<MessagesCall, Error>>),
    Safeguards(SafeguardsCall),
    Transport(TransportCall),
}

impl From<CacheCall> for MessagesHostCall {
    fn from(call: CacheCall) -> Self {
        Self::Cache(call)
    }
}

impl From<TransportCall> for MessagesHostCall {
    fn from(call: TransportCall) -> Self {
        Self::Transport(call)
    }
}

#[derive(serde::Deserialize)]
struct HostSettings {
    #[serde(flatten)]
    settings: MessagesSettings,
    model: String,
    stream_cost_model: Option<String>,
    extra_headers: Option<Map<String, Value>>,
}

/// The Python side of the Messages route: projects the prepared arguments and builds the
/// public response, chunks and exceptions.
pub(super) struct MessagesPythonHost {
    view: RequestView,
    cache: PythonCache,
    transport: HostedTransport,
    safeguards: SafeguardsHost,
    classifies: Arc<AtomicBool>,
    awaiting_safeguards: bool,
    stream_cost_model: Option<String>,
    asynchronous: bool,
    pre_request: Option<Reply<Result<MessagesCall, Error>>>,
    hooked: bool,
    hooked_request: HookedRequest,
    requested_stream: bool,
    interceptor: Option<Py<PyAny>>,
    pending_answer: Option<Reply<()>>,
    answer: Option<Py<PyAny>>,
    streamed_answer: bool,
}

impl MessagesPythonHost {
    pub(super) fn new(
        view: RequestView,
        asynchronous: bool,
        hooked_request: HookedRequest,
        classifies: Arc<AtomicBool>,
    ) -> Self {
        Self {
            view,
            cache: PythonCache::new(asynchronous),
            transport: HostedTransport::default(),
            safeguards: SafeguardsHost::default(),
            classifies,
            awaiting_safeguards: false,
            stream_cost_model: None,
            asynchronous,
            pre_request: None,
            hooked: false,
            hooked_request,
            requested_stream: false,
            interceptor: None,
            pending_answer: None,
            answer: None,
            streamed_answer: false,
        }
    }

    fn interceptor(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if !self.asynchronous {
            return Ok(None);
        }
        let interceptor = py
            .import(ROUTE_HOST_MODULE)?
            .getattr("interceptor")?
            .call1((self.view.request(py)?,))?;
        Ok((!interceptor.is_none()).then(|| interceptor.unbind()))
    }

    fn begin_answer(
        &mut self,
        py: Python<'_>,
        stream: bool,
        reply: Reply<()>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let interceptor = self
            .interceptor
            .as_ref()
            .ok_or(litellm_host_python::Error::MissingState)?;
        let pending = py.import(ROUTE_HOST_MODULE)?.getattr("answer")?.call1((
            interceptor,
            self.view.request(py)?,
            stream,
            self.hooked,
        ))?;
        self.pending_answer = Some(reply);
        self.streamed_answer = stream;
        Ok(Some(pending.unbind()))
    }

    fn host_settings(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> PyResult<HostSettings> {
        from_py(
            &py.import(ROUTE_HOST_MODULE)?
                .getattr("settings")?
                .call1((arguments, self.view.request(py)?))?,
        )
    }

    fn body_params(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> PyResult<Map<String, Value>> {
        let request = effective_py_args(self.view.request(py)?, arguments)?;
        from_py(
            &py.import(ROUTE_HOST_MODULE)?
                .getattr("body_params")?
                .call1((request, self.asynchronous, self.hooked))?,
        )
    }

    fn projection(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
        settings: HostSettings,
    ) -> PyResult<Result<MessagesCall, Error>> {
        let argument = |name: &str| self.view.argument(py, arguments, name);
        let string = |name: &str| -> PyResult<Option<String>> {
            argument(name)?.map(|value| value.extract()).transpose()
        };
        let model = settings.model;
        let messages = argument("messages")?.ok_or(ArgumentError::Required("messages"))?;
        let mut fields = self.body_params(py, arguments)?;
        let prepared = fields.remove("messages");
        let presanitized = prepared.is_some();
        let messages = match prepared {
            Some(messages) => messages,
            None => from_py(&messages)?,
        };
        let body = [
            ("model".to_string(), Value::String(model.clone())),
            ("messages".to_string(), messages),
        ]
        .into_iter()
        .chain(fields)
        .collect::<Map<String, Value>>();
        let timeout = argument("timeout")?
            .map(|value| python_timeout_seconds(py, value.unbind()))
            .transpose()?
            .flatten();
        let api_key = string("api_key")?;
        let api_base = string("api_base")?;
        // Python's handler resolves a bare model name (`claude-sonnet-4-5`) through
        // `get_llm_provider`'s model lists, which the native prefix lookup does not carry.
        let custom_llm_provider = match string("custom_llm_provider")? {
            Some(provider) => Some(provider),
            None => py
                .import(ROUTE_HOST_MODULE)?
                .getattr("inferred_provider")?
                .call1((&model, api_base.as_deref()))?
                .extract()?,
        };
        let shaping = MessagesShaping {
            capabilities: self.capabilities(py, &model, custom_llm_provider.as_deref())?,
            settings: MessagesSettings {
                whole_sse_frames: settings.stream_cost_model.is_some(),
                presanitized,
                intercepted: self.interceptor.is_some(),
                ..settings.settings
            },
        };
        let extra_headers = match settings.extra_headers {
            Some(headers) => Some(headers),
            None => self.merged_headers(py, arguments)?,
        };
        let provider_specific_header = self.provider_specific_header(py, arguments)?;
        let federation_fields: &[String] = match custom_llm_provider.as_deref() {
            Some(FEDERATING_PROVIDER) => workload_identity_fields(py)?,
            _ => &[],
        };
        let litellm_params = project_litellm_params(
            federation_fields.iter().map(String::as_str),
            argument,
            |name| module_global(py, name),
        )?;
        Ok(messages_body(body).and_then(|body| {
            Ok(MessagesCall {
                body,
                api_key,
                api_base,
                extra_headers,
                provider_specific_header,
                custom_llm_provider,
                litellm_params: litellm_params?,
                timeout: optional_timeout(timeout),
                shaping,
            })
        }))
    }

    fn merged_headers(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> PyResult<Option<Map<String, Value>>> {
        let mapping = |name: &str| -> PyResult<Option<Map<String, Value>>> {
            self.view
                .argument(py, arguments, name)?
                .map(|value| from_py(&value))
                .transpose()
        };
        Ok(merge_headers(
            mapping("headers")?,
            mapping("extra_headers")?,
        ))
    }

    fn provider_specific_header(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> PyResult<Option<ProviderSpecificHeaders>> {
        self.view
            .argument(py, arguments, "provider_specific_header")?
            .map(|value| from_py(&value))
            .transpose()
    }

    fn capabilities(
        &self,
        py: Python<'_>,
        model: &str,
        custom_llm_provider: Option<&str>,
    ) -> PyResult<MessagesModelCapabilities> {
        from_py(
            &py.import(MODEL_CAPABILITIES_MODULE)?
                .getattr("anthropic_model_capabilities")?
                .call1((model, custom_llm_provider))?,
        )
    }

    fn provider(&self, py: Python<'_>) -> String {
        self.view
            .request(py)
            .and_then(|request| request.get_item("custom_llm_provider"))
            .ok()
            .flatten()
            .and_then(|value| value.extract::<Option<String>>().ok().flatten())
            .unwrap_or_else(|| "anthropic".into())
    }

    fn map_transport_failure(&self, py: Python<'_>, error: PyErr) -> PyResult<PyErr> {
        let mapped = py
            .import(ROUTE_HOST_MODULE)?
            .getattr("map_transport_failure")?
            .call1((error.value(py), self.view.request(py)?, self.provider(py)))?;
        Ok(PyErr::from_value(mapped))
    }

    fn map_failure(&self, py: Python<'_>, error: PyErr) -> PyErr {
        if !error.is_instance_of::<PyException>(py) {
            return error;
        }
        let mapped = py
            .import(ROUTE_HOST_MODULE)
            .and_then(|module| module.getattr("map_failure"))
            .and_then(|map| map.call1((error.value(py), self.view.request(py)?, self.provider(py))))
            .and_then(|mapped| {
                mapped
                    .extract::<Py<pyo3::exceptions::PyBaseException>>()
                    .map_err(PyErr::from)
            });
        match mapped {
            Ok(mapped) => PyErr::from_value(mapped.into_bound(py).into_any()),
            Err(_) => error,
        }
    }
}

impl MessagesPythonHost {
    fn decode_call(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<MessagesCall, InvokeError<Error>> {
        let settings = self
            .host_settings(py, arguments)
            .map_err(InvokeError::Python)?;
        self.stream_cost_model = settings.stream_cost_model.clone();
        self.projection(py, arguments, settings)
            .map_err(|error| InvokeError::Python(self.map_failure(py, error)))?
            .map_err(InvokeError::Native)
    }

    fn begin_pre_request(
        &mut self,
        py: Python<'_>,
        reply: Reply<Result<MessagesCall, Error>>,
    ) -> PyResult<Py<PyAny>> {
        let hooks = py
            .import(ROUTE_HOST_MODULE)?
            .getattr("pre_request_hooks")?
            .call1((self.view.request(py)?,))?;
        self.pre_request = Some(reply);
        Ok(hooks.unbind())
    }

    fn resume_pre_request(
        &mut self,
        py: Python<'_>,
        reply: Reply<Result<MessagesCall, Error>>,
        result: PyResult<Py<PyAny>>,
    ) -> Result<(), InvokeError<Error>> {
        let hooked = result?;
        let arguments = hooked
            .bind(py)
            .cast::<PyDict>()
            .map_err(|error| InvokeError::Python(error.into()))?;
        self.view
            .resolve(py, arguments)
            .map_err(InvokeError::Python)?;
        self.hooked = true;
        let effective = self
            .view
            .request(py)
            .and_then(|request| effective_py_args(request, arguments))
            .map_err(InvokeError::Python)?;
        self.safeguards
            .rebind(py, &effective)
            .map_err(InvokeError::Python)?;
        *self
            .hooked_request
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(arguments.clone().unbind());
        self.interceptor = self.interceptor(py).map_err(InvokeError::Python)?;
        match self.decode_call(py, arguments) {
            Ok(call) => reply.send(Ok(call)),
            Err(InvokeError::Native(error)) => reply.send(Err(error)),
            Err(error) => return Err(error),
        }
        Ok(())
    }
}

impl PythonBinding for MessagesPythonHost {
    type Protocol = MessagesHosted;
    type Failure = PyErr;

    fn decode_request(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<(MessagesCall, Selection, bool), InvokeError<Error>> {
        self.view
            .resolve(py, arguments)
            .map_err(InvokeError::Python)?;
        self.interceptor = self.interceptor(py).map_err(InvokeError::Python)?;
        let function = py
            .import(ROUTE_HOST_MODULE)
            .and_then(|module| module.call_method0("cached_function"))
            .map_err(InvokeError::Python)?;
        let selection = crate::cache::configure(&mut self.cache, py, arguments, &function)
            .map_err(InvokeError::Python)?;
        let effective = self
            .view
            .request(py)
            .and_then(|request| effective_py_args(request, arguments))
            .map_err(InvokeError::Python)?;
        let classifies = self
            .safeguards
            .configure(py, &effective)
            .map_err(InvokeError::Python)?;
        self.classifies.store(classifies, Ordering::Release);
        self.requested_stream = arguments
            .get_item("stream")
            .map_err(InvokeError::Python)?
            .is_some_and(|stream| stream.is(pyo3::types::PyBool::new(py, true)));
        let request = self.decode_call(py, arguments)?;
        let hosted = self
            .transport
            .select(py, self.view.request(py).map_err(InvokeError::Python)?)
            .map_err(InvokeError::Python)?;
        Ok((request, selection, hosted))
    }

    fn encode_response(
        &mut self,
        py: Python<'_>,
        response: litellm_inference_messages::MessagesBody,
    ) -> PyResult<Py<PyAny>> {
        if let Some(answer) = self.answer.take() {
            return Ok(answer);
        }
        let public = public_response(py, ROUTE_HOST_MODULE, &response)?;
        self.safeguards.published(py, public)
    }

    fn deliver_response(&mut self, py: Python<'_>, response: Py<PyAny>) -> PyResult<Py<PyAny>> {
        if std::mem::take(&mut self.streamed_answer) {
            return self.cache.wrap_stream(py, response);
        }
        if !self.requested_stream {
            self.cache.store_response(py, response.bind(py))?;
        }
        Ok(response)
    }

    fn encode_stream_head(
        &mut self,
        py: Python<'_>,
        head: MessagesStreamHead,
    ) -> PyResult<Py<PyAny>> {
        py.import(ROUTE_HOST_MODULE)?
            .getattr("stream_hidden_params")?
            .call1((to_py(py, &head.headers)?,))
            .map(Bound::unbind)
    }

    fn encode_chunk(&mut self, py: Python<'_>, chunk: Bytes) -> PyResult<Py<PyAny>> {
        let frames = PyBytes::new(py, &chunk).into_any();
        let Some(model) = &self.stream_cost_model else {
            return Ok(frames.unbind());
        };
        py.import(ROUTE_HOST_MODULE)?
            .getattr("inject_stream_cost")?
            .call1((frames, model, self.view.request(py)?))
            .map(Bound::unbind)
    }

    fn map_error(&self, py: Python<'_>, error: Error) -> PyResult<PyErr> {
        if let Some((raised, raw)) = python_failure(py, &error) {
            return if raw {
                Ok(raised)
            } else {
                self.map_transport_failure(py, raised)
            };
        }
        route_failure(py, error, RequestMarker::Messages, |native| {
            Ok(self.map_failure(py, native))
        })
    }

    fn host_error(error: &PyErr) -> Error {
        crate::routes::inference::host_call_error(error)
    }
}

impl PythonHostCalls<MessagesHosted> for MessagesPythonHost {
    fn handle_host_call(
        &mut self,
        py: Python<'_>,
        op: MessagesHostCall,
    ) -> Result<(), InvokeError<Error>> {
        self.begin_host_call(py, op).map(|_| ())
    }

    fn begin_host_call(
        &mut self,
        py: Python<'_>,
        op: MessagesHostCall,
    ) -> Result<Option<Py<PyAny>>, InvokeError<Error>> {
        self.awaiting_safeguards = matches!(op, MessagesHostCall::Safeguards(_));
        match op {
            MessagesHostCall::Cache(op) => self.cache.begin(py, op),
            MessagesHostCall::Transport(op) => {
                let request = self.view.request(py).map_err(InvokeError::Python)?;
                self.transport.begin(py, op, request)
            }
            MessagesHostCall::PreRequest(reply) => self.begin_pre_request(py, reply).map(Some),
            MessagesHostCall::Safeguards(op) => self.safeguards.begin(py, op).map(Some),
            MessagesHostCall::Answer { stream, reply } => self.begin_answer(py, stream, reply),
        }
        .map_err(InvokeError::Python)
    }

    fn resume_host_call(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> Result<Option<Py<PyAny>>, InvokeError<Error>> {
        if let Some(reply) = self.pre_request.take() {
            return self.resume_pre_request(py, reply, result).map(|()| None);
        }
        if let Some(reply) = self.pending_answer.take() {
            self.answer = Some(result?);
            reply.send(());
            return Ok(None);
        }
        if std::mem::take(&mut self.awaiting_safeguards) {
            return self
                .safeguards
                .resume(py, result)
                .map(|()| None)
                .map_err(InvokeError::Python);
        }
        if self.transport.pending() {
            return self
                .transport
                .resume(py, result)
                .map_err(InvokeError::Python);
        }
        self.cache.resume(py, result).map_err(InvokeError::Python)
    }
}

impl PythonOwned for MessagesPythonHost {
    fn close(&mut self, _: Python<'_>) {
        self.view.close();
        self.cache.close();
        self.transport.close();
        self.pre_request = None;
        self.safeguards.close();
        self.interceptor = None;
        self.pending_answer = None;
        self.answer = None;
        self.streamed_answer = false;
    }
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.view.traverse(visit)?;
        self.cache.traverse(visit)?;
        self.transport.traverse(visit)?;
        self.safeguards.traverse(visit)?;
        visit.call(&self.interceptor)?;
        visit.call(&self.answer)
    }
}

#[cfg(test)]
mod tests {
    use litellm_host_python::present;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn map(value: Value) -> Map<String, Value> {
        serde_json::from_value(value).unwrap()
    }

    #[rstest]
    #[case::extra_over_forwarded(
        Some(json!({"X-Priority": "forwarded", "X-Forwarded-Only": "keep"})),
        Some(json!({"X-Priority": "extra", "X-Extra-Only": "also-keep"})),
        Some(json!({"X-Priority": "extra", "X-Forwarded-Only": "keep", "X-Extra-Only": "also-keep"})),
    )]
    #[case::only_forwarded(Some(json!({"X-Forwarded": "yes"})), None, Some(json!({"X-Forwarded": "yes"})))]
    #[case::only_extra_headers(
        None,
        Some(json!({"X-Custom-Header": "from-kwargs", "X-Auth-Token": "token123"})),
        Some(json!({"X-Custom-Header": "from-kwargs", "X-Auth-Token": "token123"})),
    )]
    #[case::nothing(None, Some(json!({})), None)]
    fn headers_merge_forwarded_then_extra(
        #[case] forwarded: Option<Value>,
        #[case] extra_headers: Option<Value>,
        #[case] expected: Option<Value>,
    ) {
        assert_eq!(
            merge_headers(forwarded.map(map), extra_headers.map(map)),
            expected.map(map)
        );
    }

    #[rstest]
    #[case::model_and_credentials_are_projected(
        json!({"model": "m", "api_key": "k", "api_base": "b", "api_version": "v", "timeout": 5, "messages": []}),
        Ok(json!({"model": "m", "api_key": "k", "api_base": "b", "api_version": "v"})),
    )]
    #[case::aws_keys_are_projected(
        json!({"model": "m", "aws_region_name": "eu-central-1", "aws_access_key_id": "AKIA", "messages": [{"role": "user"}]}),
        Ok(json!({"model": "m", "aws_region_name": "eu-central-1", "aws_access_key_id": "AKIA"})),
    )]
    #[case::vertex_keys_are_projected_in_both_spellings(
        json!({"model": "m", "vertex_project": "p", "vertex_ai_location": "us-east5", "vertex_credentials": {"type": "service_account"}}),
        Ok(json!({"model": "m", "vertex_project": "p", "vertex_ai_location": "us-east5", "vertex_credentials": {"type": "service_account"}})),
    )]
    #[case::an_explicit_none_is_absent(json!({"model": "m", "aws_region_name": null}), Ok(json!({"model": "m"})))]
    #[case::a_wrong_type_is_a_request_error(json!({"model": "m", "aws_region_name": 7}), Err(()))]
    #[case::a_missing_model_is_a_request_error(json!({"aws_region_name": "eu-central-1"}), Err(()))]
    fn litellm_params_are_projected_by_their_declared_names(
        #[case] kwargs: Value,
        #[case] expected: Result<Value, ()>,
    ) {
        assert_projection(kwargs, json!({}), expected);
    }

    #[rstest]
    #[case::global_fills_a_missing_vertex_project(
        json!({"model": "m"}),
        json!({"vertex_project": "from-global", "vertex_location": "us-east5"}),
        Ok(json!({"model": "m", "vertex_project": "from-global", "vertex_location": "us-east5"})),
    )]
    #[case::the_call_wins_over_the_global(
        json!({"model": "m", "vertex_project": "from-call"}),
        json!({"vertex_project": "from-global"}),
        Ok(json!({"model": "m", "vertex_project": "from-call"})),
    )]
    #[case::an_explicit_none_in_the_call_still_falls_back(
        json!({"model": "m", "vertex_project": null}),
        json!({"vertex_project": "from-global"}),
        Ok(json!({"model": "m", "vertex_project": "from-global"})),
    )]
    #[case::a_global_of_the_wrong_type_is_a_request_error(
        json!({"model": "m"}),
        json!({"vertex_location": 5}),
        Err(()),
    )]
    fn module_globals_fill_the_litellm_params_the_call_leaves_out(
        #[case] kwargs: Value,
        #[case] globals: Value,
        #[case] expected: Result<Value, ()>,
    ) {
        assert_projection(kwargs, globals, expected);
    }

    #[rstest]
    #[case::a_global_without_a_spec_is_never_read(
        json!({"model": "m"}),
        json!({"aws_region_name": "eu-central-1", "api_key": "k"}),
        Ok(json!({"model": "m"})),
    )]
    #[case::only_the_spec_named_global_is_read(
        json!({"model": "m"}),
        json!({"vertex_ai_project": "legacy-global", "vertex_project": "from-global"}),
        Ok(json!({"model": "m", "vertex_project": "from-global"})),
    )]
    fn only_the_globals_the_specs_name_are_consulted(
        #[case] kwargs: Value,
        #[case] globals: Value,
        #[case] expected: Result<Value, ()>,
    ) {
        assert_projection(kwargs, globals, expected);
    }

    #[rstest]
    #[case::named_fields_are_projected(
        json!({"model": "m", "anthropic_federation_rule_id": "fdrl_1", "anthropic_issuer_ttl_seconds": 60}),
        Ok(json!({"model": "m", "anthropic_federation_rule_id": "fdrl_1", "anthropic_issuer_ttl_seconds": 60})),
    )]
    #[case::unnamed_fields_are_not(
        json!({"model": "m", "anthropic_unrelated": "x"}),
        Ok(json!({"model": "m"})),
    )]
    fn workload_identity_fields_are_projected_by_the_names_the_host_lists(
        #[case] kwargs: Value,
        #[case] expected: Result<Value, ()>,
    ) {
        assert_projection_of(
            &[
                "anthropic_federation_rule_id",
                "anthropic_issuer_ttl_seconds",
            ],
            kwargs,
            json!({}),
            expected,
        );
    }

    fn assert_projection(kwargs: Value, globals: Value, expected: Result<Value, ()>) {
        assert_projection_of(&[], kwargs, globals, expected);
    }

    fn assert_projection_of(
        names: &[&str],
        kwargs: Value,
        globals: Value,
        expected: Result<Value, ()>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let dict = |value: &Value| {
                to_py(py, value)
                    .unwrap()
                    .into_bound(py)
                    .cast_into::<PyDict>()
                    .unwrap()
            };
            let kwargs = dict(&kwargs);
            let globals = dict(&globals);
            let bound = PyDict::new(py);
            let projected = project_litellm_params(
                names.iter().copied(),
                |name| present(&kwargs, &bound, name),
                |name| present(&globals, &bound, name),
            )
            .unwrap();
            match (projected, expected) {
                (Ok(projected), Ok(fields)) => assert_eq!(
                    projected,
                    litellm_params(serde_json::from_value(fields).unwrap()).unwrap()
                ),
                (Err(error), Err(())) => assert!(error.is_request(), "{error:?}"),
                (projected, expected) => panic!("got {projected:?}, expected {expected:?}"),
            }
        });
    }
}
