use crate::cache::{CacheCall, PythonCache, Selection};
use litellm_host_python::{PythonHostCalls, PythonOwned};

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use bytes::Bytes;
use litellm_host::protocol::Reply;
use litellm_host_python::{
    InvokeError, PythonBinding, effective_py_args, from_py, json_from_py, to_py,
};
use litellm_inference_messages::{
    DecodingError, Error, MessagesCall, MessagesSettings, MessagesShaping, UnencodableHeader,
    litellm_params, messages_body,
    route::{Messages, MessagesStreamHead},
};
use litellm_llms::base_llm::messages::context::MessagesModelCapabilities;
use litellm_llms_types::headers::ProviderSpecificHeaders;
use litellm_router_types::LitellmParams;
use pyo3::{
    exceptions::{PyException, PyUnicodeEncodeError},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    sync::PyOnceLock,
    types::{PyBytes, PyDict, PyList, PyMapping, PyString, PyTuple},
};
use serde_json::{Map, Value};

use super::agentic::HookedRequest;
use super::safeguards::{SafeguardsCall, SafeguardsHost};
use super::transport::{HostedTransport, TransportCall, python_failure};
use crate::{
    errors::{RequestMarker, route_failure},
    marshal::{
        ArgumentError, optional_timeout, project_optional_fields, public_json_response,
        python_timeout_seconds,
    },
    python_settings::PythonSettings,
    routes::RequestView,
};

type MergedHeaders = (Option<Map<String, Value>>, Option<Vec<u8>>);

fn ascii_headers<'py>(
    headers: &Bound<'py, PyAny>,
    unencodable: &mut Option<Vec<u8>>,
) -> PyResult<Bound<'py, PyAny>> {
    let Ok(headers) = headers.cast::<PyDict>() else {
        return Ok(headers.clone());
    };
    let py = headers.py();
    let kept = PyDict::new(py);
    for (name, value) in headers.iter() {
        if value.is_instance_of::<PyString>()
            && let Err(error) = value.call_method1("encode", ("ascii",))
        {
            if !error.is_instance_of::<PyUnicodeEncodeError>(py) {
                return Err(error);
            }
            if unencodable.is_none() {
                let sent = value.call_method1("encode", ("utf-8", "surrogatepass"))?;
                *unencodable = Some(sent.cast::<PyBytes>()?.as_bytes().to_vec());
            }
            continue;
        }
        kept.set_item(name, value)?;
    }
    Ok(kept.into_any())
}

const PROVIDER_SPECIFIC_HEADERS_MODULE: &str =
    "litellm.litellm_core_utils.get_provider_specific_headers";

fn ascii_scoped_headers<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let ascii_entry = |entry: Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        let Ok(entry) = entry.cast::<PyDict>() else {
            return Ok(entry);
        };
        let Some(headers) = entry.get_item("extra_headers")? else {
            return Ok(entry.clone().into_any());
        };
        let kept = entry.copy()?;
        kept.set_item("extra_headers", ascii_headers(&headers, &mut None)?)?;
        Ok(kept.into_any())
    };
    if let Ok(entries) = value.cast::<PyList>() {
        return Ok(PyList::new(
            value.py(),
            entries
                .iter()
                .map(ascii_entry)
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any());
    }
    if let Ok(entries) = value.cast::<PyTuple>() {
        return Ok(PyTuple::new(
            value.py(),
            entries
                .iter()
                .map(ascii_entry)
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any());
    }
    ascii_entry(value.clone())
}

fn json_messages(messages: &Bound<'_, PyAny>) -> PyResult<Value> {
    if messages.is_instance_of::<PyList>() || messages.is_instance_of::<PyTuple>() {
        return json_from_py(messages);
    }
    json_from_py(&messages.py().get_type::<PyList>().call1((messages,))?)
}

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
    Intercept { stream: bool, reply: Reply<bool> },
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

enum Answered {
    Stream,
    Whole { verbatim: bool },
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
    arguments: Option<Py<PyDict>>,
    interceptor: Option<Py<PyAny>>,
    pending_answer: Option<Reply<()>>,
    pending_intercept: Option<Reply<bool>>,
    pending_short_circuit: Option<(bool, Reply<bool>)>,
    answer: Option<Py<PyAny>>,
    answered: Option<Answered>,
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
            arguments: None,
            interceptor: None,
            pending_answer: None,
            pending_intercept: None,
            pending_short_circuit: None,
            answer: None,
            answered: None,
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

    fn may_intercept(&self, py: Python<'_>) -> PyResult<bool> {
        if !self.asynchronous {
            return Ok(false);
        }
        let module = py.import(ROUTE_HOST_MODULE)?;
        Ok(module.getattr("has_interceptors")?.call0()?.extract()?
            || module.getattr("short_circuits")?.call0()?.extract()?)
    }

    fn begin_intercept(
        &mut self,
        py: Python<'_>,
        stream: bool,
        reply: Reply<bool>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let module = py.import(ROUTE_HOST_MODULE)?;
        if !module
            .getattr("short_circuits")?
            .call0()?
            .extract::<bool>()?
        {
            return self.begin_interceptor(py, stream, reply);
        }
        let pending = module.getattr("short_circuit")?.call1((
            self.view.request(py)?,
            stream,
            self.hooked,
        ))?;
        self.pending_short_circuit = Some((stream, reply));
        Ok(Some(pending.unbind()))
    }

    fn resume_short_circuit(
        &mut self,
        py: Python<'_>,
        stream: bool,
        reply: Reply<bool>,
        answer: Py<PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        if answer.is_none(py) {
            return self.begin_interceptor(py, stream, reply);
        }
        self.answer = Some(answer);
        self.answered = Some(match stream {
            true => Answered::Stream,
            false => Answered::Whole { verbatim: false },
        });
        reply.send(true);
        Ok(None)
    }

    fn begin_interceptor(
        &mut self,
        py: Python<'_>,
        stream: bool,
        reply: Reply<bool>,
    ) -> PyResult<Option<Py<PyAny>>> {
        self.interceptor = self.interceptor(py)?;
        if self.interceptor.is_none() {
            reply.send(false);
            return Ok(None);
        }
        let pending = self.start_answer(py, stream)?;
        self.pending_intercept = Some(reply);
        Ok(Some(pending))
    }

    fn begin_answer(
        &mut self,
        py: Python<'_>,
        stream: bool,
        reply: Reply<()>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let pending = self.start_answer(py, stream)?;
        self.pending_answer = Some(reply);
        Ok(Some(pending))
    }

    fn start_answer(&mut self, py: Python<'_>, stream: bool) -> PyResult<Py<PyAny>> {
        let module = py.import(ROUTE_HOST_MODULE)?;
        let request = self.view.request(py)?;
        let pending = match &self.interceptor {
            Some(interceptor) => {
                module
                    .getattr("answer")?
                    .call1((interceptor, request, stream, self.hooked))?
            }
            None => module
                .getattr("gateway")?
                .call1((request, self.asynchronous, self.hooked))?,
        };
        if pending.is_none() {
            return Err(litellm_host_python::Error::MissingState.into());
        }
        self.answered = Some(match (stream, &self.interceptor) {
            (true, _) => Answered::Stream,
            (false, Some(_)) => Answered::Whole { verbatim: false },
            (false, None) => Answered::Whole { verbatim: true },
        });
        Ok(pending.unbind())
    }

    fn host_settings(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> PyResult<HostSettings> {
        from_py(&py.import(ROUTE_HOST_MODULE)?.getattr("settings")?.call1((
            arguments,
            self.view.request(py)?,
            self.asynchronous,
        ))?)
    }

    fn body_params(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> PyResult<Map<String, Value>> {
        let request = effective_py_args(self.view.request(py)?, arguments)?;
        let params = py
            .import(ROUTE_HOST_MODULE)?
            .getattr("body_params")?
            .call1((request, self.asynchronous, self.hooked))?
            .cast_into::<PyMapping>()?;
        params
            .items()?
            .iter()
            .map(|item| {
                let (name, value): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item.extract()?;
                let name: String = name.extract()?;
                let value = match name.as_str() {
                    "messages" => json_messages(&value)?,
                    _ => json_from_py(&value)?,
                };
                Ok((name, value))
            })
            .collect()
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
            None => json_messages(&messages)?,
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
        let (extra_headers, mut unencodable_request_header) = match settings.extra_headers {
            Some(headers) => (Some(headers), None),
            None => self.merged_headers(py, arguments)?,
        };
        let provider_specific_header = self.provider_specific_header(
            py,
            arguments,
            custom_llm_provider.as_deref(),
            &mut unencodable_request_header,
        )?;
        let shaping = MessagesShaping {
            capabilities: self.capabilities(py, &model, custom_llm_provider.as_deref())?,
            settings: MessagesSettings {
                whole_sse_frames: settings.stream_cost_model.is_some(),
                presanitized,
                intercepted: self.may_intercept(py)?,
                unencodable_request_header,
                ..settings.settings
            },
        };
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
    ) -> PyResult<MergedHeaders> {
        let mut unencodable = None;
        let mut mapping = |name: &str| -> PyResult<Option<Map<String, Value>>> {
            self.view
                .argument(py, arguments, name)?
                .map(|value| from_py(&ascii_headers(&value, &mut unencodable)?))
                .transpose()
        };
        let headers = mapping("headers")?;
        let extra_headers = mapping("extra_headers")?;
        Ok((merge_headers(headers, extra_headers), unencodable))
    }

    fn provider_specific_header(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
        provider: Option<&str>,
        unencodable: &mut Option<Vec<u8>>,
    ) -> PyResult<Option<ProviderSpecificHeaders>> {
        let Some(value) = self
            .view
            .argument(py, arguments, "provider_specific_header")?
        else {
            return Ok(None);
        };
        if unencodable.is_none()
            && let Some(provider) = provider
            && let Ok(scoped) = py
                .import(PROVIDER_SPECIFIC_HEADERS_MODULE)?
                .getattr("ProviderSpecificHeaderUtils")?
                .call_method1("get_provider_specific_headers", (&value, provider, true))
        {
            ascii_headers(&scoped, unencodable)?;
        }
        from_py(&ascii_scoped_headers(&value)?).map(Some)
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
            .and_then(|map| {
                map.call1((
                    error.value(py),
                    self.view.request(py)?,
                    self.provider(py),
                    self.asynchronous,
                ))
            })
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
        self.arguments = Some(arguments.clone().unbind());
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
        let public = public_json_response(py, ROUTE_HOST_MODULE, &response)?;
        self.safeguards.published(py, public)
    }

    fn deliver_response(&mut self, py: Python<'_>, response: Py<PyAny>) -> PyResult<Py<PyAny>> {
        match self.answered.take() {
            Some(Answered::Stream) => return self.cache.wrap_stream(py, response),
            Some(Answered::Whole { verbatim }) => {
                self.cache.store_response(py, response.bind(py), verbatim)?;
            }
            None if self.requested_stream => {}
            None => self.cache.store_response(py, response.bind(py), false)?,
        }
        Ok(response)
    }

    fn encode_stream_head(
        &mut self,
        py: Python<'_>,
        head: MessagesStreamHead,
    ) -> PyResult<Py<PyAny>> {
        let module = py.import(ROUTE_HOST_MODULE)?;
        let head = module
            .getattr("stream_hidden_params")?
            .call1((to_py(py, &head.headers)?,))?;
        if let Some(arguments) = self
            .arguments
            .as_ref()
            .filter(|_| self.asynchronous && !self.requested_stream)
        {
            module
                .getattr("unrequested_stream")?
                .call1((&head, arguments))?;
        }
        Ok(head.unbind())
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
        if let Some(unencodable) = litellm_error::find::<UnencodableHeader>(&error) {
            let (value, received) = match unencodable {
                UnencodableHeader::Sent(value) => (value, false),
                UnencodableHeader::Received(value) => (value, true),
            };
            let raised = py
                .import(ROUTE_HOST_MODULE)?
                .getattr("unencodable_header")?
                .call1((PyBytes::new(py, value), received))?;
            return self.map_transport_failure(py, PyErr::from_value(raised));
        }
        if let Some(undecodable) = litellm_error::find::<DecodingError>(&error) {
            let raised = py
                .import(ROUTE_HOST_MODULE)?
                .getattr("decoding_failure")?
                .call1((
                    undecodable.to_string(),
                    self.view.request(py)?,
                    self.provider(py),
                ))?;
            return Ok(PyErr::from_value(raised));
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
            MessagesHostCall::Intercept { stream, reply } => {
                self.begin_intercept(py, stream, reply)
            }
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
        if let Some((stream, reply)) = self.pending_short_circuit.take() {
            return self
                .resume_short_circuit(py, stream, reply, result?)
                .map_err(InvokeError::Python);
        }
        if let Some(reply) = self.pending_answer.take() {
            self.answer = Some(result?);
            reply.send(());
            return Ok(None);
        }
        if let Some(reply) = self.pending_intercept.take() {
            self.answer = Some(result?);
            reply.send(true);
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
        self.pending_intercept = None;
        self.pending_short_circuit = None;
        self.answer = None;
        self.answered = None;
        self.arguments = None;
    }
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.view.traverse(visit)?;
        self.cache.traverse(visit)?;
        self.transport.traverse(visit)?;
        self.safeguards.traverse(visit)?;
        visit.call(&self.interceptor)?;
        visit.call(&self.arguments)?;
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
