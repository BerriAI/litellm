use crate::cache::PythonCache;
use litellm_host_python::{PythonHostCalls, PythonOwned};

use parking_lot::Mutex;

use bytes::Bytes;
use litellm_host::protocol::Reply;
use litellm_host_python::{
    InvokeError, PythonBinding, effective_py_args, from_py, json_from_py, to_py,
};
use litellm_inference_messages::{
    DecodingError, Error, MessagesCall, MessagesHostCall, MessagesSettings, MessagesShaping,
    UnencodableHeader, UnsendableUrl, VertexShaping, litellm_params, messages_body,
    route::{Messages, MessagesStreamHead},
};
use litellm_llms::{
    anthropic::model_capabilities::with_invoke_flags,
    azure_ai::common_utils::{
        CALLABLE_TOKEN_PROVIDER, CALLER_TOKEN_PROVIDER_PARAM, ENTRA_ID_PARAMS,
    },
    base_llm::messages::context::MessagesModelCapabilities,
    bedrock::chat::invoke_handler::BedrockStreamRaise,
    bedrock::messages::claude_platform::transformation::{
        is_claude_platform_route, is_missing_region, logged_model as claude_platform_logged_model,
    },
    bedrock_mantle::messages::transformation::{mantle_capability_lookup, mantle_logged_model},
};
use litellm_llms_types::headers::ProviderSpecificHeaders;
use litellm_router_types::LitellmParams;
use litellm_secret_value::SecretValue;
use pyo3::{
    exceptions::{PyException, PyUnicodeEncodeError},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    sync::PyOnceLock,
    types::{PyBool, PyBytes, PyDict, PyList, PyMapping, PyString, PyTuple},
};
use serde_json::{Map, Value};

use super::agentic::HookedRequest;
use super::plan::MessagesPlan;
use super::safeguards::SafeguardsHost;
use super::transport::{HostedTransport, python_failure};
use crate::{
    errors::{RequestMarker, raises_bad_request, route_failure},
    marshal::{
        ArgumentError, optional_timeout, project_optional_fields, public_json_response,
        python_timeout_seconds,
    },
    python_settings::PythonSettings,
    routes::RequestView,
};

type MergedHeaders = (Option<Map<String, Value>>, Option<Vec<u8>>);

const ANTHROPIC_BETA_HEADER: &str = "anthropic-beta";

/// A config whose `should_filter_anthropic_beta_headers` holds filters `anthropic-beta` before
/// Python's client encodes it, so the wire check decides it.
fn is_filtered_beta_header(
    name: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
    filters_beta: bool,
) -> bool {
    filters_beta
        && name
            .extract::<String>()
            .is_ok_and(|name| name.eq_ignore_ascii_case(ANTHROPIC_BETA_HEADER))
        && value.call_method1("encode", ("utf-8",)).is_ok()
}

const VERTEX_LABELS_HEADER: &str = "X-Vertex-AI-Labels";

/// The caller headers the provider settles before Python's client encodes them, so a value it
/// cannot encode never reaches the ASCII check.
#[derive(Clone, Copy, Debug, Default)]
struct SettledHeaders {
    /// The provider filters `anthropic-beta`.
    filters_beta: bool,
    /// `vertex_ai` with `vertex_request_label_fields` set: the proxy replaces every caller
    /// `X-Vertex-AI-Labels`, and one Python cannot decode adds no labels.
    replaces_labels: bool,
}

impl SettledHeaders {
    fn replaces(self, name: &Bound<'_, PyAny>) -> bool {
        self.replaces_labels
            && name
                .extract::<String>()
                .is_ok_and(|name| name.eq_ignore_ascii_case(VERTEX_LABELS_HEADER))
    }
}

fn ascii_headers<'py>(
    headers: &Bound<'py, PyAny>,
    unencodable: &mut Option<Vec<u8>>,
    settled: SettledHeaders,
) -> PyResult<Bound<'py, PyAny>> {
    let Ok(headers) = headers.cast::<PyDict>() else {
        return Ok(headers.clone());
    };
    let py = headers.py();
    let kept = PyDict::new(py);
    for (name, value) in headers.iter() {
        if value.is_instance_of::<PyString>()
            && !is_filtered_beta_header(&name, &value, settled.filters_beta)
            && let Err(error) = value.call_method1("encode", ("ascii",))
        {
            if !error.is_instance_of::<PyUnicodeEncodeError>(py) {
                return Err(error);
            }
            if settled.replaces(&name) {
                // Kept while it is text, so it still stands as the caller's labels header the
                // replacement reads, and refuses, first; a value that is not text just goes.
                if value.call_method1("encode", ("utf-8",)).is_ok() {
                    kept.set_item(name, value)?;
                }
                continue;
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

fn ascii_scoped_headers<'py>(
    value: &Bound<'py, PyAny>,
    settled: SettledHeaders,
) -> PyResult<Bound<'py, PyAny>> {
    let ascii_entry = |entry: Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        let Ok(entry) = entry.cast::<PyDict>() else {
            return Ok(entry);
        };
        let Some(headers) = entry.get_item("extra_headers")? else {
            return Ok(entry.clone().into_any());
        };
        let kept = entry.copy()?;
        kept.set_item(
            "extra_headers",
            ascii_headers(&headers, &mut None, settled)?,
        )?;
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
const MESSAGES_STREAM_FAILURE: &str = "messages_stream_failure";
/// The model id a Mantle config hands the Bedrock invoke transform, which names it in its errors.
const MESSAGES_PROVIDER_MODEL: &str = "messages_provider_model";
/// The model a refusal names when it differs from the call's, as for a missing Claude Platform region.
const MESSAGES_RAISED_MODEL: &str = "messages_raised_model";
/// The model Python's handler logs a Mantle call under before the config transforms the request.
const MESSAGES_LOGGED_MODEL: &str = "messages_logged_model";
/// The provider litellm's error names when it is not the route's: none for a call refused while
/// resolving its provider, as Python's `get_llm_provider` raises it, and `bedrock` for a Mantle
/// refusal the inherited Bedrock invoke transform raises as `BadRequestError`.
const MESSAGES_PROVIDER: &str = "messages_provider";
const MESSAGES_CREDENTIAL_FAILURE: &str = "messages_credential_failure";
const REPLAYED_CREDENTIAL_FAILURE: &str = "replay";
/// No usable credentials, which the host raises as botocore's `NoCredentialsError` unless an STS call failed.
const UNAVAILABLE_CREDENTIALS: &str = "unavailable";
pub(super) const MODEL_CAPABILITIES_MODULE: &str = "litellm.rust_bridge.host.model_capabilities";

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
const AZURE_AI_PROVIDER: &str = "azure_ai";
const ENABLE_TOKEN_REFRESH: &str = "enable_azure_ad_token_refresh";

const BEDROCK_PROVIDER: &str = "bedrock";
const BEDROCK_PASSTHROUGH_PARAMS: [&str; 2] = ["model_id", "guardrailConfig"];
const BEDROCK_METADATA_PARAMS: [&str; 2] = ["metadata", "litellm_metadata"];
const BEDROCK_SPEND_LOGS_METADATA: &str = "spend_logs_metadata";
const BEDROCK_REQUEST_METADATA_FIELDS: &str = "bedrock_request_metadata_fields";

const VERTEX_AI_PROVIDER: &str = "vertex_ai";
const VERTEX_REQUEST_LABEL_FIELDS: &str = "vertex_request_label_fields";
const VERTEX_LABELS_METADATA: &str = "litellm_metadata";

fn text_pairs(value: &Bound<'_, PyAny>) -> PyResult<Map<String, Value>> {
    let Ok(mapping) = value.cast::<PyMapping>() else {
        return Ok(Map::new());
    };
    mapping
        .items()?
        .iter()
        .filter_map(|item| {
            let (key, value): (Bound<'_, PyAny>, Bound<'_, PyAny>) = match item.extract() {
                Ok(pair) => pair,
                Err(error) => return Some(Err(error)),
            };
            let key = key.cast::<PyString>().ok()?.to_str().ok()?.to_string();
            if key == BEDROCK_SPEND_LOGS_METADATA {
                return Some(text_pairs(&value).map(|pairs| (key, Value::Object(pairs))));
            }
            let value = value.cast::<PyString>().ok()?.to_str().ok()?.to_string();
            Some(Ok((key, Value::String(value))))
        })
        .collect()
}

fn bedrock_litellm_params<'py>(
    py: Python<'py>,
    argument: impl Fn(&str) -> PyResult<Option<Bound<'py, PyAny>>>,
) -> PyResult<Map<String, Value>> {
    let mut extra = project_optional_fields(BEDROCK_PASSTHROUGH_PARAMS, &argument)?;
    for name in BEDROCK_METADATA_PARAMS {
        if let Some(value) = argument(name)? {
            extra.insert(name.to_string(), Value::Object(text_pairs(&value)?));
        }
    }
    if let Some(configured) = module_global(py, BEDROCK_REQUEST_METADATA_FIELDS)?
        && (configured.is_instance_of::<PyList>() || configured.is_instance_of::<PyTuple>())
    {
        let fields = configured
            .try_iter()?
            .map(|field| Ok(Value::String(field?.str()?.to_str()?.to_string())))
            .collect::<PyResult<Vec<Value>>>()?;
        extra.insert(
            BEDROCK_REQUEST_METADATA_FIELDS.to_string(),
            Value::Array(fields),
        );
    }
    Ok(extra)
}

fn bedrock_session_identity<'py>(
    params: &mut LitellmParams,
    global: impl Fn(&str) -> PyResult<Option<Bound<'py, PyAny>>>,
    env: &dyn Fn(&str) -> Option<String>,
) -> PyResult<()> {
    use litellm_llms::bedrock::session_identity::{
        REQUIRE_SESSION_ATTRIBUTION, SessionIdentity, session_identity,
    };

    let fields = |name| -> PyResult<Vec<String>> {
        global(name)?
            .filter(|value| !value.is_none())
            .map(|value| value.extract())
            .transpose()
            .map(Option::unwrap_or_default)
    };
    let source = |name| {
        params
            .extra
            .get(name)
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
            .filter_map(|(key, value)| value.as_str().map(|text| (key.clone(), text.to_owned())))
            .collect()
    };
    let identity = SessionIdentity {
        tag_fields: fields("bedrock_session_tag_fields")?,
        name_fields: fields("bedrock_session_name_fields")?,
        metadata: source("metadata"),
        litellm_metadata: source("litellm_metadata"),
    };
    let aws = session_identity(&params.aws, &identity, env);
    let derived_name = params.aws.aws_session_name.is_none()
        && env("AWS_SESSION_NAME").is_none()
        && aws.aws_session_name.is_some();
    params.extra = params
        .extra
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .chain(std::iter::once((
            REQUIRE_SESSION_ATTRIBUTION.to_owned(),
            Value::Bool(derived_name),
        )))
        .collect();
    params.aws = aws;
    Ok(())
}

/// Python's `vertex_request_label_fields()`: a non-empty string names one field, a list or tuple
/// its string members. With fields set, the call's `litellm_metadata` goes along as the labels'
/// source.
fn vertex_litellm_params<'py>(
    py: Python<'py>,
    argument: impl Fn(&str) -> PyResult<Option<Bound<'py, PyAny>>>,
) -> PyResult<Map<String, Value>> {
    let fields = vertex_label_fields(py)?;
    if fields.is_empty() {
        return Ok(Map::new());
    }
    let metadata = argument(VERTEX_LABELS_METADATA)?
        .map(|value| text_pairs(&value))
        .transpose()?;
    Ok([(
        VERTEX_REQUEST_LABEL_FIELDS.to_string(),
        Value::Array(fields),
    )]
    .into_iter()
    .chain(metadata.map(|pairs| (VERTEX_LABELS_METADATA.to_string(), Value::Object(pairs))))
    .collect())
}

fn vertex_label_fields(py: Python<'_>) -> PyResult<Vec<Value>> {
    Ok(match module_global(py, VERTEX_REQUEST_LABEL_FIELDS)? {
        Some(configured) if configured.is_instance_of::<PyString>() => {
            let field = configured.cast::<PyString>()?.to_str()?.to_string();
            (!field.is_empty())
                .then_some(Value::String(field))
                .into_iter()
                .collect()
        }
        Some(configured)
            if configured.is_instance_of::<PyList>() || configured.is_instance_of::<PyTuple>() =>
        {
            configured
                .try_iter()?
                .filter_map(|field| match field {
                    Ok(field) => field
                        .cast::<PyString>()
                        .ok()
                        .map(|text| Ok(Value::String(text.to_str()?.to_string()))),
                    Err(error) => Some(Err(error)),
                })
                .collect::<PyResult<_>>()?
        }
        _ => Vec::new(),
    })
}

fn global_drop_params(py: Python<'_>) -> PyResult<bool> {
    Ok(module_global(py, "drop_params")?.is_some_and(|value| value.is(PyBool::new(py, true))))
}

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

/// Installs the beta map once Python has loaded it. Loading starts only when a filter has a beta
/// to filter, off the calling thread, so a call that sends no beta never fetches the remote map.
fn install_loaded_beta_headers_config(py: Python<'_>) -> PyResult<()> {
    static INSTALLED: Mutex<Option<Py<PyAny>>> = Mutex::new(None);
    litellm_llms::anthropic::beta_headers::on_beta_headers_config_needed(
        request_beta_headers_config,
    );
    let loaded = py
        .import(ROUTE_HOST_MODULE)?
        .getattr("loaded_beta_headers_config")?
        .call0()?;
    if loaded.is_none() {
        return Ok(());
    }
    if INSTALLED
        .lock()
        .as_ref()
        .is_some_and(|source| source.bind(py).is(&loaded))
    {
        return Ok(());
    }
    litellm_llms::anthropic::beta_headers::install_beta_headers_config(from_py(&loaded)?);
    *INSTALLED.lock() = Some(loaded.unbind());
    Ok(())
}

fn request_beta_headers_config() {
    Python::attach(|py| {
        if let Err(error) = py
            .import(ROUTE_HOST_MODULE)
            .and_then(|module| module.call_method0("load_beta_headers_config_in_background"))
        {
            error.write_unraisable(py, None);
        }
    });
}

fn non_empty_string(value: Option<Bound<'_, PyAny>>) -> Option<String> {
    value
        .and_then(|value| value.extract::<String>().ok())
        .filter(|value| !value.is_empty())
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
    settled: Map<String, Value>,
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
    Ok(litellm_params(
        fields.into_iter().chain(folded).chain(settled).collect(),
    ))
}

/// The refresh switch Python reads from `litellm`, and whether the call names a token provider
/// Python would call or only count as present.
fn azure_settled_params(
    azure: AzureModuleSettings,
    provider: Option<Bound<'_, PyAny>>,
) -> PyResult<Map<String, Value>> {
    let provider = provider
        .map(|provider| -> PyResult<Value> {
            Ok(Value::String(
                match provider.is_truthy()? && provider.is_callable() {
                    true => CALLABLE_TOKEN_PROVIDER,
                    false => "uncallable",
                }
                .to_string(),
            ))
        })
        .transpose()?;
    Ok([(
        ENABLE_TOKEN_REFRESH.to_string(),
        Value::Bool(azure.token_refresh),
    )]
    .into_iter()
    .chain(provider.map(|provider| (CALLER_TOKEN_PROVIDER_PARAM.to_string(), provider)))
    .collect())
}

/// What Python's Azure config reads from `litellm` itself: `litellm.api_key or litellm.azure_key`
/// after the call's own key, and `litellm.enable_azure_ad_token_refresh is True`, which Python
/// reads instead of the call's param of that name.
struct AzureModuleSettings {
    api_key: Option<SecretValue>,
    token_refresh: bool,
}

fn azure_module_settings<'py>(
    global: impl Fn(&str) -> PyResult<Option<Bound<'py, PyAny>>>,
) -> PyResult<AzureModuleSettings> {
    let key = |name: &str| -> PyResult<Option<String>> {
        Ok(global(name)?
            .and_then(|value| value.extract::<String>().ok())
            .filter(|key| !key.is_empty()))
    };
    let api_key = match key("api_key")? {
        Some(key) => Some(key),
        None => key("azure_key")?,
    };
    let token_refresh =
        global(ENABLE_TOKEN_REFRESH)?.is_some_and(|value| value.is(PyBool::new(value.py(), true)));
    Ok(AzureModuleSettings {
        api_key: api_key.map(SecretValue::new),
        token_refresh,
    })
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
    stream_opened: bool,
}

impl MessagesPythonHost {
    pub(super) fn new(
        view: RequestView,
        asynchronous: bool,
        hooked_request: HookedRequest,
    ) -> Self {
        Self {
            view,
            cache: PythonCache::new(asynchronous),
            transport: HostedTransport::default(),
            safeguards: SafeguardsHost::default(),
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
            stream_opened: false,
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
        shapes_invoke_tools: bool,
    ) -> PyResult<Map<String, Value>> {
        let request = effective_py_args(self.view.request(py)?, arguments)?;
        let params = py
            .import(ROUTE_HOST_MODULE)?
            .getattr("body_params")?
            .call1((request, self.asynchronous, self.hooked, shapes_invoke_tools))?
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
        let mantle = custom_llm_provider
            .as_deref()
            .and_then(|provider| mantle_capability_lookup(provider, &model));
        // Python's Bedrock invoke transform shapes the caller's own tool dicts in place; Rust logs
        // the tools it shapes for the Mantle routes instead.
        let claude_platform = custom_llm_provider.as_deref() == Some(BEDROCK_PROVIDER)
            && is_claude_platform_route(&model);
        let shapes_invoke_tools = custom_llm_provider.as_deref() == Some(BEDROCK_PROVIDER)
            && mantle.is_none()
            && !claude_platform;
        let mut fields = self.body_params(py, arguments, shapes_invoke_tools)?;
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
        let filters_beta: bool = py
            .import(ROUTE_HOST_MODULE)?
            .getattr("filters_beta_headers")?
            .call1((&model, custom_llm_provider.as_deref(), api_base.as_deref()))?
            .extract()?;
        install_loaded_beta_headers_config(py)?;
        let settled_headers = SettledHeaders {
            filters_beta,
            replaces_labels: custom_llm_provider.as_deref() == Some(VERTEX_AI_PROVIDER)
                && !vertex_label_fields(py)?.is_empty(),
        };
        let (extra_headers, mut unencodable_request_header) = match settings.extra_headers {
            Some(headers) => (Some(headers), None),
            None => self.merged_headers(py, arguments, settled_headers)?,
        };
        let provider_specific_header = self.provider_specific_header(
            py,
            arguments,
            custom_llm_provider.as_deref(),
            settled_headers,
            &mut unencodable_request_header,
        )?;
        let azure = match custom_llm_provider.as_deref() {
            Some(AZURE_AI_PROVIDER) => Some(azure_module_settings(|name| module_global(py, name))?),
            _ => None,
        };
        let bedrock = custom_llm_provider.as_deref() == Some(BEDROCK_PROVIDER) || mantle.is_some();
        let deployment_base_model = match non_empty_string(
            argument("model_info")?.and_then(|info| info.get_item("base_model").ok()),
        ) {
            Some(base_model) => Some(base_model),
            None => non_empty_string(argument("base_model")?),
        };
        let capabilities = match &mantle {
            Some(_) => super::model_catalog::route_capabilities(
                py,
                &model,
                custom_llm_provider.as_deref(),
                deployment_base_model.as_deref(),
                argument("model_info")?,
            )?,
            None if bedrock => {
                let requested = self.capabilities(
                    py,
                    &model,
                    custom_llm_provider.as_deref(),
                    deployment_base_model.as_deref(),
                )?;
                MessagesModelCapabilities {
                    bedrock: with_invoke_flags(
                        super::model_catalog::bedrock_capabilities(
                            py,
                            &model,
                            argument("model_info")?,
                        )?,
                        &requested,
                    ),
                    ..requested
                }
            }
            None => self.capabilities(
                py,
                &model,
                custom_llm_provider.as_deref(),
                deployment_base_model.as_deref(),
            )?,
        };
        let vertex = match custom_llm_provider.as_deref() {
            Some(VERTEX_AI_PROVIDER) => VertexShaping {
                supported_regions: super::model_catalog::vertex_supported_regions(py, &model)?,
            },
            _ => VertexShaping::default(),
        };
        let shaping = MessagesShaping {
            capabilities,
            vertex,
            settings: MessagesSettings {
                whole_sse_frames: settings.stream_cost_model.is_some(),
                inject_stream_cost: settings.stream_cost_model.is_some()
                    && settings.settings.pricing.is_some(),
                presanitized,
                intercepted: self.may_intercept(py)?,
                unencodable_request_header,
                module_api_key: azure.as_ref().and_then(|azure| azure.api_key.clone()),
                global_drop_params: global_drop_params(py)?,
                base_model: argument("base_model")?.and_then(|value| value.extract().ok()),
                ..settings.settings
            },
        };
        let provider_fields: Vec<&str> = match custom_llm_provider.as_deref() {
            Some(FEDERATING_PROVIDER) => workload_identity_fields(py)?
                .iter()
                .map(String::as_str)
                .collect(),
            Some(AZURE_AI_PROVIDER) => ENTRA_ID_PARAMS.to_vec(),
            _ => Vec::new(),
        };
        let settled = match (azure, bedrock) {
            (Some(azure), _) => {
                azure_settled_params(azure, argument(CALLER_TOKEN_PROVIDER_PARAM)?)?
            }
            (None, true) => bedrock_litellm_params(py, argument)?,
            (None, false) if custom_llm_provider.as_deref() == Some(VERTEX_AI_PROVIDER) => {
                vertex_litellm_params(py, argument)?
            }
            (None, false) => Map::new(),
        };
        let mut litellm_params =
            project_litellm_params(provider_fields, settled, argument, |name| {
                module_global(py, name)
            })?;
        if bedrock && let Ok(params) = litellm_params.as_mut() {
            bedrock_session_identity(params, |name| module_global(py, name), &|name| {
                std::env::var(name).ok()
            })?;
        }
        // Python's Claude Platform config falls back to `litellm.api_base` before its env bases.
        let api_base = match (api_base, claude_platform) {
            (None, true) => module_global(py, "api_base")?
                .and_then(|base| base.extract::<String>().ok())
                .filter(|base| !base.is_empty()),
            (api_base, _) => api_base,
        };
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
        settled: SettledHeaders,
    ) -> PyResult<MergedHeaders> {
        let mut unencodable = None;
        let mut mapping = |name: &str| -> PyResult<Option<Map<String, Value>>> {
            self.view
                .argument(py, arguments, name)?
                .map(|value| from_py(&ascii_headers(&value, &mut unencodable, settled)?))
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
        settled: SettledHeaders,
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
            ascii_headers(&scoped, unencodable, settled)?;
        }
        from_py(&ascii_scoped_headers(&value, settled)?).map(Some)
    }

    fn capabilities(
        &self,
        py: Python<'_>,
        model: &str,
        custom_llm_provider: Option<&str>,
        base_model: Option<&str>,
    ) -> PyResult<MessagesModelCapabilities> {
        from_py(
            &py.import(MODEL_CAPABILITIES_MODULE)?
                .getattr("anthropic_model_capabilities")?
                .call1((model, custom_llm_provider, base_model))?,
        )
    }

    /// The model Python's handler logs a refused `bedrock/claude_platform/` call under.
    fn claude_platform_model(&self, py: Python<'_>) -> PyResult<Option<String>> {
        let Some(model) = self.view.request(py)?.get_item("model")? else {
            return Ok(None);
        };
        let Ok(model) = model.extract::<String>() else {
            return Ok(None);
        };
        Ok(
            (self.provider(py) == BEDROCK_PROVIDER && is_claude_platform_route(&model))
                .then(|| claude_platform_logged_model(&model).to_string()),
        )
    }

    fn mantle_models(&self, py: Python<'_>) -> PyResult<Option<MantleModels>> {
        let Some(model) = self.view.request(py)?.get_item("model")? else {
            return Ok(None);
        };
        let Ok(model) = model.extract::<String>() else {
            return Ok(None);
        };
        let provider = self.provider(py);
        Ok(mantle_capability_lookup(&provider, &model)
            .zip(mantle_logged_model(&provider, &model))
            .map(|(lookup, logged)| MantleModels {
                sent: lookup.model,
                logged,
            }))
    }

    fn provider(&self, py: Python<'_>) -> String {
        self.view
            .request(py)
            .and_then(|request| {
                py.import(ROUTE_HOST_MODULE)?
                    .getattr("failure_provider")?
                    .call1((request,))?
                    .extract::<String>()
            })
            .unwrap_or_else(|_| "anthropic".into())
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
                    self.transport.read().to_vec(),
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

    fn bedrock_stream_raise(&self, py: Python<'_>, raise: &BedrockStreamRaise) -> PyResult<PyErr> {
        let (kind, arguments) = match raise {
            BedrockStreamRaise::ChecksumMismatch {
                expected,
                calculated,
            } => (
                "ChecksumMismatch",
                serde_json::json!([expected, calculated]),
            ),
            BedrockStreamRaise::InvalidHeadersLength(length) => {
                ("InvalidHeadersLength", serde_json::json!([length]))
            }
            BedrockStreamRaise::InvalidPayloadLength(length) => {
                ("InvalidPayloadLength", serde_json::json!([length]))
            }
            BedrockStreamRaise::DuplicateHeader(name) => {
                ("DuplicateHeader", serde_json::json!([name]))
            }
            BedrockStreamRaise::InvalidJson(text) => ("InvalidJson", serde_json::json!([text])),
            BedrockStreamRaise::NotAnObject(chunk) => ("NotAnObject", serde_json::json!([chunk])),
        };
        let raised = py
            .import(ROUTE_HOST_MODULE)?
            .getattr("bedrock_stream_raise")?
            .call1((kind, to_py(py, &arguments)?))?;
        Ok(PyErr::from_value(raised))
    }

    fn mapped_error(&self, py: Python<'_>, error: Error) -> PyResult<PyErr> {
        if let Some(raise) = litellm_error::find::<BedrockStreamRaise>(&error) {
            return self.bedrock_stream_raise(py, raise);
        }
        if let Some((raised, raw)) = python_failure(py, &error) {
            return if raw {
                Ok(raised)
            } else {
                self.map_transport_failure(py, raised)
            };
        }
        if let Some(raised) = super::workload_identity::python_error(py, &error)
            .or_else(|| super::workload_identity::entra_token_error(py, &error))
        {
            let mapped = py
                .import(ROUTE_HOST_MODULE)?
                .getattr("map_token_failure")?
                .call1((
                    raised.value(py),
                    self.view.request(py)?,
                    self.provider(py),
                    self.asynchronous,
                ))?;
            return Ok(PyErr::from_value(mapped));
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
        if let Some(UnsendableUrl(message)) = litellm_error::find::<UnsendableUrl>(&error) {
            let raised = py
                .import(ROUTE_HOST_MODULE)?
                .getattr("unsendable_url")?
                .call1((message.as_str(),))?;
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
        let unresolved_provider = refused_resolving_provider(&error);
        let bad_request = raises_bad_request(&error);
        let missing_platform_region = matches!(
            &error,
            Error::Auth(auth) | Error::Llm(litellm_llms::Error::Auth(auth)) if is_missing_region(auth)
        );
        route_failure(py, error, RequestMarker::Messages, |native| {
            if self.stream_opened {
                native.value(py).setattr(MESSAGES_STREAM_FAILURE, true)?;
            }
            if unresolved_provider {
                // Python never resolved the provider, so its error names the model as requested.
                let value = native.value(py);
                value.setattr(MESSAGES_PROVIDER, "")?;
                if let Some(model) = self.view.request(py)?.get_item("model")? {
                    value.setattr(MESSAGES_PROVIDER_MODEL, model)?;
                }
            } else if let Some(model) = self.claude_platform_model(py)? {
                let value = native.value(py);
                value.setattr(MESSAGES_PROVIDER_MODEL, model)?;
                if missing_platform_region {
                    value.setattr(crate::errors::MESSAGES_LATE_REJECTION, true)?;
                    value.setattr(MESSAGES_RAISED_MODEL, "")?;
                }
                if value
                    .getattr_opt(MESSAGES_CREDENTIAL_FAILURE)?
                    .is_some_and(|kind| kind.eq(REPLAYED_CREDENTIAL_FAILURE).unwrap_or(false))
                {
                    value.setattr(MESSAGES_CREDENTIAL_FAILURE, UNAVAILABLE_CREDENTIALS)?;
                }
            } else if let Some(models) = self.mantle_models(py)? {
                let value = native.value(py);
                value.setattr(MESSAGES_PROVIDER_MODEL, models.sent)?;
                value.setattr(MESSAGES_LOGGED_MODEL, models.logged)?;
                if bad_request {
                    // Python's Mantle configs inherit the Bedrock invoke transform, which raises
                    // its `BadRequestError` naming `bedrock`.
                    value.setattr(MESSAGES_PROVIDER, BEDROCK_PROVIDER)?;
                }
                // The Mantle routes map a credential Rust could not acquire from a fixed class
                // list, never by running Python's signing again.
                if value
                    .getattr_opt(MESSAGES_CREDENTIAL_FAILURE)?
                    .is_some_and(|kind| kind.eq(REPLAYED_CREDENTIAL_FAILURE).unwrap_or(false))
                {
                    value.setattr(MESSAGES_CREDENTIAL_FAILURE, UNAVAILABLE_CREDENTIALS)?;
                }
            }
            Ok(self.map_failure(py, native))
        })
    }
}

fn runs_pre_request_hooks(py: Python<'_>) -> PyResult<bool> {
    py.import(ROUTE_HOST_MODULE)?
        .getattr("runs_pre_request_hooks")?
        .call0()?
        .extract()
}

/// The models a Mantle route's errors name: the one Python's handler logs the call under before
/// its config transforms the request, and the model id that transform sends.
struct MantleModels {
    logged: String,
    sent: String,
}

/// Whether Rust refused the call where Python's `get_llm_provider` refuses it, before any provider
/// is resolved: a `bedrock_mantle/` region that is not a valid AWS region.
fn refused_resolving_provider(error: &Error) -> bool {
    matches!(
        error
            .rejected_request()
            .and_then(|rejection| rejection.downcast_ref::<litellm_llms::ErrorDetail>()),
        Some(litellm_llms::ErrorDetail::InvalidMantleRegion {
            check: litellm_llms::MantleRegionCheck::ProviderResolution,
            ..
        })
    )
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
        self.stream_cost_model = settings
            .stream_cost_model
            .clone()
            .filter(|_| settings.settings.pricing.is_none());
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
    type Protocol = Messages;
    type Failure = PyErr;
    type Decoded = MessagesPlan;

    fn decode_request(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<MessagesPlan, InvokeError<Error>> {
        self.view
            .resolve(py, arguments)
            .map_err(InvokeError::Python)?;
        let function = py
            .import(ROUTE_HOST_MODULE)
            .and_then(|module| module.call_method0("cached_function"))
            .map_err(InvokeError::Python)?;
        let cache = crate::cache::configure(&mut self.cache, py, arguments, &function)
            .map_err(InvokeError::Python)?;
        let effective = self
            .view
            .request(py)
            .and_then(|request| effective_py_args(request, arguments))
            .map_err(InvokeError::Python)?;
        let safeguards = self
            .safeguards
            .configure(py, &effective)
            .map_err(InvokeError::Python)?;
        self.requested_stream = arguments
            .get_item("stream")
            .map_err(InvokeError::Python)?
            .is_some_and(|stream| stream.is(pyo3::types::PyBool::new(py, true)));
        self.arguments = Some(arguments.clone().unbind());
        let call = self.decode_call(py, arguments)?;
        let transport = self
            .transport
            .select(py, self.view.request(py).map_err(InvokeError::Python)?)
            .map_err(InvokeError::Python)?;
        let pre_request_hooks = runs_pre_request_hooks(py).map_err(InvokeError::Python)?;
        Ok(MessagesPlan {
            call,
            cache,
            transport,
            safeguards,
            pre_request_hooks,
        })
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
        py.import(ROUTE_HOST_MODULE)?
            .getattr("record_guardrail")?
            .call1((&public, self.view.request(py)?))?;
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
        self.stream_opened = true;
        let module = py.import(ROUTE_HOST_MODULE)?;
        let head = module
            .getattr("stream_hidden_params")?
            .call1((to_py(py, &head.headers)?, head.unobserved_tail.is_some()))?;
        if let Some(arguments) = self.arguments.as_ref().filter(|_| self.asynchronous) {
            module
                .getattr("python_stream_types")?
                .call1((&head, arguments))?;
        }
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
        let mapped = self.mapped_error(py, error)?;
        if !self.stream_opened {
            return Ok(mapped);
        }
        let described = py
            .import(ROUTE_HOST_MODULE)?
            .getattr("stream_failure")?
            .call1((mapped.value(py), self.view.request(py)?, self.provider(py)))?;
        Ok(PyErr::from_value(described))
    }

    fn host_error(error: &PyErr) -> Error {
        crate::routes::inference::host_call_error(error)
    }
}

impl PythonHostCalls<Messages> for MessagesPythonHost {
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
    #[case::metadata("metadata")]
    #[case::litellm_metadata("litellm_metadata")]
    fn bedrock_identity_settings_project_into_aws_credentials(#[case] bucket: &str) {
        Python::initialize();
        Python::attach(|py| {
            let globals = to_py(
                py,
                &json!({
                    "bedrock_session_tag_fields": ["user_api_key_alias"],
                    "bedrock_session_name_fields": ["user_api_key_alias"],
                }),
            )
            .unwrap()
            .into_bound(py)
            .cast_into::<PyDict>()
            .unwrap();
            let mut params = litellm_params(map(json!({
                "model": "m", "aws_role_name": "arn:aws:iam::123456789012:role/invocation",
                (bucket): {"user_api_key_alias": "agent"},
            })))
            .unwrap();
            bedrock_session_identity(&mut params, |name| globals.get_item(name), &|_| None)
                .unwrap();
            assert_eq!(params.aws.aws_session_name.as_deref(), Some("agent"));
            assert_eq!(
                params
                    .aws
                    .aws_session_tags
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|tag| (tag.key.as_str(), tag.value.as_str()))
                    .collect::<Vec<_>>(),
                [("user_api_key_alias", "agent")]
            );
            assert_eq!(
                params
                    .extra
                    .get(litellm_llms::bedrock::session_identity::REQUIRE_SESSION_ATTRIBUTION),
                Some(&Value::Bool(true))
            );
        });
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

    const BETA_FILTERED: SettledHeaders = SettledHeaders {
        filters_beta: true,
        replaces_labels: false,
    };
    const LABELS_REPLACED: SettledHeaders = SettledHeaders {
        filters_beta: true,
        replaces_labels: true,
    };

    #[rstest]
    #[case::a_beta_is_left_to_the_provider_filter(
        c"{'anthropic-beta': 'ZWJ\\u200djoined'}",
        BETA_FILTERED,
        json!({"anthropic-beta": "ZWJ\u{200d}joined"}),
        None
    )]
    #[case::in_any_spelling(
        c"{'Anthropic-Beta': 'ZWJ\\u200djoined'}",
        BETA_FILTERED,
        json!({"Anthropic-Beta": "ZWJ\u{200d}joined"}),
        None
    )]
    #[case::a_beta_held_back_where_the_provider_does_not_filter_it(
        c"{'anthropic-beta': 'ZWJ\\u200djoined', 'x-note': 'caf\\xe9', 'x-kept': 'b'}",
        SettledHeaders::default(),
        json!({"x-kept": "b"}),
        Some("ZWJ\u{200d}joined".as_bytes().to_vec())
    )]
    #[case::a_beta_python_cannot_hand_over_as_text(
        c"{'anthropic-beta': 'a\\ud800', 'x-kept': 'b'}",
        BETA_FILTERED,
        json!({"x-kept": "b"}),
        Some(b"a\xed\xa0\x80".to_vec())
    )]
    #[case::any_other_header(
        c"{'x-note': 'ZWJ\\u200d', 'x-kept': 'b'}",
        BETA_FILTERED,
        json!({"x-kept": "b"}),
        Some("ZWJ\u{200d}".as_bytes().to_vec())
    )]
    #[case::vertex_labels_the_proxy_replaces(
        c"{'X-Vertex-AI-Labels': 'ZWJ\\u200djoined', 'x-kept': 'b'}",
        LABELS_REPLACED,
        json!({"X-Vertex-AI-Labels": "ZWJ\u{200d}joined", "x-kept": "b"}),
        None
    )]
    #[case::in_any_spelling_and_gone_when_not_text(
        c"{'x-vertex-ai-labels': 'a\\ud800', 'x-kept': 'b'}",
        LABELS_REPLACED,
        json!({"x-kept": "b"}),
        None
    )]
    #[case::vertex_labels_the_caller_still_owns(
        c"{'X-Vertex-AI-Labels': 'ZWJ\\u200djoined', 'x-kept': 'b'}",
        BETA_FILTERED,
        json!({"x-kept": "b"}),
        Some("ZWJ\u{200d}joined".as_bytes().to_vec())
    )]
    fn a_header_python_cannot_encode_is_held_back_unless_the_provider_settles_it(
        #[case] headers: &std::ffi::CStr,
        #[case] settled: SettledHeaders,
        #[case] kept: Value,
        #[case] unencodable: Option<Vec<u8>>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let headers = py.eval(headers, None, None).unwrap();
            let mut held_back = None;
            let sent: Value =
                from_py(&ascii_headers(&headers, &mut held_back, settled).unwrap()).unwrap();
            assert_eq!((sent, held_back), (kept, unencodable));
        });
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
    #[case::aws_session_tags_are_projected(
        json!({"model": "m", "aws_role_name": "r", "aws_session_tags": [{"Key": "team", "Value": "genai"}]}),
        Ok(json!({"model": "m", "aws_role_name": "r", "aws_session_tags": [{"Key": "team", "Value": "genai"}]})),
    )]
    #[case::malformed_aws_session_tags_are_a_request_error(
        json!({"model": "m", "aws_session_tags": [{"Key": "team"}]}),
        Err(()),
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

    #[rstest]
    #[case::api_key_before_azure_key(json!({"api_key": "module", "azure_key": "azure"}), Some("module"), false)]
    #[case::empty_api_key_falls_to_azure_key(json!({"api_key": "", "azure_key": "azure"}), Some("azure"), false)]
    #[case::neither(json!({}), None, false)]
    #[case::token_refresh_only_when_exactly_true(
        json!({"enable_azure_ad_token_refresh": true}),
        None,
        true
    )]
    #[case::a_truthy_token_refresh_is_not_true(json!({"enable_azure_ad_token_refresh": 1}), None, false)]
    fn azure_reads_the_module_settings_python_reads(
        #[case] globals: Value,
        #[case] api_key: Option<&str>,
        #[case] token_refresh: bool,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let globals = to_py(py, &globals)
                .unwrap()
                .into_bound(py)
                .cast_into::<PyDict>()
                .unwrap();
            let settings = azure_module_settings(|name| {
                Ok(globals.get_item(name)?.filter(|value| !value.is_none()))
            })
            .unwrap();
            assert_eq!(settings.api_key.as_ref().map(SecretValue::expose), api_key);
            assert_eq!(settings.token_refresh, token_refresh);
        });
    }

    #[rstest]
    #[case::absent(c"provider = None", None)]
    #[case::callable(c"provider = lambda: 'tok'", Some("callable"))]
    #[case::not_callable(c"provider = 'tok'", Some("uncallable"))]
    #[case::falsy_callable(
        c"class Falsy:\n    def __call__(self):\n        return 'tok'\n    def __bool__(self):\n        return False\nprovider = Falsy()\n",
        Some("uncallable")
    )]
    fn azure_marks_a_token_provider_as_python_would_treat_it(
        #[case] source: &std::ffi::CStr,
        #[case] expected: Option<&str>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(source, Some(&locals), Some(&locals)).unwrap();
            let provider = locals
                .get_item("provider")
                .unwrap()
                .filter(|provider| !provider.is_none());
            let settled = azure_settled_params(
                AzureModuleSettings {
                    api_key: None,
                    token_refresh: false,
                },
                provider,
            )
            .unwrap();
            assert_eq!(
                settled
                    .get(CALLER_TOKEN_PROVIDER_PARAM)
                    .and_then(Value::as_str),
                expected
            );
        });
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
                Map::new(),
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
