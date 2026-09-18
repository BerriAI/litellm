use std::ffi::CStr;

use litellm_auth::SecretValue;
use litellm_callbacks::event::{MachineEvent, RawResponse, RequestContext, WireRequest};
use litellm_host_python::{LifecycleEvent, LifecycleStep, PythonLifecycle, to_py};
use proptest::prelude::*;
use pyo3::prelude::*;
use rstest::rstest;
use serde_json::{Map, Value, json};

use super::LegacyLogging;
use crate::PythonLogger;
use crate::test_support::{legacy_call, local, namespace, run};

/// The payload phases of `Logging` on top of `StubLogger`, with `pre_call` handing the
/// payload to the case's `on_pre_call`.
const PAYLOAD_LOGGER: &CStr = c"
class Request:
    pass

class PayloadLogger(StubLogger):
    def update_from_kwargs(self, **update):
        self.update = update

    def pre_call(self, input, api_key, additional_args):
        self.record('pre_call', None)
        self.pre = additional_args
        self.pre_api_key = api_key
        on_pre_call(additional_args)

    def post_call(self, original_response, api_key, additional_args):
        self.record('post_call', None)
        self.post = (original_response, api_key, additional_args)

request = Request()
kwargs = {}
logger = PayloadLogger()
on_pre_call = lambda additional_args: None
check = lambda: None
";

const DOCUMENT: &str = "data:application/pdf;base64,YWJj";
const EDITED: &str = "data:application/pdf;base64,ZWRpdGVk";

fn document(source: &str) -> Value {
    json!({"type": "document_url", "document_url": source})
}

fn before_send(script: &CStr, body: Value) -> WireRequest {
    before_send_with_secrets(script, json!({}), body, &[])
}

/// Runs `before_send` over `body` for a route whose parameters are `optional_params`, with
/// the Python objects `script` binds, then delivers the provider's raw response the way the
/// driver does and runs the script's `check()`.
fn before_send_with_secrets(
    script: &CStr,
    optional_params: Value,
    body: Value,
    secret_fields: &[&str],
) -> WireRequest {
    before_send_bound(&[], script, optional_params, body, secret_fields)
}

/// [`before_send_with_secrets`] with `bindings` placed in the namespace before `script` runs.
fn before_send_bound(
    bindings: &[(&str, &Value)],
    script: &CStr,
    optional_params: Value,
    body: Value,
    secret_fields: &[&str],
) -> WireRequest {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(py, PAYLOAD_LOGGER);
        for &(name, value) in bindings {
            locals.set_item(name, to_py(py, value).unwrap()).unwrap();
        }
        run(py, &locals, script);
        let mut logging = LegacyLogging {
            logger: Some(PythonLogger::new(local(&locals, "logger").unbind())),
            ..legacy_call(py, &locals, false)
        };
        let context = RequestContext {
            model: "model".into(),
            custom_llm_provider: "provider".into(),
            optional_params,
            secret_fields: secret_fields.iter().map(|name| name.to_string()).collect(),
            api_key: Some(SecretValue::new("route-key")),
        };
        let wire = WireRequest {
            url: "https://provider.invalid/ocr".into(),
            headers: vec![("x-route".into(), "route".into())],
            body,
        };
        let step = logging.before_send(py, Box::new(wire), &context).unwrap();
        let raw = MachineEvent::ResponseReceived {
            raw: RawResponse {
                body: "raw response".into(),
            },
        };
        assert!(matches!(
            logging.emit(py, LifecycleEvent::Machine(&raw)).unwrap(),
            LifecycleStep::Done
        ));
        run(py, &locals, c"check()");
        let LifecycleStep::Wire(wire) = step else {
            panic!("before_send did not hand back the wire request");
        };
        *wire
    })
}

#[rstest]
#[case::caller_keyword(c"
document = {'type': 'document_url', 'document_url': 'data:application/pdf;base64,YWJj'}
pages = [0]
kwargs = {'document': document, 'pages': pages}
observed = []
on_pre_call = lambda args: observed.append(
    (args['complete_input_dict']['document'] is document, args['complete_input_dict']['pages'] is pages)
)
def check():
    assert observed == [(True, True)], observed
")]
#[case::request_attribute_behind_an_omitted_keyword(c"
document = {'type': 'document_url', 'document_url': 'data:application/pdf;base64,YWJj'}
pages = [0]
request.document = document
kwargs = {'pages': pages}
observed = []
on_pre_call = lambda args: observed.append(
    (args['complete_input_dict']['document'] is document, args['complete_input_dict']['pages'] is pages)
)
def check():
    assert observed == [(True, True)], observed
")]
fn passthrough_keys_reach_pre_call_as_the_callers_own_objects(#[case] script: &CStr) {
    let body = json!({"model": "model", "document": document(DOCUMENT), "pages": [0]});
    let wire = before_send(script, body.clone());
    assert_eq!(wire.body, body);
}

#[test]
fn pre_call_edit_of_a_passthrough_object_reaches_the_caller_and_the_wire() {
    let wire = before_send(
        c"
document = {'type': 'document_url', 'document_url': 'data:application/pdf;base64,YWJj'}
kwargs = {'document': document}
def on_pre_call(args):
    args['complete_input_dict']['document']['document_url'] = 'data:application/pdf;base64,ZWRpdGVk'
def check():
    assert document['document_url'] == 'data:application/pdf;base64,ZWRpdGVk'
",
        json!({"document": document(DOCUMENT)}),
    );
    assert_eq!(wire.body["document"], document(EDITED));
}

#[test]
fn a_body_key_the_route_rewrote_is_not_the_callers_object() {
    let wire = before_send(
        c"
document = {'type': 'document_url', 'document_url': 'https://example.invalid/scan.pdf'}
kwargs = {'document': document}
observed = []
def on_pre_call(args):
    observed.append(args['complete_input_dict']['document'] is document)
    args['complete_input_dict']['document']['document_name'] = 'edited.pdf'
def check():
    assert observed == [False], observed
    assert document == {'type': 'document_url', 'document_url': 'https://example.invalid/scan.pdf'}
",
        json!({"document": document(DOCUMENT)}),
    );
    assert_eq!(
        wire.body["document"],
        json!({"type": "document_url", "document_url": DOCUMENT, "document_name": "edited.pdf"})
    );
}

#[test]
fn a_caller_value_with_no_json_form_is_left_out_of_realiasing() {
    let body = json!({"pages": [0]});
    let wire = before_send(
        c"
opaque = object()
kwargs = {'pages': opaque}
observed = []
on_pre_call = lambda args: observed.append(args['complete_input_dict']['pages'])
def check():
    assert observed == [[0]], observed
",
        body.clone(),
    );
    assert_eq!(wire.body, body);
}

#[rstest]
#[case::body(
    c"
def on_pre_call(args):
    args['complete_input_dict'] = {'replacement': True}
"
)]
#[case::headers(
    c"
def on_pre_call(args):
    args['headers'] = {'x-replacement': 'yes'}
"
)]
fn rebinding_the_payload_envelope_does_not_reach_the_wire(#[case] script: &CStr) {
    let body = json!({"document": document(DOCUMENT)});
    let wire = before_send(script, body.clone());
    assert_eq!(wire.body, body);
    assert_eq!(wire.headers, [("x-route".to_string(), "route".to_string())]);
}

#[test]
fn pre_call_header_edit_reaches_the_wire() {
    let wire = before_send(
        c"
def on_pre_call(args):
    args['headers']['x-callback'] = 'edited'
",
        json!({}),
    );
    assert_eq!(
        wire.headers,
        [
            ("x-route".to_string(), "route".to_string()),
            ("x-callback".to_string(), "edited".to_string()),
        ]
    );
}

#[test]
fn pre_call_receives_the_wire_request_and_the_logger_its_redacted_request() {
    let body = json!({"model": "model", "document": document(DOCUMENT)});
    before_send_with_secrets(
        c"
logger_fn = lambda *args: None
kwargs = {
    'litellm_call_id': 'call-1',
    'client_secret': 'shh',
    'proxy_server_request': {'body': {}},
    'logger_fn': logger_fn,
    'litellm_request_debug': True,
    'ocr_cost_per_page': 0.05,
}
observed = []
on_pre_call = observed.append
def check():
    [args] = observed
    assert args['api_base'] == 'https://provider.invalid/ocr', args
    assert args['complete_input_dict'] == {
        'model': 'model',
        'document': {'type': 'document_url', 'document_url': 'data:application/pdf;base64,YWJj'},
    }, args
    update = logger.update
    assert update['model'] == 'model' and update['custom_llm_provider'] == 'provider', update
    assert update['litellm_params']['litellm_call_id'] == 'call-1', update
    assert update['litellm_params']['api_base'] == 'https://provider.invalid/ocr', update
    assert update['litellm_params']['logger_fn'] is logger_fn, update
    assert update['litellm_params']['litellm_request_debug'] is True, update
    assert update['litellm_params']['ocr_cost_per_page'] == 0.05, update
    assert update['kwargs']['client_secret'] == '****', update
    assert 'proxy_server_request' not in update['kwargs'], update
    assert update['optional_params']['client_secret'] == '****', update
",
        json!({"client_secret": "shh"}),
        body,
        &["client_secret"],
    );
}

#[rstest]
#[case::added_key(
    c"
def on_pre_call(args):
    args['complete_input_dict']['include_image_base64'] = True
",
    json!({"document": document(DOCUMENT), "include_image_base64": true})
)]
#[case::replaced_document(
    c"
document = {'type': 'document_url', 'document_url': 'data:application/pdf;base64,YWJj'}
kwargs = {'document': document}
def on_pre_call(args):
    args['complete_input_dict']['document'] = {
        'type': 'document_url', 'document_url': 'data:application/pdf;base64,ZWRpdGVk'
    }
def check():
    assert document['document_url'] == 'data:application/pdf;base64,YWJj', document
",
    json!({"document": document(EDITED)})
)]
#[case::retained_body_edited_after_rebinding(
    c"
def on_pre_call(args):
    retained = args['complete_input_dict']
    args['complete_input_dict'] = {'rebound': True}
    retained['include_image_base64'] = True
",
    json!({"document": document(DOCUMENT), "include_image_base64": true})
)]
fn pre_call_body_edits_reach_the_wire(#[case] script: &CStr, #[case] expected: Value) {
    let body = json!({"document": document(DOCUMENT)});
    let wire = before_send(script, body);
    assert_eq!(wire.body, expected);
}

#[test]
fn retained_headers_edited_after_rebinding_reach_the_wire() {
    let wire = before_send(
        c"
def on_pre_call(args):
    retained = args['headers']
    args['headers'] = {'x-rebound': 'rebound'}
    retained['x-retained'] = 'sent'
",
        json!({}),
    );
    assert_eq!(
        wire.headers,
        [
            ("x-route".to_string(), "route".to_string()),
            ("x-retained".to_string(), "sent".to_string()),
        ]
    );
}

#[test]
fn post_call_receives_the_raw_response_the_route_key_and_the_body_and_headers_pre_call_saw() {
    before_send(
        c"
def check():
    original_response, api_key, additional_args = logger.post
    assert original_response == 'raw response', original_response
    assert api_key == logger.pre_api_key == 'route-key', (api_key, logger.pre_api_key)
    assert additional_args == {
        'complete_input_dict': logger.pre['complete_input_dict'],
        'headers': logger.pre['headers'],
    }, additional_args
    assert additional_args['complete_input_dict'] is logger.pre['complete_input_dict']
    assert additional_args['headers'] is logger.pre['headers']
",
        json!({"document": document(DOCUMENT)}),
    );
}

#[test]
fn every_request_runs_the_full_pre_call_and_post_call() {
    let wire = before_send(
        c"
def on_pre_call(args):
    args['complete_input_dict']['include_image_base64'] = True
def check():
    assert logger.names() == ['pre_call', 'post_call'], logger.calls
",
        json!({"document": document(DOCUMENT)}),
    );
    assert_eq!(
        wire.body,
        json!({"document": document(DOCUMENT), "include_image_base64": true})
    );
}

/// What one pre-call callback does to the payload it is handed.
#[derive(Clone, Debug)]
enum Edit {
    Nothing,
    Set(String, Value),
    Remove(String),
    Rebind(Value),
    RebindThenSetRetained(String, Value),
}

impl Edit {
    fn script(&self) -> Value {
        match self {
            Self::Nothing => json!({"kind": "nothing"}),
            Self::Set(key, value) => json!({"kind": "set", "key": key, "value": value}),
            Self::Remove(key) => json!({"kind": "remove", "key": key}),
            Self::Rebind(value) => json!({"kind": "rebind", "value": value}),
            Self::RebindThenSetRetained(key, value) => {
                json!({"kind": "rebind_then_set_retained", "key": key, "value": value})
            }
        }
    }

    /// The legacy contract: the provider is sent the body object `pre_call` received, as
    /// the callback left it. Rebinding the envelope's key points the envelope elsewhere and
    /// leaves that object alone.
    fn sent(&self, body: &Map<String, Value>) -> Value {
        let mut sent = body.clone();
        match self {
            Self::Nothing | Self::Rebind(_) => {}
            Self::Set(key, value) | Self::RebindThenSetRetained(key, value) => {
                sent.insert(key.clone(), value.clone());
            }
            Self::Remove(key) => {
                sent.remove(key);
            }
        }
        Value::Object(sent)
    }
}

/// How the caller's keyword for a body key relates to what the route sends under it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Caller {
    PassedUnchanged,
    RewrittenByTheRoute,
    NotPassed,
}

const MODEL: &CStr = c"
aliased = {}
def on_pre_call(args):
    body = args['complete_input_dict']
    aliased.update({name: body[name] is kwargs[name] for name in unchanged})
    kind = edit['kind']
    if kind == 'set':
        body[edit['key']] = edit['value']
    elif kind == 'remove':
        body.pop(edit['key'], None)
    elif kind == 'rebind':
        args['complete_input_dict'] = edit['value']
    elif kind == 'rebind_then_set_retained':
        args['complete_input_dict'] = {}
        body[edit['key']] = edit['value']
def check():
    assert aliased == {name: True for name in unchanged}, aliased
    assert logger.names() == ['pre_call', 'post_call'], logger.calls
";

fn json_value() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::from),
        any::<i64>().prop_map(Value::from),
        any::<f64>()
            .prop_filter("JSON has no NaN or infinity", |number| number.is_finite())
            .prop_map(Value::from),
        ".{0,8}".prop_map(Value::from),
    ];
    leaf.prop_recursive(3, 24, 4, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..4).prop_map(Value::from),
            prop::collection::btree_map(key(), inner, 0..4)
                .prop_map(|fields| Value::Object(fields.into_iter().collect())),
        ]
    })
}

fn key() -> impl Strategy<Value = String> {
    "[a-z]{1,6}"
}

fn caller() -> impl Strategy<Value = Caller> {
    prop_oneof![
        Just(Caller::PassedUnchanged),
        Just(Caller::RewrittenByTheRoute),
        Just(Caller::NotPassed),
    ]
}

fn edit() -> impl Strategy<Value = Edit> {
    prop_oneof![
        Just(Edit::Nothing),
        (key(), json_value()).prop_map(|(key, value)| Edit::Set(key, value)),
        key().prop_map(Edit::Remove),
        json_value().prop_map(Edit::Rebind),
        (key(), json_value()).prop_map(|(key, value)| Edit::RebindThenSetRetained(key, value)),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// For any body, any caller keywords and any callback edit: every keyword the route
    /// sends unchanged reaches `pre_call` as the caller's own object, and the provider is
    /// sent exactly what the model says, so a callback that edits nothing changes nothing.
    #[test]
    fn the_wire_is_the_body_pre_call_received_as_the_callback_left_it(
        fields in prop::collection::btree_map(key(), (json_value(), caller()), 0..5),
        edit in edit(),
    ) {
        let body: Map<String, Value> = fields
            .iter()
            .map(|(name, (value, _))| (name.clone(), value.clone()))
            .collect();
        let kwargs: Map<String, Value> = fields
            .iter()
            .filter_map(|(name, (value, caller))| match caller {
                Caller::PassedUnchanged => Some((name.clone(), value.clone())),
                Caller::RewrittenByTheRoute => Some((name.clone(), json!([value]))),
                Caller::NotPassed => None,
            })
            .collect();
        let unchanged: Value = fields
            .iter()
            .filter(|(_, (_, caller))| *caller == Caller::PassedUnchanged)
            .map(|(name, _)| Value::from(name.clone()))
            .collect();

        let wire = before_send_bound(
            &[
                ("kwargs", &Value::Object(kwargs)),
                ("unchanged", &unchanged),
                ("edit", &edit.script()),
            ],
            MODEL,
            json!({}),
            Value::Object(body.clone()),
            &[],
        );

        prop_assert_eq!(wire.body, edit.sent(&body));
        prop_assert_eq!(wire.headers, [("x-route".to_string(), "route".to_string())]);
    }
}
