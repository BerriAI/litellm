use std::ffi::CStr;

use litellm_callbacks::event::{CallEvent, Passthrough, RawResponse, RequestContext, WireRequest};
use litellm_host_python::{AdapterStep, CallbackAdapter};
use pyo3::prelude::*;
use rstest::rstest;
use serde_json::{Value, json};

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
        on_pre_call(additional_args)

    def _pre_call(self, input, api_key, additional_args):
        self.record('_pre_call', None)

    def record_api_call_start_time(self):
        self.record('record_api_call_start_time', None)

    def post_call(self, original_response, additional_args):
        self.record('post_call', None)
        self.post = (original_response, additional_args)

    def record_post_call(self, response, *rest):
        self.record('record_post_call', response)

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

fn before_send(script: &CStr, caller: Value, body: Value) -> WireRequest {
    before_send_with_secrets(script, caller, body, &[])
}

/// Runs `before_send` over `body` for a caller whose route-side view is `caller`, with the
/// Python objects `script` binds, then delivers the provider's raw response the way the
/// driver does and runs the script's `check()`.
fn before_send_with_secrets(
    script: &CStr,
    caller: Value,
    body: Value,
    secret_fields: &[&str],
) -> WireRequest {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(py, PAYLOAD_LOGGER);
        run(py, &locals, script);
        let mut logging = LegacyLogging {
            logger: Some(PythonLogger::new(local(&locals, "logger").unbind(), true)),
            ..legacy_call(py, &locals, false)
        };
        let context = RequestContext {
            model: "model".into(),
            custom_llm_provider: "provider".into(),
            optional_params: caller.clone(),
            passthrough_fields: Passthrough::unchanged(caller.as_object().unwrap(), &body),
            secret_fields: secret_fields.iter().map(|name| name.to_string()).collect(),
        };
        let wire = WireRequest {
            url: "https://provider.invalid/ocr".into(),
            headers: vec![("x-route".into(), "route".into())],
            body,
        };
        let step = logging.before_send(py, Box::new(wire), &context).unwrap();
        let raw = CallEvent::ResponseReceived {
            raw: RawResponse {
                body: "raw response".into(),
            },
        };
        assert!(matches!(
            logging.emit(py, &raw, None).unwrap(),
            AdapterStep::Done
        ));
        run(py, &locals, c"check()");
        let AdapterStep::Wire(wire) = step else {
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
    let wire = before_send(
        script,
        json!({"document": document(DOCUMENT), "pages": [0]}),
        body.clone(),
    );
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
        json!({"document": document("https://example.invalid/scan.pdf")}),
        json!({"document": document(DOCUMENT)}),
    );
    assert_eq!(
        wire.body["document"],
        json!({"type": "document_url", "document_url": DOCUMENT, "document_name": "edited.pdf"})
    );
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
    let wire = before_send(script, json!({}), body.clone());
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
    let wire = before_send(script, json!({"document": document(DOCUMENT)}), body);
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
fn post_call_receives_the_raw_response_and_the_payload_dicts_pre_call_saw() {
    before_send(
        c"
def check():
    original_response, additional_args = logger.post
    assert original_response == 'raw response', original_response
    assert additional_args['complete_input_dict'] is logger.pre['complete_input_dict']
    assert additional_args['headers'] is logger.pre['headers']
",
        json!({}),
        json!({"document": document(DOCUMENT)}),
    );
}

#[rstest]
#[case::every_phase_listens(c"{}", &["pre_call", "post_call"])]
#[case::no_input_callback(
    c"{'input': False}",
    &["_pre_call", "record_api_call_start_time", "record_post_call"]
)]
#[case::no_payload_consumer(c"{'payload': False}", &["record_api_call_start_time"])]
fn payload_callbacks_run_only_for_the_phases_someone_listens_to(
    #[case] needed: &CStr,
    #[case] expected_calls: &[&str],
) {
    let script = std::ffi::CString::new(format!(
        "
logger.needed = {needed}
def on_pre_call(args):
    args['complete_input_dict']['include_image_base64'] = True
def check():
    assert logger.names() == {expected_calls:?}, logger.calls
",
        needed = needed.to_str().unwrap(),
        expected_calls = expected_calls,
    ))
    .unwrap();
    let body = json!({"document": document(DOCUMENT)});
    let wire = before_send(&script, json!({}), body.clone());
    let edited = json!({"document": document(DOCUMENT), "include_image_base64": true});
    assert_eq!(
        wire.body,
        if expected_calls.contains(&"pre_call") {
            edited
        } else {
            body
        }
    );
}
