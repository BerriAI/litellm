#[path = "custom_logger_contract/support.rs"]
mod support;

use pyo3::prelude::*;
use rstest::rstest;
use serde_json::{Value, json};
use support::{Case, Mode, Upstream};

#[rstest]
#[case::sync_success(Mode::Sync, Upstream::Success, &["pre_api", "post_api", "sync_success"])]
#[case::sync_failure(Mode::Sync, Upstream::Failure, &["pre_api", "sync_failure"])]
#[case::async_success(Mode::Async, Upstream::Success, &["pre_api", "post_api", "async_success"])]
#[case::async_failure(Mode::Async, Upstream::Failure, &["pre_api", "sync_failure", "async_failure"])]
fn registered_loggers_receive_each_eligible_event_once(
    #[case] mode: Mode,
    #[case] upstream: Upstream,
    #[case] expected: &[&str],
    #[values("request", "global", "both")] registration: &str,
) {
    let case = Case::new(mode, upstream, registration, None);
    case.run("complete");
    assert_eq!(case.requests().len(), 1);
    let events = case.names();
    let logging: Vec<_> = events
        .iter()
        .filter(|name| !name.starts_with("deployment_") && *name != "pre_request")
        .map(String::as_str)
        .collect();
    assert_eq!(logging, expected);
    let deployment: Vec<_> = events
        .iter()
        .filter(|name| name.starts_with("deployment_") || *name == "pre_request")
        .map(String::as_str)
        .collect();
    let expected_hooks: &[&str] = match (mode, upstream, registration) {
        (Mode::Async, Upstream::Success, "global" | "both") => {
            &["deployment_pre", "pre_request", "deployment_success"]
        }
        (Mode::Async, Upstream::Failure, "global" | "both") => {
            &["deployment_pre", "pre_request", "deployment_failure"]
        }
        _ => &[],
    };
    assert_eq!(deployment, expected_hooks);
    match upstream {
        Upstream::Success => case.assert_success(),
        Upstream::Failure => {
            case.assert_error_is("provider_error");
            Python::attach(|py| {
                let state = case.state.bind(py);
                for name in expected.iter().filter(|name| name.ends_with("failure")) {
                    assert!(
                        case.event(name)
                            .value
                            .bind(py)
                            .is(state.getattr("error").unwrap())
                    );
                }
            });
        }
        Upstream::Stream => unreachable!(),
    }
}

#[rstest]
#[case::messages_mutated_without_return("messages", "none")]
#[case::messages_mutated_with_original_kwargs("messages", "same")]
#[case::messages_mutated_with_replacement_kwargs("messages", "replacement")]
#[case::tools_replaced_without_return("tools", "none")]
#[case::tools_replaced_with_original_kwargs("tools", "same")]
#[case::tools_replaced_with_replacement_kwargs("tools", "replacement")]
fn pre_request_edits_reach_the_next_callback_and_provider(
    #[case] field: &str,
    #[case] returns: &str,
) {
    let case = Case::new(Mode::Async, Upstream::Success, "global", None);
    Python::attach(|py| {
        case.state
            .bind(py)
            .call_method1("request_edit", (field, returns))
            .unwrap();
    });
    case.run("complete");
    case.assert_success();
    let expected = match field {
        "messages" => json!([{"role": "user", "content": "edited"}]),
        "tools" => json!([{"name": "edited", "input_schema": {"type": "object"}}]),
        _ => unreachable!(),
    };
    Python::attach(|py| {
        let state = case.state.bind(py);
        let observed = state.getattr("observer").unwrap().getattr(field).unwrap();
        assert!(observed.is(state.getattr("editor").unwrap().getattr("seen").unwrap()));
        assert_eq!(
            litellm_host_python::from_py::<Value>(&observed).unwrap(),
            expected
        );
    });
    let requests = case.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].body_json::<Value>().unwrap()[field], expected);
}

#[rstest]
#[case::body_mutation("complete_input_dict", false)]
#[case::body_replacement("complete_input_dict", true)]
#[case::header_mutation("headers", false)]
#[case::header_replacement("headers", true)]
fn pre_api_mutation_reaches_the_wire_but_envelope_replacement_does_not(
    #[case] field: &str,
    #[case] replace: bool,
    #[values(Mode::Sync, Mode::Async)] mode: Mode,
) {
    let case = Case::new(mode, Upstream::Success, "request", None);
    Python::attach(|py| {
        case.state
            .bind(py)
            .call_method1("wire_edit", (field, replace))
            .unwrap();
    });
    case.run("complete");
    case.assert_success();
    Python::attach(|py| {
        let state = case.state.bind(py);
        let retained = state
            .getattr("editor")
            .unwrap()
            .getattr("retained")
            .unwrap();
        let seen = state.getattr("observer").unwrap().getattr("seen").unwrap();
        assert_eq!(seen.is(&retained), !replace);
        assert_eq!(
            seen.get_item("x-contract")
                .unwrap()
                .extract::<String>()
                .unwrap(),
            "edited"
        );
    });
    let requests = case.requests();
    assert_eq!(requests.len(), 1);
    let sent = match field {
        "headers" => requests[0]
            .headers
            .get("x-contract")
            .map(|value| value.to_str().unwrap().to_string()),
        _ => requests[0]
            .body_json::<Value>()
            .unwrap()
            .get("x-contract")
            .and_then(Value::as_str)
            .map(String::from),
    };
    assert_eq!(sent.as_deref(), if replace { None } else { Some("edited") });
}

#[rstest]
#[case::reject_before_deployment(
    "deployment_pre", false, Upstream::Success, 0,
    &["deployment_pre", "sync_failure", "async_failure"]
)]
#[case::reject_before_request(
    "pre_request", false, Upstream::Success, 0,
    &["deployment_pre", "pre_request", "deployment_failure", "sync_failure", "async_failure"]
)]
#[case::cancel_before_deployment(
    "deployment_pre", true, Upstream::Success, 0,
    &["deployment_pre"]
)]
#[case::cancel_before_request(
    "pre_request", true, Upstream::Success, 0,
    &["deployment_pre", "pre_request"]
)]
#[case::failure_logger_raises(
    "sync_failure", false, Upstream::Failure, 1,
    &["deployment_pre", "pre_request", "pre_api", "deployment_failure", "sync_failure", "async_failure"]
)]
#[case::failure_logger_cancelled(
    "sync_failure", true, Upstream::Failure, 1,
    &["deployment_pre", "pre_request", "pre_api", "deployment_failure", "sync_failure"]
)]
#[case::success_logger_raises(
    "async_success", false, Upstream::Success, 1,
    &["deployment_pre", "pre_request", "pre_api", "post_api", "deployment_success", "async_success"]
)]
#[case::pre_api_logger_raises(
    "pre_api", false, Upstream::Success, 1,
    &["deployment_pre", "pre_request", "pre_api", "post_api", "deployment_success", "async_success"]
)]
#[case::post_api_logger_raises(
    "post_api", false, Upstream::Success, 1,
    &["deployment_pre", "pre_request", "pre_api", "post_api", "deployment_success", "async_success"]
)]
#[case::deployment_success_logger_raises(
    "deployment_success", false, Upstream::Success, 1,
    &["deployment_pre", "pre_request", "pre_api", "post_api", "deployment_success", "async_success"]
)]
#[case::deployment_success_cancelled(
    "deployment_success", true, Upstream::Success, 1,
    &["deployment_pre", "pre_request", "pre_api", "post_api", "deployment_success"]
)]
#[case::async_failure_logger_raises(
    "async_failure", false, Upstream::Failure, 1,
    &["deployment_pre", "pre_request", "pre_api", "deployment_failure", "sync_failure", "async_failure"]
)]
#[case::async_failure_logger_cancelled(
    "async_failure", true, Upstream::Failure, 1,
    &["deployment_pre", "pre_request", "pre_api", "deployment_failure", "sync_failure", "async_failure"]
)]
fn callback_failures_do_not_replay_the_provider_or_switch_outcomes(
    #[case] hook: &str,
    #[case] cancelled: bool,
    #[case] upstream: Upstream,
    #[case] provider_calls: usize,
    #[case] expected: &[&str],
) {
    let case = Case::new(Mode::Async, upstream, "global", Some((hook, cancelled)));
    case.run("complete");
    assert_eq!(case.requests().len(), provider_calls);
    assert_eq!(case.names(), expected);
    match (hook, cancelled) {
        ("deployment_pre" | "pre_request", _) | (_, true) => case.assert_error_is("callback"),
        ("sync_failure" | "async_failure", false) => case.assert_error_is("provider_error"),
        _ => case.assert_success(),
    }
}

#[test]
fn deployment_response_replacement_is_returned_and_logged() {
    let case = Case::new(Mode::Async, Upstream::Success, "global", None);
    let mut replacement = support::response();
    replacement["content"] = json!([{"type": "text", "text": "replacement"}]);
    Python::attach(|py| {
        case.state
            .bind(py)
            .call_method1(
                "response_edit",
                (litellm_host_python::to_py(py, &replacement).unwrap(),),
            )
            .unwrap();
    });
    case.run("complete");
    case.assert_success();
    assert_eq!(case.requests().len(), 1);
    Python::attach(|py| {
        let state = case.state.bind(py);
        assert!(
            state.getattr("output").unwrap().is(state
                .getattr("editor")
                .unwrap()
                .getattr("replacement")
                .unwrap())
        );
        let event = case.event("async_success");
        let success = event.value.bind(py);
        assert_eq!(
            success
                .getattr("choices")
                .unwrap()
                .get_item(0)
                .unwrap()
                .getattr("message")
                .unwrap()
                .getattr("content")
                .unwrap()
                .extract::<String>()
                .unwrap(),
            "replacement"
        );
    });
}

#[test]
fn a_guardrail_rejects_the_response_without_dispatching_success() {
    let case = Case::new(Mode::Async, Upstream::Success, "global", None);
    Python::attach(|py| {
        case.state.bind(py).call_method0("block_response").unwrap();
    });
    case.run("complete");
    case.assert_error_is("blocked");
    assert_eq!(case.requests().len(), 1);
    assert_eq!(
        case.names(),
        [
            "deployment_pre",
            "pre_request",
            "pre_api",
            "post_api",
            "deployment_success",
            "sync_failure",
            "async_failure"
        ]
    );
    Python::attach(|py| {
        let state = case.state.bind(py);
        for hook in ["sync_failure", "async_failure"] {
            assert!(
                case.event(hook)
                    .value
                    .bind(py)
                    .is(state.getattr("blocked").unwrap())
            );
        }
    });
}

#[rstest]
#[case::unreleased(&[], 0)]
#[case::accepted(&[true], 1)]
#[case::accepted_twice(&[true, true], 1)]
#[case::rejected_then_accepted(&[false, true], 0)]
fn deferred_success_obeys_the_proxy_release_decision(
    #[case] releases: &[bool],
    #[case] expected: usize,
) {
    let case = Case::new(Mode::Async, Upstream::Success, "global", None);
    Python::attach(|py| {
        let state = case.state.bind(py);
        state.setattr("deferred", true).unwrap();
        state.setattr("releases", releases.to_vec()).unwrap();
    });
    case.run("complete");
    case.assert_success();
    Python::attach(|py| {
        let before: Vec<String> = case
            .state
            .bind(py)
            .getattr("before_release")
            .unwrap()
            .extract()
            .unwrap();
        assert!(!before.iter().any(|name| name == "async_success"));
    });
    assert_eq!(
        case.names()
            .iter()
            .filter(|name| *name == "async_success")
            .count(),
        expected
    );
    assert_eq!(case.requests().len(), 1);
}

#[rstest]
#[case::async_consumed(Mode::Async, "consume", false)]
#[case::async_closed_twice(Mode::Async, "close", false)]
#[case::sync_consumed(Mode::Sync, "consume", false)]
#[case::sync_closed(Mode::Sync, "close", false)]
#[case::proxy_disconnect_with_deferred_logging(Mode::Async, "disconnect", true)]
fn streamed_usage_is_logged_once_when_consumed_or_closed(
    #[case] mode: Mode,
    #[case] finish: &str,
    #[case] deferred: bool,
) {
    let case = Case::new(mode, Upstream::Stream, "global", None);
    Python::attach(|py| {
        case.state.bind(py).setattr("deferred", deferred).unwrap();
    });
    case.run(finish);
    case.assert_success();
    assert_eq!(case.requests().len(), 1);
    assert_eq!(
        case.names()
            .iter()
            .filter(|name| *name == "async_success")
            .count(),
        1
    );
    assert!(!case.names().iter().any(|name| name.ends_with("failure")));
    Python::attach(|py| {
        let before: Vec<String> = case
            .state
            .bind(py)
            .getattr("before_finish")
            .unwrap()
            .extract()
            .unwrap();
        assert!(
            !before.iter().any(|name| name.ends_with("success")),
            "premature logging: {before:?}"
        );
        let event = case.event("async_success");
        let success = event.value.bind(py);
        let usage = success.getattr("usage").unwrap();
        if finish == "consume" {
            let chunks: Vec<Vec<u8>> = case
                .state
                .bind(py)
                .getattr("chunks")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(chunks.concat(), support::stream_events().as_bytes());
            assert_eq!(
                usage
                    .getattr("completion_tokens")
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                support::response()["usage"]["output_tokens"]
                    .as_u64()
                    .unwrap()
            );
        }

        assert_eq!(
            usage
                .getattr("prompt_tokens")
                .unwrap()
                .extract::<u64>()
                .unwrap(),
            support::response()["usage"]["input_tokens"]
                .as_u64()
                .unwrap()
        );
    });
}

#[derive(Clone, Copy, Debug)]
enum Delivery {
    Caller,
    Executor,
    LoggingWorker,
}

#[rstest]
#[case::sync_pre_api(Mode::Sync, Upstream::Success, "pre_api", Delivery::Caller)]
#[case::sync_post_api(Mode::Sync, Upstream::Success, "post_api", Delivery::Caller)]
#[case::sync_success_is_submitted(
    Mode::Sync,
    Upstream::Success,
    "sync_success",
    Delivery::Executor
)]
#[case::async_deployment_hook(Mode::Async, Upstream::Success, "deployment_pre", Delivery::Caller)]
#[case::async_pre_request_hook(Mode::Async, Upstream::Success, "pre_request", Delivery::Caller)]
#[case::async_pre_api(Mode::Async, Upstream::Success, "pre_api", Delivery::Caller)]
#[case::async_success_is_queued(
    Mode::Async,
    Upstream::Success,
    "async_success",
    Delivery::LoggingWorker
)]
#[case::sync_failure_is_inline(Mode::Async, Upstream::Failure, "sync_failure", Delivery::Caller)]
#[case::async_failure_is_awaited(Mode::Async, Upstream::Failure, "async_failure", Delivery::Caller)]
fn callbacks_run_in_the_required_execution_context(
    #[case] mode: Mode,
    #[case] upstream: Upstream,
    #[case] hook: &str,
    #[case] delivery: Delivery,
) {
    let case = Case::new(mode, upstream, "global", None);
    case.run("complete");
    assert_eq!(case.requests().len(), 1);
    Python::attach(|py| {
        let state = case.state.bind(py);
        let event = case.event(hook);
        let caller_thread: u64 = state.getattr("caller_thread").unwrap().extract().unwrap();
        let caller_task: Option<u64> = state.getattr("caller_task").unwrap().extract().unwrap();
        let thread = event.thread;
        let task = event.task;
        assert_eq!(event.context, "caller");
        match delivery {
            Delivery::Caller => {
                assert_eq!(thread, caller_thread);
                assert_eq!(task, caller_task);
            }
            Delivery::Executor => {
                assert_ne!(thread, caller_thread);
                assert_eq!(task, None);
            }
            Delivery::LoggingWorker => {
                assert_eq!(thread, caller_thread);
                assert!(task.is_some());
                assert_ne!(task, caller_task);
            }
        }
    });
}
