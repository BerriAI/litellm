pub use crate::lifecycle::ocr::*;
pub use crate::lifecycle::{ErrorDisposition, Outcome};

#[cfg(test)]
use crate::Error;
#[cfg(test)]
use crate::ocr::{OcrRequest, prepare};

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;
    use crate::ocr::types::OcrDocument;

    fn request() -> OcrRequest {
        OcrRequest {
            model: "mistral/requested-model".into(),
            custom_llm_provider: None,
            api_key: Some("test-key".into()),
            api_base: Some("https://example.test".into()),
            extra_headers: vec![],
            timeout_seconds: 2.0,
            request_format: None,
            document: OcrDocument::DocumentUrl {
                document_url: "https://example.test/doc.pdf".into(),
            },
            azure_ad_token: None,
            vertex_project: None,
            vertex_location: None,
            stream: false,
        }
    }

    fn machine(asynchronous: bool) -> Lifecycle {
        let NativeOutcome::Completed(machine) = Lifecycle::new(
            &request(),
            Options {
                asynchronous,
                ..Options::default()
            },
        )
        .unwrap() else {
            panic!("expected admission")
        };
        machine
    }

    fn observed() -> Observations {
        Observations {
            logger_available: true,
            has_fallbacks: false,
        }
    }

    fn reach(machine: &mut Lifecycle, operation: Operation) {
        for _ in 0..12 {
            if machine.operation() == operation {
                return;
            }
            machine.advance(Outcome::Success, observed()).unwrap();
        }
        panic!("operation not reached: {operation:?}")
    }

    #[test]
    fn success_sequences_and_completion_are_core_selected() {
        use Operation::*;
        for (asynchronous, expected) in [
            (false, vec![Setup, Prepare, Send, SyncSuccess, Restore]),
            (
                true,
                vec![
                    Setup,
                    DeploymentPre,
                    Prepare,
                    Send,
                    DeploymentSuccess,
                    AsyncSuccess,
                    SyncSuccessIfNeeded,
                    Restore,
                ],
            ),
        ] {
            let mut machine = machine(asynchronous);
            for operation in expected {
                assert_eq!(machine.operation(), operation);
                assert_eq!(
                    machine.advance(Outcome::Success, observed()).unwrap().error,
                    ErrorDisposition::Preserve
                );
            }
            assert_eq!(machine.operation(), Complete(Outcome::Success));
            assert!(machine.advance(Outcome::Success, observed()).is_err());
        }
    }

    #[test]
    fn ordinary_errors_and_cancellation_at_every_execution_stage() {
        use Operation::*;
        for asynchronous in [false, true] {
            let stages = if asynchronous {
                vec![
                    Setup,
                    DeploymentPre,
                    Prepare,
                    Send,
                    DeploymentSuccess,
                    AsyncSuccess,
                    SyncSuccessIfNeeded,
                ]
            } else {
                vec![Setup, Prepare, Send, SyncSuccess]
            };
            for stage in stages {
                for outcome in [Outcome::Failure, Outcome::Abort] {
                    let mut machine = machine(asynchronous);
                    reach(&mut machine, stage);
                    let transition = machine.advance(outcome, observed()).unwrap();
                    assert_eq!(transition.error, ErrorDisposition::Replace);
                    let expected = if outcome == Outcome::Abort {
                        Restore
                    } else if asynchronous && matches!(stage, Prepare | Send) {
                        DeploymentFailure
                    } else {
                        SyncFailure
                    };
                    assert_eq!(transition.operation, expected, "{stage:?}, {outcome:?}");
                    if expected == DeploymentFailure {
                        assert_eq!(
                            machine
                                .advance(Outcome::Success, observed())
                                .unwrap()
                                .operation,
                            SyncFailure
                        );
                    }
                    if outcome == Outcome::Failure {
                        assert_eq!(
                            machine
                                .advance(Outcome::Success, observed())
                                .unwrap()
                                .operation,
                            if asynchronous { AsyncFailure } else { Restore }
                        );
                        if asynchronous {
                            assert_eq!(
                                machine
                                    .advance(Outcome::Success, observed())
                                    .unwrap()
                                    .operation,
                                Restore
                            );
                        }
                    }
                    assert_eq!(
                        machine
                            .advance(Outcome::Success, observed())
                            .unwrap()
                            .operation,
                        Complete(outcome)
                    );
                }
            }
        }
    }

    #[test]
    fn deployment_observer_preserves_opaque_original_error_even_on_abort() {
        for observer_outcome in [Outcome::Success, Outcome::Failure, Outcome::Abort] {
            let mut machine = machine(true);
            reach(&mut machine, Operation::Send);
            let original = Rc::new("original opaque error");
            let mut retained = Rc::clone(&original);
            assert_eq!(
                machine
                    .advance(Outcome::Failure, observed())
                    .unwrap()
                    .operation,
                Operation::DeploymentFailure
            );
            let transition = machine.advance(observer_outcome, observed()).unwrap();
            if transition.error == ErrorDisposition::Replace {
                retained = Rc::new("observer error");
            }
            assert!(Rc::ptr_eq(&original, &retained));
            assert_eq!(transition.operation, Operation::SyncFailure);
            reach(&mut machine, Operation::Restore);
            assert_eq!(
                machine
                    .advance(Outcome::Success, observed())
                    .unwrap()
                    .operation,
                Operation::Complete(Outcome::Failure)
            );
        }
    }

    #[test]
    fn failure_handlers_and_restore_propagate_their_own_errors_without_recursion() {
        for stage in [
            Operation::SyncFailure,
            Operation::AsyncFailure,
            Operation::Restore,
        ] {
            for outcome in [Outcome::Failure, Outcome::Abort] {
                let mut machine = machine(true);
                machine.advance(Outcome::Failure, observed()).unwrap();
                reach(&mut machine, stage);
                let transition = machine.advance(outcome, observed()).unwrap();
                assert_eq!(transition.error, ErrorDisposition::Replace);
                if stage != Operation::Restore {
                    assert_eq!(transition.operation, Operation::Restore);
                    machine.advance(Outcome::Success, observed()).unwrap();
                }
                assert_eq!(machine.operation(), Operation::Complete(outcome));
            }
        }
    }

    #[test]
    fn logger_availability_internal_calls_and_fallbacks_control_logging_only() {
        let mut failed_setup = machine(true);
        assert_eq!(
            failed_setup
                .advance(Outcome::Failure, Observations::default())
                .unwrap()
                .operation,
            Operation::Restore
        );
        for internal_call in [false, true] {
            for has_fallbacks in [false, true] {
                let NativeOutcome::Completed(mut machine) = Lifecycle::new(
                    &request(),
                    Options {
                        asynchronous: true,
                        internal_call,
                        ..Options::default()
                    },
                )
                .unwrap() else {
                    panic!("expected admission")
                };
                reach(&mut machine, Operation::DeploymentSuccess);
                let next = machine
                    .advance(
                        Outcome::Success,
                        Observations {
                            has_fallbacks,
                            ..observed()
                        },
                    )
                    .unwrap();
                assert_eq!(
                    next.operation,
                    if internal_call || has_fallbacks {
                        Operation::SyncSuccessIfNeeded
                    } else {
                        Operation::AsyncSuccess
                    }
                );
            }
        }
        let NativeOutcome::Completed(mut internal) = Lifecycle::new(
            &request(),
            Options {
                asynchronous: true,
                internal_call: true,
                ..Options::default()
            },
        )
        .unwrap() else {
            panic!("expected admission")
        };
        reach(&mut internal, Operation::Prepare);
        assert_eq!(
            internal
                .advance(Outcome::Failure, observed())
                .unwrap()
                .operation,
            Operation::DeploymentFailure
        );
        assert_eq!(
            internal
                .advance(Outcome::Abort, observed())
                .unwrap()
                .operation,
            Operation::Restore
        );
    }

    #[test]
    fn decline_is_admission_only_and_file_is_an_inert_descriptor() {
        for request in [
            OcrRequest {
                document: OcrDocument::File,
                ..request()
            },
            OcrRequest {
                document: OcrDocument::Unsupported,
                ..request()
            },
            OcrRequest {
                stream: true,
                ..request()
            },
            OcrRequest {
                request_format: Some("native".into()),
                ..request()
            },
            OcrRequest {
                model: "openai/model".into(),
                ..request()
            },
        ] {
            assert!(matches!(
                Lifecycle::new(&request, Options::default()),
                Ok(NativeOutcome::Declined(_))
            ));
        }
        assert!(matches!(
            Lifecycle::new(
                &request(),
                Options {
                    credential_method: CredentialMethod::Acquisition,
                    ..Options::default()
                }
            ),
            Ok(NativeOutcome::Declined(_))
        ));
        assert!(matches!(
            Lifecycle::new(
                &OcrRequest {
                    timeout_seconds: f64::NAN,
                    ..request()
                },
                Options::default()
            ),
            Err(Error::InvalidRequest(_))
        ));
        let mut machine = machine(true);
        reach(&mut machine, Operation::Prepare);
        assert!(matches!(
            prepare::prepare(OcrRequest {
                document: OcrDocument::File,
                ..request()
            }),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(
            machine
                .advance(Outcome::Failure, observed())
                .unwrap()
                .operation,
            Operation::DeploymentFailure
        );
        reach(&mut machine, Operation::Restore);
        assert_eq!(
            machine
                .advance(Outcome::Success, observed())
                .unwrap()
                .operation,
            Operation::Complete(Outcome::Failure)
        );
    }

    #[test]
    fn identity_keeps_supplied_provenance_across_hook_replacement_and_sdk_attempts() {
        for _ in 0..2 {
            let NativeOutcome::Completed(mut machine) = Lifecycle::new(
                &request(),
                Options {
                    asynchronous: true,
                    call_id: Some("logical-call".into()),
                    trace_id: Some("trace".into()),
                    ..Options::default()
                },
            )
            .unwrap() else {
                panic!("expected admission")
            };
            reach(&mut machine, Operation::Prepare);
            let prepared = prepare::prepare(OcrRequest {
                model: "mistral/replacement".into(),
                ..request()
            })
            .unwrap();
            assert_eq!(prepared.model, "replacement");
            assert_eq!(
                machine.identity(),
                &Identity {
                    requested_model: "mistral/requested-model".into(),
                    call_id: "logical-call".into(),
                    trace_id: Some("trace".into()),
                    generated_call_id: false,
                }
            );
            reach(&mut machine, Operation::Restore);
            assert_eq!(machine.identity().call_id, "logical-call");
        }
        let first = machine(false);
        let second = machine(false);
        assert!(first.identity().generated_call_id);
        assert_eq!(first.identity().call_id.len(), 36);
        assert_eq!(&first.identity().call_id[14..15], "4");
        assert_ne!(first.identity().call_id, second.identity().call_id);
        assert_eq!(first.identity().trace_id, None);
    }
}
