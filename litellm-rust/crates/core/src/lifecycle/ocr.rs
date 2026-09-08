use crate::Error;
use crate::ocr::{OcrAdmissionRequest, request};

use super::program::{CallProgram, ProgramOptions, actions_for};
use super::{ActionBinding, LifecycleRoute, Outcome};

pub use super::program::{Observations, Operation, Transition};

#[derive(Debug)]
pub enum NativeOutcome<T> {
    Completed(T),
    Declined(Decline),
}

#[derive(Debug, PartialEq, Eq)]
pub struct Decline(&'static str);

impl Decline {
    pub fn reason(&self) -> &'static str {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CredentialMethod {
    #[default]
    Configured,
    Acquisition,
}

#[derive(Default)]
pub struct Options {
    pub asynchronous: bool,
    pub internal_call: bool,
    pub call_id: Option<String>,
    pub trace_id: Option<String>,
    pub credential_method: CredentialMethod,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Identity {
    pub requested_model: String,
    pub call_id: String,
    pub trace_id: Option<String>,
    pub generated_call_id: bool,
}

#[derive(Debug)]
pub struct OcrState {
    program: CallProgram,
    identity: Identity,
}

#[derive(Debug)]
pub struct OcrRoute;

pub type Lifecycle = super::Lifecycle<OcrRoute>;

impl Lifecycle {
    pub fn new(
        admission: &OcrAdmissionRequest,
        options: Options,
    ) -> Result<NativeOutcome<Self>, Error> {
        <super::Lifecycle<OcrRoute>>::admit(admission, options).map(|admission| match admission {
            Ok(lifecycle) => NativeOutcome::Completed(lifecycle),
            Err(decline) => NativeOutcome::Declined(decline),
        })
    }

    pub fn identity(&self) -> &Identity {
        &self.state.identity
    }

    pub fn commitment(&self) -> super::Commitment {
        self.state.program.commitment()
    }

    pub fn failure_stage(&self) -> Option<super::FailureStage> {
        self.state.program.failure_stage()
    }
}

impl LifecycleRoute for OcrRoute {
    type Admission = OcrAdmissionRequest;
    type Options = Options;
    type Context = Observations;
    type Operation = Operation;
    type Observation = Observations;
    type Outcome = Outcome;
    type Transition = Transition;
    type Error = Error;
    type Decline = Decline;
    type State = OcrState;

    fn admit(
        admission: &Self::Admission,
        options: Self::Options,
    ) -> Result<Result<Self::State, Self::Decline>, Self::Error> {
        match request::admission_capabilities(admission) {
            Err(Error::Unsupported(reason)) => return Ok(Err(Decline(reason))),
            Err(error) => return Err(error),
            Ok(()) => {}
        }
        if options.credential_method == CredentialMethod::Acquisition {
            return Ok(Err(Decline("OCR credential acquisition")));
        }
        let generated_call_id = options.call_id.is_none();
        let call_id = options.call_id.unwrap_or_else(generate_call_id);
        Ok(Ok(OcrState {
            program: CallProgram::new(ProgramOptions {
                asynchronous: options.asynchronous,
                internal_call: options.internal_call,
            }),
            identity: Identity {
                requested_model: admission.model.clone(),
                call_id,
                trace_id: options.trace_id,
                generated_call_id,
            },
        }))
    }

    fn operation(state: &Self::State) -> Self::Operation {
        state.program.operation()
    }

    fn advance(
        state: &mut Self::State,
        outcome: Self::Outcome,
        observations: Self::Observation,
    ) -> Result<Self::Transition, Self::Error> {
        state
            .program
            .advance(outcome, observations)
            .ok_or_else(|| Error::InvalidRequest("OCR lifecycle is already complete".into()))
    }

    fn actions_for(
        operation: Self::Operation,
        _context: &Self::Context,
    ) -> &'static [ActionBinding] {
        actions_for(operation)
    }
}

fn generate_call_id() -> String {
    let id = (rand::random::<u128>() & !(0xf000_u128 << 64 | 0xc000_u128 << 48))
        | (0x4000_u128 << 64 | 0x8000_u128 << 48);
    let hex = format!("{id:032x}");
    format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    )
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;
    use crate::lifecycle::ErrorDisposition;
    use crate::ocr::types::OcrDocument;

    fn request() -> OcrAdmissionRequest {
        OcrAdmissionRequest {
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
        let NativeOutcome::Completed(supplied) = Lifecycle::new(
            &request(),
            Options {
                asynchronous,
                ..Options::default()
            },
        )
        .unwrap() else {
            panic!("expected admission")
        };
        supplied
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
            (
                false,
                vec![Setup, BuildRequest, PreCall, Send, SyncSuccess, Restore],
            ),
            (
                true,
                vec![
                    Setup,
                    DeploymentPre,
                    BuildRequest,
                    PreCall,
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
    fn failures_and_cancellation_transition_without_recursion() {
        use Operation::*;
        for asynchronous in [false, true] {
            let stages = if asynchronous {
                vec![
                    Setup,
                    DeploymentPre,
                    BuildRequest,
                    PreCall,
                    Send,
                    DeploymentSuccess,
                    AsyncSuccess,
                    SyncSuccessIfNeeded,
                ]
            } else {
                vec![Setup, BuildRequest, PreCall, Send, SyncSuccess]
            };
            for stage in stages {
                for outcome in [Outcome::Failure, Outcome::Abort] {
                    let mut machine = machine(asynchronous);
                    reach(&mut machine, stage);
                    let transition = machine.advance(outcome, observed()).unwrap();
                    assert_eq!(transition.error, ErrorDisposition::Replace);
                    let expected = if outcome == Outcome::Abort {
                        Restore
                    } else if asynchronous && matches!(stage, BuildRequest | PreCall | Send) {
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
    fn deployment_failure_observer_preserves_the_original_error() {
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
            assert_eq!(
                machine.failure_stage(),
                Some(crate::lifecycle::FailureStage::ProviderCall)
            );
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
                assert_eq!(
                    machine
                        .advance(
                            Outcome::Success,
                            Observations {
                                has_fallbacks,
                                ..observed()
                            },
                        )
                        .unwrap()
                        .operation,
                    if internal_call || has_fallbacks {
                        Operation::SyncSuccessIfNeeded
                    } else {
                        Operation::AsyncSuccess
                    }
                );
            }
        }
    }

    #[test]
    fn admission_declines_unsupported_requests_without_effects() {
        for request in [
            OcrAdmissionRequest {
                document: OcrDocument::File,
                ..request()
            },
            OcrAdmissionRequest {
                document: OcrDocument::Unsupported,
                ..request()
            },
            OcrAdmissionRequest {
                stream: true,
                ..request()
            },
            OcrAdmissionRequest {
                request_format: Some("native".into()),
                ..request()
            },
            OcrAdmissionRequest {
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
                &OcrAdmissionRequest {
                    timeout_seconds: f64::NAN,
                    ..request()
                },
                Options::default()
            ),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[test]
    fn identity_preserves_supplied_provenance_and_generates_uuid_v4() {
        let NativeOutcome::Completed(supplied) = Lifecycle::new(
            &request(),
            Options {
                call_id: Some("logical-call".into()),
                trace_id: Some("trace".into()),
                ..Options::default()
            },
        )
        .unwrap() else {
            panic!("expected admission")
        };
        assert_eq!(
            supplied.identity(),
            &Identity {
                requested_model: "mistral/requested-model".into(),
                call_id: "logical-call".into(),
                trace_id: Some("trace".into()),
                generated_call_id: false,
            }
        );
        let first = machine(false);
        let second = machine(false);
        assert!(first.identity().generated_call_id);
        assert_eq!(first.identity().call_id.len(), 36);
        assert_eq!(&first.identity().call_id[14..15], "4");
        assert_ne!(first.identity().call_id, second.identity().call_id);
    }
}
