//! One call's v1 protocol, with no I/O and no runtime: which envelope each moment of the
//! call produces, who receives it, and how interceptor patches fold into the wire
//! request. A host projects its own values into facts, asks the session what to deliver,
//! and runs the handlers however it can: inline in CPython, or across a process boundary.
//!
//! The two planes are visible in the types. An [`Emission`] has no way back: nothing an
//! observer does or returns re-enters the session. An [`Interception`] is answered, one
//! patch per interceptor, before the request leaves.

use std::collections::BTreeSet;

use litellm_host::event::{FailureOrigin, RequestContext, Timing, WireRequest};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    envelope::{CallFacts, Envelope, ErrorFacts, Event, EventKind, RequestFacts, Sequencer},
    patch::{PatchError, WirePatch, apply},
};

/// What one subscriber asked for, as data: no handler, no runtime object. It is what a
/// host that runs handlers elsewhere reports back once it has loaded them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Subscription {
    pub name: String,
    pub events: BTreeSet<EventKind>,
    pub observes: bool,
    pub intercepts: bool,
}

/// One envelope and the subscribers it goes to, as indexes into the session's
/// subscriptions, in registration order.
#[derive(Clone, Debug, PartialEq)]
pub struct Emission {
    pub envelope: Envelope,
    pub observers: Vec<usize>,
}

/// An interceptor's turn: who is asked, and the request as the earlier ones left it.
#[derive(Clone, Debug, PartialEq)]
pub struct Turn {
    pub subscriber: usize,
    pub request: RequestFacts,
}

/// The wire request on its way through the interceptors, in registration order.
#[derive(Debug)]
pub struct Interception {
    wire: WireRequest,
    context: RequestContext,
    interceptors: Vec<usize>,
    position: usize,
}

pub struct CallSession {
    sequencer: Sequencer,
    subscriptions: Vec<Subscription>,
    streamed: bool,
}

impl Interception {
    /// The interceptor to ask next, or `None` once every one has answered.
    pub fn turn(&self) -> Option<Turn> {
        self.interceptors
            .get(self.position)
            .map(|&subscriber| Turn {
                subscriber,
                request: RequestFacts::new(&self.wire, &self.context),
            })
    }

    /// The current interceptor's answer; an invalid patch ends the interception.
    pub fn patched(self, patch: WirePatch) -> Result<Self, PatchError> {
        Ok(Self {
            wire: apply(self.wire, patch)?,
            position: self.position + 1,
            ..self
        })
    }
}

impl CallSession {
    /// Opens the call. `call.started` is always envelope zero.
    pub fn start(
        call_id: String,
        call_type: &'static str,
        subscriptions: Vec<Subscription>,
        facts: CallFacts,
    ) -> (Self, Emission) {
        let mut session = Self {
            sequencer: Sequencer::new(call_id, call_type),
            subscriptions,
            streamed: false,
        };
        let emission = session.emission(Event::CallStarted(facts));
        (session, emission)
    }

    fn emission(&mut self, event: Event) -> Emission {
        let kind = event.kind();
        Emission {
            envelope: self.sequencer.envelope(event),
            observers: self
                .subscriptions
                .iter()
                .enumerate()
                .filter(|(_, subscription)| {
                    subscription.observes && subscription.events.contains(&kind)
                })
                .map(|(index, _)| index)
                .collect(),
        }
    }

    pub fn interception(&self, wire: WireRequest, context: RequestContext) -> Interception {
        Interception {
            wire,
            context,
            interceptors: self
                .subscriptions
                .iter()
                .enumerate()
                .filter(|(_, subscription)| subscription.intercepts)
                .map(|(index, _)| index)
                .collect(),
            position: 0,
        }
    }

    /// Closes an interception: `request.sending` describes the wire as the last
    /// interceptor left it, which is the wire the host must send.
    pub fn request_sending(&mut self, interception: Interception) -> (Emission, WireRequest) {
        let facts = RequestFacts::new(&interception.wire, &interception.context);
        (
            self.emission(Event::RequestSending(facts)),
            interception.wire,
        )
    }

    pub fn response_received(&mut self, body: String) -> Emission {
        self.emission(Event::ResponseReceived { body })
    }

    /// The call's stream was handed to the caller; the terminal envelope says so.
    pub fn opened(&mut self) {
        self.streamed = true;
    }

    /// A terminal envelope consumes the session, so a call has exactly one, last.
    pub fn succeeded(
        mut self,
        timing: Timing,
        response: Value,
        response_error: Option<String>,
    ) -> Emission {
        let streamed = self.streamed;
        self.emission(Event::CallSucceeded {
            timing: timing.into(),
            streamed,
            response,
            response_error,
        })
    }

    pub fn failed(mut self, timing: Timing, origin: FailureOrigin, error: ErrorFacts) -> Emission {
        let streamed = self.streamed;
        self.emission(Event::CallFailed {
            timing: timing.into(),
            streamed,
            origin: origin.into(),
            error,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, json};
    use strum::VariantArray;

    use super::*;

    const TIMING: Timing = Timing {
        start_time: 1.0,
        end_time: 2.0,
    };

    fn subscription(name: &str, events: &[EventKind], intercepts: bool) -> Subscription {
        Subscription {
            name: name.to_string(),
            events: events.iter().copied().collect(),
            observes: !events.is_empty(),
            intercepts,
        }
    }

    fn facts() -> CallFacts {
        CallFacts {
            start_time: 1.0,
            asynchronous: true,
            metadata: Map::new(),
            metadata_dropped: Vec::new(),
        }
    }

    fn wire() -> WireRequest {
        WireRequest {
            url: "https://provider.example".to_string(),
            headers: Vec::new(),
            body: json!({"input": "original"}),
        }
    }

    fn context() -> RequestContext {
        RequestContext {
            model: "model".to_string(),
            custom_llm_provider: "provider".to_string(),
            optional_params: json!({}),
            secret_fields: Vec::new(),
            api_key: None,
        }
    }

    fn body(value: Value) -> WirePatch {
        serde_json::from_value(json!({"body": value})).unwrap()
    }

    fn seen_body(turn: &Turn) -> Value {
        serde_json::to_value(&turn.request).unwrap()["body"].clone()
    }

    fn error() -> ErrorFacts {
        ErrorFacts {
            class: "builtins.ValueError".to_string(),
            message: "bad".to_string(),
            status_code: Some(400),
        }
    }

    fn start(subscriptions: Vec<Subscription>) -> (CallSession, Emission) {
        CallSession::start("call".to_string(), "ocr", subscriptions, facts())
    }

    #[test]
    fn a_call_numbers_its_envelopes_from_the_start_to_one_terminal() {
        let (mut session, started) = start(vec![subscription("all", EventKind::VARIANTS, false)]);
        let (sending, _) = {
            let interception = session.interception(wire(), context());
            session.request_sending(interception)
        };
        let received = session.response_received("{}".to_string());
        let terminal = session.succeeded(TIMING, json!({"id": "response"}), None);

        let seen: Vec<(u32, EventKind)> = [started, sending, received, terminal]
            .iter()
            .map(|emission| (emission.envelope.seq, emission.envelope.event.kind()))
            .collect();
        assert_eq!(
            seen,
            vec![
                (0, EventKind::CallStarted),
                (1, EventKind::RequestSending),
                (2, EventKind::ResponseReceived),
                (3, EventKind::CallSucceeded),
            ]
        );
    }

    #[test]
    fn an_envelope_goes_only_to_observers_subscribed_to_its_kind_in_registration_order() {
        let (session, started) = start(vec![
            subscription("terminal", &[EventKind::CallFailed], false),
            subscription("interceptor", &[], true),
            subscription(
                "both",
                &[EventKind::CallStarted, EventKind::CallFailed],
                false,
            ),
            subscription("started", &[EventKind::CallStarted], true),
        ]);

        assert_eq!(started.observers, vec![2, 3]);
        assert_eq!(
            session
                .failed(TIMING, FailureOrigin::Call, error())
                .observers,
            vec![0, 2]
        );
    }

    #[test]
    fn each_interceptor_sees_the_request_as_the_earlier_ones_left_it() {
        let (mut session, _) = start(vec![
            subscription("first", &[], true),
            subscription("observer", &[EventKind::RequestSending], false),
            subscription("second", &[], true),
        ]);
        let interception = session.interception(wire(), context());

        let first = interception.turn().unwrap();
        assert_eq!(
            (first.subscriber, seen_body(&first)),
            (0, json!({"input": "original"}))
        );
        let interception = interception
            .patched(body(json!({"input": "first"})))
            .unwrap();

        let second = interception.turn().unwrap();
        assert_eq!(
            (second.subscriber, seen_body(&second)),
            (2, json!({"input": "first"}))
        );
        let interception = interception.patched(WirePatch::default()).unwrap();
        assert_eq!(interception.turn(), None);

        let (sending, sent) = session.request_sending(interception);
        assert_eq!(sent.body, json!({"input": "first"}));
        assert_eq!(sending.observers, vec![1]);
        assert_eq!(
            serde_json::to_value(&sending.envelope).unwrap()["event"]["body"],
            sent.body
        );
    }

    #[test]
    fn an_invalid_patch_ends_the_interception() {
        let (session, _) = start(vec![subscription("interceptor", &[], true)]);
        let patch: WirePatch =
            serde_json::from_value(json!({"headers": {"set": [["Authorization", "stolen"]]}}))
                .unwrap();

        assert_eq!(
            session
                .interception(wire(), context())
                .patched(patch)
                .unwrap_err(),
            PatchError::ProtectedHeader("Authorization".to_string())
        );
    }

    #[test]
    fn a_terminal_envelope_says_whether_the_call_streamed() {
        let (mut streamed, _) = start(Vec::new());
        streamed.opened();
        let (plain, _) = start(Vec::new());

        for (session, expected) in [(streamed, true), (plain, false)] {
            let Event::CallFailed { streamed, .. } = session
                .failed(TIMING, FailureOrigin::Host, error())
                .envelope
                .event
            else {
                panic!("call.failed expected");
            };
            assert_eq!(streamed, expected);
        }
    }

    #[test]
    fn a_subscription_is_plain_data_a_remote_host_can_report() {
        let reported = subscription("remote", &[EventKind::CallSucceeded], true);
        let wire = serde_json::to_value(&reported).unwrap();

        assert_eq!(
            wire,
            json!({"name": "remote", "events": ["call.succeeded"], "observes": true, "intercepts": true})
        );
        assert_eq!(
            serde_json::from_value::<Subscription>(wire).unwrap(),
            reported
        );
    }
}
