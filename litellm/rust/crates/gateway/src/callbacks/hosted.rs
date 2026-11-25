//! A callback's turn when the proxy answers its host calls, as `callbacks-python` runs one with
//! Python behind it: the callback's hosted handler runs as a coroutine, and each host call it
//! yields is answered by the gateway's [`Answers`] for it before the handler resumes.

use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    CallbackError, ErasedCallback, HookInput, HookOutput, HostReply, HostRequest, RequestState,
    hosted,
};
use litellm_coroutine::CoroutineState;
use serde_json::Value;

/// What answers a callback's host calls, by the call's name and its request.
pub trait Answers: Send + Sync + 'static {
    fn answer(&self, call: &'static str, request: Value) -> BoxFuture<'_, HostReply>;
}

/// `callback`'s handler on `input`, each host call it makes answered by `answers`.
pub(crate) async fn turn(
    callback: Arc<dyn ErasedCallback>,
    input: HookInput,
    state: Option<RequestState>,
    answers: &dyn Answers,
) -> HookOutput {
    let mut body =
        hosted(
            move |host| async move { callback.handle_hosted(&input, state.as_ref(), host).await },
        );
    loop {
        match body.resume().await {
            Ok(CoroutineState::Yielded(HostRequest {
                call,
                request,
                reply,
            })) => reply.send(answers.answer(call, request).await),
            Ok(CoroutineState::Complete(output)) => return output,
            Err(error) => return Err(CallbackError::Failed(error.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use litellm_callbacks::{
        Callback, Field, FieldSet, FieldValues, Hook, Host, HostCall, HostFault, Manifest, Patch,
        Selection, Subscription, Verdict,
    };
    use rstest::rstest;
    use serde::Serialize;
    use serde_json::json;

    use super::*;

    #[derive(Serialize)]
    struct Spend {
        key: String,
    }

    impl HostCall for Spend {
        const NAME: &'static str = "spend";
        type Reply = f64;
    }

    /// Asks its host for a key's spend twice and publishes the sum as response headers.
    struct Asking(Manifest);

    impl Callback for Asking {
        type State = ();

        fn manifest(&self) -> &Manifest {
            &self.0
        }

        fn handle<'a>(&'a self, _: &'a HookInput, _: Option<&'a ()>) -> BoxFuture<'a, HookOutput> {
            Box::pin(async { Ok(None) })
        }

        fn handle_hosted<'a>(
            &'a self,
            _: &'a HookInput,
            _: Option<&'a ()>,
            host: Host,
        ) -> BoxFuture<'a, HookOutput> {
            Box::pin(async move {
                let first = host
                    .call(Spend {
                        key: "a".to_owned(),
                    })
                    .await?;
                let second = host
                    .call(Spend {
                        key: "b".to_owned(),
                    })
                    .await?;
                Ok(Some(Verdict::Patch(Patch::default().set(
                    Field::ResponseHeaders,
                    json!({"x-spend": first + second}),
                ))))
            })
        }
    }

    struct ByKey;

    impl Answers for ByKey {
        fn answer(&self, call: &'static str, request: Value) -> BoxFuture<'_, HostReply> {
            Box::pin(async move {
                match (call, request["key"].as_str()) {
                    ("spend", Some("a")) => Ok(json!(1.5)),
                    ("spend", Some("b")) => Ok(json!(2.0)),
                    _ => Err(HostFault::Malformed(format!("no {call}"))),
                }
            })
        }
    }

    struct Refusing;

    impl Answers for Refusing {
        fn answer(&self, call: &'static str, _: Value) -> BoxFuture<'_, HostReply> {
            Box::pin(async move { Err(HostFault::Malformed(format!("no {call}"))) })
        }
    }

    fn input() -> HookInput {
        HookInput {
            hook: Hook::AttemptEnded,
            fields: FieldValues::new(FieldSet::of(&[])),
            call: None,
        }
    }

    fn asking() -> Arc<dyn ErasedCallback> {
        Arc::new(Asking(Manifest {
            selection: Selection::named("asking"),
            subscriptions: vec![Subscription::on(Hook::AttemptEnded)],
            has_request_state: false,
        }))
    }

    #[rstest]
    #[tokio::test]
    async fn each_host_call_is_answered_before_the_callback_goes_on() {
        let verdict = turn(asking(), input(), None, &ByKey).await.unwrap();
        assert_eq!(
            verdict,
            Some(Verdict::Patch(
                Patch::default().set(Field::ResponseHeaders, json!({"x-spend": 3.5}))
            ))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_host_call_the_gateway_does_not_answer_fails_the_callback() {
        let failed = turn(asking(), input(), None, &Refusing).await;
        assert!(matches!(failed, Err(CallbackError::Host(_))), "{failed:?}");
    }
}
