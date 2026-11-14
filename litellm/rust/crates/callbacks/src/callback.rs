use std::{any::type_name, sync::Arc};

use futures_util::future::BoxFuture;

use crate::{
    host::{Host, HostCallError},
    input::{FieldError, HookInput, RequestState},
    manifest::Manifest,
    verdict::Verdict,
};

#[derive(Debug, thiserror::Error)]
pub enum CallbackError {
    #[error(transparent)]
    Field(#[from] FieldError),
    #[error(transparent)]
    Host(#[from] HostCallError),
    #[error("the request state handed back is not a {0}")]
    StateType(&'static str),
    #[error("{0}")]
    Failed(String),
}

pub type HookOutput = Result<Option<Verdict>, CallbackError>;

pub trait Callback: Send + Sync + 'static {
    type State: Send + Sync + 'static;

    fn manifest(&self) -> &Manifest;

    fn request_state(&self) -> Option<Self::State> {
        None
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a Self::State>,
    ) -> BoxFuture<'a, HookOutput>;

    fn handle_hosted<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a Self::State>,
        _host: Host,
    ) -> BoxFuture<'a, HookOutput> {
        self.handle(input, state)
    }

    // No caller yet: a port that queues (s3_v2) loses unsent rows at shutdown until the host calls this.
    fn flush(&self) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
}

/// The object-safe form of [`Callback`] that a host stores behind `dyn`. Request state crosses
/// it type-erased and is downcast back to the callback's own `State` in one place, here.
pub trait ErasedCallback: Send + Sync + 'static {
    fn manifest(&self) -> &Manifest;

    fn request_state(&self) -> Option<RequestState>;

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a RequestState>,
    ) -> BoxFuture<'a, HookOutput>;

    fn handle_hosted<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a RequestState>,
        host: Host,
    ) -> BoxFuture<'a, HookOutput>;

    fn flush(&self) -> BoxFuture<'_, ()>;
}

fn typed<S: 'static>(state: Option<&RequestState>) -> Result<Option<&S>, CallbackError> {
    state
        .map(|state| {
            state
                .downcast_ref::<S>()
                .ok_or(CallbackError::StateType(type_name::<S>()))
        })
        .transpose()
}

impl<C: Callback> ErasedCallback for C {
    fn manifest(&self) -> &Manifest {
        Callback::manifest(self)
    }

    fn request_state(&self) -> Option<RequestState> {
        Callback::request_state(self).map(|state| Arc::new(state) as RequestState)
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a RequestState>,
    ) -> BoxFuture<'a, HookOutput> {
        match typed::<C::State>(state) {
            Ok(state) => Callback::handle(self, input, state),
            Err(error) => Box::pin(async move { Err(error) }),
        }
    }

    fn handle_hosted<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a RequestState>,
        host: Host,
    ) -> BoxFuture<'a, HookOutput> {
        match typed::<C::State>(state) {
            Ok(state) => Callback::handle_hosted(self, input, state, host),
            Err(error) => Box::pin(async move { Err(error) }),
        }
    }

    fn flush(&self) -> BoxFuture<'_, ()> {
        Callback::flush(self)
    }
}

#[cfg(test)]
mod tests {
    use futures_util::FutureExt;
    use rstest::rstest;

    use super::*;
    use crate::{
        hook::Hook,
        input::FieldValues,
        manifest::{Selection, Subscription},
        verdict::Patch,
    };

    struct Counting(Manifest);

    impl Callback for Counting {
        type State = u32;

        fn manifest(&self) -> &Manifest {
            &self.0
        }

        fn request_state(&self) -> Option<u32> {
            Some(7)
        }

        fn handle<'a>(
            &'a self,
            _input: &'a HookInput,
            state: Option<&'a u32>,
        ) -> BoxFuture<'a, HookOutput> {
            let seen = state.copied();
            Box::pin(async move {
                Ok(seen.map(|seen| {
                    Verdict::Patch(Patch::default().set(
                        crate::field::Field::Model,
                        serde_json::json!(seen.to_string()),
                    ))
                }))
            })
        }
    }

    fn counting() -> Arc<dyn ErasedCallback> {
        Arc::new(Counting(Manifest {
            selection: Selection::named("counting"),
            subscriptions: vec![Subscription::on(Hook::PreCall)],
            has_request_state: true,
        }))
    }

    fn input() -> HookInput {
        HookInput {
            hook: Hook::PreCall,
            fields: FieldValues::default(),
            call: None,
        }
    }

    #[rstest]
    fn the_state_a_callback_creates_comes_back_to_it_typed() {
        let callback = counting();
        let state = callback.request_state();
        let verdict = callback
            .handle(&input(), state.as_ref())
            .now_or_never()
            .unwrap()
            .unwrap();
        assert_eq!(
            verdict,
            Some(Verdict::Patch(
                Patch::default().set(crate::field::Field::Model, serde_json::json!("7"))
            ))
        );
    }

    #[rstest]
    fn a_callback_with_no_state_sees_none() {
        let verdict = counting()
            .handle(&input(), None)
            .now_or_never()
            .unwrap()
            .unwrap();
        assert_eq!(verdict, None);
    }

    #[rstest]
    fn state_of_another_type_is_an_error_not_a_panic() {
        let foreign: RequestState = Arc::new(String::from("not a u32"));
        let error = counting()
            .handle_hosted(&input(), Some(&foreign), Host::detached())
            .now_or_never()
            .unwrap()
            .unwrap_err();
        assert!(matches!(error, CallbackError::StateType("u32")), "{error}");
    }
}
