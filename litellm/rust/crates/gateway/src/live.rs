//! What the gateway serves from its config, held so a new config can replace it while serving.
//!
//! [`Served`] is everything built from one config that has to agree: the [`AppState`], the
//! [`Admit`] built over it and the native app built from both. [`Live`] holds the current one and
//! replaces it whole, so a request is never admitted by one config and answered by another. Each
//! admitted request carries the [`Served`] that admitted it into the native app, which answers it
//! from that one even when a swap lands in between, and a request already in the native app keeps
//! the app it started in.

use std::{convert::Infallible, sync::Arc};

use arc_swap::ArcSwap;
use axum::{extract::Request, response::Response};
use tower::{Service, ServiceExt, util::BoxCloneSyncService};

use crate::{Admit, AppState, Reason, Refusal, Refused};

type Native = BoxCloneSyncService<Request, Response, Infallible>;

/// Everything built from one config: the state, which requests the native app answers, and the
/// native app itself. `admit` is `None` when the native app answers nothing, as in front of Python
/// when the config Python runs with cannot be loaded.
pub struct Served {
    pub state: AppState,
    pub admit: Option<Arc<Admit>>,
    native: Native,
}

/// The [`Served`] that admitted a request, which the native app answers it from.
#[derive(Clone)]
struct AdmittedBy(Arc<Served>);

impl Served {
    pub fn new<S>(state: AppState, admit: Option<Arc<Admit>>, native: S) -> Self
    where
        S: Service<Request, Response = Response, Error = Infallible>
            + Clone
            + Send
            + Sync
            + 'static,
        S::Future: Send + 'static,
    {
        Self {
            state,
            admit,
            native: BoxCloneSyncService::new(native),
        }
    }

    /// A request refused as denied carries this [`Served`] too, since the native app that
    /// refuses it has to be the one `admit` judged it for.
    async fn admit(self: Arc<Self>, request: Request) -> Result<Request, (Box<Request>, Reason)> {
        let Some(admit) = self.admit.clone() else {
            return Err((
                Box::new(request),
                Reason::new(Refused::Configuration, "configuration"),
            ));
        };
        match admit.admit(request).await {
            Ok(mut admitted) => {
                admitted.extensions_mut().insert(AdmittedBy(self));
                Ok(admitted)
            }
            Err((mut refused, reason)) => {
                if reason.refusal == Refusal::Denied {
                    refused.extensions_mut().insert(AdmittedBy(self));
                }
                Err((refused, reason))
            }
        }
    }
}

/// The [`Served`] the gateway answers new requests from. A swap stores a whole new one, so its
/// parts are always replaced together.
pub struct Live {
    current: ArcSwap<Served>,
    /// Held from loading the current [`Served`] to storing the next, so a swap built from the
    /// current one never loses another swap that landed in between.
    writer: tokio::sync::Mutex<()>,
}

impl Live {
    pub fn new(served: Served) -> Self {
        Self {
            current: ArcSwap::from_pointee(served),
            writer: tokio::sync::Mutex::new(()),
        }
    }

    /// What new requests are served from.
    pub fn current(&self) -> Arc<Served> {
        self.current.load_full()
    }

    /// Replaces what new requests are served from with what `next` builds from the current one,
    /// and gives back the one it replaced. A build that fails replaces nothing. Swaps run one at a
    /// time, so `next` always sees the latest. Requests already admitted finish on the one that
    /// admitted them.
    pub async fn swap<F, E>(&self, next: impl FnOnce(Arc<Served>) -> F) -> Result<Arc<Served>, E>
    where
        F: Future<Output = Result<Served, E>>,
    {
        let _writer = self.writer.lock().await;
        let current = self.current.load_full();
        let built = next(current.clone()).await?;
        self.current.store(Arc::new(built));
        Ok(current)
    }

    /// Puts `request` to the current [`Admit`]. An admitted request carries what admitted it, so
    /// the native app answers it from the same config. A refused one comes back as the client
    /// sent it, with why.
    pub async fn admitted(
        self: Arc<Self>,
        request: Request,
    ) -> Result<Request, (Box<Request>, Reason)> {
        self.current().admit(request).await
    }

    /// The native app of whatever admitted each request, or of the current [`Served`] for a
    /// request [`Live::admitted`] did not admit.
    pub fn native(
        self: Arc<Self>,
    ) -> impl Service<Request, Response = Response, Error = Infallible, Future: Send + 'static>
    + Clone
    + Send
    + Sync
    + 'static {
        tower::service_fn(move |mut request: Request| {
            let served = request
                .extensions_mut()
                .remove::<AdmittedBy>()
                .map_or_else(|| self.current(), |AdmittedBy(served)| served);
            served.native.clone().oneshot(request)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::body::Body;
    use litellm_config::{Config, Environment};
    use litellm_core_utils::get_llm_provider_logic::LlmProviders;
    use rstest::{fixture, rstest};
    use tokio::sync::Semaphore;

    use super::*;

    /// A native app that answers every request with its own `name`, once `release` has a permit.
    fn answering(
        name: &'static str,
        release: Arc<Semaphore>,
    ) -> impl Service<Request, Response = Response, Error = Infallible, Future: Send + 'static>
    + Clone
    + Send
    + Sync
    + 'static {
        tower::service_fn(move |_: Request| {
            let release = release.clone();
            async move {
                drop(release.acquire().await.unwrap());
                Ok::<_, Infallible>(Response::new(Body::from(name)))
            }
        })
    }

    #[fixture]
    fn held() -> Arc<Semaphore> {
        Arc::new(Semaphore::new(0))
    }

    #[fixture]
    fn released() -> Arc<Semaphore> {
        Arc::new(Semaphore::new(1))
    }

    /// What `name`'s config serves: it deploys only `model`, and its native app answers with
    /// `name` once `release` lets it.
    async fn served(name: &'static str, model: &str, release: &Arc<Semaphore>) -> Served {
        let config = Config::from_yaml(&format!(
            "model_list:\n  - model_name: {model}\n    litellm_params:\n      model: anthropic/claude-haiku-5-5\n      api_key: provider-key\ngeneral_settings:\n  master_key: gateway-key\n"
        ))
        .unwrap();
        let state = AppState::build(&Config::default(), &Environment::default())
            .await
            .unwrap();
        let admit = Admit::over(
            &config,
            vec![(model.to_owned(), Ok(LlmProviders::Anthropic))],
        );
        Served::new(
            state,
            Some(Arc::new(admit)),
            answering(name, release.clone()),
        )
    }

    fn asking_for(model: &str) -> Request {
        Request::post("/v1/messages")
            .header("authorization", "Bearer gateway-key")
            .body(Body::from(format!(
                r#"{{"model":"{model}","max_tokens":8,"messages":[]}}"#
            )))
            .unwrap()
    }

    async fn answered_by(response: Response) -> String {
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(body.to_vec()).unwrap()
    }

    async fn swapped_to(live: &Live, next: Served) {
        live.swap(|_| async { Ok::<_, Infallible>(next) })
            .await
            .unwrap();
    }

    #[rstest]
    #[tokio::test]
    async fn a_request_admitted_before_a_swap_is_answered_by_the_native_app_that_admitted_it(
        released: Arc<Semaphore>,
    ) {
        let live = Arc::new(Live::new(served("before", "claude", &released).await));
        let admitted = live.clone().admitted(asking_for("claude")).await.unwrap();

        swapped_to(&live, served("after", "sonnet", &released).await).await;
        let response = live.clone().native().oneshot(admitted).await.unwrap();

        assert_eq!(answered_by(response).await, "before");
    }

    #[rstest]
    #[case::a_model_only_the_old_config_deploys("claude", Some(Refused::Model))]
    #[case::a_model_only_the_new_config_deploys("sonnet", None)]
    #[tokio::test]
    async fn a_request_after_a_swap_is_admitted_by_the_new_config(
        released: Arc<Semaphore>,
        #[case] model: &str,
        #[case] refused: Option<Refused>,
    ) {
        let live = Arc::new(Live::new(served("before", "claude", &released).await));

        swapped_to(&live, served("after", "sonnet", &released).await).await;
        let admitted = live.clone().admitted(asking_for(model)).await;

        assert_eq!(
            admitted.as_ref().err().map(|(_, reason)| reason.why),
            refused
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_request_admitted_after_a_swap_is_answered_by_the_new_native_app(
        released: Arc<Semaphore>,
    ) {
        let live = Arc::new(Live::new(served("before", "claude", &released).await));
        swapped_to(&live, served("after", "sonnet", &released).await).await;

        let admitted = live.clone().admitted(asking_for("sonnet")).await.unwrap();
        let response = live.clone().native().oneshot(admitted).await.unwrap();

        assert_eq!(answered_by(response).await, "after");
    }

    #[rstest]
    #[tokio::test]
    async fn a_request_in_flight_during_a_swap_finishes_on_the_native_app_it_started_in(
        held: Arc<Semaphore>,
        released: Arc<Semaphore>,
    ) {
        let live = Arc::new(Live::new(served("before", "claude", &held).await));
        let admitted = live.clone().admitted(asking_for("claude")).await.unwrap();
        let in_flight = tokio::spawn(live.clone().native().oneshot(admitted));
        tokio::time::sleep(Duration::from_millis(10)).await;

        swapped_to(&live, served("after", "claude", &released).await).await;
        let after = live.clone().admitted(asking_for("claude")).await.unwrap();
        let after = live.clone().native().oneshot(after).await.unwrap();
        held.add_permits(1);
        let before = in_flight.await.unwrap().unwrap();

        assert_eq!(
            (answered_by(before).await, answered_by(after).await),
            ("before".to_owned(), "after".to_owned())
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_swap_that_fails_to_build_keeps_serving_what_it_was_replacing(
        released: Arc<Semaphore>,
    ) {
        let live = Arc::new(Live::new(served("before", "claude", &released).await));

        let failed = live
            .swap(|_| async { Err::<Served, _>("unbuildable") })
            .await;
        let admitted = live.clone().admitted(asking_for("claude")).await.unwrap();
        let response = live.clone().native().oneshot(admitted).await.unwrap();

        assert_eq!(
            (failed.err(), answered_by(response).await.as_str()),
            (Some("unbuildable"), "before")
        );
    }

    /// The name of what `served`'s native app answers with.
    async fn name_of(served: &Served) -> String {
        let response = served
            .native
            .clone()
            .oneshot(asking_for("claude"))
            .await
            .unwrap();
        answered_by(response).await
    }

    /// A swap built from the current [`Served`], whose native app answers with the name of the
    /// one it replaced and one more `+`, the first name in `names` that is.
    async fn one_more(live: &Live, names: &'static [&'static str]) {
        live.swap(|current| async move {
            let previous = name_of(&current).await;
            tokio::task::yield_now().await;
            let name = names
                .iter()
                .find(|name| name.strip_suffix('+') == Some(previous.as_str()))
                .unwrap();
            Ok::<_, Infallible>(served(name, "claude", &released()).await)
        })
        .await
        .unwrap();
    }

    #[rstest]
    #[tokio::test]
    async fn swaps_built_from_the_current_state_never_lose_one_another(released: Arc<Semaphore>) {
        const NAMES: &[&str] = &["base+", "base++"];
        let live = Live::new(served("base", "claude", &released).await);

        tokio::join!(one_more(&live, NAMES), one_more(&live, NAMES));

        assert_eq!(name_of(&live.current()).await, "base++");
    }
}
