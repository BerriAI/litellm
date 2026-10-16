//! Tells the mounting server which deployment a request resolved to, so its metrics layer can
//! label by model group and deployment without parsing the request body a second time.
//!
//! The server wraps a request in [`scope`]; the handler records the deployment it picked through
//! a task-local slot. Outside a scope recording is a no-op.

use std::{cell::RefCell, future::Future};

use crate::Deployment;

/// The deployment a request was routed to.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Resolved {
    /// The public model name the client asked for (the config's `model_name`).
    pub model_group: String,
    pub api_base: Option<String>,
}

tokio::task_local! {
    static RESOLVED: RefCell<Option<Resolved>>;
}

/// Runs `future` with a fresh slot and returns its output with what the handler recorded.
pub async fn scope<F: Future>(future: F) -> (F::Output, Option<Resolved>) {
    RESOLVED
        .scope(RefCell::new(None), async {
            let output = future.await;
            (output, RESOLVED.with(RefCell::take))
        })
        .await
}

pub(crate) fn record(model_group: &str, deployment: &Deployment) {
    let _ = RESOLVED.try_with(|slot| {
        *slot.borrow_mut() = Some(Resolved {
            model_group: model_group.to_owned(),
            api_base: deployment.api_base.clone(),
        });
    });
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn deployment() -> Deployment {
        Deployment {
            api_base: Some("http://mock:9000/d/chat-0-0".into()),
            ..Deployment::default()
        }
    }

    #[rstest]
    #[tokio::test]
    async fn scope_returns_the_recorded_deployment() {
        let ((), resolved) = scope(async { record("gpt-group", &deployment()) }).await;

        assert_eq!(
            resolved,
            Some(Resolved {
                model_group: "gpt-group".into(),
                api_base: Some("http://mock:9000/d/chat-0-0".into()),
            })
        );
    }

    #[rstest]
    #[tokio::test]
    async fn scope_without_a_record_returns_none() {
        let ((), resolved) = scope(async {}).await;

        assert_eq!(resolved, None);
    }

    #[rstest]
    fn recording_outside_a_scope_is_a_no_op() {
        record("gpt-group", &deployment());
    }
}
