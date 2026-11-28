//! The request fields a Messages call sent without, because its provider would have refused them.
//!
//! A transformation that takes a field out of the caller's request names it here with [`note`],
//! as the caller wrote it (`temperature`, `output_config.effort`). The route runs a call inside
//! [`scoped`] and reads the names back with [`noted`], so a drop made several functions down
//! reaches the response without every transformation in between returning it. Outside a call
//! nothing is kept.

use std::{cell::RefCell, future::Future};

tokio::task_local! {
    static DROPPED: RefCell<Vec<String>>;
}

/// Runs `future` as one call, with nothing dropped yet.
pub async fn scoped<F: Future>(future: F) -> F::Output {
    DROPPED.scope(RefCell::default(), future).await
}

/// The fields the call this code runs in dropped so far, each once, in the order they went.
pub fn noted() -> Vec<String> {
    DROPPED
        .try_with(|dropped| dropped.borrow().clone())
        .unwrap_or_default()
}

/// Names `field` as dropped from the request of the call this code runs in.
pub fn note(field: &str) {
    let _ = DROPPED.try_with(|dropped| {
        let mut dropped = dropped.borrow_mut();
        if !dropped.iter().any(|noted| noted == field) {
            dropped.push(field.to_owned());
        }
    });
}

/// What `run` returned and the fields it dropped, for a test of one transformation.
#[cfg(test)]
pub(crate) fn dropped_by<R>(run: impl FnOnce() -> R) -> (R, Vec<String>) {
    DROPPED.sync_scope(RefCell::default(), || {
        let output = run();
        (output, noted())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_call_reads_back_each_field_it_dropped_once_in_order() {
        let ((), dropped) = dropped_by(|| {
            note("top_k");
            note("temperature");
            note("top_k");
        });
        assert_eq!(dropped, ["top_k", "temperature"]);
    }

    #[tokio::test]
    async fn a_call_keeps_its_drops_across_an_await_and_apart_from_the_next_call() {
        let first = scoped(async {
            note("speed");
            tokio::task::yield_now().await;
            noted()
        })
        .await;
        let second = scoped(async { noted() }).await;
        assert_eq!((first, second), (vec!["speed".to_owned()], Vec::new()));
    }

    #[test]
    fn outside_a_call_a_drop_is_not_kept() {
        note("speed");
        assert_eq!(noted(), Vec::<String>::new());
    }
}
