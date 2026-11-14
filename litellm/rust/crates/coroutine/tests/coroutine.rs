use std::{
    future::Future,
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_coroutine::{Abandoned, Co, Coroutine, CoroutineState, Reply, ResumeError, reply};
use rstest::rstest;
use tokio::time::timeout;

#[derive(Debug)]
enum Ask {
    Name(Reply<&'static str>),
    Count(Reply<u32>),
}

type Test<C> = Coroutine<Ask, C>;

fn yielded<C>(state: Result<CoroutineState<Ask, C>, ResumeError>) -> Ask {
    match state {
        Ok(CoroutineState::Yielded(ask)) => ask,
        Ok(CoroutineState::Complete(_)) => panic!("expected a yield, the body returned"),
        Err(error) => panic!("expected a yield, resume failed: {error}"),
    }
}

fn complete<C>(state: Result<CoroutineState<Ask, C>, ResumeError>) -> C {
    match state {
        Ok(CoroutineState::Complete(output)) => output,
        Ok(CoroutineState::Yielded(ask)) => panic!("expected completion, got {ask:?}"),
        Err(error) => panic!("expected completion, resume failed: {error}"),
    }
}

fn name(ask: Ask) -> Reply<&'static str> {
    match ask {
        Ask::Name(reply) => reply,
        other => panic!("expected a name ask, got {other:?}"),
    }
}

fn count(ask: Ask) -> Reply<u32> {
    match ask {
        Ask::Count(reply) => reply,
        other => panic!("expected a count ask, got {other:?}"),
    }
}

/// A body parked at one name ask, with nothing else going on.
fn suspended_once() -> Test<Result<&'static str, Abandoned>> {
    Coroutine::new(|co| async move { co.yield_(Ask::Name).await })
}

#[tokio::test]
async fn each_typed_answer_resumes_the_yield_that_asked_for_it() {
    let mut coroutine: Test<String> = Coroutine::new(|co| async move {
        let first = co.yield_(Ask::Name).await.unwrap();
        let second = co.yield_(Ask::Count).await.unwrap();
        format!("{first}+{second}")
    });

    name(yielded(coroutine.resume().await)).send("a");
    count(yielded(coroutine.resume().await)).send(2);

    assert_eq!(complete(coroutine.resume().await), "a+2");
}

/// A driver that polls `resume` once, inline, sees every yield the body makes during
/// that poll instead of being sent back to its event loop.
#[test]
fn a_yield_made_while_resuming_is_returned_by_that_same_poll() {
    let mut coroutine: Test<u32> = Coroutine::new(|co| async move {
        let first = co.yield_(Ask::Count).await.unwrap();
        let second = co.yield_(Ask::Count).await.unwrap();
        first + second
    });
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let mut poll_once =
        |coroutine: &mut Test<u32>| match std::pin::pin!(coroutine.resume()).poll(&mut context) {
            std::task::Poll::Ready(state) => state,
            std::task::Poll::Pending => panic!("resume needed a second poll"),
        };

    count(yielded(poll_once(&mut coroutine))).send(1);
    count(yielded(poll_once(&mut coroutine))).send(2);

    assert_eq!(complete(poll_once(&mut coroutine)), 3);
}

#[tokio::test]
async fn the_body_awaits_real_futures_between_yields() {
    let mut coroutine: Test<u32> = Coroutine::new(|co| async move {
        tokio::time::sleep(Duration::from_millis(5)).await;
        co.yield_(Ask::Count).await.unwrap()
    });

    count(yielded(coroutine.resume().await)).send(7);

    assert_eq!(complete(coroutine.resume().await), 7);
}

#[tokio::test]
async fn concurrent_yields_come_out_in_order_and_are_answered_separately() {
    let mut coroutine: Test<(&str, u32)> = Coroutine::new(|co| async move {
        let (first, second) = tokio::join!(co.yield_(Ask::Name), co.yield_(Ask::Count));
        (first.unwrap(), second.unwrap())
    });

    name(yielded(coroutine.resume().await)).send("one");
    count(yielded(coroutine.resume().await)).send(2);

    assert_eq!(complete(coroutine.resume().await), ("one", 2));
}

#[tokio::test]
async fn resuming_before_the_reply_is_settled_is_refused_and_keeps_the_yield_waiting() {
    let mut coroutine = suspended_once();
    let reply = name(yielded(coroutine.resume().await));

    assert_eq!(
        coroutine.resume().await.unwrap_err(),
        ResumeError::Unanswered
    );

    reply.send("real");
    assert_eq!(complete(coroutine.resume().await), Ok("real"));
}

#[tokio::test]
async fn a_dropped_reply_abandons_its_yield() {
    let mut coroutine = suspended_once();
    drop(yielded(coroutine.resume().await));

    assert_eq!(complete(coroutine.resume().await), Err(Abandoned));
}

#[tokio::test]
async fn an_answer_the_yield_no_longer_awaits_is_discarded() {
    let mut coroutine: Test<&str> = Coroutine::new(|co| async move {
        tokio::select! {
            biased;
            _ = co.yield_(Ask::Name) => unreachable!("the answer comes after the body moved on"),
            () = std::future::ready(()) => {}
        }
        co.yield_(Ask::Name).await.unwrap()
    });
    let stale = name(yielded(coroutine.resume().await));
    stale.send("stale");

    name(yielded(coroutine.resume().await)).send("fresh");

    assert_eq!(complete(coroutine.resume().await), "fresh");
}

#[rstest]
#[case::returned(false)]
#[case::cancelled(true)]
#[tokio::test]
async fn a_finished_coroutine_refuses_to_resume(#[case] cancel: bool) {
    let mut coroutine = suspended_once();
    let reply = name(yielded(coroutine.resume().await));
    if cancel {
        coroutine.cancel();
    } else {
        reply.send("done");
        complete(coroutine.resume().await).unwrap();
    }

    assert_eq!(coroutine.resume().await.unwrap_err(), ResumeError::Finished);
}

#[tokio::test]
async fn a_dropped_resume_leaves_the_coroutine_resumable() {
    let mut coroutine: Test<u32> = Coroutine::new(|co| async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        co.yield_(Ask::Count).await.unwrap()
    });
    assert!(
        timeout(Duration::from_millis(1), coroutine.resume())
            .await
            .is_err()
    );

    count(yielded(coroutine.resume().await)).send(3);

    assert_eq!(complete(coroutine.resume().await), 3);
}

struct Dropped(Arc<Mutex<bool>>);

impl Drop for Dropped {
    fn drop(&mut self) {
        *self.0.lock().unwrap() = true;
    }
}

#[tokio::test]
async fn cancel_drops_the_body() {
    let dropped = Arc::new(Mutex::new(false));
    let guard = Dropped(Arc::clone(&dropped));
    let mut coroutine: Test<()> = Coroutine::new(|co| async move {
        let _guard = guard;
        co.yield_(Ask::Count).await.unwrap();
    });
    let _reply = yielded(coroutine.resume().await);

    coroutine.cancel();

    assert!(*dropped.lock().unwrap());
}

#[rstest]
#[case::cancelled(true)]
#[case::dropped(false)]
#[tokio::test]
async fn a_co_that_escaped_the_body_is_abandoned_once_the_coroutine_ends(#[case] cancel: bool) {
    let escaped: Arc<Mutex<Option<Co<Ask>>>> = Arc::default();
    let slot = Arc::clone(&escaped);
    let mut coroutine: Test<()> = Coroutine::new(move |co| {
        *slot.lock().unwrap() = Some(co.clone());
        async move {
            co.yield_(Ask::Count).await.unwrap();
        }
    });
    let _reply = yielded(coroutine.resume().await);
    let co = escaped.lock().unwrap().take().unwrap();
    let waiting = tokio::spawn(async move { co.yield_(Ask::Name).await });
    tokio::task::yield_now().await;

    if cancel {
        coroutine.cancel();
    } else {
        drop(coroutine);
    }

    let outcome = timeout(Duration::from_secs(1), waiting)
        .await
        .expect("an escaped yield waits forever")
        .unwrap();
    assert_eq!(outcome, Err(Abandoned));
}

#[rstest]
#[case::sent(true)]
#[case::dropped(false)]
#[tokio::test]
async fn a_detached_reply_settles_its_answer(#[case] send: bool) {
    let (reply, answer) = reply::<u32>();
    if send {
        reply.send(5);
    } else {
        drop(reply);
    }

    assert_eq!(answer.await, if send { Ok(5) } else { Err(Abandoned) });
}
