use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, PoisonError},
};

use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

pub enum Limiter {
    Fixed(Arc<Semaphore>),
    Adaptive(Arc<Adaptive>),
}

pub enum Permit {
    Fixed(#[allow(dead_code)] OwnedSemaphorePermit),
    Adaptive(Arc<Adaptive>),
}

impl Drop for Permit {
    fn drop(&mut self) {
        if let Self::Adaptive(limiter) = self {
            limiter.release();
        }
    }
}

impl Limiter {
    pub fn new(width: usize, adaptive: bool, ceiling: usize) -> Self {
        if adaptive {
            Self::Adaptive(Arc::new(Adaptive::new(width, width, width.max(ceiling))))
        } else {
            Self::Fixed(Arc::new(Semaphore::new(width.min(Semaphore::MAX_PERMITS))))
        }
    }

    pub async fn acquire(&self) -> Permit {
        match self {
            Self::Fixed(semaphore) => Permit::Fixed(
                Arc::clone(semaphore)
                    .acquire_owned()
                    .await
                    .unwrap_or_else(|_| unreachable!("the upload semaphore is never closed")),
            ),
            Self::Adaptive(limiter) => {
                limiter.acquire().await;
                Permit::Adaptive(Arc::clone(limiter))
            }
        }
    }

    pub fn record(&self, throttled: bool) {
        if let Self::Adaptive(limiter) = self {
            limiter.record(throttled);
        }
    }

    #[cfg(test)]
    pub fn limit(&self) -> Option<usize> {
        match self {
            Self::Fixed(_) => None,
            Self::Adaptive(limiter) => Some(limiter.state().limit),
        }
    }
}

struct State {
    limit: usize,
    clean_streak: usize,
    in_flight: usize,
    waiters: VecDeque<oneshot::Sender<()>>,
}

pub struct Adaptive {
    floor: usize,
    ceiling: usize,
    state: Mutex<State>,
}

impl Adaptive {
    fn new(initial: usize, floor: usize, ceiling: usize) -> Self {
        Self {
            floor,
            ceiling,
            state: Mutex::new(State {
                limit: initial.max(floor).min(ceiling),
                clean_streak: 0,
                in_flight: 0,
                waiters: VecDeque::new(),
            }),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    async fn acquire(&self) {
        let waiter = {
            let mut state = self.state();
            if state.in_flight < state.limit {
                state.in_flight += 1;
                return;
            }
            let (sender, receiver) = oneshot::channel();
            state.waiters.push_back(sender);
            receiver
        };
        let _ = waiter.await;
    }

    fn grant(state: &mut State) {
        while state.in_flight < state.limit {
            let Some(waiter) = state.waiters.pop_front() else {
                return;
            };
            if waiter.send(()).is_ok() {
                state.in_flight += 1;
            }
        }
    }

    fn release(&self) {
        let mut state = self.state();
        state.in_flight = state.in_flight.saturating_sub(1);
        Self::grant(&mut state);
    }

    fn record(&self, throttled: bool) {
        let mut state = self.state();
        if throttled {
            state.limit = self.floor.max(state.limit / 2);
            state.clean_streak = 0;
            return;
        }
        state.clean_streak += 1;
        if state.clean_streak >= state.limit && state.limit < self.ceiling {
            state.limit += 1;
            state.clean_streak = 0;
            Self::grant(&mut state);
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn the_limit_grows_after_a_clean_streak_and_halves_on_a_throttle() {
        let limiter = Limiter::new(2, true, 8);
        for _ in 0..2 {
            limiter.record(false);
        }
        assert_eq!(limiter.limit(), Some(3));
        for _ in 0..3 {
            limiter.record(false);
        }
        assert_eq!(limiter.limit(), Some(4));
        limiter.record(true);
        assert_eq!(limiter.limit(), Some(2));
        limiter.record(true);
        assert_eq!(limiter.limit(), Some(2));
    }

    #[rstest]
    #[tokio::test]
    async fn a_bound_past_the_semaphore_maximum_acts_as_no_bound() {
        let limiter = Limiter::new(usize::MAX, false, 0);
        let _first = limiter.acquire().await;
        let _second = limiter.acquire().await;
    }

    #[rstest]
    #[tokio::test]
    async fn a_full_limit_admits_the_next_waiter_when_a_permit_drops() {
        let limiter = Arc::new(Limiter::new(1, true, 1));
        let first = limiter.acquire().await;
        let waiting = {
            let limiter = Arc::clone(&limiter);
            tokio::spawn(async move {
                let _permit = limiter.acquire().await;
            })
        };
        tokio::task::yield_now().await;
        assert!(!waiting.is_finished());
        drop(first);
        waiting.await.unwrap();
    }

    fn adaptive(initial: usize, floor: usize, ceiling: usize) -> Arc<Limiter> {
        Arc::new(Limiter::Adaptive(Arc::new(Adaptive::new(
            initial, floor, ceiling,
        ))))
    }

    fn in_flight(limiter: &Limiter) -> usize {
        match limiter {
            Limiter::Fixed(_) => 0,
            Limiter::Adaptive(adaptive) => adaptive.state().in_flight,
        }
    }

    fn record(limiter: &Limiter, throttled: bool, times: usize) {
        for _ in 0..times {
            limiter.record(throttled);
        }
    }

    #[rstest]
    #[case(4, 1, 16, &[(false, 4)], 5)]
    #[case(4, 1, 16, &[(false, 3)], 4)]
    #[case(16, 1, 16, &[(true, 1)], 8)]
    #[case(4, 4, 16, &[(true, 1)], 4)]
    #[case(15, 1, 16, &[(false, 1000)], 16)]
    #[case(4, 1, 32, &[(false, 3), (true, 1), (false, 1)], 2)]
    fn the_limit_follows_the_samples(
        #[case] initial: usize,
        #[case] floor: usize,
        #[case] ceiling: usize,
        #[case] samples: &[(bool, usize)],
        #[case] expected: usize,
    ) {
        let limiter = adaptive(initial, floor, ceiling);
        for &(throttled, times) in samples {
            record(&limiter, throttled, times);
        }
        assert_eq!(limiter.limit(), Some(expected));
    }

    #[rstest]
    #[tokio::test]
    async fn growing_the_limit_wakes_a_waiting_acquirer() {
        let limiter = adaptive(1, 1, 4);
        let holder = limiter.acquire().await;
        let waiting = {
            let limiter = Arc::clone(&limiter);
            tokio::spawn(async move {
                let _permit = limiter.acquire().await;
            })
        };
        tokio::task::yield_now().await;
        assert!(!waiting.is_finished());
        limiter.record(false);
        waiting.await.unwrap();
        drop(holder);
        assert_eq!(in_flight(&limiter), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn releasing_a_slot_wakes_exactly_one_waiter() {
        let limiter = adaptive(1, 1, 4);
        let holder = limiter.acquire().await;
        let acquired = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (release, released) = tokio::sync::watch::channel(false);
        let waiters: Vec<_> = (0..3)
            .map(|_| {
                let limiter = Arc::clone(&limiter);
                let acquired = Arc::clone(&acquired);
                let mut released = released.clone();
                tokio::spawn(async move {
                    let _permit = limiter.acquire().await;
                    acquired.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let _ = released.wait_for(|released| *released).await;
                })
            })
            .collect();
        tokio::task::yield_now().await;
        drop(holder);
        tokio::task::yield_now().await;
        assert_eq!(acquired.load(std::sync::atomic::Ordering::SeqCst), 1);
        for _ in 0..3 {
            limiter.record(false);
            tokio::task::yield_now().await;
        }
        assert_eq!(acquired.load(std::sync::atomic::Ordering::SeqCst), 3);
        release.send(true).unwrap();
        for waiter in waiters {
            waiter.await.unwrap();
        }
        assert_eq!(in_flight(&limiter), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn a_cancelled_waiter_is_skipped_and_leaves_in_flight_at_zero() {
        let limiter = adaptive(1, 1, 1);
        let holder = limiter.acquire().await;
        let spawn_waiter = |limiter: Arc<Limiter>| {
            tokio::spawn(async move {
                let _permit = limiter.acquire().await;
            })
        };
        let doomed = spawn_waiter(Arc::clone(&limiter));
        let next = spawn_waiter(Arc::clone(&limiter));
        tokio::task::yield_now().await;
        doomed.abort();
        assert!(doomed.await.unwrap_err().is_cancelled());
        drop(holder);
        next.await.unwrap();
        assert_eq!(in_flight(&limiter), 0);
    }
}
