use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{Machine, Observer, Shared};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub failure_threshold: u32,
    pub recovery_timeout: Duration,
    pub timeout_min_duration: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(60),
            timeout_min_duration: Duration::from_secs(5),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Failure {
    Hard,
    Timeout,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum State {
    Closed {
        hard: u32,
        timeouts: u32,
        streak_started: Option<Instant>,
    },
    Open {
        since: Instant,
    },
    HalfOpen,
}

impl State {
    const CLOSED: Self = Self::Closed {
        hard: 0,
        timeouts: 0,
        streak_started: None,
    };
}

#[derive(Clone, Copy, Debug)]
pub enum Event {
    Admit(Instant),
    Succeeded,
    Failed(Failure, Instant),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    Admitted,
    Probe,
    Rejected,
    Recorded,
}

impl Machine for Config {
    type State = State;
    type Event = Event;
    type Output = Output;

    fn step(&self, state: &State, event: Event) -> (State, Output) {
        match (state, event) {
            (State::Closed { .. }, Event::Admit(_)) => (state.clone(), Output::Admitted),
            (State::Open { since }, Event::Admit(now))
                if now.duration_since(*since) > self.recovery_timeout =>
            {
                (State::HalfOpen, Output::Probe)
            }
            (State::Open { .. } | State::HalfOpen, Event::Admit(_)) => {
                (state.clone(), Output::Rejected)
            }
            (State::Closed { .. } | State::HalfOpen, Event::Succeeded) => {
                (State::CLOSED, Output::Recorded)
            }
            (State::Open { .. }, Event::Succeeded) => (state.clone(), Output::Recorded),
            (State::Closed { hard, .. }, Event::Failed(Failure::Hard, now)) => {
                let hard = hard + 1;
                let next = match hard >= self.failure_threshold {
                    true => State::Open { since: now },
                    false => State::Closed {
                        hard,
                        timeouts: 0,
                        streak_started: None,
                    },
                };
                (next, Output::Recorded)
            }
            (
                State::Closed {
                    hard,
                    timeouts,
                    streak_started,
                },
                Event::Failed(Failure::Timeout, now),
            ) => {
                let timeouts = timeouts + 1;
                let started = streak_started.unwrap_or(now);
                let opens = timeouts >= self.failure_threshold
                    && now.duration_since(started) >= self.timeout_min_duration;
                let next = match opens {
                    true => State::Open { since: now },
                    false => State::Closed {
                        hard: *hard,
                        timeouts,
                        streak_started: Some(started),
                    },
                };
                (next, Output::Recorded)
            }
            (State::Open { .. } | State::HalfOpen, Event::Failed(_, now)) => {
                (State::Open { since: now }, Output::Recorded)
            }
        }
    }
}

#[derive(Clone)]
pub struct Breaker(Arc<Shared<Config>>);

#[derive(Debug, PartialEq, Eq)]
pub struct Rejected;

pub struct Admission {
    breaker: Breaker,
    generation: u64,
    probe: bool,
    settled: bool,
}

impl Breaker {
    pub fn new(config: Config, observer: Observer<State>) -> Self {
        Self(Arc::new(Shared::new(config, State::CLOSED, observer)))
    }

    pub fn admit(&self, now: Instant) -> Result<Admission, Rejected> {
        let (output, generation) = self.0.fire(Event::Admit(now));
        let probe = match output {
            Output::Admitted => false,
            Output::Probe => true,
            Output::Rejected | Output::Recorded => return Err(Rejected),
        };
        Ok(Admission {
            breaker: self.clone(),
            generation,
            probe,
            settled: false,
        })
    }

    pub fn failed(&self, failure: Failure, now: Instant) {
        self.0.fire(Event::Failed(failure, now));
    }

    pub fn state(&self) -> State {
        self.0.state()
    }
}

impl Admission {
    pub fn probe(&self) -> bool {
        self.probe
    }

    pub fn succeeded(mut self) {
        self.settled = true;
        self.breaker.0.fire_if(self.generation, Event::Succeeded);
    }

    pub fn failed(mut self, failure: Failure, now: Instant) {
        self.settled = true;
        self.breaker.failed(failure, now);
    }
}

impl Drop for Admission {
    fn drop(&mut self) {
        if self.probe && !self.settled {
            self.breaker.0.fire_if(
                self.generation,
                Event::Failed(Failure::Hard, Instant::now()),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex},
        time::{Duration, Instant},
    };

    use proptest::prelude::*;
    use rstest::rstest;

    use super::{Admission, Breaker, Config, Failure, State};
    use crate::Transition;

    #[derive(Clone, Copy, Debug)]
    enum Step {
        Fail(f64, Failure),
        Admit(f64, bool),
        Succeed(usize),
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Phase {
        Closed,
        Open,
        HalfOpen,
    }

    use Phase::{Closed as C, HalfOpen as H, Open as O};

    #[rstest]
    fn dropping_an_old_probe_does_not_undo_a_newer_recovery() {
        let breaker = Breaker::new(
            Config {
                failure_threshold: 1,
                recovery_timeout: Duration::from_secs(1),
                ..Config::default()
            },
            Box::new(|_| {}),
        );
        let now = Instant::now();
        let pending = breaker.admit(now).unwrap();
        breaker.failed(Failure::Hard, now);
        let old_probe = breaker.admit(now + Duration::from_secs(2)).unwrap();
        assert!(old_probe.probe());
        pending.failed(Failure::Hard, now + Duration::from_secs(2));
        let new_probe = breaker.admit(now + Duration::from_secs(4)).unwrap();
        assert!(new_probe.probe());
        new_probe.succeeded();
        assert_eq!(breaker.state(), State::CLOSED);

        drop(old_probe);

        assert_eq!(breaker.state(), State::CLOSED);
        assert!(!breaker.admit(now + Duration::from_secs(4)).unwrap().probe());
    }

    fn hard(at: f64) -> Step {
        Step::Fail(at, Failure::Hard)
    }

    fn timeout(at: f64) -> Step {
        Step::Fail(at, Failure::Timeout)
    }

    fn five_hard_at_zero() -> Vec<Step> {
        vec![hard(0.0); 5]
    }

    fn phase(state: &State) -> Phase {
        match state {
            State::Closed { .. } => Phase::Closed,
            State::Open { .. } => Phase::Open,
            State::HalfOpen => Phase::HalfOpen,
        }
    }

    fn quiet() -> Breaker {
        Breaker::new(Config::default(), Box::new(|_| {}))
    }

    #[rstest]
    #[case::five_hard_failures_open(vec![hard(0.0), hard(1.0), hard(2.0), hard(3.0), hard(4.0)], vec![C, C, C, C, O])]
    #[case::a_success_resets_the_hard_failure_count(
        vec![hard(0.0), hard(1.0), hard(2.0), hard(3.0), Step::Admit(4.0, true), Step::Succeed(4), hard(5.0), hard(6.0), hard(7.0), hard(8.0)],
        vec![C; 10]
    )]
    #[case::a_timeout_streak_opens_only_after_the_min_duration(
        vec![timeout(0.0), timeout(1.0), timeout(2.0), timeout(3.0), timeout(4.9), timeout(5.0)],
        vec![C, C, C, C, C, O]
    )]
    #[case::a_hard_failure_restarts_the_timeout_streak(
        vec![timeout(0.0), timeout(1.0), timeout(2.0), timeout(3.0), hard(4.0), timeout(5.0), timeout(6.0), timeout(7.0), timeout(8.0), timeout(9.9), timeout(10.0)],
        vec![C, C, C, C, C, C, C, C, C, C, O]
    )]
    #[case::the_probe_waits_until_strictly_past_the_recovery_timeout(
        [five_hard_at_zero(), vec![Step::Admit(60.0, false), Step::Admit(60.5, true)]].concat(),
        vec![C, C, C, C, O, O, H]
    )]
    #[case::a_probe_success_closes(
        [five_hard_at_zero(), vec![Step::Admit(61.0, true), Step::Succeed(5)]].concat(),
        vec![C, C, C, C, O, H, C]
    )]
    #[case::a_probe_failure_reopens_with_a_fresh_timer(
        [five_hard_at_zero(), vec![Step::Admit(61.0, true), hard(62.0), Step::Admit(122.0, false), Step::Admit(122.5, true)]].concat(),
        vec![C, C, C, C, O, H, O, O, H]
    )]
    #[case::only_one_probe_at_a_time(
        [five_hard_at_zero(), vec![Step::Admit(61.0, true), Step::Admit(61.5, false)]].concat(),
        vec![C, C, C, C, O, H, H]
    )]
    #[case::a_late_failure_while_open_pushes_back_the_probe(
        [five_hard_at_zero(), vec![hard(30.0), Step::Admit(61.0, false), Step::Admit(90.5, true)]].concat(),
        vec![C, C, C, C, O, O, O, H]
    )]
    #[case::a_success_from_before_the_breaker_moved_is_ignored(
        vec![Step::Admit(0.0, true), hard(1.0), hard(1.0), hard(1.0), hard(1.0), hard(1.0), Step::Admit(62.0, true), Step::Succeed(0), Step::Succeed(6)],
        vec![C, C, C, C, C, O, H, H, C]
    )]
    fn matches_the_python_redis_circuit_breaker(
        #[case] steps: Vec<Step>,
        #[case] expected: Vec<Phase>,
    ) {
        let breaker = quiet();
        let t0 = Instant::now();
        let mut admissions: HashMap<usize, Admission> = HashMap::new();

        let phases: Vec<Phase> = steps
            .iter()
            .enumerate()
            .map(|(index, step)| {
                match *step {
                    Step::Fail(at, failure) => {
                        breaker.failed(failure, t0 + Duration::from_secs_f64(at))
                    }
                    Step::Admit(at, admitted) => {
                        let admission = breaker.admit(t0 + Duration::from_secs_f64(at));
                        assert_eq!(admission.is_ok(), admitted, "step {index}");
                        admissions.extend(admission.ok().map(|admission| (index, admission)));
                    }
                    Step::Succeed(admitted_at) => {
                        admissions.remove(&admitted_at).unwrap().succeeded()
                    }
                }
                phase(&breaker.state())
            })
            .collect();

        assert_eq!(phases, expected);
    }

    #[rstest]
    fn a_dropped_probe_reopens_instead_of_wedging_half_open() {
        let breaker = quiet();
        let t0 = Instant::now();
        (0..5).for_each(|_| breaker.failed(Failure::Hard, t0));
        let probe = breaker.admit(t0 + Duration::from_secs(61)).unwrap();
        assert!(probe.probe());

        drop(probe);

        assert!(matches!(breaker.state(), State::Open { .. }));
    }

    #[rstest]
    fn a_dropped_regular_call_records_nothing() {
        let breaker = quiet();
        let t0 = Instant::now();
        breaker.failed(Failure::Hard, t0);

        drop(breaker.admit(t0).unwrap());

        assert_eq!(
            breaker.state(),
            State::Closed {
                hard: 1,
                timeouts: 0,
                streak_started: None
            }
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum Action {
        Admit,
        SucceedOldest,
        FailOldest(Failure),
        Fail(Failure),
    }

    fn action() -> impl Strategy<Value = (u64, Action)> {
        let failure = prop_oneof![Just(Failure::Hard), Just(Failure::Timeout)];
        let action = prop_oneof![
            Just(Action::Admit),
            Just(Action::SucceedOldest),
            failure.clone().prop_map(Action::FailOldest),
            failure.prop_map(Action::Fail),
        ];
        (0u64..30_000, action)
    }

    proptest! {
        #[test]
        fn open_never_closes_directly_and_each_half_open_visit_has_one_probe(
            actions in prop::collection::vec(action(), 1..200)
        ) {
            let transitions: Arc<Mutex<Vec<Transition<State>>>> = Arc::default();
            let seen = Arc::clone(&transitions);
            let config = Config {
                failure_threshold: 3,
                recovery_timeout: Duration::from_secs(10),
                timeout_min_duration: Duration::from_secs(2),
            };
            let breaker = Breaker::new(config, Box::new(move |t| seen.lock().unwrap().push(t.clone())));
            let t0 = Instant::now();
            let mut now = t0;
            let mut pending: Vec<Admission> = Vec::new();
            let mut probes_per_visit: Vec<u32> = Vec::new();

            for (advance_ms, action) in actions {
                now += Duration::from_millis(advance_ms);
                match action {
                    Action::Admit => {
                        if let Ok(admission) = breaker.admit(now) {
                            if admission.probe() {
                                probes_per_visit.push(1);
                            }
                            pending.push(admission);
                        }
                    }
                    Action::SucceedOldest if !pending.is_empty() => pending.remove(0).succeeded(),
                    Action::FailOldest(failure) if !pending.is_empty() => pending.remove(0).failed(failure, now),
                    Action::SucceedOldest | Action::FailOldest(_) => {}
                    Action::Fail(failure) => breaker.failed(failure, now),
                }
            }

            let transitions = transitions.lock().unwrap();
            let open_to_closed = transitions
                .iter()
                .filter(|t| matches!((&t.from, &t.to), (State::Open { .. }, State::Closed { .. })))
                .count();
            let half_open_visits = transitions.iter().filter(|t| t.to == State::HalfOpen).count();
            prop_assert_eq!(open_to_closed, 0);
            prop_assert_eq!(probes_per_visit.len(), half_open_visits);
        }
    }
}
