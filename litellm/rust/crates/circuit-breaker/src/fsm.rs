use std::{
    fmt::Debug,
    mem,
    sync::{Mutex, PoisonError},
};

pub trait Machine {
    type State: Clone + Debug;
    type Event;
    type Output;

    fn step(&self, state: &Self::State, event: Self::Event) -> (Self::State, Self::Output);
}

#[derive(Clone, Debug)]
pub struct Transition<S> {
    pub from: S,
    pub to: S,
    pub generation: u64,
}

pub type Observer<S> = Box<dyn Fn(&Transition<S>) + Send + Sync>;

pub struct Shared<M: Machine> {
    machine: M,
    current: Mutex<(M::State, u64)>,
    observer: Observer<M::State>,
}

impl<M: Machine> Shared<M> {
    pub fn new(machine: M, initial: M::State, observer: Observer<M::State>) -> Self {
        Self {
            machine,
            current: Mutex::new((initial, 0)),
            observer,
        }
    }

    pub fn fire(&self, event: M::Event) -> (M::Output, u64) {
        let mut current = self.current.lock().unwrap_or_else(PoisonError::into_inner);
        self.apply(&mut current, event)
    }

    pub fn fire_if(&self, generation: u64, event: M::Event) -> Option<M::Output> {
        let mut current = self.current.lock().unwrap_or_else(PoisonError::into_inner);
        (current.1 == generation).then(|| self.apply(&mut current, event).0)
    }

    pub fn state(&self) -> M::State {
        self.current
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .0
            .clone()
    }

    fn apply(&self, current: &mut (M::State, u64), event: M::Event) -> (M::Output, u64) {
        let (to, output) = self.machine.step(&current.0, event);
        if mem::discriminant(&to) != mem::discriminant(&current.0) {
            let from = mem::replace(&mut current.0, to.clone());
            current.1 += 1;
            (self.observer)(&Transition {
                from,
                to,
                generation: current.1,
            });
        } else {
            current.0 = to;
        }
        (output, current.1)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use rstest::rstest;

    use super::{Machine, Shared, Transition};

    struct Wrapping(u32);

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Counter {
        Low(u32),
        High(u32),
    }

    enum Tick {
        Up,
        Stay,
    }

    impl Machine for Wrapping {
        type State = Counter;
        type Event = Tick;
        type Output = ();

        fn step(&self, state: &Counter, event: Tick) -> (Counter, ()) {
            let value = match (state, event) {
                (Counter::Low(_) | Counter::High(_), Tick::Stay) => return (state.clone(), ()),
                (Counter::Low(value) | Counter::High(value), Tick::Up) => (value + 1) % self.0,
            };
            match value < self.0 / 2 {
                true => (Counter::Low(value), ()),
                false => (Counter::High(value), ()),
            }
        }
    }

    type Seen = Arc<Mutex<Vec<Transition<Counter>>>>;

    fn recorded(wrap: u32) -> (Shared<Wrapping>, Seen) {
        let transitions: Seen = Arc::default();
        let seen = Arc::clone(&transitions);
        let shared = Shared::new(
            Wrapping(wrap),
            Counter::Low(0),
            Box::new(move |transition| seen.lock().unwrap().push(transition.clone())),
        );
        (shared, transitions)
    }

    #[rstest]
    fn concurrent_events_are_all_applied_and_transitions_are_observed_in_generation_order() {
        let (shared, transitions) = recorded(4);

        std::thread::scope(|scope| {
            for _ in 0..64 {
                scope.spawn(|| {
                    for _ in 0..100 {
                        shared.fire(Tick::Up);
                    }
                });
            }
        });

        assert_eq!(shared.state(), Counter::Low(6400 % 4));
        let generations: Vec<u64> = transitions
            .lock()
            .unwrap()
            .iter()
            .map(|t| t.generation)
            .collect();
        assert_eq!(generations, (1..=3200).collect::<Vec<u64>>());
    }

    #[rstest]
    fn a_step_that_keeps_the_variant_neither_counts_nor_notifies() {
        let (shared, transitions) = recorded(4);

        let (_, after_inner_change) = shared.fire(Tick::Up);
        let (_, after_self_loop) = shared.fire(Tick::Stay);

        assert_eq!(shared.state(), Counter::Low(1));
        assert_eq!((after_inner_change, after_self_loop), (0, 0));
        assert!(transitions.lock().unwrap().is_empty());
    }

    #[rstest]
    fn fire_if_skips_an_event_from_an_older_generation() {
        let (shared, _) = recorded(4);
        shared.fire(Tick::Up);
        let (_, generation) = shared.fire(Tick::Up);

        let stale = shared.fire_if(generation - 1, Tick::Up);
        let current = shared.fire_if(generation, Tick::Up);

        assert!(stale.is_none());
        assert!(current.is_some());
        assert_eq!(shared.state(), Counter::High(3));
    }
}
