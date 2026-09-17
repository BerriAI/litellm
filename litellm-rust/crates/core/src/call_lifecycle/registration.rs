use std::collections::HashSet;

use super::callbacks::CallbackId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Registry {
    Input,
    AsyncInput,
    Success,
    AsyncSuccess,
    Failure,
    AsyncFailure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Registration {
    Named { known: bool, async_only: bool },
    Object { asynchronous: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    pub id: CallbackId,
    pub registration: Registration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub resolved: Option<Entry>,
    pub duplicate_type: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct RegistrationFacts {
    pub candidates: Vec<Candidate>,
    pub input: Vec<Entry>,
    pub success: Vec<Entry>,
    pub failure: Vec<Entry>,
    pub async_success: Vec<CallbackId>,
    pub async_failure: Vec<CallbackId>,
    pub bootstrap_pending: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NamedEvent {
    Success,
    Failure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegistryMutation {
    Append(Registry, CallbackId),
    Remove(Registry, CallbackId),
    ExpandNamed(NamedEvent, CallbackId),
    Bootstrap,
}

struct Planner {
    input: Vec<Entry>,
    success: Vec<Entry>,
    failure: Vec<Entry>,
    async_success: HashSet<CallbackId>,
    async_failure: HashSet<CallbackId>,
    mutations: Vec<RegistryMutation>,
}

impl Planner {
    fn contains(&self, registry: Registry, id: CallbackId) -> bool {
        match registry {
            Registry::Input => self.input.iter().any(|entry| entry.id == id),
            Registry::Success => self.success.iter().any(|entry| entry.id == id),
            Registry::Failure => self.failure.iter().any(|entry| entry.id == id),
            Registry::AsyncSuccess => self.async_success.contains(&id),
            Registry::AsyncFailure => self.async_failure.contains(&id),
            Registry::AsyncInput => false,
        }
    }

    fn append(&mut self, registry: Registry, entry: Entry) {
        if self.contains(registry, entry.id) {
            return;
        }
        match registry {
            Registry::Input => self.input.push(entry),
            Registry::Success => self.success.push(entry),
            Registry::Failure => self.failure.push(entry),
            Registry::AsyncSuccess => {
                self.async_success.insert(entry.id);
            }
            Registry::AsyncFailure => {
                self.async_failure.insert(entry.id);
            }
            Registry::AsyncInput => {}
        }
        self.mutations
            .push(RegistryMutation::Append(registry, entry.id));
    }

    fn record(&mut self, mutation: RegistryMutation) {
        self.mutations.push(mutation);
    }
}

fn is_asynchronous(registration: Registration) -> bool {
    matches!(registration, Registration::Object { asynchronous: true })
}

pub fn plan_registration(facts: &RegistrationFacts) -> Vec<RegistryMutation> {
    let mut planner = Planner {
        input: facts.input.clone(),
        success: facts.success.clone(),
        failure: facts.failure.clone(),
        async_success: facts.async_success.iter().copied().collect(),
        async_failure: facts.async_failure.iter().copied().collect(),
        mutations: Vec::new(),
    };

    for candidate in &facts.candidates {
        let Some(entry) = candidate.resolved else {
            continue;
        };
        if candidate.duplicate_type {
            continue;
        }
        planner.append(Registry::Input, entry);
        if !is_asynchronous(entry.registration) {
            planner.append(Registry::Success, entry);
            planner.append(Registry::Failure, entry);
        }
        planner.append(Registry::AsyncSuccess, entry);
        planner.append(Registry::AsyncFailure, entry);
    }

    if facts.bootstrap_pending
        && !(planner.input.is_empty() && planner.success.is_empty() && planner.failure.is_empty())
    {
        planner.record(RegistryMutation::Bootstrap);
    }

    let input = planner.input.clone();
    for entry in input
        .iter()
        .filter(|entry| is_asynchronous(entry.registration))
    {
        planner.record(RegistryMutation::Append(Registry::AsyncInput, entry.id));
        planner.record(RegistryMutation::Remove(Registry::Input, entry.id));
    }

    let success = planner.success.clone();
    for entry in &success {
        match entry.registration {
            Registration::Object { asynchronous: true }
            | Registration::Named {
                async_only: true, ..
            } => {
                planner.append(Registry::AsyncSuccess, *entry);
                planner.record(RegistryMutation::Remove(Registry::Success, entry.id));
            }
            Registration::Named { known: true, .. } => {
                planner.record(RegistryMutation::ExpandNamed(NamedEvent::Success, entry.id));
            }
            _ => {}
        }
    }

    let failure = planner.failure.clone();
    for entry in &failure {
        match entry.registration {
            Registration::Object { asynchronous: true } => {
                planner.append(Registry::AsyncFailure, *entry);
                planner.record(RegistryMutation::Remove(Registry::Failure, entry.id));
            }
            Registration::Named { known: true, .. } => {
                planner.record(RegistryMutation::ExpandNamed(NamedEvent::Failure, entry.id));
            }
            _ => {}
        }
    }

    planner.mutations
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DynamicSuccessSlot {
    Sync,
    Async,
}

pub fn classify_dynamic_success(entry: Entry, named_async: bool) -> DynamicSuccessSlot {
    match entry.registration {
        Registration::Object { asynchronous: true } => DynamicSuccessSlot::Async,
        Registration::Named { .. } if named_async => DynamicSuccessSlot::Async,
        _ => DynamicSuccessSlot::Sync,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(id: u64, asynchronous: bool) -> Entry {
        Entry {
            id: CallbackId(id),
            registration: Registration::Object { asynchronous },
        }
    }

    fn named(id: u64, known: bool, async_only: bool) -> Entry {
        Entry {
            id: CallbackId(id),
            registration: Registration::Named { known, async_only },
        }
    }

    fn candidate(entry: Entry) -> Candidate {
        Candidate {
            resolved: Some(entry),
            duplicate_type: false,
        }
    }

    #[test]
    fn sync_callback_in_callbacks_registers_in_every_list_once() {
        let facts = RegistrationFacts {
            candidates: vec![candidate(object(1, false)), candidate(object(1, false))],
            ..RegistrationFacts::default()
        };
        assert_eq!(
            plan_registration(&facts),
            [
                RegistryMutation::Append(Registry::Input, CallbackId(1)),
                RegistryMutation::Append(Registry::Success, CallbackId(1)),
                RegistryMutation::Append(Registry::Failure, CallbackId(1)),
                RegistryMutation::Append(Registry::AsyncSuccess, CallbackId(1)),
                RegistryMutation::Append(Registry::AsyncFailure, CallbackId(1)),
            ]
        );
    }

    #[test]
    fn async_callable_in_callbacks_skips_sync_lists_and_moves_out_of_input() {
        let facts = RegistrationFacts {
            candidates: vec![candidate(object(2, true))],
            ..RegistrationFacts::default()
        };
        assert_eq!(
            plan_registration(&facts),
            [
                RegistryMutation::Append(Registry::Input, CallbackId(2)),
                RegistryMutation::Append(Registry::AsyncSuccess, CallbackId(2)),
                RegistryMutation::Append(Registry::AsyncFailure, CallbackId(2)),
                RegistryMutation::Append(Registry::AsyncInput, CallbackId(2)),
                RegistryMutation::Remove(Registry::Input, CallbackId(2)),
            ]
        );
    }

    #[test]
    fn unresolved_and_duplicate_type_named_candidates_are_skipped() {
        let facts = RegistrationFacts {
            candidates: vec![
                Candidate {
                    resolved: None,
                    duplicate_type: false,
                },
                Candidate {
                    resolved: Some(object(3, false)),
                    duplicate_type: true,
                },
            ],
            ..RegistrationFacts::default()
        };
        assert!(plan_registration(&facts).is_empty());
    }

    #[test]
    fn already_registered_callbacks_are_not_appended_again() {
        let facts = RegistrationFacts {
            candidates: vec![candidate(object(1, false))],
            input: vec![object(1, false)],
            success: vec![object(1, false)],
            failure: vec![object(1, false)],
            async_success: vec![CallbackId(1)],
            async_failure: vec![CallbackId(1)],
            ..RegistrationFacts::default()
        };
        assert!(plan_registration(&facts).is_empty());
    }

    #[test]
    fn bootstrap_runs_once_when_any_public_list_is_populated() {
        let empty = RegistrationFacts {
            bootstrap_pending: true,
            ..RegistrationFacts::default()
        };
        assert!(plan_registration(&empty).is_empty());
        let populated = RegistrationFacts {
            bootstrap_pending: true,
            candidates: vec![candidate(object(1, false))],
            ..RegistrationFacts::default()
        };
        assert!(plan_registration(&populated).contains(&RegistryMutation::Bootstrap));
        let already = RegistrationFacts {
            bootstrap_pending: false,
            success: vec![object(1, false)],
            ..RegistrationFacts::default()
        };
        assert!(!plan_registration(&already).contains(&RegistryMutation::Bootstrap));
    }

    #[test]
    fn success_safety_net_moves_async_and_async_only_names_and_expands_known_names() {
        let facts = RegistrationFacts {
            success: vec![
                object(1, true),
                named(2, false, true),
                named(3, true, false),
                named(4, false, false),
                object(5, false),
            ],
            ..RegistrationFacts::default()
        };
        assert_eq!(
            plan_registration(&facts),
            [
                RegistryMutation::Append(Registry::AsyncSuccess, CallbackId(1)),
                RegistryMutation::Remove(Registry::Success, CallbackId(1)),
                RegistryMutation::Append(Registry::AsyncSuccess, CallbackId(2)),
                RegistryMutation::Remove(Registry::Success, CallbackId(2)),
                RegistryMutation::ExpandNamed(NamedEvent::Success, CallbackId(3)),
            ]
        );
    }

    #[test]
    fn failure_safety_net_ignores_async_only_names() {
        let facts = RegistrationFacts {
            failure: vec![
                object(1, true),
                named(2, false, true),
                named(3, true, false),
            ],
            async_failure: vec![CallbackId(1)],
            ..RegistrationFacts::default()
        };
        assert_eq!(
            plan_registration(&facts),
            [
                RegistryMutation::Remove(Registry::Failure, CallbackId(1)),
                RegistryMutation::ExpandNamed(NamedEvent::Failure, CallbackId(3)),
            ]
        );
    }

    #[test]
    fn dynamic_success_split_follows_async_callables_and_selected_names() {
        assert_eq!(
            classify_dynamic_success(object(1, true), false),
            DynamicSuccessSlot::Async
        );
        assert_eq!(
            classify_dynamic_success(object(1, false), false),
            DynamicSuccessSlot::Sync
        );
        assert_eq!(
            classify_dynamic_success(named(2, true, false), true),
            DynamicSuccessSlot::Async
        );
        assert_eq!(
            classify_dynamic_success(named(2, true, true), false),
            DynamicSuccessSlot::Sync
        );
    }
}
