use std::sync::{Arc, Mutex};

use litellm_host::{
    hooks::NativeHooks,
    lifecycle::{CallEvent, FailureOrigin, Timing},
    native_chain::NativeChain,
};
use rstest::rstest;

struct Recorder {
    index: usize,
    events: Arc<Mutex<Vec<usize>>>,
}

impl NativeHooks for Recorder {
    fn on_event(&self, _: &CallEvent) {
        self.events.lock().unwrap().push(self.index);
    }
}

const TIMING: Timing = Timing {
    start_time: 1.0,
    end_time: 2.0,
};

#[rstest]
#[case::started(CallEvent::Started { start_time: 1.0 }, &[0, 1])]
#[case::succeeded(CallEvent::Succeeded { timing: TIMING, response: () }, &[1, 0])]
#[case::failed(CallEvent::Failed { timing: TIMING, origin: FailureOrigin::Call, error: () }, &[1, 0])]
#[case::cancelled(CallEvent::Cancelled { timing: TIMING }, &[1, 0])]
fn events_follow_the_same_onion_order_as_python_hooks(
    #[case] event: CallEvent,
    #[case] expected: &[usize],
) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let chain = NativeChain::new([
        Box::new(Recorder {
            index: 0,
            events: events.clone(),
        }) as Box<dyn NativeHooks>,
        Box::new(Recorder {
            index: 1,
            events: events.clone(),
        }),
    ]);

    chain.on_event(&event);

    assert_eq!(*events.lock().unwrap(), expected);
}
