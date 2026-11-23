//! The lowest level each Python logger takes, as the last call into the bridge read it. An event is filtered against
//! it without asking Python, which a thread that does not hold the GIL cannot do cheaply. Python checks the level
//! again when a line is delivered, so a stale floor only loses lines between a level being lowered and the next
//! call.

use std::sync::atomic::{AtomicU8, Ordering};

use strum::EnumCount;

use super::route::PythonLogger;

const WARNING: u8 = 30;

pub(super) struct Floors([AtomicU8; PythonLogger::COUNT]);

impl Floors {
    pub(super) const fn new() -> Self {
        Self([const { AtomicU8::new(WARNING) }; PythonLogger::COUNT])
    }

    pub(super) fn refresh(&self, floors: [u8; PythonLogger::COUNT]) {
        self.0
            .iter()
            .zip(floors)
            .for_each(|(slot, floor)| slot.store(floor, Ordering::Relaxed));
    }

    pub(super) fn enabled(&self, logger: PythonLogger, level: u8) -> bool {
        level >= self.0[logger as usize].load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::{Floors, PythonLogger};

    #[rstest::rstest]
    fn a_logger_takes_warnings_until_a_call_reads_its_level() {
        let floors = Floors::new();
        assert!(floors.enabled(PythonLogger::Sdk, 30));
        assert!(!floors.enabled(PythonLogger::Sdk, 20));
    }

    #[rstest::rstest]
    #[case::debug_on(10, 10, true)]
    #[case::below_the_floor(20, 10, false)]
    #[case::at_the_floor(40, 40, true)]
    #[case::nothing_enabled(51, 40, false)]
    fn an_event_is_enabled_at_or_above_its_loggers_floor(
        #[case] floor: u8,
        #[case] level: u8,
        #[case] enabled: bool,
    ) {
        let floors = Floors::new();
        floors.refresh([30, floor]);
        assert_eq!(floors.enabled(PythonLogger::Proxy, level), enabled);
        assert_eq!(floors.enabled(PythonLogger::Sdk, level), level >= 30);
    }
}
