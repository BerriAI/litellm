//! A call's cache of Python level checks, so a call that logs many events asks Python once per level.

use std::sync::atomic::{AtomicU8, Ordering};

use strum::EnumCount;

use super::route::PythonLogger;

#[derive(Default)]
pub(super) struct Levels([[AtomicU8; 4]; PythonLogger::COUNT]);

const UNKNOWN: u8 = 0;
const DISABLED: u8 = 1;
const ENABLED: u8 = 2;

impl Levels {
    /// Each Python logger has its own level, so each gets its own slots.
    pub(super) fn enabled(
        &self,
        logger: PythonLogger,
        level: u8,
        query: impl FnOnce() -> bool,
    ) -> bool {
        let slot = &self.0[logger as usize][usize::from(level / 10 - 1)];
        match slot.load(Ordering::Relaxed) {
            UNKNOWN => {
                let enabled = query();
                slot.store(if enabled { ENABLED } else { DISABLED }, Ordering::Relaxed);
                enabled
            }
            state => state == ENABLED,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::{Levels, PythonLogger};

    #[rstest::rstest]
    #[case::debug(10)]
    #[case::info(20)]
    #[case::warning(30)]
    #[case::error(40)]
    fn call_levels_ask_python_once_per_level(
        #[case] level: u8,
        #[values(true, false)] enabled: bool,
    ) {
        let levels = Levels::default();
        let queries = Cell::new(0);
        let query = || {
            queries.set(queries.get() + 1);
            enabled
        };
        assert_eq!(levels.enabled(PythonLogger::Sdk, level, query), enabled);
        assert_eq!(levels.enabled(PythonLogger::Sdk, level, query), enabled);
        assert_eq!(levels.enabled(PythonLogger::Sdk, level, query), enabled);
        assert_eq!(queries.get(), 1);
    }

    #[rstest::rstest]
    fn each_python_logger_keeps_its_own_cached_levels() {
        let levels = Levels::default();
        assert!(levels.enabled(PythonLogger::Sdk, 20, || true));
        assert!(!levels.enabled(PythonLogger::Proxy, 20, || false));
        assert!(levels.enabled(PythonLogger::Sdk, 20, || false));
        assert!(!levels.enabled(PythonLogger::Proxy, 20, || true));
    }
}
