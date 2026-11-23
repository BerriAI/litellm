//! A call's cache of Python level checks, so a call that logs many events asks Python once per level.

use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Default)]
pub(super) struct Levels([AtomicU8; 8]);

const UNKNOWN: u8 = 0;
const DISABLED: u8 = 1;
const ENABLED: u8 = 2;

impl Levels {
    /// Python routes proxy targets to its proxy logger, which has its own level, so each logger gets its own slots.
    pub(super) fn enabled(&self, level: u8, proxy: bool, query: impl FnOnce() -> bool) -> bool {
        let slot = &self.0[usize::from(level / 10 - 1) + if proxy { 4 } else { 0 }];
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

    use super::Levels;

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
        assert_eq!(levels.enabled(level, false, query), enabled);
        assert_eq!(levels.enabled(level, false, query), enabled);
        assert_eq!(levels.enabled(level, false, query), enabled);
        assert_eq!(queries.get(), 1);
    }

    #[rstest::rstest]
    fn the_proxy_logger_keeps_its_own_cached_levels() {
        let levels = Levels::default();
        assert!(levels.enabled(20, false, || true));
        assert!(!levels.enabled(20, true, || false));
        assert!(levels.enabled(20, false, || false));
        assert!(!levels.enabled(20, true, || true));
    }
}
