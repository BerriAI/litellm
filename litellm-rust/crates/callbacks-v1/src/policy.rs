#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionMode {
    Sync,
    Async,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandlerSelection {
    Sync,
    Async,
    Skip,
}

pub fn select_handler(mode: ExecutionMode, has_sync: bool, has_async: bool) -> HandlerSelection {
    match (mode, has_sync, has_async) {
        (ExecutionMode::Async, _, true) => HandlerSelection::Async,
        (_, true, _) => HandlerSelection::Sync,
        _ => HandlerSelection::Skip,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(ExecutionMode::Async, true, true, HandlerSelection::Async)]
    #[case(ExecutionMode::Async, false, true, HandlerSelection::Async)]
    #[case(ExecutionMode::Async, true, false, HandlerSelection::Sync)]
    #[case(ExecutionMode::Sync, true, true, HandlerSelection::Sync)]
    #[case(ExecutionMode::Sync, true, false, HandlerSelection::Sync)]
    #[case(ExecutionMode::Sync, false, true, HandlerSelection::Skip)]
    #[case(ExecutionMode::Sync, false, false, HandlerSelection::Skip)]
    fn handler_selection_never_selects_async_for_sync_calls(
        #[case] mode: ExecutionMode,
        #[case] has_sync: bool,
        #[case] has_async: bool,
        #[case] expected: HandlerSelection,
    ) {
        assert_eq!(select_handler(mode, has_sync, has_async), expected);
    }
}
