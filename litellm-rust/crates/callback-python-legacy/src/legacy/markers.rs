use super::vocabulary::LoggedMarker;

pub const fn marker_key(marker: LoggedMarker) -> &'static str {
    match marker {
        LoggedMarker::SyncSuccess => "has_logged_sync_success",
        LoggedMarker::AsyncSuccess => "has_logged_async_success",
        LoggedMarker::SyncFailure => "has_logged_sync_failure",
        LoggedMarker::AsyncFailure => "has_logged_async_failure",
    }
}
