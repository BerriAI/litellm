export interface UsageFetchState {
  loading: boolean;
  isFetchingMore: boolean;
  cancelled: boolean;
  failed: boolean;
}

/** Why exporting what is on screen would under-report, or undefined once it covers the whole range. */
export const getExportBlockedReason = ({
  loading,
  isFetchingMore,
  cancelled,
  failed,
}: UsageFetchState): string | undefined => {
  if (failed) return "Some spend data failed to load, so an export would under-report. Reload the page to try again.";
  if (cancelled)
    return "Loading was stopped before the whole range arrived, so an export would under-report. Reload the page to load it all.";
  if (loading || isFetchingMore)
    return "Spend data is still loading, so an export would under-report. Wait for it to finish.";
  return undefined;
};
