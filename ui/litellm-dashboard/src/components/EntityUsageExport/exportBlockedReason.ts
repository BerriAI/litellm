export interface UsageFetchState {
  coversRange: boolean;
  cancelled: boolean;
  failed: boolean;
}

export const getExportBlockedReason = ({ coversRange, cancelled, failed }: UsageFetchState): string | undefined => {
  if (failed) return "Some spend data failed to load, so an export would under-report. Reload the page to try again.";
  if (cancelled)
    return "Loading was stopped before the whole range arrived, so an export would under-report. Reload the page to load it all.";
  if (!coversRange) return "Spend data is still loading, so an export would under-report. Wait for it to finish.";
  return undefined;
};
