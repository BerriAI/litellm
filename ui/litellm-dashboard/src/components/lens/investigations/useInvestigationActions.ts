"use client";

import { useLensUpdate, useSaveLens, type LensWrite } from "../data/mutations";
import type { Finding, Lens, RunWindow, Settings } from "../model/types";

/** Every write the Investigations screens can ask for, so leaves only ever express intent. */
export function useInvestigationActions() {
  const update = useLensUpdate();
  const save = useSaveLens();
  const attempt = (write: LensWrite): Promise<boolean> =>
    update.mutateAsync(write).then(
      () => true,
      () => false,
    );
  return {
    busy: update.isPending,
    error: update.error,
    reset: update.reset,
    save: (input: { id?: string; settings: Settings }): Promise<Lens> => save.mutateAsync(input),
    pause: (lens: Lens) => attempt((api) => api.saveLens(lens.id, { ...lens.settings, enabled: false })),
    cancelRun: (lens: Lens) => attempt((api) => api.cancelRun(lens.id)),
    startRun: (lens: Lens, request: RunWindow) => attempt((api) => api.startRun(lens.id, request)),
    review: (lens: Lens, finding: Finding, status: Finding["status"], reason: string) =>
      attempt((api) => api.reviewFinding(lens.id, finding.id, status, reason)),
  };
}
