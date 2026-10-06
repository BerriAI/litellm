"use client";

import { useMutation } from "@tanstack/react-query";
import { useInvalidateLenses, useLensUpdate, useSaveLens, type LensWrite } from "../data/mutations";
import { useLensApi } from "../data/LensServices";
import type { Finding, Lens, RunWindow, Settings } from "../model/types";
import type { InboxRow, OwnedFinding } from "../model/inbox";

type InboxReview = { readonly row: InboxRow; readonly status: Finding["status"]; readonly reason: string };
type InboxReviewResult =
  | { readonly ok: true }
  | { readonly ok: false; readonly error: Error; readonly source: OwnedFinding };

const reviewError = (error: unknown): Error => (error instanceof Error ? error : new Error(String(error)));

/** Every write the Investigations screens can ask for, so leaves only ever express intent. */
export function useInvestigationActions() {
  const api = useLensApi();
  const invalidate = useInvalidateLenses();
  const update = useLensUpdate();
  const save = useSaveLens();
  const inboxReview = useMutation({
    retry: false,
    mutationFn: async ({ row, status, reason }: InboxReview): Promise<InboxReviewResult> => {
      const results = await Promise.all(
        row.sources.map(async (source): Promise<InboxReviewResult> => {
          try {
            await api.reviewFinding(source.lens.id, source.finding.id, status, reason);
            return { ok: true };
          } catch (error) {
            return { ok: false, error: reviewError(error), source };
          }
        }),
      );
      return results.find((result) => !result.ok) ?? { ok: true };
    },
    onSettled: invalidate,
  });
  const attempt = (write: LensWrite): Promise<boolean> =>
    update.mutateAsync(write).then(
      () => true,
      () => false,
    );
  return {
    busy: update.isPending || inboxReview.isPending,
    error: update.error ?? (inboxReview.data?.ok === false ? inboxReview.data.error : null),
    reset: () => {
      update.reset();
      inboxReview.reset();
    },
    save: (input: { id?: string; settings: Settings }): Promise<Lens> => save.mutateAsync(input),
    pause: (lens: Lens) => attempt((api) => api.saveLens(lens.id, { ...lens.settings, enabled: false })),
    cancelRun: (lens: Lens) => attempt((api) => api.cancelRun(lens.id)),
    startRun: (lens: Lens, request: RunWindow) => attempt((api) => api.startRun(lens.id, request)),
    review: (lens: Lens, finding: Finding, status: Finding["status"], reason: string) =>
      attempt((api) => api.reviewFinding(lens.id, finding.id, status, reason)),
    reviewInbox: (row: InboxRow, status: Finding["status"], reason: string) =>
      inboxReview.mutateAsync({ row, status, reason }),
  };
}
