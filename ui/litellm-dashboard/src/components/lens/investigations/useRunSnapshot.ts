"use client";

import { useQuery } from "@tanstack/react-query";
import { useLensApi } from "../data/LensServices";
import { lensQueries } from "../data/queries";
import { mergeFeedback } from "../model/findings";
import type { Finding, Job, Lens, Settings } from "../model/types";
import { useRunRoute } from "../route";

export interface RunSnapshot {
  readonly batchId: string;
  readonly job: Job | undefined;
  readonly missingSnapshot: boolean;
  readonly batchSettings: Settings | undefined;
  readonly batchFindings: readonly Finding[];
  readonly error: Error | null;
  readonly refetch: () => void;
}

const isAggregate = (batchId: string) => batchId === "latest" || batchId === "all";

/** The run a detail view is looking at: the latest job from the list, or one older job fetched on its own. */
export function useRunSnapshot(lens: Lens | undefined): RunSnapshot {
  const api = useLensApi();
  const { batchId } = useRunRoute();
  const historical = useQuery(lensQueries.run(api, lens?.id, batchId));
  const job = isAggregate(batchId) ? lens?.jobs[0] : historical.data;
  const current = lens?.findings ?? [];
  return {
    batchId,
    job,
    missingSnapshot: job?.status === "completed" && job.findings == null && batchId !== "all",
    batchSettings: job?.settings ?? lens?.settings,
    batchFindings: mergeFeedback(batchId === "all" ? current : job?.findings ?? [], current),
    error: historical.error,
    refetch: () => void historical.refetch(),
  };
}
