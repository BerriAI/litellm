"use client";

import { useNow } from "@/hooks/useNow";
import { analysisElapsed } from "../../model/progress";
import { nextCheckStatus } from "../../model/status";
import type { Job, Lens } from "../../model/types";

export function NextCheck({ lens }: { lens: Lens }) {
  const now = useNow(15000);
  const label = nextCheckStatus(lens, now);
  if (!label) return null;
  return <p className="mt-1 text-xs text-muted-foreground">{label}</p>;
}

export function ScanDuration({ job }: { job: Job }) {
  if (!job.finished_at) return null;
  return (
    <span title="Total time, including any wait for an analyzer">
      {" · Took "}
      {analysisElapsed(job.created_at, Date.parse(job.finished_at))}
    </span>
  );
}
