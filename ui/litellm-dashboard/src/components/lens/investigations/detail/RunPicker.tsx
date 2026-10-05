"use client";

import { Info } from "lucide-react";
import { Button } from "@/components/ui/button";

import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import { ScanDuration } from "./JobMeta";
import { useRunHistory } from "./useRunHistory";
import type { Lens, Job } from "../../model/types";
import { useRunRoute } from "../../route";

import { money, when } from "../../model/format";

export interface RunPickerProps {
  readonly lens: Lens;
  readonly job: Job | undefined;
}

export function RunPicker({ lens, job }: RunPickerProps) {
  const { batchId, selectRun } = useRunRoute();
  const history = useRunHistory(lens, 0);
  const options = history.data ?? lens.jobs;
  const aggregate = batchId === "latest" || batchId === "all";
  const outsideHistory = !aggregate && !options.some((j) => j.id === batchId);
  return (
    <div className="flex min-w-0 items-center gap-1 pb-1">
      <select
        aria-label="Investigation run"
        className="h-8 w-44 max-w-full truncate rounded-md border-0 bg-transparent px-2 text-xs text-muted-foreground hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
        value={batchId}
        onChange={(e) => selectRun(e.target.value)}
      >
        <option value="latest">Latest run</option>
        {job && outsideHistory && (
          <option value={batchId}>
            {when(job.created_at)} · {job.status}
          </option>
        )}
        {options.map((j) => (
          <option key={j.id} value={j.id}>
            {when(j.created_at)} · {j.status}
          </option>
        ))}
        <option value="all">All accumulated findings</option>
      </select>
      {job && batchId !== "all" && (
        <Popover key={job.id}>
          <PopoverTrigger
            aria-label="Run details"
            render={<Button variant="ghost" size="icon" className="size-7 text-muted-foreground" />}
          >
            <Info className="size-3.5" />
          </PopoverTrigger>
          <PopoverContent align="end" className="gap-3">
            <PopoverTitle>Run details</PopoverTitle>
            <p className="text-xs text-muted-foreground">
              {job.coverage?.screened ?? 0} / {job.coverage?.selected ?? 0} selected runs reviewed
              <ScanDuration job={job} />
            </p>
            <p className="text-xs text-muted-foreground">
              {job.coverage?.partial ?? 0} partial traces · {job.coverage?.unassessable ?? 0} could not be assessed ·{" "}
              {job.coverage?.inconclusive ?? 0} inconclusive
            </p>
            <dl className="space-y-2 text-xs">
              <div>
                <dt className="text-muted-foreground">Activity window</dt>
                <dd className="mt-1">
                  {when(job.start)} to {when(job.end)}
                </dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="text-muted-foreground">Analysis cost</dt>
                <dd>{money(job.cost ?? 0)}</dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="text-muted-foreground">Status</dt>
                <dd className="capitalize">{job.status}</dd>
              </div>
            </dl>
          </PopoverContent>
        </Popover>
      )}
    </div>
  );
}
