"use client";

import { useState } from "react";

import { modelsUsed } from "../../model/inbox";
import { analysisModel, doneLine, inFlight, issueCount, nowLine, stripState } from "../../model/live";
import { queueReasonText } from "../../model/status";
import type { Job, Review } from "../../model/types";
import { QueueReasonText, WorkerTasks } from "../QueueReasonText";
import { useQueueReason, type QueueContext } from "../useQueueReason";
import { LiveDrawer } from "./LiveDrawer";
import { LiveStrip } from "./LiveStrip";
import { useStripOpen } from "./useLivePanels";

const MAX_LANES = 4;

export function LiveRun({
  job,
  reviews,
  name,
  queue,
}: {
  job: Job;
  reviews: readonly Review[];
  name: string;
  queue?: QueueContext;
}) {
  const reason = useQueueReason(job, queue);
  const [stripOpen, setStripOpen] = useStripOpen();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const reading = inFlight(job);
  const withReviews = { ...job, reviews: [...reviews] };
  const model = analysisModel([...reviews.map((r) => r.model), ...modelsUsed(job.steps), job.settings.model]);
  const issues = issueCount(withReviews);
  const open = () => setDrawerOpen(true);
  const waiting = reason && (
    <>
      <QueueReasonText reason={reason} onConnect={queue?.onConnect} />
      <WorkerTasks reason={reason} queue={queue} />
    </>
  );

  return (
    <>
      {stripOpen ? (
        <LiveStrip
          model={model}
          state={stripState(withReviews, model, reason ? queueReasonText(reason) : undefined)}
          waiting={waiting}
          reviews={reviews}
          reviewed={job.reviewed}
          selected={job.coverage.selected}
          issues={issues}
          cost={job.cost}
          onOpen={open}
          onClose={() => setStripOpen(false)}
        />
      ) : (
        <button
          type="button"
          onClick={() => setStripOpen(true)}
          className="self-start text-xs text-muted-foreground hover:text-foreground"
        >
          Show live trace results
        </button>
      )}
      <LiveDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        name={name}
        model={model}
        status={job.status}
        reviewed={job.reviewed}
        reviews={reviews}
        reading={reading}
        counter={nowLine(job, reading.length)}
        done={job.status === "completed" ? doneLine(job) : null}
        slots={Math.max(1, Math.min(job.settings.concurrency, MAX_LANES))}
        checks={job.settings.checks}
        scope={
          job.reviewed > reviews.length ? `From the latest ${reviews.length} of ${job.reviewed} reviewed traces` : ""
        }
        waiting={waiting}
      />
    </>
  );
}
