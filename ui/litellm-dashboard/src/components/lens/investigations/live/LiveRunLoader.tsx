"use client";

import type { Job } from "../../model/types";
import type { QueueContext } from "../useQueueReason";
import { LiveRun } from "./LiveRun";
import { useJobReviews } from "./useJobReviews";

export function LiveRunLoader({
  lensId,
  job,
  name,
  queue,
}: {
  lensId: string;
  job: Job;
  name: string;
  queue: QueueContext;
}) {
  const reviews = useJobReviews(queue.api, lensId, job);
  return <LiveRun job={job} reviews={reviews} name={name} queue={queue} />;
}
