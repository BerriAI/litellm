"use client";

import { queueReasonText, workerTaskText, type QueueReason } from "../model/status";
import type { QueueContext } from "./useQueueReason";

export function QueueReasonText({ reason, onConnect }: { reason: QueueReason; onConnect?: () => void }) {
  if (reason.kind !== "no_worker" || !onConnect) return <>{queueReasonText(reason)}</>;
  return (
    <>
      No worker connected. Start one from{" "}
      <button type="button" onClick={onConnect} className="font-medium text-foreground underline underline-offset-2">
        Connect worker
      </button>
      .
    </>
  );
}

export function WorkerTasks({ reason, queue }: { reason: QueueReason; queue?: QueueContext }) {
  if (reason.kind === "no_worker" || !reason.tasks.length) return null;
  return (
    <ol aria-label="What the worker is doing" className="flex flex-col">
      {reason.tasks.map((task) => (
        <li key={task.lensId}>
          <button
            type="button"
            disabled={!queue}
            onClick={() => queue?.onOpenLens(task.lensId)}
            className="w-full truncate rounded px-1.5 py-0.5 text-left tabular-nums hover:bg-muted hover:text-foreground"
          >
            {workerTaskText(task)}
          </button>
        </li>
      ))}
    </ol>
  );
}
