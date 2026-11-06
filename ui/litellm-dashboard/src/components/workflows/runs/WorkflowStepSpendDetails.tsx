"use client";

import { z } from "zod";
import { useLogDetails } from "@/app/(dashboard)/hooks/logDetails/useLogDetails";
import { Button } from "@/components/ui/button";
import { TracePayload } from "./TracePayload";

export interface WorkflowStepSpendDetailsProps {
  readonly requestId: string;
  readonly startedAt: string;
}

export function WorkflowStepSpendDetails({ requestId, startedAt }: WorkflowStepSpendDetailsProps) {
  const details = useLogDetails(requestId, startedAt, true);
  const payload = z.json().safeParse(details.data);
  if (details.isError) {
    return (
      <div role="alert" className="rounded-lg border border-destructive/30 p-4 text-sm">
        <p>
          Spend record unavailable. It may not be stored yet, may have expired, or your account may not have access.
        </p>
        <Button variant="outline" size="sm" className="mt-3" onClick={() => void details.refetch()}>
          Retry spend record
        </Button>
      </div>
    );
  }
  if (details.isPending || details.isFetching)
    return (
      <p role="status" className="text-sm text-muted-foreground">
        Loading authorized spend record…
      </p>
    );
  if (!payload.success || payload.data === null)
    return <p className="text-sm text-muted-foreground">No stored details are available for this request.</p>;
  return <TracePayload label={`Spend record ${requestId}`} value={payload.data} />;
}
