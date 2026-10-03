"use client";
import { ListRow } from "@/components/shared/ListRow";

import { ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";

import { TabsContent } from "@/components/ui/tabs";
import { ScanDuration } from "../InvestigationProgress";
import type { Lens, Job } from "../../model/types";

import { money, when } from "../../model/format";

export function HistoryTab({
  history,
  historyError,
  refetchHistory,
  lens,
  openBatch,
  historyOffset,
  setHistoryOffset,
}: {
  history: Job[] | undefined;
  historyError: Error | null | undefined;
  refetchHistory: () => void;
  lens: Lens;
  openBatch: (id: string) => void;
  historyOffset: number;
  setHistoryOffset: (offset: number) => void;
}) {
  return (
    <TabsContent value="activity" className="pt-4 space-y-4">
      {historyError && (
        <p role="alert" className="text-sm text-destructive">
          Could not load run history.{" "}
          <Button variant="link" size="sm" onClick={() => void refetchHistory()}>
            Retry
          </Button>
        </p>
      )}
      <div className="divide-y border-y">
        {(history ?? lens.jobs)?.map((j) => (
          <ListRow
            key={j.id}
            onClick={() => openBatch(j.id)}
            className="flex w-full items-center gap-4 py-4 text-left hover:bg-muted/30 focus-visible:outline-2 focus-visible:outline-ring"
          >
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium">{when(j.created_at)}</p>
              <p className="mt-1 text-xs text-muted-foreground">
                {j.coverage?.screened ?? 0} runs reviewed
                {j.findings != null && <> · {j.findings.length} findings</>}
                <ScanDuration job={j} />
              </p>
              {j.error && <p className="mt-2 line-clamp-2 text-xs text-destructive">{j.error}</p>}
            </div>
            <div className="text-right text-xs text-muted-foreground">
              <p
                data-state={j.status === "failed" ? "failed" : "other"}
                className="capitalize data-[state=failed]:text-destructive"
              >
                {j.status}
              </p>
              <p className="mt-1">{money(j.cost ?? 0)}</p>
            </div>
            <ChevronRight className="size-4 shrink-0 text-muted-foreground" />
          </ListRow>
        ))}
      </div>
      {(historyOffset > 0 || (history?.length ?? 0) >= 50) && (
        <div className="flex justify-between">
          <Button
            variant="ghost"
            size="sm"
            disabled={!historyOffset}
            onClick={() => setHistoryOffset(Math.max(0, historyOffset - 50))}
          >
            Newer runs
          </Button>
          <Button
            variant="ghost"
            size="sm"
            disabled={(history?.length ?? 0) < 50}
            onClick={() => setHistoryOffset(historyOffset + 50)}
          >
            Older runs
          </Button>
        </div>
      )}
    </TabsContent>
  );
}
