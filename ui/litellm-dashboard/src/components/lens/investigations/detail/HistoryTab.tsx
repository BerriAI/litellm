"use client";
import { ListRow } from "@/components/shared/ListRow";

import { useState } from "react";
import { ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";

import { TabsContent } from "@/components/ui/tabs";
import { HistoryTimeline } from "./HistoryTimeline";
import { ScanDuration } from "./JobMeta";
import { useRunHistory } from "./useRunHistory";
import type { Lens } from "../../model/types";
import { useRunRoute } from "../../route";

import { money, when } from "../../model/format";

const PAGE = 50;

export interface HistoryTabProps {
  readonly lens: Lens;
}

export function HistoryTab({ lens }: HistoryTabProps) {
  const [offset, setOffset] = useState(0);
  const history = useRunHistory(lens, offset);
  const { openRun } = useRunRoute();
  const rows = history.data ?? lens.jobs;
  return (
    <TabsContent value="activity" className="pt-4 space-y-4">
      {history.error && (
        <p role="alert" className="text-sm text-destructive">
          Could not load run history.{" "}
          <Button variant="link" size="sm" onClick={() => void history.refetch()}>
            Retry
          </Button>
        </p>
      )}
      {rows.length > 0 && <HistoryTimeline jobs={rows} slots={PAGE} onOpen={openRun} />}
      <div className="divide-y border-y">
        {rows.map((j) => (
          <ListRow
            key={j.id}
            onClick={() => openRun(j.id)}
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
      {(offset > 0 || (history.data?.length ?? 0) >= PAGE) && (
        <div className="flex justify-between">
          <Button variant="ghost" size="sm" disabled={!offset} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
            Newer runs
          </Button>
          <Button
            variant="ghost"
            size="sm"
            disabled={(history.data?.length ?? 0) < PAGE}
            onClick={() => setOffset(offset + PAGE)}
          >
            Older runs
          </Button>
        </div>
      )}
    </TabsContent>
  );
}
