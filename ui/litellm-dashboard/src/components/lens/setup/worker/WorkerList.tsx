"use client";

import { Button } from "@/components/ui/button";

import { CheckCircle2 } from "lucide-react";
import { workerConnected } from "../../model/status";
import { AnalysisKeyDetails } from "./AnalysisKeyDetails";
import type { LensList } from "../../model/types";

function workerStatus(worker: LensList["workers"][number], now: number): string {
  if (!worker.analysis_key_id) return "Billing key required";
  return workerConnected(worker, now) ? "Connected" : "Not connected";
}

export function WorkerList({
  workers,
  now,
  accessToken,
  editBilling,
  revoke,
}: {
  workers: LensList["workers"];
  now: number;
  accessToken: string;
  editBilling: (worker: LensList["workers"][number]) => void;
  revoke: (id: string) => Promise<void>;
}) {
  return (
    <>
      {workers
        .filter((w) => !w.revoked)
        .map((worker) => {
          const connected = workerConnected(worker, now);
          return (
            <section key={worker.id} className="space-y-5 text-sm">
              <div className="flex items-center justify-between gap-3">
                {workers.filter((w) => !w.revoked).length > 1 && <h3 className="font-medium">{worker.name}</h3>}
                <span
                  data-state={connected ? "active" : "inactive"}
                  className="flex items-center gap-2 text-muted-foreground data-[state=active]:text-emerald-700 dark:data-[state=active]:text-emerald-400"
                >
                  {connected && <CheckCircle2 className="size-4" />}
                  {workerStatus(worker, now)}
                </span>
              </div>
              {worker.analysis_key_id && (
                <AnalysisKeyDetails accessToken={accessToken} keyId={worker.analysis_key_id} showName />
              )}
              <div className="flex flex-wrap items-center justify-between gap-2">
                <Button variant="outline" size="sm" onClick={() => editBilling(worker)}>
                  Settings
                </Button>
                <Button variant="ghost" size="sm" onClick={() => revoke(worker.id)}>
                  Revoke access
                </Button>
              </div>
            </section>
          );
        })}
    </>
  );
}
