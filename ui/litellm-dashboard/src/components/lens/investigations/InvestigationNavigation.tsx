"use client";
import { StatusDot } from "@/components/shared/StatusDot";

import { ArrowLeft, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Lens } from "../model/types";

export function InvestigationNavigation({
  lens,
  showActions,
  activityReady,
  connected,
  ready,
  selectLens,
  setWorkerSetup,
  setEditing,
}: {
  lens: Lens | undefined;
  showActions: boolean;
  activityReady: boolean;
  connected: boolean;
  ready: boolean;
  selectLens: (id: string | null) => void;
  setWorkerSetup: (open: boolean) => void;
  setEditing: (mode: "new") => void;
}) {
  return (
    <header className={lens ? "flex flex-wrap items-center justify-between gap-2" : "flex items-center gap-2"}>
      {lens ? (
        <Button variant="ghost" size="sm" className="-ml-3" onClick={() => selectLens(null)}>
          <ArrowLeft className="size-4" /> Back
        </Button>
      ) : null}
      {showActions && (
        <div className="flex flex-wrap gap-2">
          <Button
            variant="ghost"
            size="sm"
            disabled={!activityReady}
            onClick={() => setWorkerSetup(true)}
            title={connected ? "Worker connected" : "Connect worker"}
            className="text-muted-foreground"
          >
            <StatusDot state={connected ? "ok" : "warn"} />
            {connected ? "Worker" : "Connect worker"}
          </Button>
          <Button size="sm" variant={lens ? "outline" : "default"} disabled={!ready} onClick={() => setEditing("new")}>
            <Plus className="size-4" /> New investigation
          </Button>
        </div>
      )}
    </header>
  );
}
