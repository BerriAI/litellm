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
    <header className="flex flex-wrap items-center justify-between gap-3">
      {lens ? (
        <Button variant="ghost" size="sm" className="-ml-3" onClick={() => selectLens(null)}>
          <ArrowLeft className="size-4" /> All investigations
        </Button>
      ) : (
        <h2 className="text-lg font-semibold">Investigations</h2>
      )}
      {showActions && (
        <div className="flex flex-wrap gap-2">
          <Button variant="ghost" disabled={!activityReady} onClick={() => setWorkerSetup(true)}>
            <StatusDot state={connected ? "ok" : "warn"} />
            {connected ? "Worker connected" : "Connect worker"}
          </Button>
          <Button variant={lens ? "outline" : "default"} disabled={!ready} onClick={() => setEditing("new")}>
            <Plus className="size-4" /> New investigation
          </Button>
        </div>
      )}
    </header>
  );
}
