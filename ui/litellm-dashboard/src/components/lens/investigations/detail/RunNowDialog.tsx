"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";

import { lensQueries } from "../../data/queries";
import { useLensApi } from "../../data/LensServices";
import type { Lens, RunWindow } from "../../model/types";
import { RUN_PRESETS, runRequest, type RunChoice, type RunPreset } from "../../model/runRequest";

const localInput = (date: Date) =>
  new Date(date.getTime() - date.getTimezoneOffset() * 60_000).toISOString().slice(0, 16);

export function RunNowDialog({
  lens,
  busy,
  onClose,
  onRun,
}: {
  lens: Lens;
  busy: boolean;
  onClose: () => void;
  onRun: (request: RunWindow) => Promise<void>;
}) {
  const api = useLensApi();
  const agentsQuery = useQuery(lensQueries.agents(api, "traces"));
  const agents = Array.isArray(agentsQuery.data) ? agentsQuery.data : [];
  const now = new Date();
  const [preset, setPreset] = useState<RunPreset>(null);
  const [agent, setAgent] = useState(lens.settings.agent_name ?? "");
  const [start, setStart] = useState(localInput(new Date(now.getTime() - 3_600_000)));
  const [end, setEnd] = useState(localInput(now));
  const [error, setError] = useState("");
  const submit = async () => {
    const choice: RunChoice = { preset, agent, saved: lens.settings.agent_name ?? "", start, end };
    const request = runRequest(choice);
    if (typeof request === "string") {
      setError(request);
      return;
    }
    setError("");
    await onRun(request);
  };
  return (
    <Dialog open onOpenChange={(open) => !open && !busy && onClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Run now</DialogTitle>
          <DialogDescription>
            Runs once with these choices. The saved schedule and settings stay the same.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <label className="grid gap-1.5 text-sm font-medium">
            Agent
            <Input
              list="run-now-agents"
              value={agent}
              placeholder="All agents"
              onChange={(e) => setAgent(e.target.value)}
              aria-label="Agent"
            />
            <datalist id="run-now-agents">
              {agents.map((name) => (
                <option key={name} value={name} />
              ))}
            </datalist>
          </label>
          <fieldset className="space-y-2">
            <legend className="text-sm font-medium">Traces to review</legend>
            <div className="flex flex-wrap gap-1.5">
              {RUN_PRESETS.map((p) => (
                <button
                  key={p.label}
                  type="button"
                  aria-pressed={preset === p.hours}
                  onClick={() => setPreset(p.hours)}
                  className="rounded-md bg-muted/60 px-2.5 py-1 text-xs text-muted-foreground hover:bg-muted aria-pressed:bg-background aria-pressed:text-foreground aria-pressed:ring-[1.5px] aria-pressed:ring-foreground aria-pressed:ring-inset"
                >
                  {p.label}
                </button>
              ))}
            </div>
            {preset === -1 && (
              <div className="grid grid-cols-2 gap-2">
                <label className="grid gap-1 text-xs text-muted-foreground">
                  From
                  <Input type="datetime-local" value={start} onChange={(e) => setStart(e.target.value)} />
                </label>
                <label className="grid gap-1 text-xs text-muted-foreground">
                  To
                  <Input type="datetime-local" value={end} onChange={(e) => setEnd(e.target.value)} />
                </label>
              </div>
            )}
          </fieldset>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
        </div>
        <DialogFooter>
          <Button variant="outline" disabled={busy} onClick={onClose}>
            Cancel
          </Button>
          <Button disabled={busy} onClick={() => void submit()}>
            Run now
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
