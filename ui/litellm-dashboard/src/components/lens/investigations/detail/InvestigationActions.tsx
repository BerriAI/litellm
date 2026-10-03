"use client";

import { Pause, Play, Settings2, MoreHorizontal, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import { type Lens, type Job } from "../../model/types";

export function InvestigationActions({
  lens,
  ready,
  busy,
  active,
  setEditing,
  setMonitoring,
  update,
}: {
  lens: Lens;
  ready: boolean;
  busy: boolean;
  active: Job | undefined;
  setEditing: (mode: "new" | "edit" | "duplicate") => void;
  setMonitoring: (open: boolean) => void;
  update: (path: string, body: unknown, method?: "post" | "put" | "patch") => Promise<void>;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      <DropdownMenu>
        <DropdownMenuTrigger render={<Button variant="ghost" size="icon" aria-label="Investigation actions" />}>
          <MoreHorizontal className="size-4" />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-48">
          <DropdownMenuItem onClick={() => setEditing("edit")}>
            <Settings2 />
            Edit investigation
          </DropdownMenuItem>
          <DropdownMenuItem disabled={!ready} onClick={() => setEditing("duplicate")}>
            <Copy />
            Duplicate
          </DropdownMenuItem>
          {lens.settings.enabled ? (
            <DropdownMenuItem
              disabled={busy}
              onClick={() => update(`/lens/${lens.id}`, { ...lens.settings, enabled: false }, "put")}
            >
              <Pause />
              Pause monitoring
            </DropdownMenuItem>
          ) : (
            <DropdownMenuItem disabled={!ready || busy} onClick={() => setMonitoring(true)}>
              <Play />
              Enable monitoring
            </DropdownMenuItem>
          )}
        </DropdownMenuContent>
      </DropdownMenu>
      <Button disabled={busy || !!active || !ready} onClick={() => update(`/lens/${lens.id}/runs`, {})}>
        <Play className="size-3" />
        Run now
      </Button>
    </div>
  );
}
