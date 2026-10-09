"use client";

import { Pause, Play, Settings2, MoreHorizontal, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import { hasActiveJob } from "../../model/status";
import { type Lens } from "../../model/types";

export interface InvestigationIntents {
  readonly onEdit: () => void;
  readonly onDuplicate: () => void;
  readonly onPause: () => void;
  readonly onEnableMonitoring: () => void;
  readonly onCancelRun: () => void;
  readonly onRunNow: () => void;
  readonly onConnectWorker: () => void;
}

export type InvestigationActionsProps = Omit<InvestigationIntents, "onCancelRun" | "onConnectWorker"> & {
  readonly lens: Lens;
  readonly ready: boolean;
  readonly busy: boolean;
};

export function InvestigationActions({
  lens,
  ready,
  busy,
  onEdit,
  onDuplicate,
  onPause,
  onEnableMonitoring,
  onRunNow,
}: InvestigationActionsProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger render={<Button variant="ghost" size="icon" aria-label="Investigation actions" />}>
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48">
        <DropdownMenuItem disabled={busy || hasActiveJob(lens.jobs) || !ready} onClick={onRunNow}>
          <Play />
          Run now
        </DropdownMenuItem>
        <DropdownMenuItem onClick={onEdit}>
          <Settings2 />
          Edit investigation
        </DropdownMenuItem>
        <DropdownMenuItem disabled={!ready} onClick={onDuplicate}>
          <Copy />
          Duplicate
        </DropdownMenuItem>
        {lens.settings.enabled ? (
          <DropdownMenuItem disabled={busy} onClick={onPause}>
            <Pause />
            Pause monitoring
          </DropdownMenuItem>
        ) : (
          <DropdownMenuItem disabled={!ready || busy} onClick={onEnableMonitoring}>
            <Play />
            Enable monitoring
          </DropdownMenuItem>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
