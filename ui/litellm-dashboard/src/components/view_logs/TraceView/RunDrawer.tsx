"use client";

import { useCallback } from "react";

import { SidePanel } from "@/components/shared/SidePanel";

import { RunView } from "./TraceDrawer";
import { type RunSelection, type TraceRef, traceKey, traceRefOf } from "./traceRouting";
import type { TraceSummary } from "./traceTypes";

interface RunDrawerProps {
  trace: TraceRef | null;
  runs: readonly TraceSummary[];
  accessToken: string;
  selection: RunSelection;
  onSelect: (trace: TraceRef | null) => void;
  fullScreen: boolean;
  onFullScreenChange: (fullScreen: boolean) => void;
}

/** Right-side drawer over the runs list: resizable, keeps the list clickable, swaps runs in place. */
export function RunDrawer({
  trace,
  runs,
  accessToken,
  selection,
  onSelect,
  fullScreen,
  onFullScreenChange,
}: RunDrawerProps) {
  const index = trace === null ? -1 : runs.findIndex((run) => traceKey(traceRefOf(run)) === traceKey(trace));
  const step = useCallback(
    (delta: number) => {
      const next = runs[index + delta];
      if (next) onSelect(traceRefOf(next));
    },
    [runs, index, onSelect],
  );
  const close = useCallback(() => onSelect(null), [onSelect]);
  return (
    <SidePanel
      item={trace}
      itemKey={traceKey}
      noun="trace"
      label="Trace details"
      testId="run-drawer"
      index={index}
      total={runs.length}
      onStep={step}
      onClose={close}
      fullScreen={fullScreen}
      onFullScreenChange={onFullScreenChange}
    >
      {(shown) => (
        <RunView
          traceId={shown.traceId}
          traceRef={shown.traceRef}
          selection={selection}
          accessToken={accessToken}
          onBack={close}
          embedded
        />
      )}
    </SidePanel>
  );
}
