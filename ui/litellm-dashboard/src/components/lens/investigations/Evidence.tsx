"use client";

import { Inspector } from "@/components/shared/Inspector";
import { RunView } from "@/components/lens/traces/detail/run/RunView";
import { useLocalRunSelection } from "@/components/lens/traces/routing";

import { useLensAccessToken } from "../data/LensServices";
import { evidenceTarget, type EvidenceTarget } from "../model/findings";
import type { EvidenceRef } from "../route";

export interface EvidenceViewProps {
  readonly evidence: EvidenceRef;
  /** Shown as a back link above the evidence when it is stacked over something else. */
  readonly backLabel?: string;
  readonly onBack?: () => void;
}

/** The original trace step a piece of evidence points at, filling the panel. */
export function EvidenceView({ evidence, backLabel, onBack }: EvidenceViewProps) {
  const target = evidenceTarget(evidence.id);
  return (
    <div className="flex min-h-0 flex-1 flex-col" data-testid="evidence-view">
      {onBack && backLabel && (
        <div className="flex h-9 shrink-0 items-center border-b px-3">
          <Inspector.BackLink label={backLabel} onClick={onBack} />
        </div>
      )}
      <EvidenceBody key={`${evidence.id}:${evidence.span}`} target={target} evidence={evidence} onBack={onBack} />
    </div>
  );
}

function EvidenceBody({
  target,
  evidence,
  onBack = () => {},
}: {
  target: EvidenceTarget | null;
  evidence: EvidenceRef;
  onBack?: () => void;
}) {
  if (target)
    return (
      <TraceEvidence
        traceId={target.id}
        traceRef={target.traceRef}
        initialSpanId={evidence.span || null}
        onBack={onBack}
      />
    );
  return <p className="p-4 text-sm text-muted-foreground">This evidence is no longer available.</p>;
}

export function TraceEvidence({
  traceId,
  traceRef,
  initialSpanId,
  onBack,
}: {
  traceId: string;
  traceRef?: string;
  initialSpanId: string | null;
  onBack: () => void;
}) {
  const selection = useLocalRunSelection(initialSpanId);
  const accessToken = useLensAccessToken();
  return (
    <RunView
      traceId={traceId}
      traceRef={traceRef}
      selection={selection}
      accessToken={accessToken}
      onBack={onBack}
      embedded
    />
  );
}
