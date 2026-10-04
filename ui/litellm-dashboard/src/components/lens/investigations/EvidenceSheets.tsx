"use client";

import { evidenceTarget } from "../model/findings";
import { findFinding } from "../model/inbox";
import type { Lens } from "../model/types";
import { useIssueRoute } from "../route";
import { RequestEvidenceSheet } from "./RequestEvidenceSheet";
import { useEvidenceRoute } from "./resultRoute";
import { TraceSheet } from "./TraceSheet";

export interface EvidenceSheetsProps {
  readonly lenses: readonly Lens[];
  readonly lens: Lens | undefined;
  readonly accessToken: string;
}

/** The original trace step or request a finding's evidence points at, over whichever screen opened it. */
export function EvidenceSheets({ lenses, lens, accessToken }: EvidenceSheetsProps) {
  const { issueKey } = useIssueRoute();
  const { evidence, setEvidence } = useEvidenceRoute();
  const owner = (issueKey ? findFinding(lenses, issueKey)?.lens : undefined) ?? lens;
  const target = evidence ? evidenceTarget(evidence.id) : null;
  return (
    <>
      {target?.source === "traces" && (
        <TraceSheet
          open
          traceId={target.id}
          traceRef={target.traceRef}
          initialSpanId={evidence?.span}
          accessToken={accessToken}
          onClose={() => setEvidence(null)}
        />
      )}
      <RequestEvidenceSheet lensId={owner?.id} target={target} onClose={() => setEvidence(null)} />
    </>
  );
}
