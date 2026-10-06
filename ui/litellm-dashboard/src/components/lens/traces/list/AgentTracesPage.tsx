"use client";

import { useRelativeRange } from "@/components/shared/timeRange/useRelativeRange";

import { AgentTracesSection } from "./AgentTracesSection";
import { useTracesLive } from "../api";
import type { InvestigateScope } from "../../route";

export default function AgentTracesPage({
  accessToken,
  isActive = true,
  readOnly = false,
  canMintTracingKey = false,
  canViewFindings = true,
  onInvestigate,
}: {
  accessToken: string;
  isActive?: boolean;
  readOnly?: boolean;
  canMintTracingKey?: boolean;
  canViewFindings?: boolean;
  onInvestigate?: (scope: InvestigateScope) => void;
}) {
  const time = useRelativeRange(useTracesLive());
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <AgentTracesSection
        accessToken={accessToken}
        isActive={isActive}
        range={time.range}
        readOnly={readOnly}
        canMintTracingKey={canMintTracingKey}
        canViewFindings={canViewFindings}
        onInvestigate={onInvestigate}
        timeControls={time}
      />
    </div>
  );
}
