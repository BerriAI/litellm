import React from "react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { formatGuardrailMode, formatLoggingOnlyScope, modeIncludesLoggingOnly } from "./guardrail_info_helpers";

type GuardrailModeParams = {
  mode?: unknown;
  default_on?: boolean;
  logging_only_scope?: string | null;
};

export const GuardrailModeCard: React.FC<{ litellmParams: GuardrailModeParams }> = ({ litellmParams }) => (
  <Card className="block p-6">
    <p>Mode</p>
    <div className="mt-2">
      <h3 className="text-lg font-medium">{formatGuardrailMode(litellmParams.mode) || "-"}</h3>
      <Badge variant={litellmParams.default_on ? "secondary" : "outline"}>
        {litellmParams.default_on ? "Default On" : "Default Off"}
      </Badge>
    </div>
    {modeIncludesLoggingOnly(litellmParams.mode) && (
      <div className="mt-4">
        <p>Logging only scope</p>
        <h3 className="text-lg font-medium">{formatLoggingOnlyScope(litellmParams.logging_only_scope)}</h3>
      </div>
    )}
  </Card>
);

export const GuardrailModeRows: React.FC<{ litellmParams: GuardrailModeParams }> = ({ litellmParams }) => (
  <>
    <div>
      <p className="font-medium">Mode</p>
      <div>{formatGuardrailMode(litellmParams.mode) || "-"}</div>
    </div>
    {modeIncludesLoggingOnly(litellmParams.mode) && (
      <div>
        <p className="font-medium">Logging only scope</p>
        <div>{formatLoggingOnlyScope(litellmParams.logging_only_scope)}</div>
      </div>
    )}
  </>
);
