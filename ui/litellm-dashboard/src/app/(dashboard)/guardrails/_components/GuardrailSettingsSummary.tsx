import React from "react";
import type { Guardrail } from "@/components/guardrails/types";
import { Badge } from "@/components/ui/badge";
import { formatGuardrailMode, samplingPercentageToForm } from "./guardrail_info_helpers";
import ToolPermissionRulesEditor, { ToolPermissionConfig } from "./tool_permission/ToolPermissionRulesEditor";

interface GuardrailSettingsSummaryProps {
  guardrailData: Pick<Guardrail, "guardrail_id" | "guardrail_name" | "litellm_params" | "created_at" | "updated_at">;
  displayName: string;
  formatDate: (dateString?: string) => string;
  toolPermissionConfig: ToolPermissionConfig;
}

const GuardrailSettingsSummary: React.FC<GuardrailSettingsSummaryProps> = ({
  guardrailData,
  displayName,
  formatDate,
  toolPermissionConfig,
}) => (
  <div className="space-y-4">
    <div>
      <p className="font-medium">Guardrail ID</p>
      <div className="font-mono">{guardrailData.guardrail_id}</div>
    </div>
    <div>
      <p className="font-medium">Guardrail Name</p>
      <div>{guardrailData.guardrail_name || "Unnamed Guardrail"}</div>
    </div>
    <div>
      <p className="font-medium">Provider</p>
      <div>{displayName}</div>
    </div>
    <div>
      <p className="font-medium">Mode</p>
      <div>{formatGuardrailMode(guardrailData.litellm_params?.mode) || "-"}</div>
    </div>
    <div>
      <p className="font-medium">Default On</p>
      <Badge variant={guardrailData.litellm_params?.default_on ? "secondary" : "outline"}>
        {guardrailData.litellm_params?.default_on ? "Yes" : "No"}
      </Badge>
    </div>
    <div>
      <p className="font-medium">Sampling percentage</p>
      <div>{samplingPercentageToForm(guardrailData.litellm_params?.sampling_percentage)}%</div>
    </div>

    {guardrailData.litellm_params?.pii_entities_config &&
      Object.keys(guardrailData.litellm_params.pii_entities_config).length > 0 && (
        <div>
          <p className="font-medium">PII Protection</p>
          <div className="mt-2">
            <Badge variant="secondary">
              {Object.keys(guardrailData.litellm_params.pii_entities_config).length} PII entities configured
            </Badge>
          </div>
        </div>
      )}

    <div>
      <p className="font-medium">Created At</p>
      <div>{formatDate(guardrailData.created_at)}</div>
    </div>
    <div>
      <p className="font-medium">Last Updated</p>
      <div>{formatDate(guardrailData.updated_at)}</div>
    </div>

    {guardrailData.litellm_params?.guardrail === "tool_permission" && (
      <ToolPermissionRulesEditor value={toolPermissionConfig} disabled />
    )}
  </div>
);

export default GuardrailSettingsSummary;
