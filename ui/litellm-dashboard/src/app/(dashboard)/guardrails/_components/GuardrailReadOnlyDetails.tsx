import { Badge } from "@/components/ui/badge";
import { formatGuardrailMode } from "./guardrail_info_helpers";
import { GuardrailStreamScopeDetail } from "./StreamScopeFields";
import ToolPermissionRulesEditor, { type ToolPermissionConfig } from "./tool_permission/ToolPermissionRulesEditor";

export const GuardrailReadOnlyDetails = ({
  guardrailId,
  guardrailName,
  displayName,
  mode,
  streamScope,
  defaultOn,
  piiEntityCount,
  createdAt,
  updatedAt,
  showToolPermission,
  toolPermissionConfig,
}: {
  guardrailId: string;
  guardrailName: string;
  displayName: string;
  mode: unknown;
  streamScope: unknown;
  defaultOn: boolean | undefined;
  piiEntityCount: number;
  createdAt: string;
  updatedAt: string;
  showToolPermission: boolean;
  toolPermissionConfig: ToolPermissionConfig;
}) => (
  <div className="space-y-4">
    <div>
      <p className="font-medium">Guardrail ID</p>
      <div className="font-mono">{guardrailId}</div>
    </div>
    <div>
      <p className="font-medium">Guardrail Name</p>
      <div>{guardrailName || "Unnamed Guardrail"}</div>
    </div>
    <div>
      <p className="font-medium">Provider</p>
      <div>{displayName}</div>
    </div>
    <div>
      <p className="font-medium">Mode</p>
      <div>{formatGuardrailMode(mode) || "-"}</div>
    </div>
    <GuardrailStreamScopeDetail raw={streamScope} />
    <div>
      <p className="font-medium">Default On</p>
      <Badge variant={defaultOn ? "secondary" : "outline"}>{defaultOn ? "Yes" : "No"}</Badge>
    </div>
    {piiEntityCount > 0 && (
      <div>
        <p className="font-medium">PII Protection</p>
        <div className="mt-2">
          <Badge variant="secondary">{piiEntityCount} PII entities configured</Badge>
        </div>
      </div>
    )}
    <div>
      <p className="font-medium">Created At</p>
      <div>{createdAt}</div>
    </div>
    <div>
      <p className="font-medium">Last Updated</p>
      <div>{updatedAt}</div>
    </div>
    {showToolPermission && <ToolPermissionRulesEditor value={toolPermissionConfig} disabled />}
  </div>
);
