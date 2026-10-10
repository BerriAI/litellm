import { Badge } from "@/components/ui/badge";
import { GuardrailModeRows } from "./GuardrailModeDisplay";
import { GuardrailStreamScopeDetail } from "./StreamScopeFields";
import ToolPermissionRulesEditor, { type ToolPermissionConfig } from "./tool_permission/ToolPermissionRulesEditor";

export const GuardrailReadOnlyDetails = ({
  guardrailId,
  guardrailName,
  displayName,
  litellmParams,
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
  litellmParams: { mode?: unknown; logging_only_scope?: string | null };
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
    <GuardrailModeRows litellmParams={litellmParams} />
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
