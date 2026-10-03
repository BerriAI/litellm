import React from "react";
import { ChevronDown } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { MountedFormField } from "@/components/common_components/MountedFormField";
import { tagsControl, textControl } from "./mcpFieldRules";

export const APPROVAL_POLICY_FIELD_NAMES = [
  "approval_policy_tools",
  "approval_policy_issuer",
  "approval_policy_jwks_url",
  "approval_policy_audience",
] as const;

const ApprovalPolicySection: React.FC = () => (
  <Collapsible className="rounded-lg border border-border px-4">
    <CollapsibleTrigger className="group flex w-full items-center justify-between gap-4 py-3 text-left">
      <span className="text-sm font-semibold text-foreground">Approval references (optional)</span>
      <ChevronDown className="size-4 text-muted-foreground transition-transform group-data-[panel-open]:rotate-180" />
    </CollapsibleTrigger>
    <CollapsibleContent keepMounted className="space-y-4 pb-4">
      <p className="text-sm text-muted-foreground">
        Calls to these tools must carry a signed, unexpired approval reference issued by the approval service below.
        References must set mcp_server to this server&apos;s ID. Other tools are unaffected.
      </p>
      <MountedFormField label="Tools requiring approval" name="approval_policy_tools">
        {(control) => <MultiSelect {...tagsControl(control)} placeholder="Add tool names" className="rounded-lg" />}
      </MountedFormField>
      <MountedFormField label="Issuer" name="approval_policy_issuer">
        {(control) => (
          <Input
            {...textControl(control)}
            placeholder="https://approvals.example.com"
            className="rounded-lg border-border focus:border-info focus:ring-ring"
          />
        )}
      </MountedFormField>
      <MountedFormField
        label="JWKS URL"
        name="approval_policy_jwks_url"
        help="Must use https, http is allowed only for localhost"
      >
        {(control) => (
          <Input
            {...textControl(control)}
            placeholder="https://approvals.example.com/.well-known/jwks.json"
            className="rounded-lg border-border focus:border-info focus:ring-ring"
          />
        )}
      </MountedFormField>
      <MountedFormField label="Audience (optional)" name="approval_policy_audience">
        {(control) => (
          <Input
            {...textControl(control)}
            placeholder="mcp-gateway"
            className="rounded-lg border-border focus:border-info focus:ring-ring"
          />
        )}
      </MountedFormField>
    </CollapsibleContent>
  </Collapsible>
);

export default ApprovalPolicySection;
