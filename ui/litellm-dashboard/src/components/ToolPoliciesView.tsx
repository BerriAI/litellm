"use client";

import { parseAsString, useQueryState } from "nuqs";
import React, { useCallback } from "react";
import useCan from "@/app/(dashboard)/hooks/useCan";
import { ToolDetail } from "@/components/ToolDetail";
import { ToolPoliciesPanel } from "@/components/ToolPolicies/ToolPoliciesPanel";

const TOOL_PARAM = parseAsString.withOptions({ history: "push" });

interface ToolPoliciesViewProps {
  accessToken: string | null;
}

export default function ToolPoliciesView({ accessToken }: ToolPoliciesViewProps) {
  const canViewToolPolicies = useCan("viewToolPolicies");
  const [toolName, setToolName] = useQueryState("tool", TOOL_PARAM);

  const handleSelectTool = useCallback((name: string) => void setToolName(name), [setToolName]);

  const handleBack = useCallback(() => void setToolName(null), [setToolName]);

  if (!canViewToolPolicies) {
    return (
      <div className="p-6 w-full min-w-0 flex-1">
        <h1 className="text-2xl font-semibold text-foreground mb-2">Tool Policies</h1>
        <p className="text-sm text-muted-foreground">Tool Policies is only available to admin users.</p>
      </div>
    );
  }

  return (
    <div className="p-6 w-full min-w-0 flex-1">
      {toolName ? (
        <ToolDetail key={toolName} toolName={toolName} onBack={handleBack} accessToken={accessToken} />
      ) : (
        <ToolPoliciesPanel accessToken={accessToken} onSelectTool={handleSelectTool} />
      )}
    </div>
  );
}
