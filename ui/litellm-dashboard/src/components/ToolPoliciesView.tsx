"use client";

import React, { useCallback } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { ToolDetail } from "@/components/ToolDetail";
import { ToolPoliciesPanel } from "@/components/ToolPolicies/ToolPoliciesPanel";

export const TOOL_QUERY_PARAM = "tool";

interface ToolPoliciesViewProps {
  accessToken: string | null;
}

export default function ToolPoliciesView({ accessToken }: ToolPoliciesViewProps) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const selectedTool = searchParams.get(TOOL_QUERY_PARAM);

  const navigateToTool = useCallback(
    (toolName: string | null) => {
      const params = new URLSearchParams(searchParams.toString());
      if (toolName) {
        params.set(TOOL_QUERY_PARAM, toolName);
      } else {
        params.delete(TOOL_QUERY_PARAM);
      }
      const query = params.toString();
      router.push(query ? `${pathname}?${query}` : pathname);
    },
    [pathname, router, searchParams],
  );

  const handleSelectTool = useCallback((toolName: string) => navigateToTool(toolName), [navigateToTool]);
  const handleBack = useCallback(() => navigateToTool(null), [navigateToTool]);

  return (
    <div className="p-6 w-full min-w-0 flex-1">
      {selectedTool ? (
        <ToolDetail toolName={selectedTool} onBack={handleBack} accessToken={accessToken} />
      ) : (
        <ToolPoliciesPanel accessToken={accessToken} onSelectTool={handleSelectTool} />
      )}
    </div>
  );
}
