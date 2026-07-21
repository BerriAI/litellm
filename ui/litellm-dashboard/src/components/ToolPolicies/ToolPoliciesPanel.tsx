"use client";

import { useQuery, useQueryClient, type UseQueryOptions } from "@tanstack/react-query";
import React, { useCallback, useMemo, useState } from "react";

import { MetricCard } from "@/components/GuardrailsMonitor/MetricCard";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { fetchToolsList, ToolRow, updateToolPolicy } from "@/components/networking";

import { ToolPoliciesTable } from "./ToolPoliciesTable";

function getUTCDateKey(date: Date): string {
  return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}-${String(date.getUTCDate()).padStart(2, "0")}`;
}

function isCreatedInUTCDay(createdAt: string | undefined, utcDateKey: string): boolean {
  if (!createdAt) return false;
  try {
    return getUTCDateKey(new Date(createdAt)) === utcDateKey;
  } catch {
    return false;
  }
}

function countToolsInUTCDay(tools: ToolRow[], utcDateKey: string): number {
  return tools.filter((tool) => isCreatedInUTCDay(tool.created_at, utcDateKey)).length;
}

function getTrendSubtitle(newToday: number, newYesterday: number): string | undefined {
  const diff = newToday - newYesterday;
  if (diff === 0) return undefined;
  return diff > 0 ? `+${diff} since yesterday` : `${diff} since yesterday`;
}

function toMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

const TOOLS_QUERY_KEY = "tool-policies";

interface ToolPoliciesPanelProps {
  accessToken: string | null;
  onSelectTool: (toolName: string) => void;
}

export const ToolPoliciesPanel: React.FC<ToolPoliciesPanelProps> = ({ accessToken, onSelectTool }) => {
  const queryClient = useQueryClient();
  const [savingInput, setSavingInput] = useState<string | null>(null);
  const [savingOutput, setSavingOutput] = useState<string | null>(null);

  const queryKey = useMemo(() => [TOOLS_QUERY_KEY, accessToken], [accessToken]);

  const queryOptions: UseQueryOptions<ToolRow[]> = {
    queryKey,
    queryFn: async () => (accessToken === null ? [] : fetchToolsList(accessToken)),
    enabled: accessToken !== null,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  };
  const query = useQuery(queryOptions);

  const tools = useMemo(() => query.data ?? [], [query.data]);

  const patchTool = useCallback(
    (toolName: string, patch: Partial<ToolRow>) => {
      queryClient.setQueryData<ToolRow[]>(queryKey, (previous) =>
        (previous ?? []).map((tool) => (tool.tool_name === toolName ? { ...tool, ...patch } : tool)),
      );
    },
    [queryClient, queryKey],
  );

  const handleInputPolicyChange = useCallback(
    async (toolName: string, newPolicy: string) => {
      if (accessToken === null) return;
      setSavingInput(toolName);
      try {
        await updateToolPolicy(accessToken, toolName, { input_policy: newPolicy });
        patchTool(toolName, { input_policy: newPolicy });
      } catch (e) {
        NotificationsManager.fromBackend(`Failed to update input policy: ${toMessage(e, "unknown error")}`);
      } finally {
        setSavingInput(null);
      }
    },
    [accessToken, patchTool],
  );

  const handleOutputPolicyChange = useCallback(
    async (toolName: string, newPolicy: string) => {
      if (accessToken === null) return;
      setSavingOutput(toolName);
      try {
        await updateToolPolicy(accessToken, toolName, { output_policy: newPolicy });
        patchTool(toolName, { output_policy: newPolicy });
      } catch (e) {
        NotificationsManager.fromBackend(`Failed to update output policy: ${toMessage(e, "unknown error")}`);
      } finally {
        setSavingOutput(null);
      }
    },
    [accessToken, patchTool],
  );

  const { newToday, trendSubtitle, totalTools, blockedCount, activeTeamsCount, needsReviewTools } = useMemo(() => {
    const now = new Date();
    const todayKey = getUTCDateKey(now);
    const yesterday = new Date(now);
    yesterday.setUTCDate(yesterday.getUTCDate() - 1);
    const today = countToolsInUTCDay(tools, todayKey);

    return {
      newToday: today,
      trendSubtitle: getTrendSubtitle(today, countToolsInUTCDay(tools, getUTCDateKey(yesterday))),
      totalTools: tools.length,
      blockedCount: tools.filter((tool) => tool.input_policy === "blocked").length,
      activeTeamsCount: new Set(tools.map((tool) => tool.team_id).filter(Boolean)).size,
      needsReviewTools: tools.filter(
        (tool) => isCreatedInUTCDay(tool.created_at, todayKey) && tool.input_policy === "untrusted",
      ),
    };
  }, [tools]);

  const scrollToToolRow = (toolId: string) => {
    document.querySelector(`[data-row-id="${CSS.escape(toolId)}"]`)?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  };

  return (
    <div className="w-full">
      <h1 className="text-2xl font-semibold text-gray-900 mb-6">Tool Policies</h1>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <MetricCard
          label="New Today"
          value={newToday}
          valueColor="text-green-600"
          subtitle={trendSubtitle}
          icon={
            <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          }
        />
        <MetricCard label="Total Tools Discovered" value={totalTools} />
        <MetricCard
          label="Blocked Tools"
          value={blockedCount}
          valueColor={blockedCount > 0 ? "text-red-600" : undefined}
        />
        <MetricCard label="Active Teams" value={activeTeamsCount > 0 ? activeTeamsCount : "—"} />
      </div>

      {needsReviewTools.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-6">
          <h2 className="text-sm font-semibold text-amber-900 mb-1">Needs Review</h2>
          <p className="text-sm text-amber-800 mb-3">
            {needsReviewTools.length} new tool{needsReviewTools.length !== 1 ? "s" : ""} discovered that require policy
            decisions.
          </p>
          <div className="flex flex-wrap gap-2">
            {needsReviewTools.map((tool) => (
              <span
                key={tool.tool_id}
                className="inline-flex items-center gap-2 px-3 py-1.5 bg-white border border-amber-200 rounded-md text-sm"
              >
                <span className="font-mono text-amber-900 truncate max-w-[200px]" title={tool.tool_name}>
                  {tool.tool_name}
                </span>
                <button
                  type="button"
                  onClick={() => scrollToToolRow(tool.tool_id)}
                  className="text-amber-700 hover:text-amber-900 font-medium text-xs whitespace-nowrap"
                >
                  Review
                </button>
              </span>
            ))}
          </div>
        </div>
      )}

      {query.isError && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-sm text-sm text-red-700" role="alert">
          {toMessage(query.error, "Failed to load tools")}
        </div>
      )}

      <ToolPoliciesTable
        data={tools}
        isLoading={query.isLoading}
        isRefreshing={query.isFetching}
        onRefresh={() => void query.refetch()}
        onSelectTool={onSelectTool}
        savingInput={savingInput}
        savingOutput={savingOutput}
        onInputPolicyChange={handleInputPolicyChange}
        onOutputPolicyChange={handleOutputPolicyChange}
      />
    </div>
  );
};
