"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import React, { useCallback, useMemo, useState } from "react";

import useCan from "@/app/(dashboard)/hooks/useCan";
import { MetricCard } from "@/components/GuardrailsMonitor/MetricCard";
import { toast } from "@/lib/toast";
import { ToolRow, updateToolPolicy } from "@/components/networking";

import { toolPoliciesListOptions } from "./toolPoliciesQueries";
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

const withTool = (names: ReadonlySet<string>, toolName: string): ReadonlySet<string> => new Set([...names, toolName]);

const withoutTool = (names: ReadonlySet<string>, toolName: string): ReadonlySet<string> =>
  new Set([...names].filter((name) => name !== toolName));

interface ToolPoliciesPanelProps {
  accessToken: string | null;
  onSelectTool: (toolName: string) => void;
}

export const ToolPoliciesPanel: React.FC<ToolPoliciesPanelProps> = ({ accessToken, onSelectTool }) => {
  const queryClient = useQueryClient();
  const canViewToolPolicies = useCan("viewToolPolicies");
  const [savingInput, setSavingInput] = useState<ReadonlySet<string>>(() => new Set());
  const [savingOutput, setSavingOutput] = useState<ReadonlySet<string>>(() => new Set());

  const listOptions = useMemo(() => toolPoliciesListOptions(accessToken), [accessToken]);
  const query = useQuery({ ...listOptions, enabled: canViewToolPolicies && accessToken !== null });

  const tools = useMemo(() => query.data ?? [], [query.data]);

  // Cancel first: a list fetch that started before this save would otherwise resolve afterwards
  // and overwrite the row we just wrote with its pre-save snapshot.
  const patchTool = useCallback(
    async (toolName: string, patch: Partial<ToolRow>) => {
      await queryClient.cancelQueries({ queryKey: listOptions.queryKey });
      queryClient.setQueryData(listOptions.queryKey, (previous) =>
        (previous ?? []).map((tool) => (tool.tool_name === toolName ? { ...tool, ...patch } : tool)),
      );
    },
    [queryClient, listOptions],
  );

  const handleInputPolicyChange = useCallback(
    async (toolName: string, newPolicy: string) => {
      if (accessToken === null) return;
      setSavingInput((previous) => withTool(previous, toolName));
      try {
        await updateToolPolicy(accessToken, toolName, { input_policy: newPolicy });
        await patchTool(toolName, { input_policy: newPolicy });
      } catch (e) {
        toast.fromError(`Failed to update input policy: ${toMessage(e, "unknown error")}`);
      } finally {
        setSavingInput((previous) => withoutTool(previous, toolName));
      }
    },
    [accessToken, patchTool],
  );

  const handleOutputPolicyChange = useCallback(
    async (toolName: string, newPolicy: string) => {
      if (accessToken === null) return;
      setSavingOutput((previous) => withTool(previous, toolName));
      try {
        await updateToolPolicy(accessToken, toolName, { output_policy: newPolicy });
        await patchTool(toolName, { output_policy: newPolicy });
      } catch (e) {
        toast.fromError(`Failed to update output policy: ${toMessage(e, "unknown error")}`);
      } finally {
        setSavingOutput((previous) => withoutTool(previous, toolName));
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
      <h1 className="text-2xl font-semibold text-foreground mb-6">Tool Policies</h1>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <MetricCard
          label="New Today"
          value={newToday}
          valueColor="text-success"
          subtitle={trendSubtitle}
          icon={
            <svg className="w-4 h-4 text-success" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          }
        />
        <MetricCard label="Total Tools Discovered" value={totalTools} />
        <MetricCard
          label="Blocked Tools"
          value={blockedCount}
          valueColor={blockedCount > 0 ? "text-destructive" : undefined}
        />
        <MetricCard label="Active Teams" value={activeTeamsCount > 0 ? activeTeamsCount : "—"} />
      </div>

      {needsReviewTools.length > 0 && (
        <div className="bg-warning/10 border border-warning/20 rounded-lg p-4 mb-6">
          <h2 className="text-sm font-semibold text-warning mb-1">Needs Review</h2>
          <p className="text-sm text-warning mb-3">
            {needsReviewTools.length} new tool{needsReviewTools.length !== 1 ? "s" : ""} discovered that require policy
            decisions.
          </p>
          <div className="flex flex-wrap gap-2">
            {needsReviewTools.map((tool) => (
              <span
                key={tool.tool_id}
                className="inline-flex items-center gap-2 px-3 py-1.5 bg-card border border-warning/20 rounded-md text-sm"
              >
                <span className="font-mono text-warning truncate max-w-[200px]" title={tool.tool_name}>
                  {tool.tool_name}
                </span>
                <button
                  type="button"
                  onClick={() => scrollToToolRow(tool.tool_id)}
                  className="text-warning hover:text-warning/80 font-medium text-xs whitespace-nowrap"
                >
                  Review
                </button>
              </span>
            ))}
          </div>
        </div>
      )}

      {query.isError && (
        <div
          className="mb-4 p-3 bg-destructive/10 border border-destructive/20 rounded-sm text-sm text-destructive"
          role="alert"
        >
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
