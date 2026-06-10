"use client";

import React, { useCallback, useDeferredValue, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Button, Switch, Tooltip } from "antd";
import { Table, TableHead, TableHeaderCell, TableBody, TableRow, TableCell } from "@tremor/react";
import { TimeCell } from "./view_logs/time_cell";
import type { SortState } from "./common_components/TableHeaderSortDropdown/TableHeaderSortDropdown";
import { TableHeaderSortDropdown } from "./common_components/TableHeaderSortDropdown/TableHeaderSortDropdown";
import FilterComponent, { FilterOption } from "./molecules/filter";
import { MetricCard } from "./GuardrailsMonitor/MetricCard";
import { PolicySelect, INPUT_POLICY_OPTIONS, OUTPUT_POLICY_OPTIONS } from "./ToolPolicies/PolicySelect";
import { fetchToolsList, updateToolPolicy, ToolRow } from "./networking";

function getUTCDateKey(date: Date): string {
  return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}-${String(date.getUTCDate()).padStart(2, "0")}`;
}

function isCreatedInUTCDay(createdAt: string | undefined, utcDateKey: string): boolean {
  if (!createdAt) return false;
  try {
    const d = new Date(createdAt);
    return getUTCDateKey(d) === utcDateKey;
  } catch {
    return false;
  }
}

function countToolsInUTCDay(tools: ToolRow[], utcDateKey: string): number {
  return tools.filter((t) => isCreatedInUTCDay(t.created_at, utcDateKey)).length;
}

function getTrendSubtitle(
  newToday: number,
  newYesterday: number,
  t: (key: string, opts?: Record<string, unknown>) => string,
): string | undefined {
  const diff = newToday - newYesterday;
  if (diff === 0) return undefined;
  if (diff > 0) return t("toolPolicies.trendPositive", { diff });
  return t("toolPolicies.trendNegative", { diff });
}

type SortField = "tool_name" | "input_policy" | "output_policy" | "team_id" | "key_alias" | "created_at" | "call_count";

interface FilterValues {
  [key: string]: string;
}

interface ToolPoliciesProps {
  accessToken: string | null;
  userRole?: string;
  onSelectTool?: (toolName: string) => void;
}

export const ToolPolicies: React.FC<ToolPoliciesProps> = ({ accessToken, onSelectTool }) => {
  const { t } = useTranslation();
  const [tools, setTools] = useState<ToolRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [isFetching, setIsFetching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savingInput, setSavingInput] = useState<string | null>(null);
  const [savingOutput, setSavingOutput] = useState<string | null>(null);

  const [searchTerm, setSearchTerm] = useState("");
  const [sortField, setSortField] = useState<SortField>("created_at");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [currentPage, setCurrentPage] = useState(1);
  const [isLiveTail, setIsLiveTail] = useState(true);
  const [activeFilters, setActiveFilters] = useState<FilterValues>({});
  const pageSize = 50;

  const isFetchingDeferred = useDeferredValue(isFetching);
  const isButtonLoading = isFetching || isFetchingDeferred;

  const load = useCallback(async () => {
    if (!accessToken) return;
    setIsFetching(true);
    setError(null);
    try {
      const rows = await fetchToolsList(accessToken);
      setTools(rows);
    } catch (e: any) {
      setError(e.message ?? t("toolPolicies.failedToLoadTools"));
    } finally {
      setIsFetching(false);
      setLoading(false);
    }
  }, [accessToken, t]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (!isLiveTail) return;
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, [isLiveTail, load]);

  const handleInputPolicyChange = async (toolName: string, newPolicy: string) => {
    if (!accessToken) return;
    setSavingInput(toolName);
    try {
      await updateToolPolicy(accessToken, toolName, { input_policy: newPolicy });
      setTools((prev) => prev.map((t) => (t.tool_name === toolName ? { ...t, input_policy: newPolicy } : t)));
    } catch (e: any) {
      alert(t("toolPolicies.updateInputPolicyFailed", { error: e.message }));
    } finally {
      setSavingInput(null);
    }
  };

  const handleOutputPolicyChange = async (toolName: string, newPolicy: string) => {
    if (!accessToken) return;
    setSavingOutput(toolName);
    try {
      await updateToolPolicy(accessToken, toolName, { output_policy: newPolicy });
      setTools((prev) => prev.map((t) => (t.tool_name === toolName ? { ...t, output_policy: newPolicy } : t)));
    } catch (e: any) {
      alert(t("toolPolicies.updateOutputPolicyFailed", { error: e.message }));
    } finally {
      setSavingOutput(null);
    }
  };

  const handleSortChange = (field: SortField, newState: SortState) => {
    if (newState === false) {
      setSortField("created_at");
      setSortOrder("desc");
    } else {
      setSortField(field);
      setSortOrder(newState);
    }
    setCurrentPage(1);
  };

  const handleApplyFilters = (filters: FilterValues) => {
    setActiveFilters(filters);
    setCurrentPage(1);
  };

  const handleResetFilters = () => {
    setActiveFilters({});
    setCurrentPage(1);
  };

  const teamOptions = Array.from(new Set(tools.map((t) => t.team_id).filter(Boolean))).map((v) => ({
    label: v as string,
    value: v as string,
  }));
  const keyAliasOptions = Array.from(new Set(tools.map((t) => t.key_alias).filter(Boolean))).map((v) => ({
    label: v as string,
    value: v as string,
  }));

  const filterOptions: FilterOption[] = [
    {
      name: "Input Policy",
      label: t("toolPolicies.inputPolicy"),
      options: INPUT_POLICY_OPTIONS.map((o) => ({ label: o.label, value: o.value })),
    },
    {
      name: "Output Policy",
      label: t("toolPolicies.outputPolicy"),
      options: OUTPUT_POLICY_OPTIONS.map((o) => ({ label: o.label, value: o.value })),
    },
    {
      name: "Team Name",
      label: t("toolPolicies.teamName"),
      options: teamOptions,
    },
    {
      name: "Key Name",
      label: t("toolPolicies.keyName"),
      options: keyAliasOptions,
    },
  ];

  const { newToday, newYesterday, trendSubtitle, totalTools, blockedCount, activeTeamsCount, needsReviewTools } =
    useMemo(() => {
      const now = new Date();
      const todayKey = getUTCDateKey(now);
      const yesterday = new Date(now);
      yesterday.setUTCDate(yesterday.getUTCDate() - 1);
      const yesterdayKey = getUTCDateKey(yesterday);

      const newToday = countToolsInUTCDay(tools, todayKey);
      const newYesterday = countToolsInUTCDay(tools, yesterdayKey);
      const trendSubtitle = getTrendSubtitle(newToday, newYesterday, t);

      const totalTools = tools.length;
      const blockedCount = tools.filter((t) => t.input_policy === "blocked").length;
      const activeTeamsCount = new Set(tools.map((t) => t.team_id).filter(Boolean)).size;

      const needsReviewTools = tools.filter(
        (t) => isCreatedInUTCDay(t.created_at, todayKey) && t.input_policy === "untrusted",
      );

      return {
        newToday,
        newYesterday,
        trendSubtitle,
        totalTools,
        blockedCount,
        activeTeamsCount,
        needsReviewTools,
      };
    }, [tools, t]);

  const SortHeader = ({ label, field }: { label: string; field: SortField }) => (
    <div className="flex items-center gap-1">
      <span>{label}</span>
      <TableHeaderSortDropdown
        sortState={sortField === field ? sortOrder : false}
        onSortChange={(s) => handleSortChange(field, s)}
      />
    </div>
  );

  const filtered = tools.filter((t) => {
    if (searchTerm) {
      const q = searchTerm.toLowerCase();
      const matchesSearch =
        t.tool_name.toLowerCase().includes(q) ||
        (t.team_id ?? "").toLowerCase().includes(q) ||
        (t.key_alias ?? "").toLowerCase().includes(q) ||
        (t.key_hash ?? "").toLowerCase().includes(q) ||
        t.input_policy.toLowerCase().includes(q) ||
        t.output_policy.toLowerCase().includes(q);
      if (!matchesSearch) return false;
    }
    if (activeFilters["Input Policy"] && t.input_policy !== activeFilters["Input Policy"]) return false;
    if (activeFilters["Output Policy"] && t.output_policy !== activeFilters["Output Policy"]) return false;
    if (activeFilters["Team Name"] && t.team_id !== activeFilters["Team Name"]) return false;
    if (activeFilters["Key Name"] && t.key_alias !== activeFilters["Key Name"]) return false;
    return true;
  });

  const sorted = [...filtered].sort((a, b) => {
    const av = (a as any)[sortField] ?? "";
    const bv = (b as any)[sortField] ?? "";
    if (av < bv) return sortOrder === "desc" ? 1 : -1;
    if (av > bv) return sortOrder === "desc" ? -1 : 1;
    return 0;
  });

  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const paginated = sorted.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  const scrollToToolRow = (toolId: string) => {
    const idx = sorted.findIndex((t) => t.tool_id === toolId);
    if (idx >= 0) {
      const page = Math.floor(idx / pageSize) + 1;
      if (page !== currentPage) setCurrentPage(page);
      requestAnimationFrame(() => {
        setTimeout(() => {
          document.getElementById(`tool-row-${toolId}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
        }, 100);
      });
    }
  };

  return (
    <div className="w-full">
      <h1 className="text-2xl font-semibold text-gray-900 mb-6">{t("toolPolicies.title")}</h1>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <MetricCard
          label={t("toolPolicies.newToday")}
          value={newToday}
          valueColor="text-green-600"
          subtitle={trendSubtitle}
          icon={
            <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          }
        />
        <MetricCard label={t("toolPolicies.totalToolsDiscovered")} value={totalTools} />
        <MetricCard
          label={t("toolPolicies.blockedTools")}
          value={blockedCount}
          valueColor={blockedCount > 0 ? "text-red-600" : undefined}
        />
        <MetricCard label={t("toolPolicies.activeTeams")} value={activeTeamsCount > 0 ? activeTeamsCount : "—"} />
      </div>

      {needsReviewTools.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-6">
          <h2 className="text-sm font-semibold text-amber-900 mb-1">{t("toolPolicies.needsReview")}</h2>
          <p className="text-sm text-amber-800 mb-3">
            {t("toolPolicies.needsReviewDesc", { count: needsReviewTools.length })}
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
                  {t("toolPolicies.review")}
                </button>
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="bg-white rounded-lg shadow w-full max-w-full box-border">
        <div className="border-b px-6 py-4 w-full max-w-full box-border">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between space-y-4 md:space-y-0 w-full max-w-full box-border">
            <div className="flex flex-wrap items-center gap-3">
              <div className="relative w-64">
                <input
                  type="text"
                  placeholder={t("toolPolicies.searchByToolName")}
                  className="w-full px-3 py-2 pl-8 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  value={searchTerm}
                  onChange={(e) => {
                    setSearchTerm(e.target.value);
                    setCurrentPage(1);
                  }}
                />
                <svg
                  className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                  />
                </svg>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-gray-900">{t("toolPolicies.liveTail")}</span>
                <Switch checked={isLiveTail} onChange={setIsLiveTail} />
              </div>

              <button
                onClick={load}
                disabled={isButtonLoading}
                className="flex items-center gap-1.5 px-3 py-2 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-60"
              >
                <svg
                  className={`w-4 h-4 ${isButtonLoading ? "animate-spin" : ""}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                  />
                </svg>
                {isButtonLoading ? t("toolPolicies.fetching") : t("toolPolicies.fetch")}
              </button>
            </div>

            <div className="flex items-center gap-4 text-sm text-gray-600 whitespace-nowrap">
              <span>
                {t("toolPolicies.showing", {
                  from: filtered.length === 0 ? 0 : (currentPage - 1) * pageSize + 1,
                  to: Math.min(currentPage * pageSize, filtered.length),
                  total: filtered.length,
                })}
              </span>
              <span>{t("toolPolicies.page", { current: currentPage, total: totalPages })}</span>
              <div className="flex gap-1">
                <button
                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="px-3 py-1.5 border rounded-md text-sm hover:bg-gray-50 disabled:opacity-40"
                >
                  {t("common.previous")}
                </button>
                <button
                  onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                  className="px-3 py-1.5 border rounded-md text-sm hover:bg-gray-50 disabled:opacity-40"
                >
                  {t("common.next")}
                </button>
              </div>
            </div>
          </div>

          <div className="mt-3">
            <FilterComponent
              options={filterOptions}
              onApplyFilters={handleApplyFilters}
              onResetFilters={handleResetFilters}
              buttonLabel={t("toolPolicies.filtersButton")}
            />
          </div>
        </div>

        {isLiveTail && (
          <div className="bg-green-50 border-b border-green-100 px-6 py-2 flex items-center justify-between">
            <span className="text-sm text-green-700">{t("toolPolicies.autoRefreshing")}</span>
            <button onClick={() => setIsLiveTail(false)} className="text-xs text-green-600 underline">
              {t("toolPolicies.stop")}
            </button>
          </div>
        )}

        {error && (
          <div className="mx-6 mt-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">{error}</div>
        )}

        <Table className="[&_td]:py-0.5 [&_th]:py-1 w-full">
          <TableHead>
            <TableRow>
              <TableHeaderCell className="py-1 h-8">
                <SortHeader label={t("toolPolicies.colDiscovered")} field="created_at" />
              </TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">
                <SortHeader label={t("toolPolicies.colToolName")} field="tool_name" />
              </TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">
                <SortHeader label={t("toolPolicies.inputPolicy")} field="input_policy" />
              </TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">
                <SortHeader label={t("toolPolicies.outputPolicy")} field="output_policy" />
              </TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">
                <SortHeader label={t("toolPolicies.colCalls")} field="call_count" />
              </TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">
                <SortHeader label={t("toolPolicies.teamName")} field="team_id" />
              </TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">{t("toolPolicies.colKeyHash")}</TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">
                <SortHeader label={t("toolPolicies.keyName")} field="key_alias" />
              </TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">{t("toolPolicies.colUserAgent")}</TableHeaderCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={9} className="h-8 text-center text-gray-500">
                  {t("toolPolicies.loadingTools")}
                </TableCell>
              </TableRow>
            ) : paginated.length === 0 ? (
              <TableRow>
                <TableCell colSpan={9} className="h-8 text-center text-gray-500">
                  {t("toolPolicies.noToolsDiscovered")}
                </TableCell>
              </TableRow>
            ) : (
              paginated.map((tool) => (
                <TableRow key={tool.tool_id} id={`tool-row-${tool.tool_id}`} className="h-8 hover:bg-gray-50">
                  <TableCell className="py-0.5 max-h-8 overflow-hidden whitespace-nowrap">
                    <TimeCell utcTime={tool.created_at ?? ""} />
                  </TableCell>
                  <TableCell className="py-0.5 max-h-8 overflow-hidden">
                    <button
                      type="button"
                      onClick={() => onSelectTool?.(tool.tool_name)}
                      className="text-left w-full font-mono text-xs max-w-[20ch] truncate block font-medium text-blue-600 hover:text-blue-800 hover:underline focus:outline-none focus:ring-0"
                    >
                      <Tooltip title={onSelectTool ? t("toolPolicies.clickToViewDetails") : tool.tool_name}>
                        <span>{tool.tool_name}</span>
                      </Tooltip>
                    </button>
                  </TableCell>
                  <TableCell className="py-0.5 max-h-8">
                    <PolicySelect
                      value={tool.input_policy}
                      toolName={tool.tool_name}
                      saving={savingInput === tool.tool_name}
                      onChange={handleInputPolicyChange}
                      policyType="input"
                    />
                  </TableCell>
                  <TableCell className="py-0.5 max-h-8">
                    <PolicySelect
                      value={tool.output_policy}
                      toolName={tool.tool_name}
                      saving={savingOutput === tool.tool_name}
                      onChange={handleOutputPolicyChange}
                      policyType="output"
                    />
                  </TableCell>
                  <TableCell className="py-0.5 max-h-8">
                    <div className="flex items-center justify-end h-8 tabular-nums text-sm font-mono text-gray-700">
                      {(tool.call_count ?? 0).toLocaleString()}
                    </div>
                  </TableCell>
                  <TableCell className="py-0.5 max-h-8 overflow-hidden whitespace-nowrap">
                    <Tooltip title={tool.team_id ?? "-"}>
                      <span className="max-w-[15ch] truncate block">{tool.team_id ?? "-"}</span>
                    </Tooltip>
                  </TableCell>
                  <TableCell className="py-0.5 max-h-8 overflow-hidden whitespace-nowrap">
                    <Tooltip title={tool.key_hash ?? "-"}>
                      <span className="font-mono max-w-[15ch] truncate block text-blue-600">
                        {tool.key_hash ?? "-"}
                      </span>
                    </Tooltip>
                  </TableCell>
                  <TableCell className="py-0.5 max-h-8 overflow-hidden whitespace-nowrap">
                    <Tooltip title={tool.key_alias ?? "-"}>
                      <span className="max-w-[15ch] truncate block">{tool.key_alias ?? "-"}</span>
                    </Tooltip>
                  </TableCell>
                  <TableCell className="py-0.5 max-h-8 overflow-hidden whitespace-nowrap">
                    <Tooltip title={tool.user_agent ?? "-"}>
                      <span className="font-mono max-w-[20ch] truncate block text-xs text-gray-500">
                        {tool.user_agent ?? "-"}
                      </span>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>

        {totalPages > 1 && (
          <div className="border-t px-6 py-3 flex items-center justify-between text-sm text-gray-600">
            <span>
              {t("toolPolicies.showing", {
                from: (currentPage - 1) * pageSize + 1,
                to: Math.min(currentPage * pageSize, sorted.length),
                total: sorted.length,
              })}
            </span>
            <div className="flex gap-1">
              <button
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1.5 border rounded-md hover:bg-gray-50 disabled:opacity-40"
              >
                {t("common.previous")}
              </button>
              <button
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-1.5 border rounded-md hover:bg-gray-50 disabled:opacity-40"
              >
                {t("common.next")}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ToolPolicies;
