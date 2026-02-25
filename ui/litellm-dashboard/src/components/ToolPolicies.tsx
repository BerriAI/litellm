"use client";

import React, { useCallback, useDeferredValue, useEffect, useState } from "react";
import { Switch } from "@tremor/react";
import { Drawer, Select, Tooltip } from "antd";
import {
  Table,
  TableHead,
  TableHeaderCell,
  TableBody,
  TableRow,
  TableCell,
} from "@tremor/react";
import { TimeCell } from "./view_logs/time_cell";
import { TableHeaderSortDropdown } from "./common_components/TableHeaderSortDropdown/TableHeaderSortDropdown";
import type { SortState } from "./common_components/TableHeaderSortDropdown/TableHeaderSortDropdown";
import FilterComponent, { FilterOption } from "./molecules/filter";
import {
  fetchToolsList,
  fetchToolCallLogs,
  fetchToolSettings,
  updateToolPolicy,
  updateToolSettings,
  ToolRow,
  ToolCallLog,
} from "./networking";
import { FormattedToolView } from "./view_logs/ToolsSection/FormattedToolView";
import type { ParsedTool } from "./view_logs/ToolsSection/types";

const POLICY_OPTIONS = [
  { value: "trusted",   label: "trusted",   color: "#065f46", bg: "#d1fae5", border: "#6ee7b7" },
  { value: "blocked",   label: "blocked",   color: "#991b1b", bg: "#fee2e2", border: "#fca5a5" },
] as const;

type PolicyValue = "trusted" | "blocked";

const policyStyle = (p: string) =>
  POLICY_OPTIONS.find((o) => o.value === p) ?? POLICY_OPTIONS[1];

type SortField = "tool_name" | "call_policy" | "team_id" | "key_alias" | "created_at" | "call_count";

interface FilterValues {
  [key: string]: string;
}

interface ToolPoliciesProps {
  accessToken: string | null;
  userRole?: string;
}

const PolicySelect: React.FC<{
  value: string;
  toolName: string;
  saving: boolean;
  onChange: (toolName: string, policy: string) => void;
}> = ({ value, toolName, saving, onChange }) => {
  const style = policyStyle(value);
  return (
    <Select
      size="small"
      value={value}
      disabled={saving}
      loading={saving}
      onChange={(v) => onChange(toolName, v)}
      onClick={(e) => e.stopPropagation()}
      style={{
        minWidth: 110,
        fontWeight: 500,
      }}
      styles={{
        selector: {
          backgroundColor: style.bg,
          borderColor: style.border,
          color: style.color,
          borderRadius: 999,
          fontSize: 11,
          fontWeight: 600,
          paddingLeft: 8,
          paddingRight: 4,
        },
      }}
      popupMatchSelectWidth={false}
      options={POLICY_OPTIONS.map((o) => ({
        value: o.value,
        label: (
          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              fontSize: 12,
              fontWeight: 500,
              color: o.color,
            }}
          >
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                backgroundColor: o.color,
                display: "inline-block",
                flexShrink: 0,
              }}
            />
            {o.label}
          </span>
        ),
      }))}
    />
  );
};

/** Build a ParsedTool from captured tool_arguments for display. */
function buildParsedTool(toolArguments: Record<string, unknown> | null | undefined, toolName: string): ParsedTool | null {
  if (!toolArguments) return null;
  return {
    index: 1,
    name: toolName,
    description: "",
    parameters: {},
    called: true,
    callData: { id: "", name: toolName, arguments: toolArguments as Record<string, any> },
  };
}

export const ToolPolicies: React.FC<ToolPoliciesProps> = ({ accessToken }) => {
  const [tools, setTools] = useState<ToolRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [isFetching, setIsFetching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState<string | null>(null);

  const [searchTerm, setSearchTerm] = useState("");
  const [sortField, setSortField] = useState<SortField>("created_at");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [currentPage, setCurrentPage] = useState(1);
  const [isLiveTail, setIsLiveTail] = useState(true);
  const [activeFilters, setActiveFilters] = useState<FilterValues>({});
  const pageSize = 50;

  // Settings state
  const [showSettings, setShowSettings] = useState(false);
  const [storeToolCallLogs, setStoreToolCallLogs] = useState(true);
  const [savingSettings, setSavingSettings] = useState(false);

  // Call log drawer state
  const [selectedTool, setSelectedTool] = useState<ToolRow | null>(null);
  const [callLogs, setCallLogs] = useState<ToolCallLog[]>([]);
  const [callLogsTotal, setCallLogsTotal] = useState(0);
  const [callLogsLoading, setCallLogsLoading] = useState(false);
  const [callLogsPage, setCallLogsPage] = useState(1);
  const callLogsPageSize = 20;

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
      setError(e.message ?? "Failed to load tools");
    } finally {
      setIsFetching(false);
      setLoading(false);
    }
  }, [accessToken]);

  // Load settings once on mount
  useEffect(() => {
    if (!accessToken) return;
    fetchToolSettings(accessToken)
      .then((s) => setStoreToolCallLogs(s.store_tool_call_logs))
      .catch(() => {/* ignore, default true */});
  }, [accessToken]);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    if (!isLiveTail) return;
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, [isLiveTail, load]);

  const handlePolicyChange = async (toolName: string, newPolicy: string) => {
    if (!accessToken) return;
    setSaving(toolName);
    try {
      await updateToolPolicy(accessToken, toolName, newPolicy);
      setTools((prev) =>
        prev.map((t) => (t.tool_name === toolName ? { ...t, call_policy: newPolicy } : t))
      );
    } catch (e: any) {
      alert(`Failed to update policy: ${e.message}`);
    } finally {
      setSaving(null);
    }
  };

  const handleSettingsToggle = async (val: boolean) => {
    if (!accessToken) return;
    setSavingSettings(true);
    try {
      const result = await updateToolSettings(accessToken, val);
      setStoreToolCallLogs(result.store_tool_call_logs);
    } catch {
      // revert on error
      setStoreToolCallLogs(!val);
    } finally {
      setSavingSettings(false);
    }
  };

  const loadCallLogs = useCallback(
    async (tool: ToolRow, page: number) => {
      if (!accessToken) return;
      setCallLogsLoading(true);
      try {
        const offset = (page - 1) * callLogsPageSize;
        const result = await fetchToolCallLogs(accessToken, tool.tool_name, callLogsPageSize, offset);
        setCallLogs(result.calls);
        setCallLogsTotal(result.total);
      } catch {
        setCallLogs([]);
        setCallLogsTotal(0);
      } finally {
        setCallLogsLoading(false);
      }
    },
    [accessToken]
  );

  const handleRowClick = (tool: ToolRow) => {
    setSelectedTool(tool);
    setCallLogsPage(1);
    loadCallLogs(tool, 1);
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

  // Build unique team/key options from loaded data
  const teamOptions = Array.from(new Set(tools.map((t) => t.team_id).filter(Boolean))).map(
    (v) => ({ label: v as string, value: v as string })
  );
  const keyAliasOptions = Array.from(new Set(tools.map((t) => t.key_alias).filter(Boolean))).map(
    (v) => ({ label: v as string, value: v as string })
  );

  const filterOptions: FilterOption[] = [
    {
      name: "Policy",
      label: "Policy",
      options: POLICY_OPTIONS.map((o) => ({ label: o.label, value: o.value })),
    },
    {
      name: "Team Name",
      label: "Team Name",
      options: teamOptions,
    },
    {
      name: "Key Name",
      label: "Key Name",
      options: keyAliasOptions,
    },
  ];

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
        t.call_policy.toLowerCase().includes(q);
      if (!matchesSearch) return false;
    }
    if (activeFilters["Policy"] && t.call_policy !== activeFilters["Policy"]) return false;
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
  const callLogsTotalPages = Math.max(1, Math.ceil(callLogsTotal / callLogsPageSize));

  return (
    <div className="p-6 w-full">
      <h1 className="text-2xl font-semibold text-gray-900 mb-6">Tool Policies</h1>
      <div className="bg-white rounded-lg shadow w-full max-w-full box-border">

        {/* Toolbar */}
        <div className="border-b px-6 py-4 w-full max-w-full box-border">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between space-y-4 md:space-y-0 w-full max-w-full box-border">
            <div className="flex flex-wrap items-center gap-3">
              <div className="relative w-64">
                <input
                  type="text"
                  placeholder="Search by Tool Name"
                  className="w-full px-3 py-2 pl-8 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  value={searchTerm}
                  onChange={(e) => { setSearchTerm(e.target.value); setCurrentPage(1); }}
                />
                <svg className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-gray-900">Live Tail</span>
                <Switch color="green" checked={isLiveTail} onChange={setIsLiveTail} />
              </div>

              <button
                onClick={load}
                disabled={isButtonLoading}
                className="flex items-center gap-1.5 px-3 py-2 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-60"
              >
                <svg className={`w-4 h-4 ${isButtonLoading ? "animate-spin" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                {isButtonLoading ? "Fetching" : "Fetch"}
              </button>

              {/* Settings gear */}
              <button
                onClick={() => setShowSettings(true)}
                className="flex items-center gap-1.5 px-3 py-2 text-sm border rounded-md hover:bg-gray-50"
                title="Tool log settings"
              >
                <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </button>
            </div>

            <div className="flex items-center gap-4 text-sm text-gray-600 whitespace-nowrap">
              <span>
                Showing {filtered.length === 0 ? 0 : (currentPage - 1) * pageSize + 1} - {Math.min(currentPage * pageSize, filtered.length)} of {filtered.length} results
              </span>
              <span>Page {currentPage} of {totalPages}</span>
              <div className="flex gap-1">
                <button onClick={() => setCurrentPage((p) => Math.max(1, p - 1))} disabled={currentPage === 1}
                  className="px-3 py-1.5 border rounded-md text-sm hover:bg-gray-50 disabled:opacity-40">Previous</button>
                <button onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))} disabled={currentPage === totalPages}
                  className="px-3 py-1.5 border rounded-md text-sm hover:bg-gray-50 disabled:opacity-40">Next</button>
              </div>
            </div>
          </div>

          {/* Filter row */}
          <div className="mt-3">
            <FilterComponent
              options={filterOptions}
              onApplyFilters={handleApplyFilters}
              onResetFilters={handleResetFilters}
              buttonLabel="Filters"
            />
          </div>
        </div>

        {/* Auto-refresh banner */}
        {isLiveTail && (
          <div className="bg-green-50 border-b border-green-100 px-6 py-2 flex items-center justify-between">
            <span className="text-sm text-green-700">Auto-refreshing every 15 seconds</span>
            <button onClick={() => setIsLiveTail(false)} className="text-xs text-green-600 underline">Stop</button>
          </div>
        )}

        {error && (
          <div className="mx-6 mt-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">{error}</div>
        )}

        {/* Table */}
        <Table className="[&_td]:py-0.5 [&_th]:py-1 w-full">
          <TableHead>
            <TableRow>
              <TableHeaderCell className="py-1 h-8"><SortHeader label="Discovered" field="created_at" /></TableHeaderCell>
              <TableHeaderCell className="py-1 h-8"><SortHeader label="Tool Name" field="tool_name" /></TableHeaderCell>
              <TableHeaderCell className="py-1 h-8"><SortHeader label="Policy" field="call_policy" /></TableHeaderCell>
              <TableHeaderCell className="py-1 h-8"><SortHeader label="# Calls" field="call_count" /></TableHeaderCell>
              <TableHeaderCell className="py-1 h-8"><SortHeader label="Team Name" field="team_id" /></TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">Key Hash</TableHeaderCell>
              <TableHeaderCell className="py-1 h-8"><SortHeader label="Key Name" field="key_alias" /></TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">Origin</TableHeaderCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={8} className="h-8 text-center text-gray-500">Loading tools…</TableCell>
              </TableRow>
            ) : paginated.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} className="h-8 text-center text-gray-500">
                  No tools discovered yet. Make a chat completion that returns tool_calls to start auto-discovery.
                </TableCell>
              </TableRow>
            ) : (
              paginated.map((tool) => (
                <TableRow
                  key={tool.tool_id}
                  className="h-8 hover:bg-gray-50 cursor-pointer"
                  onClick={() => handleRowClick(tool)}
                >
                  <TableCell className="py-0.5 max-h-8 overflow-hidden whitespace-nowrap">
                    <TimeCell utcTime={tool.created_at ?? ""} />
                  </TableCell>
                  <TableCell className="py-0.5 max-h-8 overflow-hidden">
                    <Tooltip title={tool.tool_name}>
                      <span className="font-mono text-xs max-w-[20ch] truncate block font-medium text-blue-600 hover:underline">
                        {tool.tool_name}
                      </span>
                    </Tooltip>
                  </TableCell>
                  <TableCell className="py-0.5 max-h-8">
                    <PolicySelect
                      value={tool.call_policy}
                      toolName={tool.tool_name}
                      saving={saving === tool.tool_name}
                      onChange={handlePolicyChange}
                    />
                  </TableCell>
                  <TableCell className="py-0.5 max-h-8 text-right tabular-nums text-sm font-mono text-gray-700">
                    {(tool.call_count ?? 0).toLocaleString()}
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
                    <Tooltip title={tool.origin ?? "-"}>
                      <span className="max-w-[15ch] truncate block">{tool.origin ?? "-"}</span>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>

        {/* Bottom pagination (only when > 1 page) */}
        {totalPages > 1 && (
          <div className="border-t px-6 py-3 flex items-center justify-between text-sm text-gray-600">
            <span>Showing {(currentPage - 1) * pageSize + 1} - {Math.min(currentPage * pageSize, sorted.length)} of {sorted.length}</span>
            <div className="flex gap-1">
              <button onClick={() => setCurrentPage((p) => Math.max(1, p - 1))} disabled={currentPage === 1}
                className="px-3 py-1.5 border rounded-md hover:bg-gray-50 disabled:opacity-40">Previous</button>
              <button onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))} disabled={currentPage === totalPages}
                className="px-3 py-1.5 border rounded-md hover:bg-gray-50 disabled:opacity-40">Next</button>
            </div>
          </div>
        )}
      </div>

      {/* Settings drawer */}
      <Drawer
        title="Tool Log Settings"
        open={showSettings}
        onClose={() => setShowSettings(false)}
        width={360}
      >
        <div className="space-y-6">
          <div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-900">Store Tool Call Logs</p>
                <p className="text-xs text-gray-500 mt-0.5">
                  Record each tool invocation. When enabled, click any tool row to see its call history.
                </p>
              </div>
              <Switch
                color="green"
                checked={storeToolCallLogs}
                disabled={savingSettings}
                onChange={handleSettingsToggle}
              />
            </div>
          </div>
          {!storeToolCallLogs && (
            <div className="p-3 bg-yellow-50 border border-yellow-200 rounded text-xs text-yellow-800">
              Call log recording is off. New invocations won&apos;t be tracked until re-enabled.
            </div>
          )}
          <div className="text-xs text-gray-400">
            To see request/response args, also enable <strong>Store Prompts in Spend Logs</strong> on the Logs settings page.
          </div>
        </div>
      </Drawer>

      {/* Call log drawer */}
      <Drawer
        title={
          <div>
            <span className="font-mono text-sm">{selectedTool?.tool_name}</span>
            <span className="ml-2 text-xs text-gray-500 font-normal">call history</span>
          </div>
        }
        open={!!selectedTool}
        onClose={() => { setSelectedTool(null); setCallLogs([]); }}
        width={680}
      >
        {callLogsLoading ? (
          <div className="text-center text-gray-500 py-8">Loading…</div>
        ) : callLogs.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            <p>No call logs found.</p>
            {!storeToolCallLogs && (
              <p className="text-xs mt-2 text-yellow-700">
                Store Tool Call Logs is currently off — enable it in Settings to capture future calls.
              </p>
            )}
          </div>
        ) : (
          <>
            <div className="mb-3 text-xs text-gray-500">{callLogsTotal} total call{callLogsTotal !== 1 ? "s" : ""}</div>
            <div className="space-y-3">
              {callLogs.map((log) => {
                const parsedTool = buildParsedTool(log.tool_arguments, selectedTool?.tool_name ?? "");
                return (
                  <div key={log.id} className="border rounded-lg p-3 bg-gray-50 text-xs">
                    <div className="flex items-center justify-between mb-2">
                      <TimeCell utcTime={log.created_at ?? ""} />
                      <div className="flex gap-3 text-gray-400">
                        {log.request_id && (
                          <Tooltip title={log.request_id}>
                            <span className="font-mono truncate max-w-[18ch]">req: {log.request_id.slice(0, 8)}…</span>
                          </Tooltip>
                        )}
                        {log.key_hash && (
                          <Tooltip title={log.key_hash}>
                            <span className="font-mono truncate max-w-[12ch] text-blue-500">{log.key_hash.slice(0, 8)}…</span>
                          </Tooltip>
                        )}
                      </div>
                    </div>
                    {parsedTool ? (
                      <FormattedToolView tool={parsedTool} />
                    ) : (
                      <p className="text-gray-400 italic">No arguments captured for this call</p>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Pagination */}
            {callLogsTotalPages > 1 && (
              <div className="mt-4 flex items-center justify-between text-xs text-gray-500">
                <span>Page {callLogsPage} of {callLogsTotalPages}</span>
                <div className="flex gap-1">
                  <button
                    disabled={callLogsPage === 1}
                    onClick={() => {
                      const p = callLogsPage - 1;
                      setCallLogsPage(p);
                      if (selectedTool) loadCallLogs(selectedTool, p);
                    }}
                    className="px-2 py-1 border rounded hover:bg-gray-100 disabled:opacity-40"
                  >Previous</button>
                  <button
                    disabled={callLogsPage === callLogsTotalPages}
                    onClick={() => {
                      const p = callLogsPage + 1;
                      setCallLogsPage(p);
                      if (selectedTool) loadCallLogs(selectedTool, p);
                    }}
                    className="px-2 py-1 border rounded hover:bg-gray-100 disabled:opacity-40"
                  >Next</button>
                </div>
              </div>
            )}
          </>
        )}
      </Drawer>
    </div>
  );
};

export default ToolPolicies;
