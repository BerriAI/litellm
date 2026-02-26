"use client";

import { ArrowLeftOutlined, HistoryOutlined, ToolOutlined } from "@ant-design/icons";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Button, Select, Spin } from "antd";
import React, { useCallback, useMemo, useState } from "react";
import TeamDropdown from "@/components/common_components/team_dropdown";
import { LogViewer } from "@/components/GuardrailsMonitor/LogViewer";
import type { LogEntry } from "@/components/GuardrailsMonitor/mockData";
import { PolicySelect } from "@/components/ToolPolicies/PolicySelect";
import {
  deleteToolPolicyOverride,
  fetchToolDetail,
  getToolUsageLogs,
  keyListCall,
  teamListCall,
  updateToolPolicy,
  type ToolPolicyOverrideRow,
} from "@/components/networking";
import type { Team } from "@/components/key_team_helpers/key_list";

interface ToolDetailProps {
  toolName: string;
  onBack: () => void;
  accessToken: string | null;
}

interface TeamOption {
  team_id: string;
  team_alias?: string;
}

interface KeyOption {
  token: string;
  key_alias?: string;
}

const TOOL_DETAIL_QUERY_KEY = "tool-detail";

const LOGS_PAGE_SIZE = 50;

function getDefaultLogsDateRange(): { start: string; end: string } {
  const end = new Date();
  const start = new Date();
  start.setDate(start.getDate() - 90);
  const fmt = (d: Date) =>
    d.toISOString().slice(0, 19).replace("T", " ");
  return { start: fmt(start), end: fmt(end) };
}

export function ToolDetail({ toolName, onBack, accessToken }: ToolDetailProps) {
  const queryClient = useQueryClient();
  const [overrideSaving, setOverrideSaving] = useState(false);
  const [policySaving, setPolicySaving] = useState(false);
  const [blockScope, setBlockScope] = useState<"team" | "key">("team");
  const [blockTeamId, setBlockTeamId] = useState<string | null>(null);
  const [blockKey, setBlockKey] = useState<KeyOption | null>(null);

  const logsDateRange = useMemo(() => getDefaultLogsDateRange(), []);

  const { data: detail, isLoading: detailLoading, error: detailError } = useQuery({
    queryKey: [TOOL_DETAIL_QUERY_KEY, toolName],
    queryFn: () => fetchToolDetail(accessToken!, toolName),
    enabled: !!accessToken && !!toolName,
  });

  const { data: teamsData } = useQuery({
    queryKey: ["teams-list-tool-detail"],
    queryFn: () => teamListCall(accessToken!, null, null),
    enabled: !!accessToken,
  });

  const { data: keysData } = useQuery({
    queryKey: ["keys-list-tool-detail"],
    queryFn: () => keyListCall(accessToken!, null, null, null, null, null, 1, 100),
    enabled: !!accessToken,
  });

  const { data: logsData, isLoading: logsLoading } = useQuery({
    queryKey: ["tool-usage-logs", toolName, logsDateRange.start, logsDateRange.end],
    queryFn: () =>
      getToolUsageLogs(accessToken!, toolName, {
        page: 1,
        pageSize: LOGS_PAGE_SIZE,
        startDate: logsDateRange.start,
        endDate: logsDateRange.end,
      }),
    enabled: !!accessToken && !!toolName,
  });

  const logs: LogEntry[] = useMemo(() => {
    const list = logsData?.logs ?? [];
    return list.map((l) => ({
      id: l.id,
      timestamp: l.timestamp,
      action: "passed" as const,
      model: l.model ?? undefined,
      input_snippet: l.input_snippet ?? undefined,
    }));
  }, [logsData?.logs]);

  const teams: Team[] = useMemo(() => {
    const arr = Array.isArray(teamsData) ? teamsData : teamsData?.data ?? [];
    return arr.map((t: { team_id?: string; id?: string; team_alias?: string }) => ({
      team_id: t.team_id ?? t.id ?? "",
      team_alias: t.team_alias ?? t.team_id ?? "",
      models: [],
      max_budget: null,
      budget_duration: null,
      tpm_limit: null,
      rpm_limit: null,
      organization_id: "",
      created_at: "",
      keys: [],
      members_with_roles: [],
      spend: 0,
    }));
  }, [teamsData]);

  const keys: KeyOption[] = useMemo(() => {
    const keysRes = keysData?.keys ?? keysData?.data ?? [];
    return keysRes.map((k: { token?: string; api_key?: string; key_hash?: string; key_alias?: string }) => ({
      token: k.token ?? k.api_key ?? k.key_hash ?? "",
      key_alias: k.key_alias ?? (k.token ?? k.api_key ?? k.key_hash)?.toString?.()?.substring?.(0, 8),
    }));
  }, [keysData]);

  const invalidateDetail = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: [TOOL_DETAIL_QUERY_KEY, toolName] });
  }, [queryClient, toolName]);

  const handlePolicyChange = useCallback(
    async (name: string, newPolicy: string) => {
      if (!accessToken) return;
      setPolicySaving(true);
      try {
        await updateToolPolicy(accessToken, name, newPolicy);
        invalidateDetail();
      } catch (e: unknown) {
        alert(`Failed to update policy: ${e instanceof Error ? e.message : String(e)}`);
      } finally {
        setPolicySaving(false);
      }
    },
    [accessToken, invalidateDetail]
  );

  const handleAddOverride = useCallback(async () => {
    if (!accessToken || !toolName) return;
    const isTeam = blockScope === "team";
    if (isTeam && !blockTeamId) return;
    if (!isTeam && !blockKey?.token) return;
    setOverrideSaving(true);
    try {
      await updateToolPolicy(accessToken, toolName, "blocked", {
        team_id: isTeam ? blockTeamId : undefined,
        key_hash: !isTeam ? blockKey!.token : undefined,
        key_alias: !isTeam ? blockKey!.key_alias : undefined,
      });
      invalidateDetail();
      setBlockTeamId(null);
      setBlockKey(null);
    } catch (e: unknown) {
      alert(`Failed to add override: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setOverrideSaving(false);
    }
  }, [accessToken, toolName, blockScope, blockTeamId, blockKey, invalidateDetail]);

  const handleRemoveOverride = useCallback(
    async (override: ToolPolicyOverrideRow) => {
      if (!accessToken || !toolName) return;
      setOverrideSaving(true);
      try {
        await deleteToolPolicyOverride(accessToken, toolName, {
          team_id: override.team_id ?? undefined,
          key_hash: override.key_hash ?? undefined,
        });
        invalidateDetail();
      } catch (e: unknown) {
        alert(`Failed to remove override: ${e instanceof Error ? e.message : String(e)}`);
      } finally {
        setOverrideSaving(false);
      }
    },
    [accessToken, toolName, invalidateDetail]
  );

  if (detailLoading && !detail) {
    return (
      <div className="flex items-center justify-center py-12">
        <Spin size="large" />
      </div>
    );
  }

  if (detailError && !detail) {
    return (
      <div>
        <Button type="link" icon={<ArrowLeftOutlined />} onClick={onBack} className="pl-0 mb-4">
          Back to Tool Policies
        </Button>
        <p className="text-red-600">Failed to load tool details.</p>
      </div>
    );
  }

  if (!detail) {
    return null;
  }

  const { tool, overrides } = detail;

  return (
    <div>
      <div className="mb-6">
        <Button
          type="link"
          icon={<ArrowLeftOutlined />}
          onClick={onBack}
          className="pl-0 mb-4"
        >
          Back to Tool Policies
        </Button>

        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-3 mb-1 flex-wrap">
              <ToolOutlined className="text-xl text-gray-400" />
              <h1 className="text-xl font-semibold text-gray-900 font-mono">{tool.tool_name}</h1>
              <span className="inline-flex items-center px-2.5 py-1 text-xs font-medium rounded-md bg-gray-100 text-gray-700 border border-gray-200">
                {tool.origin ?? "—"}
              </span>
              <span className="inline-flex items-center px-2.5 py-1 text-xs font-medium rounded-md bg-indigo-50 text-indigo-700 border border-indigo-200">
                {(tool.call_count ?? 0).toLocaleString()} calls
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="space-y-6">
        <section className="bg-white rounded-lg border border-gray-200 p-5 shadow-sm">
          <h2 className="text-sm font-semibold text-gray-700 mb-3">Global policy</h2>
          <PolicySelect
            value={tool.call_policy}
            toolName={tool.tool_name}
            saving={policySaving}
            onChange={handlePolicyChange}
            size="middle"
            minWidth={140}
            stopPropagation={false}
          />
        </section>

        {overrides.length > 0 && (
          <section className="bg-white rounded-lg border border-gray-200 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-gray-700 mb-3">Blocked for team or key</h2>
            <ul className="border rounded-md divide-y divide-gray-100 bg-red-50/30">
              {overrides.map((ov) => (
                <li
                  key={ov.override_id}
                  className="flex items-center justify-between px-3 py-2.5 text-sm"
                >
                  <span className="text-gray-700">
                    {ov.team_id ? `Team: ${ov.team_id}` : ""}
                    {ov.team_id && ov.key_hash ? " · " : ""}
                    {ov.key_hash ? `Key: ${ov.key_alias || ov.key_hash.substring(0, 8)}` : ""}
                    {!ov.team_id && !ov.key_hash ? "—" : ""}
                  </span>
                  <Button
                    type="link"
                    danger
                    size="small"
                    disabled={overrideSaving}
                    onClick={() => handleRemoveOverride(ov)}
                  >
                    Remove
                  </Button>
                </li>
              ))}
            </ul>
          </section>
        )}

        <section className="bg-white rounded-lg border border-gray-200 p-5 shadow-sm">
          <h2 className="text-sm font-semibold text-gray-700 mb-3">Block for team or key</h2>
          <div className="flex flex-col gap-4 max-w-md">
            <div>
              <span className="text-sm font-medium text-gray-700 block mb-2">Scope</span>
              <div className="flex items-center gap-6">
                <label className="flex items-center gap-2 cursor-pointer text-sm text-gray-700">
                  <input
                    type="radio"
                    checked={blockScope === "team"}
                    onChange={() => setBlockScope("team")}
                    className="align-middle"
                  />
                  Team
                </label>
                <label className="flex items-center gap-2 cursor-pointer text-sm text-gray-700">
                  <input
                    type="radio"
                    checked={blockScope === "key"}
                    onChange={() => setBlockScope("key")}
                    className="align-middle"
                  />
                  Key
                </label>
              </div>
            </div>
            <div>
              <span className="text-sm font-medium text-gray-700 block mb-2">
                {blockScope === "team" ? "Team" : "Key"}
              </span>
              {blockScope === "team" ? (
                <TeamDropdown
                  teams={teams}
                  value={blockTeamId ?? undefined}
                  onChange={(id) => setBlockTeamId(id || null)}
                />
              ) : (
                <Select
                  placeholder="Select key"
                  allowClear
                  showSearch
                  optionFilterProp="label"
                  value={blockKey ? blockKey.token : undefined}
                  onChange={(token) => {
                    const k = keys.find((x) => x.token === token);
                    setBlockKey(k ?? null);
                  }}
                  options={keys.map((k) => ({
                    value: k.token,
                    label: k.key_alias || k.token?.substring?.(0, 12) || k.token,
                  }))}
                  className="w-full"
                  style={{ minWidth: 200 }}
                />
              )}
            </div>
            <Button
              type="primary"
              danger
              disabled={overrideSaving || (blockScope === "team" ? !blockTeamId : !blockKey?.token)}
              loading={overrideSaving}
              onClick={handleAddOverride}
            >
              Block for {blockScope}
            </Button>
          </div>
        </section>

        <section className="bg-white rounded-lg border border-gray-200 p-5 shadow-sm">
          <h2 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
            <HistoryOutlined />
            Recent logs
          </h2>
          <LogViewer
            guardrailName={tool.tool_name}
            filterAction="passed"
            logs={logs}
            logsLoading={logsLoading}
            totalLogs={logsData?.total ?? 0}
            accessToken={accessToken}
            startDate={logsDateRange.start}
            endDate={logsDateRange.end}
          />
        </section>
      </div>
    </div>
  );
}
