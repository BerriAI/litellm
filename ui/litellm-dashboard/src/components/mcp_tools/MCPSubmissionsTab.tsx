"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  SearchIcon,
  CheckIcon,
  XIcon,
  AlertCircleIcon,
  ServerIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  SettingsIcon,
} from "lucide-react";
import {
  fetchMCPSubmissions,
  approveMCPServer,
  rejectMCPServer,
  getGeneralSettingsCall,
  updateConfigFieldSetting,
} from "@/components/networking";
import { MCPServer, MCPSubmissionsSummary } from "./types";
import { FIELD_GROUPS, MCP_REQUIRED_FIELD_DEFS, SETTINGS_KEY } from "./MCPStandardsSettings";
import NotificationsManager from "@/components/molecules/notifications_manager";

type MCPStatus = "active" | "pending_review" | "rejected";

const STATUS_CONFIG: Record<
  MCPStatus,
  { label: string; bg: string; text: string; dot: string }
> = {
  active: {
    label: "Active",
    bg: "bg-green-50",
    text: "text-green-700",
    dot: "bg-green-500",
  },
  pending_review: {
    label: "Pending Review",
    bg: "bg-yellow-50",
    text: "text-yellow-700",
    dot: "bg-yellow-500",
  },
  rejected: {
    label: "Rejected",
    bg: "bg-red-50",
    text: "text-red-700",
    dot: "bg-red-500",
  },
};

function formatDate(value: string | null | undefined): string {
  if (!value) return "—";
  try {
    const d = new Date(value);
    return isNaN(d.getTime()) ? value : d.toISOString().slice(0, 10);
  } catch {
    return value;
  }
}

function StatCard({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg px-4 py-3">
      <div className={`text-2xl font-bold ${color}`}>{value}</div>
      <div className="text-xs text-gray-500 mt-0.5">{label}</div>
    </div>
  );
}

type ConfirmDialogProps = {
  action: "approve" | "reject";
  serverName: string;
  isCurrentlyActive?: boolean;
  onConfirm: (reviewNotes?: string) => void;
  onCancel: () => void;
};

function ConfirmDialog({ action, serverName, isCurrentlyActive, onConfirm, onCancel }: ConfirmDialogProps) {
  const [reviewNotes, setReviewNotes] = useState("");
  const isApprove = action === "approve";
  const rejectBody = isCurrentlyActive
    ? "This server is currently live. Rejecting it will immediately remove it from the proxy runtime."
    : "This will mark the submission as rejected.";
  return (
    <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl p-6 max-w-sm w-full mx-4">
        <div
          className={`w-10 h-10 rounded-full flex items-center justify-center mb-4 ${
            isApprove ? "bg-green-100" : "bg-red-100"
          }`}
        >
          {isApprove ? (
            <CheckIcon className="h-5 w-5 text-green-600" />
          ) : (
            <AlertCircleIcon className="h-5 w-5 text-red-600" />
          )}
        </div>
        <h3 className="text-base font-semibold text-gray-900 mb-1">
          {isApprove ? "Approve MCP Server" : "Reject MCP Server"}
        </h3>
        <p className="text-sm text-gray-500 mb-4">
          Are you sure you want to {action}{" "}
          <span className="font-medium text-gray-700">&quot;{serverName}&quot;</span>?{" "}
          {isApprove
            ? "This will make it active and available for use."
            : rejectBody}
        </p>
        {!isApprove && (
          <textarea
            placeholder="Reason for rejection (optional)"
            value={reviewNotes}
            onChange={(e) => setReviewNotes(e.target.value)}
            className="w-full border border-gray-200 rounded-md px-3 py-2 text-sm text-gray-700 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-blue-500 mb-4 resize-none"
            rows={3}
          />
        )}
        <div className="flex gap-3">
          <button
            type="button"
            onClick={onCancel}
            className="flex-1 border border-gray-300 text-gray-700 hover:bg-gray-50 text-sm font-medium py-2 rounded-md transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => onConfirm(isApprove ? undefined : reviewNotes || undefined)}
            className={`flex-1 text-white text-sm font-medium py-2 rounded-md transition-colors ${
              isApprove ? "bg-green-500 hover:bg-green-600" : "bg-red-500 hover:bg-red-600"
            }`}
          >
            {isApprove ? "Approve" : "Reject"}
          </button>
        </div>
      </div>
    </div>
  );
}

type SubmissionRulesPanelProps = {
  requiredFields: string[];
  onChange: (fields: string[]) => void;
  onSave: () => Promise<void>;
  isSaving: boolean;
};

function SubmissionRulesPanel({ requiredFields, onChange, onSave, isSaving }: SubmissionRulesPanelProps) {
  const [expanded, setExpanded] = useState(false);
  const activeLabels = MCP_REQUIRED_FIELD_DEFS.filter((f) => requiredFields.includes(f.key));

  const toggle = (key: string) => {
    onChange(requiredFields.includes(key) ? requiredFields.filter((k) => k !== key) : [...requiredFields, key]);
  };

  return (
    <div className="mb-5 border border-gray-200 rounded-lg bg-white overflow-hidden">
      {/* Header — always visible */}
      <div
        className="flex items-center justify-between px-4 py-3 cursor-pointer select-none"
        onClick={() => setExpanded((v) => !v)}
      >
        <div className="flex items-center gap-2">
          <SettingsIcon className="h-4 w-4 text-gray-400" />
          <span className="text-sm font-semibold text-gray-800">Submission Rules</span>
          {activeLabels.length > 0 ? (
            <span className="text-xs text-gray-500">
              ({activeLabels.length} required field{activeLabels.length !== 1 ? "s" : ""})
            </span>
          ) : (
            <span className="text-xs text-gray-400 italic">no rules set</span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {/* Active rule chips — collapsed view */}
          {!expanded && activeLabels.length > 0 && (
            <div className="flex flex-wrap gap-1.5 max-w-md">
              {activeLabels.map((f) => (
                <span
                  key={f.key}
                  className="inline-flex items-center gap-1 text-xs bg-blue-50 text-blue-700 border border-blue-200 px-2 py-0.5 rounded-full"
                >
                  <CheckIcon className="h-3 w-3" />
                  {f.label}
                </span>
              ))}
            </div>
          )}
          {expanded ? (
            <ChevronUpIcon className="h-4 w-4 text-gray-400" />
          ) : (
            <ChevronDownIcon className="h-4 w-4 text-gray-400" />
          )}
        </div>
      </div>

      {/* Expanded editor */}
      {expanded && (
        <div className="border-t border-gray-100 px-4 pt-4 pb-4">
          <p className="text-xs text-gray-500 mb-4">
            Select which fields must be filled in before a submission is considered compliant.
            LiteLLM will show ✓ / ✗ for each rule on every submission card below.
          </p>
          <div className="grid grid-cols-2 gap-x-8 gap-y-5">
            {FIELD_GROUPS.map((group) => (
              <div key={group.label}>
                <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                  {group.label}
                </div>
                <div className="space-y-2">
                  {group.fields.map((field) => {
                    const active = requiredFields.includes(field.key);
                    return (
                      <label
                        key={field.key}
                        className="flex items-start gap-2.5 cursor-pointer group"
                      >
                        <input
                          type="checkbox"
                          checked={active}
                          onChange={() => toggle(field.key)}
                          className="mt-0.5 h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
                        />
                        <div>
                          <div className="text-sm font-medium text-gray-800 group-hover:text-blue-700 transition-colors">
                            {field.label}
                          </div>
                          <div className="text-xs text-gray-400">{field.description}</div>
                        </div>
                      </label>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
          <div className="mt-5 flex items-center gap-3">
            <button
              type="button"
              disabled={isSaving}
              onClick={async () => {
                await onSave();
                setExpanded(false);
              }}
              className="px-4 py-1.5 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded-md transition-colors"
            >
              {isSaving ? "Saving…" : "Save Rules"}
            </button>
            <button
              type="button"
              onClick={() => setExpanded(false)}
              className="px-4 py-1.5 text-sm font-medium text-gray-600 hover:text-gray-900 border border-gray-200 rounded-md hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

type MCPServerCardProps = {
  server: MCPServer;
  onApprove: () => void;
  onReject: () => void;
  requiredFields: string[];
};

function MCPServerCard({ server, onApprove, onReject, requiredFields }: MCPServerCardProps) {
  const approvalStatus = (server.approval_status ?? "active") as MCPStatus;
  const statusCfg = STATUS_CONFIG[approvalStatus] ?? STATUS_CONFIG["active"];

  const checks = MCP_REQUIRED_FIELD_DEFS.filter((f) => requiredFields.includes(f.key)).map((f) => ({
    key: f.key,
    label: f.label,
    description: f.description,
    passed: f.check(server),
  }));
  const passCount = checks.filter((c) => c.passed).length;
  const failCount = checks.length - passCount;
  const allPassed = checks.length > 0 && failCount === 0;

  return (
    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
      {/* Server info */}
      <div className="px-4 pt-4 pb-3">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1.5">
              <span
                className={`inline-flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 rounded-full ${statusCfg.bg} ${statusCfg.text}`}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${statusCfg.dot}`} />
                {statusCfg.label}
              </span>
            </div>
            <h3 className="text-sm font-semibold text-gray-900">
              {server.alias ?? server.server_name ?? server.server_id}
            </h3>
            {server.description && (
              <p className="text-xs text-gray-500 mt-0.5 line-clamp-1">{server.description}</p>
            )}
            {server.url && (
              <div className="flex items-center gap-1.5 mt-1.5">
                <ServerIcon className="h-3.5 w-3.5 text-gray-400 flex-shrink-0" />
                <code className="text-xs text-gray-500 font-mono truncate">{server.url}</code>
              </div>
            )}
            <div className="flex items-center gap-3 mt-1.5 text-xs text-gray-400">
              <span>Transport: <span className="text-gray-600">{server.transport ?? "sse"}</span></span>
              <span>·</span>
              <span>Submitted by: <span className="text-gray-600">{server.submitted_by ?? "—"}</span></span>
              <span>·</span>
              <span>{formatDate(server.submitted_at)}</span>
            </div>
            {approvalStatus === "rejected" && server.review_notes && (
              <p className="text-xs text-red-600 mt-1.5">Rejection reason: {server.review_notes}</p>
            )}
          </div>
          {/* Approve/Reject when no checks panel (no rules configured) */}
          {checks.length === 0 && approvalStatus !== "rejected" && (
            <div className="flex items-center gap-2 flex-shrink-0">
              {approvalStatus !== "active" && (
                <button
                  type="button"
                  onClick={onApprove}
                  className="text-xs bg-green-500 hover:bg-green-600 text-white px-3 py-1.5 rounded-md transition-colors font-medium"
                >
                  Approve
                </button>
              )}
              <button
                type="button"
                onClick={onReject}
                className="text-xs border border-red-300 text-red-600 hover:bg-red-50 px-3 py-1.5 rounded-md transition-colors font-medium"
              >
                Reject
              </button>
            </div>
          )}
          {checks.length === 0 && approvalStatus === "rejected" && (
            <div className="flex items-center gap-2 flex-shrink-0">
              <button
                type="button"
                onClick={onApprove}
                className="text-xs bg-green-500 hover:bg-green-600 text-white px-3 py-1.5 rounded-md transition-colors font-medium"
              >
                Re-approve
              </button>
            </div>
          )}
        </div>
      </div>

      {/* GitHub-style checks panel */}
      {checks.length > 0 && (
        <div className="border-t border-gray-200">
          {/* Overall status header */}
          <div
            className={`flex items-center gap-3 px-4 py-3 ${
              allPassed ? "bg-green-50 border-b border-green-100" : "bg-red-50 border-b border-red-100"
            }`}
          >
            {/* Large status circle */}
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                allPassed ? "bg-green-500" : "bg-red-500"
              }`}
            >
              {allPassed ? (
                <CheckIcon className="h-4 w-4 text-white" />
              ) : (
                <XIcon className="h-4 w-4 text-white" />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <div className={`text-sm font-semibold leading-tight ${allPassed ? "text-green-800" : "text-red-800"}`}>
                {allPassed
                  ? "All checks passed"
                  : `${failCount} check${failCount !== 1 ? "s" : ""} failed`}
              </div>
              <div className="text-xs text-gray-500 mt-0.5">
                {passCount} passing, {failCount} failing
              </div>
            </div>
            {/* Approve / Reject in header */}
            <div className="flex items-center gap-2 flex-shrink-0">
              {approvalStatus !== "active" && approvalStatus !== "rejected" && (
                <button
                  type="button"
                  onClick={onApprove}
                  className="text-xs bg-green-600 hover:bg-green-700 text-white px-3 py-1.5 rounded-md transition-colors font-medium"
                >
                  Approve
                </button>
              )}
              {approvalStatus === "rejected" && (
                <button
                  type="button"
                  onClick={onApprove}
                  className="text-xs bg-green-600 hover:bg-green-700 text-white px-3 py-1.5 rounded-md transition-colors font-medium"
                >
                  Re-approve
                </button>
              )}
              {approvalStatus !== "rejected" && (
                <button
                  type="button"
                  onClick={onReject}
                  className="text-xs border border-red-300 text-red-600 hover:bg-red-50 bg-white px-3 py-1.5 rounded-md transition-colors font-medium"
                >
                  Reject
                </button>
              )}
            </div>
          </div>

          {/* Individual check rows */}
          <div className="divide-y divide-gray-100">
            {checks.map((c) => (
              <div key={c.key} className="flex items-center gap-3 px-4 py-2.5">
                {/* Small circle icon */}
                <div
                  className={`w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 ${
                    c.passed ? "bg-green-100" : "bg-red-100"
                  }`}
                >
                  {c.passed ? (
                    <CheckIcon className="h-3 w-3 text-green-600" />
                  ) : (
                    <XIcon className="h-3 w-3 text-red-600" />
                  )}
                </div>
                <span className={`text-sm flex-1 ${c.passed ? "text-gray-700" : "text-gray-800"}`}>
                  {c.label}
                </span>
                <span className={`text-xs ${c.passed ? "text-green-600" : "text-red-500"}`}>
                  {c.passed ? "Passes" : "Missing"}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

interface MCPSubmissionsTabProps {
  accessToken: string | null;
}

export function MCPSubmissionsTab({ accessToken }: MCPSubmissionsTabProps) {
  const [summary, setSummary] = useState<MCPSubmissionsSummary>({
    total: 0,
    pending_review: 0,
    active: 0,
    rejected: 0,
    items: [],
  });
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | MCPStatus>("all");
  const [confirmAction, setConfirmAction] = useState<{
    serverId: string;
    serverName: string;
    action: "approve" | "reject";
    isCurrentlyActive?: boolean;
  } | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [requiredFields, setRequiredFields] = useState<string[]>([]);
  const [isSavingRules, setIsSavingRules] = useState(false);

  const fetchData = useCallback(async () => {
    if (!accessToken) {
      setIsLoading(false);
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const [res, settings] = await Promise.all([
        fetchMCPSubmissions(accessToken),
        getGeneralSettingsCall(accessToken).catch((err) => {
          console.warn("MCPSubmissionsTab: failed to load general settings, compliance rules will be empty:", err);
          return null;
        }),
      ]);
      setSummary(res);
      if (settings?.data && Array.isArray(settings.data)) {
        const row = settings.data.find(
          (r: { field_name: string; field_value: unknown }) => r.field_name === SETTINGS_KEY,
        );
        if (row && Array.isArray(row.field_value)) {
          setRequiredFields(row.field_value as string[]);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load submissions");
    } finally {
      setIsLoading(false);
    }
  }, [accessToken]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleSaveRules = async () => {
    if (!accessToken) return;
    setIsSavingRules(true);
    try {
      await updateConfigFieldSetting(accessToken, SETTINGS_KEY, requiredFields);
      NotificationsManager.success("Submission rules saved");
    } catch {
      NotificationsManager.fromBackend("Failed to save submission rules");
    } finally {
      setIsSavingRules(false);
    }
  };

  const filtered = summary.items.filter((s) => {
    if (statusFilter !== "all" && s.approval_status !== statusFilter) return false;
    if (search.trim()) {
      const q = search.toLowerCase();
      const name = (s.alias ?? s.server_name ?? s.server_id ?? "").toLowerCase();
      const url = (s.url ?? "").toLowerCase();
      return name.includes(q) || url.includes(q);
    }
    return true;
  });

  async function handleApprove(serverId: string, serverName: string) {
    if (!accessToken) return;
    try {
      await approveMCPServer(accessToken, serverId);
      setConfirmAction(null);
      await fetchData();
      NotificationsManager.success(`MCP server "${serverName}" approved`);
    } catch {
      NotificationsManager.fromBackend("Failed to approve MCP server");
    }
  }

  async function handleReject(serverId: string, serverName: string, reviewNotes?: string) {
    if (!accessToken) return;
    try {
      await rejectMCPServer(accessToken, serverId, reviewNotes);
      setConfirmAction(null);
      await fetchData();
      NotificationsManager.success(`MCP server "${serverName}" rejected`);
    } catch {
      NotificationsManager.fromBackend("Failed to reject MCP server");
    }
  }

  return (
    <div className="p-6">
      {/* Submission Rules panel */}
      <SubmissionRulesPanel
        requiredFields={requiredFields}
        onChange={setRequiredFields}
        onSave={handleSaveRules}
        isSaving={isSavingRules}
      />

      <div className="grid grid-cols-4 gap-4 mb-6">
        <StatCard label="Total Submitted" value={summary.total} color="text-gray-900" />
        <StatCard label="Pending Review" value={summary.pending_review} color="text-yellow-600" />
        <StatCard label="Active" value={summary.active} color="text-green-600" />
        <StatCard label="Rejected" value={summary.rejected} color="text-red-600" />
      </div>

      <div className="flex items-center gap-3 mb-5">
        <div className="relative flex-1 max-w-xs">
          <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search MCP servers..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-4 py-2 border border-gray-200 rounded-md text-sm text-gray-700 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value as typeof statusFilter)}
          className="border border-gray-200 rounded-md px-3 py-2 text-sm text-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 bg-white"
        >
          <option value="all">All Status</option>
          <option value="pending_review">Pending Review</option>
          <option value="active">Active</option>
          <option value="rejected">Rejected</option>
        </select>
      </div>

      <div className="space-y-3">
        {isLoading && (
          <div className="text-center py-12 text-gray-500 text-sm">Loading submissions…</div>
        )}
        {error && (
          <div className="text-center py-12 text-red-600 text-sm">{error}</div>
        )}
        {!isLoading && !error && filtered.length === 0 && (
          <div className="text-center py-12 text-gray-400 text-sm">
            No MCP server submissions match your filters.
          </div>
        )}
        {!isLoading &&
          !error &&
          filtered.map((server) => (
            <MCPServerCard
              key={server.server_id}
              server={server}
              requiredFields={requiredFields}
              onApprove={() =>
                setConfirmAction({
                  serverId: server.server_id,
                  serverName: server.alias ?? server.server_name ?? server.server_id,
                  action: "approve",
                })
              }
              onReject={() =>
                setConfirmAction({
                  serverId: server.server_id,
                  serverName: server.alias ?? server.server_name ?? server.server_id,
                  action: "reject",
                  isCurrentlyActive: server.approval_status === "active",
                })
              }
            />
          ))}
      </div>

      {confirmAction && (
        <ConfirmDialog
          action={confirmAction.action}
          serverName={confirmAction.serverName}
          isCurrentlyActive={confirmAction.isCurrentlyActive}
          onConfirm={(reviewNotes) =>
            confirmAction.action === "approve"
              ? handleApprove(confirmAction.serverId, confirmAction.serverName)
              : handleReject(confirmAction.serverId, confirmAction.serverName, reviewNotes)
          }
          onCancel={() => setConfirmAction(null)}
        />
      )}
    </div>
  );
}
