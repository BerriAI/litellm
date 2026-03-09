"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  SearchIcon,
  CheckIcon,
  XIcon,
  AlertCircleIcon,
  ServerIcon,
} from "lucide-react";
import {
  fetchMCPSubmissions,
  approveMCPServer,
  rejectMCPServer,
  getGeneralSettingsCall,
} from "@/components/networking";
import { MCPServer, MCPSubmissionsSummary } from "./types";
import { MCP_REQUIRED_FIELD_DEFS } from "./MCPStandardsSettings";
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
  onConfirm: (reviewNotes?: string) => void;
  onCancel: () => void;
};

function ConfirmDialog({ action, serverName, onConfirm, onCancel }: ConfirmDialogProps) {
  const [reviewNotes, setReviewNotes] = useState("");
  const isApprove = action === "approve";
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
            : "This will mark it as rejected."}
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
    passed: f.check(server),
  }));
  const passCount = checks.filter((c) => c.passed).length;
  const allPassed = checks.length > 0 && passCount === checks.length;

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5">
            <span
              className={`inline-flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 rounded-full ${statusCfg.bg} ${statusCfg.text}`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${statusCfg.dot}`} />
              {statusCfg.label}
            </span>
            {checks.length > 0 && (
              <span
                className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                  allPassed ? "bg-green-50 text-green-700" : "bg-amber-50 text-amber-700"
                }`}
              >
                {passCount}/{checks.length} checks
              </span>
            )}
          </div>
          <h3 className="text-sm font-semibold text-gray-900 mb-1">
            {server.alias ?? server.server_name ?? server.server_id}
          </h3>
          {server.description && (
            <p className="text-xs text-gray-500 mb-2 line-clamp-1">{server.description}</p>
          )}
          {server.url && (
            <div className="flex items-center gap-1.5 mb-2">
              <ServerIcon className="h-3.5 w-3.5 text-gray-400 flex-shrink-0" />
              <code className="text-xs text-gray-500 font-mono truncate">{server.url}</code>
            </div>
          )}
          <div className="flex items-center gap-4 text-xs text-gray-500">
            <span>
              Transport:{" "}
              <span className="font-medium text-gray-700">{server.transport ?? "sse"}</span>
            </span>
            <span>
              Submitted by:{" "}
              <span className="font-medium text-gray-700">{server.submitted_by ?? "—"}</span>
            </span>
            <span>
              Date:{" "}
              <span className="font-medium text-gray-700">{formatDate(server.submitted_at)}</span>
            </span>
          </div>
          {approvalStatus === "rejected" && server.review_notes && (
            <p className="text-xs text-red-600 mt-1">
              Rejection reason: {server.review_notes}
            </p>
          )}
          {checks.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-2 pt-2 border-t border-gray-100">
              {checks.map((c) => (
                <span
                  key={c.key}
                  className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full ${
                    c.passed ? "bg-green-50 text-green-700" : "bg-red-50 text-red-700"
                  }`}
                >
                  {c.passed ? (
                    <CheckIcon className="h-3 w-3" />
                  ) : (
                    <XIcon className="h-3 w-3" />
                  )}
                  {c.label}
                </span>
              ))}
            </div>
          )}
        </div>
        {approvalStatus === "pending_review" && (
          <div className="flex items-center gap-2 flex-shrink-0">
            <button
              type="button"
              onClick={onApprove}
              className="text-xs bg-green-500 hover:bg-green-600 text-white px-3 py-1.5 rounded-md transition-colors font-medium"
            >
              Approve
            </button>
            <button
              type="button"
              onClick={onReject}
              className="text-xs border border-red-300 text-red-600 hover:bg-red-50 px-3 py-1.5 rounded-md transition-colors font-medium"
            >
              Reject
            </button>
          </div>
        )}
      </div>
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
  } | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [requiredFields, setRequiredFields] = useState<string[]>([]);

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
        getGeneralSettingsCall(accessToken).catch(() => null),
      ]);
      setSummary(res);
      if (settings?.data && Array.isArray(settings.data)) {
        const row = settings.data.find(
          (r: { field_name: string; field_value: unknown }) => r.field_name === "mcp_required_fields",
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
                })
              }
            />
          ))}
      </div>

      {confirmAction && (
        <ConfirmDialog
          action={confirmAction.action}
          serverName={confirmAction.serverName}
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
