"use client";

import React, { useState } from "react";
import {
  SearchIcon,
  PlusIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  XIcon,
  CheckIcon,
  ExternalLinkIcon,
  KeyIcon,
  ServerIcon,
  AlertCircleIcon,
  InfoIcon,
} from "lucide-react";

type GuardrailStatus = "active" | "pending" | "rejected";

type TeamGuardrail = {
  id: string;
  team: string;
  name: string;
  endpoint: string;
  status: GuardrailStatus;
  model: string;
  forwardKey: boolean;
  description: string;
  method: "POST" | "GET";
  customHeaders: {
    key: string;
    value: string;
  }[];
  submittedAt: string;
  submittedBy: string;
};

const SAMPLE_GUARDRAILS: TeamGuardrail[] = [
  {
    id: "1",
    team: "ML Platform",
    name: "Prompt Injection Detector",
    endpoint: "https://guardrails.ml-platform.internal/validate",
    status: "active",
    model: "gpt-4o-mini",
    forwardKey: true,
    description:
      "Detects prompt injection attacks and jailbreak attempts before they reach the model.",
    method: "POST",
    customHeaders: [
      { key: "X-Service-Name", value: "ml-platform-guardrail" },
      { key: "X-Environment", value: "production" },
    ],
    submittedAt: "2024-01-15",
    submittedBy: "alice@company.com",
  },
  {
    id: "2",
    team: "Data Science",
    name: "PII Redaction Guard",
    endpoint: "https://ds-guardrails.company.com/pii-check",
    status: "active",
    model: "claude-3-haiku",
    forwardKey: true,
    description:
      "Identifies and redacts personally identifiable information from prompts and responses.",
    method: "POST",
    customHeaders: [{ key: "X-Team", value: "data-science" }],
    submittedAt: "2024-01-18",
    submittedBy: "bob@company.com",
  },
  {
    id: "3",
    team: "Security",
    name: "SQL Injection Preventer",
    endpoint: "https://security-gd.internal/sql-guard",
    status: "pending",
    model: "gpt-4o",
    forwardKey: false,
    description:
      "Prevents SQL injection patterns from being passed through AI-generated queries.",
    method: "POST",
    customHeaders: [],
    submittedAt: "2024-02-01",
    submittedBy: "charlie@company.com",
  },
  {
    id: "4",
    team: "Customer Success",
    name: "Tone & Brand Compliance",
    endpoint: "https://cs-guardrails.company.com/brand",
    status: "pending",
    model: "gpt-4o-mini",
    forwardKey: true,
    description:
      "Ensures AI responses align with brand voice guidelines and customer-facing tone standards.",
    method: "POST",
    customHeaders: [
      { key: "X-Brand-Version", value: "v2.1" },
      { key: "X-Strictness", value: "high" },
    ],
    submittedAt: "2024-02-05",
    submittedBy: "diana@company.com",
  },
  {
    id: "5",
    team: "Legal",
    name: "Legal Disclaimer Enforcer",
    endpoint: "https://legal-gd.internal/compliance",
    status: "active",
    model: "gpt-4-turbo",
    forwardKey: false,
    description:
      "Ensures all AI-generated content includes required legal disclaimers and compliance notices.",
    method: "POST",
    customHeaders: [{ key: "X-Jurisdiction", value: "US" }],
    submittedAt: "2024-01-10",
    submittedBy: "eve@company.com",
  },
  {
    id: "6",
    team: "Finance",
    name: "Financial Advice Blocker",
    endpoint: "https://finance-gd.company.com/validate",
    status: "rejected",
    model: "gpt-4o-mini",
    forwardKey: true,
    description:
      "Blocks specific financial advice patterns not covered by the built-in LiteLLM filter.",
    method: "POST",
    customHeaders: [],
    submittedAt: "2024-01-28",
    submittedBy: "frank@company.com",
  },
];

const STATUS_CONFIG: Record<
  GuardrailStatus,
  { label: string; bg: string; text: string; dot: string }
> = {
  active: {
    label: "Active",
    bg: "bg-green-50",
    text: "text-green-700",
    dot: "bg-green-500",
  },
  pending: {
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

const TEAM_COLORS: Record<string, string> = {
  "ML Platform": "bg-purple-100 text-purple-700",
  "Data Science": "bg-blue-100 text-blue-700",
  Security: "bg-red-100 text-red-700",
  "Customer Success": "bg-orange-100 text-orange-700",
  Legal: "bg-gray-100 text-gray-700",
  Finance: "bg-green-100 text-green-700",
};

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

function Toggle({
  enabled,
  onToggle,
}: {
  enabled: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      role="switch"
      aria-checked={enabled}
      className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 ${
        enabled ? "bg-blue-500" : "bg-gray-200"
      }`}
    >
      <span
        className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform ${
          enabled ? "translate-x-4" : "translate-x-0.5"
        }`}
      />
    </button>
  );
}

type GuardrailCardProps = {
  guardrail: TeamGuardrail;
  isSelected: boolean;
  isHeadersExpanded: boolean;
  onSelect: () => void;
  onToggleForwardKey: () => void;
  onToggleHeaders: () => void;
  onApprove: () => void;
  onReject: () => void;
};

function GuardrailCard({
  guardrail: g,
  isSelected,
  isHeadersExpanded,
  onSelect,
  onToggleForwardKey,
  onToggleHeaders,
  onApprove,
  onReject,
}: GuardrailCardProps) {
  const status = STATUS_CONFIG[g.status];
  const teamColor = TEAM_COLORS[g.team] ?? "bg-gray-100 text-gray-700";
  return (
    <div
      className={`bg-white border rounded-lg p-4 transition-all ${
        isSelected ? "border-blue-400 ring-1 ring-blue-200" : "border-gray-200"
      }`}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span
              className={`text-xs font-medium px-2 py-0.5 rounded-full ${teamColor}`}
            >
              Team: {g.team}
            </span>
            <span
              className={`inline-flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 rounded-full ${status.bg} ${status.text}`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${status.dot}`} />
              {status.label}
            </span>
          </div>
          <h3 className="text-sm font-semibold text-gray-900 mb-1">{g.name}</h3>
          <p className="text-xs text-gray-500 mb-2 line-clamp-1">
            {g.description}
          </p>
          <div className="flex items-center gap-1.5 mb-2">
            <ServerIcon className="h-3.5 w-3.5 text-gray-400 flex-shrink-0" />
            <code className="text-xs text-gray-500 font-mono truncate">
              {g.endpoint}
            </code>
          </div>
          <div className="flex items-center gap-4 text-xs text-gray-500">
            <span>
              Model: <span className="font-medium text-gray-700">{g.model}</span>
            </span>
            <span>
              Submitted:{" "}
              <span className="font-medium text-gray-700">{g.submittedAt}</span>
            </span>
          </div>
        </div>
        <div className="flex flex-col items-end gap-2 flex-shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500 whitespace-nowrap">
              Forward API Key
            </span>
            <Toggle enabled={g.forwardKey} onToggle={onToggleForwardKey} />
          </div>
          <div className="flex items-center gap-2 mt-1">
            <button
              type="button"
              onClick={onSelect}
              className="text-xs border border-gray-300 text-gray-600 hover:bg-gray-50 px-3 py-1.5 rounded-md transition-colors font-medium"
            >
              {isSelected ? "Close" : "Review"}
            </button>
            {g.status === "pending" && (
              <>
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
              </>
            )}
          </div>
        </div>
      </div>
      <div className="mt-3 pt-3 border-t border-gray-100">
        <button
          type="button"
          onClick={onToggleHeaders}
          className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-700 transition-colors"
        >
          {isHeadersExpanded ? (
            <ChevronUpIcon className="h-3.5 w-3.5" />
          ) : (
            <ChevronDownIcon className="h-3.5 w-3.5" />
          )}
          Custom Headers
          {g.customHeaders.length > 0 && (
            <span className="ml-1 bg-gray-100 text-gray-600 rounded-full px-1.5 py-0.5 text-xs">
              {g.customHeaders.length}
            </span>
          )}
        </button>
        {isHeadersExpanded && (
          <div className="mt-2">
            {g.customHeaders.length === 0 ? (
              <p className="text-xs text-gray-400 italic">
                No custom headers configured.
              </p>
            ) : (
              <div className="space-y-1">
                {g.customHeaders.map((h, i) => (
                  <div
                    key={`${h.key}-${i}`}
                    className="flex items-center gap-2 text-xs font-mono"
                  >
                    <span className="text-gray-500 bg-gray-50 border border-gray-200 rounded px-2 py-0.5">
                      {h.key}
                    </span>
                    <span className="text-gray-400">:</span>
                    <span className="text-gray-700 bg-gray-50 border border-gray-200 rounded px-2 py-0.5">
                      {h.value}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function ConfigRow({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="text-xs font-semibold text-gray-500 mb-1">{label}</div>
      <div>{children}</div>
    </div>
  );
}

type DetailPanelProps = {
  guardrail: TeamGuardrail;
  onClose: () => void;
  onApprove: () => void;
  onReject: () => void;
  onToggleForwardKey: () => void;
};

function DetailPanel({
  guardrail: g,
  onClose,
  onApprove,
  onReject,
  onToggleForwardKey,
}: DetailPanelProps) {
  const status = STATUS_CONFIG[g.status];
  const teamColor = TEAM_COLORS[g.team] ?? "bg-gray-100 text-gray-700";
  return (
    <div className="w-96 flex-shrink-0 bg-white overflow-auto">
      <div className="p-5">
        <div className="flex items-start justify-between mb-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span
                className={`text-xs font-medium px-2 py-0.5 rounded-full ${teamColor}`}
              >
                Team: {g.team}
              </span>
              <span
                className={`inline-flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 rounded-full ${status.bg} ${status.text}`}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${status.dot}`} />
                {status.label}
              </span>
            </div>
            <h2 className="text-base font-semibold text-gray-900">{g.name}</h2>
            <p className="text-xs text-gray-500 mt-0.5">
              Submitted by {g.submittedBy} on {g.submittedAt}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
            aria-label="Close detail panel"
          >
            <XIcon className="h-4 w-4" />
          </button>
        </div>
        <p className="text-sm text-gray-600 mb-5">{g.description}</p>
        <div className="space-y-4">
          <ConfigRow label="Endpoint">
            <div className="flex items-center gap-1.5">
              <code className="text-xs font-mono text-gray-700 break-all">
                {g.endpoint}
              </code>
              <a
                href={g.endpoint}
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-blue-500 flex-shrink-0"
              >
                <ExternalLinkIcon className="h-3.5 w-3.5" />
              </a>
            </div>
          </ConfigRow>
          <ConfigRow label="Method">
            <span className="text-xs font-mono font-medium text-gray-700 bg-gray-100 px-2 py-0.5 rounded">
              {g.method}
            </span>
          </ConfigRow>
          <ConfigRow label="Validation Model">
            <span className="text-xs font-medium text-gray-700">{g.model}</span>
          </ConfigRow>
          <div className="border border-blue-100 bg-blue-50 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-1.5">
                <KeyIcon className="h-3.5 w-3.5 text-blue-500" />
                <span className="text-xs font-semibold text-blue-800">
                  Forward LiteLLM API Key
                </span>
              </div>
              <Toggle enabled={g.forwardKey} onToggle={onToggleForwardKey} />
            </div>
            <p className="text-xs text-blue-700 leading-relaxed">
              When enabled, the caller&apos;s LiteLLM API key is forwarded as an{" "}
              <code className="font-mono bg-blue-100 px-1 rounded">
                Authorization
              </code>{" "}
              header to your guardrail endpoint. This allows your guardrail to
              authenticate model calls using the original caller&apos;s
              credentials.
            </p>
          </div>
          <div>
            <div className="flex items-center gap-1.5 mb-2">
              <span className="text-xs font-semibold text-gray-700">
                Custom Headers
              </span>
              {g.customHeaders.length > 0 && (
                <span className="bg-gray-100 text-gray-600 rounded-full px-1.5 py-0.5 text-xs">
                  {g.customHeaders.length}
                </span>
              )}
            </div>
            {g.customHeaders.length === 0 ? (
              <p className="text-xs text-gray-400 italic">
                No custom headers configured.
              </p>
            ) : (
              <div className="border border-gray-200 rounded-md overflow-hidden">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200">
                      <th className="text-left px-3 py-2 text-gray-500 font-medium">
                        Key
                      </th>
                      <th className="text-left px-3 py-2 text-gray-500 font-medium">
                        Value
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {g.customHeaders.map((h, i) => (
                      <tr
                        key={`${h.key}-${i}`}
                        className={
                          i < g.customHeaders.length - 1
                            ? "border-b border-gray-100"
                            : ""
                        }
                      >
                        <td className="px-3 py-2 font-mono text-gray-700">
                          {h.key}
                        </td>
                        <td className="px-3 py-2 font-mono text-gray-600">
                          {h.value}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
          <div className="flex items-start gap-2 bg-gray-50 border border-gray-200 rounded-lg p-3">
            <InfoIcon className="h-3.5 w-3.5 text-gray-400 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-gray-500 leading-relaxed">
              This guardrail runs on a separate instance. It receives the user
              request, validates it using{" "}
              <span className="font-medium">{g.model}</span>, and forwards the
              result to the next step in the pipeline. See{" "}
              <a
                href="https://docs.litellm.ai/docs/adding_provider/generic_guardrail_api"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-500 hover:underline"
              >
                LiteLLM Generic Guardrail API docs
              </a>{" "}
              for configuration details.
            </p>
          </div>
        </div>
        <div className="mt-5 pt-4 border-t border-gray-100 space-y-2">
          <button
            type="button"
            className="w-full flex items-center justify-center gap-2 border border-gray-300 text-gray-700 hover:bg-gray-50 text-sm font-medium py-2 rounded-md transition-colors"
          >
            <ExternalLinkIcon className="h-4 w-4" />
            Test Endpoint
          </button>
          {g.status === "pending" && (
            <div className="flex gap-2">
              <button
                type="button"
                onClick={onApprove}
                className="flex-1 flex items-center justify-center gap-1.5 bg-green-500 hover:bg-green-600 text-white text-sm font-medium py-2 rounded-md transition-colors"
              >
                <CheckIcon className="h-4 w-4" />
                Approve
              </button>
              <button
                type="button"
                onClick={onReject}
                className="flex-1 flex items-center justify-center gap-1.5 border border-red-300 text-red-600 hover:bg-red-50 text-sm font-medium py-2 rounded-md transition-colors"
              >
                <XIcon className="h-4 w-4" />
                Reject
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

type ConfirmDialogProps = {
  action: "approve" | "reject";
  guardrailName: string;
  onConfirm: () => void;
  onCancel: () => void;
};

function ConfirmDialog({
  action,
  guardrailName,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
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
          {isApprove ? "Approve Guardrail" : "Reject Guardrail"}
        </h3>
        <p className="text-sm text-gray-500 mb-5">
          Are you sure you want to {action}{" "}
          <span className="font-medium text-gray-700">&quot;{guardrailName}&quot;</span>?{" "}
          {isApprove
            ? "This will make it active and available for use."
            : "This will mark it as rejected and notify the team."}
        </p>
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
            onClick={onConfirm}
            className={`flex-1 text-white text-sm font-medium py-2 rounded-md transition-colors ${
              isApprove
                ? "bg-green-500 hover:bg-green-600"
                : "bg-red-500 hover:bg-red-600"
            }`}
          >
            {isApprove ? "Approve" : "Reject"}
          </button>
        </div>
      </div>
    </div>
  );
}

export function TeamGuardrailsTab() {
  const [guardrails, setGuardrails] = useState<TeamGuardrail[]>(SAMPLE_GUARDRAILS);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<
    "all" | GuardrailStatus
  >("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [expandedHeaders, setExpandedHeaders] = useState<Set<string>>(new Set());
  const [confirmAction, setConfirmAction] = useState<{
    id: string;
    action: "approve" | "reject";
  } | null>(null);

  const filtered = guardrails.filter((g) => {
    const matchesSearch =
      g.name.toLowerCase().includes(search.toLowerCase()) ||
      g.team.toLowerCase().includes(search.toLowerCase()) ||
      g.endpoint.toLowerCase().includes(search.toLowerCase());
    const matchesStatus = statusFilter === "all" || g.status === statusFilter;
    return matchesSearch && matchesStatus;
  });
  const selected = guardrails.find((g) => g.id === selectedId) ?? null;
  const totalCount = guardrails.length;
  const pendingCount = guardrails.filter((g) => g.status === "pending").length;
  const activeCount = guardrails.filter((g) => g.status === "active").length;
  const rejectedCount = guardrails.filter((g) => g.status === "rejected").length;

  function toggleForwardKey(id: string) {
    setGuardrails((prev) =>
      prev.map((g) =>
        g.id === id ? { ...g, forwardKey: !g.forwardKey } : g
      )
    );
  }

  function handleApprove(id: string) {
    setGuardrails((prev) =>
      prev.map((g) => (g.id === id ? { ...g, status: "active" as const } : g))
    );
    setConfirmAction(null);
    if (selectedId === id) setSelectedId(null);
  }

  function handleReject(id: string) {
    setGuardrails((prev) =>
      prev.map((g) =>
        g.id === id ? { ...g, status: "rejected" as const } : g
      )
    );
    setConfirmAction(null);
    if (selectedId === id) setSelectedId(null);
  }

  function toggleHeaders(id: string) {
    setExpandedHeaders((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <div className="flex h-full">
      <div
        className={`flex-1 min-w-0 p-6 overflow-auto ${
          selected ? "border-r border-gray-200" : ""
        }`}
      >
        <div className="grid grid-cols-4 gap-4 mb-6">
          <StatCard label="Total Submitted" value={totalCount} color="text-gray-900" />
          <StatCard
            label="Pending Review"
            value={pendingCount}
            color="text-yellow-600"
          />
          <StatCard label="Active" value={activeCount} color="text-green-600" />
          <StatCard label="Rejected" value={rejectedCount} color="text-red-600" />
        </div>
        <div className="flex items-center gap-3 mb-5">
          <div className="relative flex-1 max-w-xs">
            <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search guardrails..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-9 pr-4 py-2 border border-gray-200 rounded-md text-sm text-gray-700 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          <select
            value={statusFilter}
            onChange={(e) =>
              setStatusFilter(e.target.value as typeof statusFilter)
            }
            className="border border-gray-200 rounded-md px-3 py-2 text-sm text-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 bg-white"
          >
            <option value="all">All Status</option>
            <option value="pending">Pending Review</option>
            <option value="active">Active</option>
            <option value="rejected">Rejected</option>
          </select>
          <button
            type="button"
            className="ml-auto flex items-center gap-2 bg-blue-500 hover:bg-blue-600 text-white text-sm font-medium px-4 py-2 rounded-md transition-colors"
          >
            <PlusIcon className="h-4 w-4" />
            Add Guardrail
          </button>
        </div>
        <div className="space-y-3">
          {filtered.length === 0 && (
            <div className="text-center py-12 text-gray-400 text-sm">
              No guardrails match your filters.
            </div>
          )}
          {filtered.map((g) => (
            <GuardrailCard
              key={g.id}
              guardrail={g}
              isSelected={selectedId === g.id}
              isHeadersExpanded={expandedHeaders.has(g.id)}
              onSelect={() => setSelectedId(selectedId === g.id ? null : g.id)}
              onToggleForwardKey={() => toggleForwardKey(g.id)}
              onToggleHeaders={() => toggleHeaders(g.id)}
              onApprove={() => setConfirmAction({ id: g.id, action: "approve" })}
              onReject={() => setConfirmAction({ id: g.id, action: "reject" })}
            />
          ))}
        </div>
      </div>
      {selected && (
        <DetailPanel
          guardrail={selected}
          onClose={() => setSelectedId(null)}
          onApprove={() =>
            setConfirmAction({ id: selected.id, action: "approve" })
          }
          onReject={() =>
            setConfirmAction({ id: selected.id, action: "reject" })
          }
          onToggleForwardKey={() => toggleForwardKey(selected.id)}
        />
      )}
      {confirmAction && (
        <ConfirmDialog
          action={confirmAction.action}
          guardrailName={
            guardrails.find((g) => g.id === confirmAction.id)?.name ?? ""
          }
          onConfirm={() =>
            confirmAction.action === "approve"
              ? handleApprove(confirmAction.id)
              : handleReject(confirmAction.id)
          }
          onCancel={() => setConfirmAction(null)}
        />
      )}
    </div>
  );
}
