"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
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
  CircleHelp,
} from "lucide-react";
import { z } from "zod/v4";
import {
  listGuardrailSubmissions,
  approveGuardrailSubmission,
  rejectGuardrailSubmission,
  updateGuardrailCall,
  type GuardrailSubmissionItem,
} from "@/components/networking";
import { toast } from "@/lib/toast";
import TeamDropdown from "@/components/common_components/team_dropdown";
import { useRegisterGuardrail } from "@/app/(dashboard)/hooks/guardrails/useRegisterGuardrail";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { isValidUrl } from "@/lib/forms/urlValidation";
import { useZodForm } from "@/lib/forms/useZodForm";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

const GUARDRAIL_MODES = [
  { value: "pre_call", label: "Pre Call" },
  { value: "post_call", label: "Post Call" },
  { value: "during_call", label: "During Call" },
] as const;

const submitGuardrailSchema = z.object({
  team_id: z.string().min(1, "Select a team"),
  guardrail_name: z.string().min(1, "Enter a guardrail name"),
  mode: z.string().min(1, "Select a mode"),
  api_base: z.string().min(1, "Enter the API base URL").refine(isValidUrl, "Must be a valid URL"),
  extra_litellm_params: z.string().superRefine((value, ctx) => {
    if (!value) return;
    try {
      const parsed: unknown = JSON.parse(value);
      if (typeof parsed !== "object" || Array.isArray(parsed)) {
        ctx.addIssue({ code: "custom", message: "Must be a JSON object" });
      }
    } catch {
      ctx.addIssue({ code: "custom", message: "Invalid JSON" });
    }
  }),
  guardrail_info: z.string().superRefine((value, ctx) => {
    if (!value) return;
    try {
      JSON.parse(value);
    } catch {
      ctx.addIssue({ code: "custom", message: "Invalid JSON" });
    }
  }),
});

type SubmitGuardrailValues = z.output<typeof submitGuardrailSchema>;

const EMPTY_SUBMIT_VALUES: SubmitGuardrailValues = {
  team_id: "",
  guardrail_name: "",
  mode: "pre_call",
  api_base: "",
  extra_litellm_params: "",
  guardrail_info: "",
};

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

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
  extraHeaders: string[];
  submittedAt: string;
  submittedBy: string;
  mode?: string;
  unreachable_fallback?: string;
  additionalProviderParams?: Record<string, unknown>;
  guardrailType?: string;
};

function mapStatus(apiStatus: string): GuardrailStatus {
  if (apiStatus === "pending_review") return "pending";
  if (apiStatus === "active" || apiStatus === "rejected") return apiStatus;
  return "active";
}

function formatSubmissionDate(value: string | null | undefined): string {
  if (!value) return "—";
  try {
    const d = new Date(value);
    return isNaN(d.getTime()) ? value : d.toISOString().slice(0, 10);
  } catch {
    return value;
  }
}

function submissionToTeamGuardrail(item: GuardrailSubmissionItem): TeamGuardrail {
  const params = item.litellm_params ?? {};
  const info = item.guardrail_info ?? {};
  const headers = params.headers;
  const customHeaders: { key: string; value: string }[] = Array.isArray(headers)
    ? headers.map((h: { key?: string; name?: string; value: string }) => ({
        key: (h.key ?? h.name ?? "").toString(),
        value: String(h.value ?? ""),
      }))
    : typeof headers === "object" && headers !== null
      ? Object.entries(headers).map(([key, value]) => ({
          key,
          value: String(value ?? ""),
        }))
      : [];
  const endpoint = (params.api_base as string) ?? (params.url as string) ?? "";
  const model = (info.model as string) ?? (params.model as string) ?? "—";
  const forwardKey = (params.forward_api_key as boolean) ?? true;
  const extraHeaders = Array.isArray(params.extra_headers)
    ? (params.extra_headers as string[]).filter((h): h is string => typeof h === "string")
    : [];
  return {
    id: item.guardrail_id,
    team: item.team_id ?? "—",
    name: item.guardrail_name,
    endpoint,
    status: mapStatus(item.status),
    model,
    forwardKey,
    description: (info.description as string) ?? "",
    method: (params.method as "POST" | "GET") ?? "POST",
    customHeaders,
    extraHeaders,
    submittedAt: formatSubmissionDate(item.submitted_at),
    submittedBy: item.submitted_by_email ?? item.submitted_by_user_id ?? "—",
    mode: params.mode as string | undefined,
    unreachable_fallback: params.unreachable_fallback as string | undefined,
    additionalProviderParams: params.additional_provider_specific_params as Record<string, unknown> | undefined,
    guardrailType: params.guardrail as string | undefined,
  };
}

const STATUS_CONFIG: Record<GuardrailStatus, { label: string; bg: string; text: string; dot: string }> = {
  active: {
    label: "Active",
    bg: "bg-success/10",
    text: "text-success",
    dot: "bg-success",
  },
  pending: {
    label: "Pending Review",
    bg: "bg-warning/10",
    text: "text-warning",
    dot: "bg-warning",
  },
  rejected: {
    label: "Rejected",
    bg: "bg-destructive/10",
    text: "text-destructive",
    dot: "bg-destructive",
  },
};

const TEAM_COLORS: Record<string, string> = {
  "ML Platform": "bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300",
  "Data Science": "bg-info/15 text-info",
  Security: "bg-destructive/15 text-destructive",
  "Customer Success": "bg-warning/15 text-warning",
  Legal: "bg-muted text-foreground",
  Finance: "bg-success/15 text-success",
};

function buildEquivalentConfigYaml(g: TeamGuardrail): string {
  const lines: string[] = [
    "litellm_settings:",
    "  guardrails:",
    `    - guardrail_name: "${g.name.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`,
    "      litellm_params:",
    `        guardrail: ${g.guardrailType ?? "generic_guardrail_api"}`,
    `        mode: ${g.mode ?? "pre_call"}  # or post_call, during_call`,
    `        api_base: ${g.endpoint || "https://your-guardrail-api.com"}`,
    "        api_key: os.environ/YOUR_GUARDRAIL_API_KEY  # optional",
    `        unreachable_fallback: ${g.unreachable_fallback ?? "fail_closed"}  # default: fail_closed. Set to fail_open to proceed if the guardrail endpoint is unreachable.`,
    `        forward_api_key: ${g.forwardKey}`,
  ];
  if (g.model && g.model !== "—") {
    lines.push(`        model: "${g.model}"  # LLM model name sent to the guardrail for context`);
  }
  if (g.customHeaders.length > 0) {
    lines.push("        headers:  # static headers (sent with every request)");
    for (const h of g.customHeaders) {
      lines.push(`          ${h.key}: "${String(h.value).replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`);
    }
  }
  if (g.extraHeaders.length > 0) {
    lines.push("        extra_headers:  # forward these client request headers to the guardrail");
    for (const name of g.extraHeaders) {
      lines.push(`          - ${name}`);
    }
  }
  if (g.additionalProviderParams && Object.keys(g.additionalProviderParams).length > 0) {
    lines.push("        additional_provider_specific_params:");
    for (const [k, v] of Object.entries(g.additionalProviderParams)) {
      const val = typeof v === "string" ? `"${v}"` : String(v);
      lines.push(`          ${k}: ${val}`);
    }
  }
  return lines.join("\n");
}

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="bg-card border border-border rounded-lg px-4 py-3">
      <div className={`text-2xl font-bold ${color}`}>{value}</div>
      <div className="text-xs text-muted-foreground mt-0.5">{label}</div>
    </div>
  );
}

function Toggle({
  enabled,
  onToggle,
  disabled = false,
}: {
  enabled: boolean;
  onToggle: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      role="switch"
      aria-checked={enabled}
      disabled={disabled}
      className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-hidden focus:ring-2 focus:ring-ring focus:ring-offset-1 ${
        enabled ? "bg-info" : "bg-muted"
      } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
    >
      <span
        className={`inline-block h-3.5 w-3.5 transform rounded-full bg-card shadow transition-transform ${
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
  isAdmin: boolean;
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
  isAdmin,
  onSelect,
  onToggleForwardKey,
  onToggleHeaders,
  onApprove,
  onReject,
}: GuardrailCardProps) {
  const status = STATUS_CONFIG[g.status];
  const teamColor = TEAM_COLORS[g.team] ?? "bg-muted text-foreground";
  return (
    <div
      className={`bg-card border rounded-lg p-4 transition-all ${
        isSelected ? "border-info ring-1 ring-info/30" : "border-border"
      }`}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${teamColor}`}>Team: {g.team}</span>
            <span
              className={`inline-flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 rounded-full ${status.bg} ${status.text}`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${status.dot}`} />
              {status.label}
            </span>
          </div>
          <h3 className="text-sm font-semibold text-foreground mb-1">{g.name}</h3>
          <p className="text-xs text-muted-foreground mb-2 line-clamp-1">{g.description}</p>
          <div className="flex items-center gap-1.5 mb-2">
            <ServerIcon className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
            <code className="text-xs text-muted-foreground font-mono truncate">{g.endpoint}</code>
          </div>
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <span>
              Model: <span className="font-medium text-foreground">{g.model}</span>
            </span>
            <span>
              Submitted: <span className="font-medium text-foreground">{g.submittedAt}</span>
            </span>
          </div>
        </div>
        <div className="flex flex-col items-end gap-2 shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground whitespace-nowrap">Forward API Key</span>
            <Toggle enabled={g.forwardKey} onToggle={onToggleForwardKey} disabled={!isAdmin} />
          </div>
          <div className="flex items-center gap-2 mt-1">
            <button
              type="button"
              onClick={onSelect}
              className="text-xs border border-border text-muted-foreground hover:bg-muted px-3 py-1.5 rounded-md transition-colors font-medium"
            >
              {isSelected ? "Close" : "Review"}
            </button>
            {isAdmin && g.status === "pending" && (
              <>
                <button
                  type="button"
                  onClick={onApprove}
                  className="text-xs bg-success hover:bg-success/80 text-success-foreground px-3 py-1.5 rounded-md transition-colors font-medium"
                >
                  Approve
                </button>
                <button
                  type="button"
                  onClick={onReject}
                  className="text-xs border border-destructive/30 text-destructive hover:bg-destructive/10 px-3 py-1.5 rounded-md transition-colors font-medium"
                >
                  Reject
                </button>
              </>
            )}
          </div>
        </div>
      </div>
      <div className="mt-3 pt-3 border-t border-border">
        <button
          type="button"
          onClick={onToggleHeaders}
          className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {isHeadersExpanded ? <ChevronUpIcon className="h-3.5 w-3.5" /> : <ChevronDownIcon className="h-3.5 w-3.5" />}
          Static headers
          {g.customHeaders.length > 0 && (
            <span className="ml-1 bg-muted text-muted-foreground rounded-full px-1.5 py-0.5 text-xs">
              {g.customHeaders.length}
            </span>
          )}
        </button>
        {isHeadersExpanded && (
          <div className="mt-2">
            {g.customHeaders.length === 0 ? (
              <p className="text-xs text-muted-foreground italic">No static headers configured.</p>
            ) : (
              <div className="space-y-1">
                {g.customHeaders.map((h, i) => (
                  <div key={`${h.key}-${i}`} className="flex items-center gap-2 text-xs font-mono">
                    <span className="text-muted-foreground bg-muted border border-border rounded-sm px-2 py-0.5">
                      {h.key}
                    </span>
                    <span className="text-muted-foreground">:</span>
                    <span className="text-foreground bg-muted border border-border rounded-sm px-2 py-0.5">
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

function ConfigRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs font-semibold text-muted-foreground mb-1">{label}</div>
      <div>{children}</div>
    </div>
  );
}

type DetailPanelProps = {
  guardrail: TeamGuardrail;
  isAdmin: boolean;
  onClose: () => void;
  onApprove: () => void;
  onReject: () => void;
  onToggleForwardKey: () => void;
  onUpdateCustomHeaders: (customHeaders: { key: string; value: string }[]) => Promise<void>;
  onUpdateExtraHeaders: (extraHeaders: string[]) => Promise<void>;
};

function DetailPanel({
  guardrail: g,
  isAdmin,
  onClose,
  onApprove,
  onReject,
  onToggleForwardKey,
  onUpdateCustomHeaders,
  onUpdateExtraHeaders,
}: DetailPanelProps) {
  const [configExpanded, setConfigExpanded] = useState(false);
  const [newExtraHeader, setNewExtraHeader] = useState("");
  const [newStaticHeaderKey, setNewStaticHeaderKey] = useState("");
  const [newStaticHeaderValue, setNewStaticHeaderValue] = useState("");
  const status = STATUS_CONFIG[g.status];
  const teamColor = TEAM_COLORS[g.team] ?? "bg-muted text-foreground";
  return (
    <div className="w-96 shrink-0 bg-card overflow-auto">
      <div className="p-5">
        <div className="flex items-start justify-between mb-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${teamColor}`}>Team: {g.team}</span>
              <span
                className={`inline-flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 rounded-full ${status.bg} ${status.text}`}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${status.dot}`} />
                {status.label}
              </span>
            </div>
            <h2 className="text-base font-semibold text-foreground">{g.name}</h2>
            <p className="text-xs text-muted-foreground mt-0.5">
              Submitted by {g.submittedBy} on {g.submittedAt}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground transition-colors"
            aria-label="Close detail panel"
          >
            <XIcon className="h-4 w-4" />
          </button>
        </div>
        <p className="text-sm text-muted-foreground mb-5">{g.description}</p>
        <div className="space-y-4">
          <ConfigRow label="Endpoint">
            <div className="flex items-center gap-1.5">
              <code className="text-xs font-mono text-foreground break-all">{g.endpoint}</code>
              <a
                href={g.endpoint}
                target="_blank"
                rel="noopener noreferrer"
                className="text-muted-foreground hover:text-info shrink-0"
              >
                <ExternalLinkIcon className="h-3.5 w-3.5" />
              </a>
            </div>
          </ConfigRow>
          <ConfigRow label="Method">
            <span className="text-xs font-mono font-medium text-foreground bg-muted px-2 py-0.5 rounded-sm">
              {g.method}
            </span>
          </ConfigRow>
          <div className="border border-info/15 bg-info/10 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-1.5">
                <KeyIcon className="h-3.5 w-3.5 text-info" />
                <span className="text-xs font-semibold text-info">Forward LiteLLM API Key</span>
              </div>
              <Toggle enabled={g.forwardKey} onToggle={onToggleForwardKey} disabled={!isAdmin} />
            </div>
            <p className="text-xs text-info leading-relaxed">
              When enabled, the caller&apos;s LiteLLM API key is forwarded as an{" "}
              <code className="font-mono bg-info/15 px-1 rounded-sm">Authorization</code> header to your guardrail
              endpoint. This allows your guardrail to authenticate model calls using the original caller&apos;s
              credentials.
            </p>
          </div>
          <div>
            <div className="flex items-center gap-1.5 mb-2">
              <span className="text-xs font-semibold text-foreground">Static headers</span>
              {g.customHeaders.length > 0 && (
                <span className="bg-muted text-muted-foreground rounded-full px-1.5 py-0.5 text-xs">
                  {g.customHeaders.length}
                </span>
              )}
            </div>
            <p className="text-xs text-muted-foreground mb-2">Sent with every request to the guardrail.</p>
            {g.customHeaders.length === 0 ? (
              <p className="text-xs text-muted-foreground italic mb-2">No static headers configured.</p>
            ) : (
              <ul className="list-none space-y-1 mb-2">
                {g.customHeaders.map((h, i) => (
                  <li
                    key={`${h.key}-${i}`}
                    className="flex items-center justify-between gap-2 text-xs font-mono bg-muted border border-border rounded-sm px-2 py-1.5"
                  >
                    <span className="text-foreground truncate">
                      {h.key}: {h.value}
                    </span>
                    {isAdmin && (
                      <button
                        type="button"
                        onClick={() => onUpdateCustomHeaders(g.customHeaders.filter((_, idx) => idx !== i))}
                        className="text-muted-foreground hover:text-destructive shrink-0"
                        aria-label={`Remove ${h.key}`}
                      >
                        <XIcon className="h-3.5 w-3.5" />
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            )}
            {isAdmin && (
              <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
                <input
                  type="text"
                  value={newStaticHeaderKey}
                  onChange={(e) => setNewStaticHeaderKey(e.target.value)}
                  placeholder="Header name (e.g. X-API-Key)"
                  className="flex-1 min-w-0 text-xs font-mono border border-border rounded-sm px-2 py-1.5 text-foreground placeholder:text-muted-foreground focus:outline-hidden focus:ring-1 focus:ring-ring"
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      const key = newStaticHeaderKey.trim();
                      const value = newStaticHeaderValue.trim();
                      if (key && !g.customHeaders.some((h) => h.key.toLowerCase() === key.toLowerCase())) {
                        onUpdateCustomHeaders([...g.customHeaders, { key, value }]);
                        setNewStaticHeaderKey("");
                        setNewStaticHeaderValue("");
                      }
                    }
                  }}
                />
                <input
                  type="text"
                  value={newStaticHeaderValue}
                  onChange={(e) => setNewStaticHeaderValue(e.target.value)}
                  placeholder="Value"
                  className="flex-1 min-w-0 text-xs font-mono border border-border rounded-sm px-2 py-1.5 text-foreground placeholder:text-muted-foreground focus:outline-hidden focus:ring-1 focus:ring-ring"
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      const key = newStaticHeaderKey.trim();
                      const value = newStaticHeaderValue.trim();
                      if (key && !g.customHeaders.some((h) => h.key.toLowerCase() === key.toLowerCase())) {
                        onUpdateCustomHeaders([...g.customHeaders, { key, value }]);
                        setNewStaticHeaderKey("");
                        setNewStaticHeaderValue("");
                      }
                    }
                  }}
                />
                <button
                  type="button"
                  onClick={() => {
                    const key = newStaticHeaderKey.trim();
                    const value = newStaticHeaderValue.trim();
                    if (key && !g.customHeaders.some((h) => h.key.toLowerCase() === key.toLowerCase())) {
                      onUpdateCustomHeaders([...g.customHeaders, { key, value }]);
                      setNewStaticHeaderKey("");
                      setNewStaticHeaderValue("");
                    }
                  }}
                  className="text-xs font-medium text-info border border-info/20 bg-info/10 hover:bg-info/15 px-2 py-1.5 rounded-sm transition-colors shrink-0"
                >
                  Add
                </button>
              </div>
            )}
          </div>
          <div>
            <div className="flex items-center gap-1.5 mb-2">
              <span className="text-xs font-semibold text-foreground">Forward client headers</span>
              {g.extraHeaders.length > 0 && (
                <span className="bg-muted text-muted-foreground rounded-full px-1.5 py-0.5 text-xs">
                  {g.extraHeaders.length}
                </span>
              )}
            </div>
            <p className="text-xs text-muted-foreground mb-2">
              Allowed header names to forward from the client request to the guardrail (e.g. x-request-id).
            </p>
            {g.extraHeaders.length === 0 ? (
              <p className="text-xs text-muted-foreground italic mb-2">No forward client headers configured.</p>
            ) : (
              <ul className="list-none space-y-1 mb-2">
                {g.extraHeaders.map((name, i) => (
                  <li
                    key={`${name}-${i}`}
                    className="flex items-center justify-between gap-2 text-xs font-mono bg-muted border border-border rounded-sm px-2 py-1.5"
                  >
                    <span className="text-foreground truncate">{name}</span>
                    {isAdmin && (
                      <button
                        type="button"
                        onClick={() => onUpdateExtraHeaders(g.extraHeaders.filter((_, idx) => idx !== i))}
                        className="text-muted-foreground hover:text-destructive shrink-0"
                        aria-label={`Remove ${name}`}
                      >
                        <XIcon className="h-3.5 w-3.5" />
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            )}
            {isAdmin && (
              <div className="flex gap-2">
                <input
                  type="text"
                  value={newExtraHeader}
                  onChange={(e) => setNewExtraHeader(e.target.value)}
                  placeholder="e.g. x-request-id"
                  className="flex-1 min-w-0 text-xs font-mono border border-border rounded-sm px-2 py-1.5 text-foreground placeholder:text-muted-foreground focus:outline-hidden focus:ring-1 focus:ring-ring"
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      const name = newExtraHeader.trim().toLowerCase();
                      if (name && !g.extraHeaders.map((h) => h.toLowerCase()).includes(name)) {
                        onUpdateExtraHeaders([...g.extraHeaders, name]);
                        setNewExtraHeader("");
                      }
                    }
                  }}
                />
                <button
                  type="button"
                  onClick={() => {
                    const name = newExtraHeader.trim().toLowerCase();
                    if (name && !g.extraHeaders.map((h) => h.toLowerCase()).includes(name)) {
                      onUpdateExtraHeaders([...g.extraHeaders, name]);
                      setNewExtraHeader("");
                    }
                  }}
                  className="text-xs font-medium text-info border border-info/20 bg-info/10 hover:bg-info/15 px-2 py-1.5 rounded-sm transition-colors"
                >
                  Add
                </button>
              </div>
            )}
          </div>
          <div className="border border-border rounded-lg overflow-hidden">
            <button
              type="button"
              onClick={() => setConfigExpanded(!configExpanded)}
              className="w-full flex items-center justify-between px-3 py-2 text-left text-xs font-semibold text-foreground bg-muted hover:bg-border transition-colors"
            >
              <span>Equivalent config</span>
              {configExpanded ? (
                <ChevronUpIcon className="h-3.5 w-3.5 text-muted-foreground" />
              ) : (
                <ChevronDownIcon className="h-3.5 w-3.5 text-muted-foreground" />
              )}
            </button>
            {configExpanded && (
              <pre className="p-3 text-xs font-mono text-foreground bg-card border-t border-border overflow-x-auto whitespace-pre-wrap break-all">
                {buildEquivalentConfigYaml(g)}
              </pre>
            )}
          </div>
          <div className="flex items-start gap-2 bg-muted border border-border rounded-lg p-3">
            <InfoIcon className="h-3.5 w-3.5 text-muted-foreground shrink-0 mt-0.5" />
            <p className="text-xs text-muted-foreground leading-relaxed">
              This guardrail runs on a separate instance. It receives the user request and forwards the result to the
              next step in the pipeline. See{" "}
              <a
                href="https://docs.litellm.ai/docs/adding_provider/generic_guardrail_api"
                target="_blank"
                rel="noopener noreferrer"
                className="text-info hover:underline"
              >
                LiteLLM Generic Guardrail API docs
              </a>{" "}
              for configuration details.
            </p>
          </div>
        </div>
        <div className="mt-5 pt-4 border-t border-border space-y-2">
          <button
            type="button"
            className="w-full flex items-center justify-center gap-2 border border-border text-foreground hover:bg-muted text-sm font-medium py-2 rounded-md transition-colors"
          >
            <ExternalLinkIcon className="h-4 w-4" />
            Test Endpoint
          </button>
          {isAdmin && g.status === "pending" && (
            <div className="flex gap-2">
              <button
                type="button"
                onClick={onApprove}
                className="flex-1 flex items-center justify-center gap-1.5 bg-success hover:bg-success/80 text-success-foreground text-sm font-medium py-2 rounded-md transition-colors"
              >
                <CheckIcon className="h-4 w-4" />
                Approve
              </button>
              <button
                type="button"
                onClick={onReject}
                className="flex-1 flex items-center justify-center gap-1.5 border border-destructive/30 text-destructive hover:bg-destructive/10 text-sm font-medium py-2 rounded-md transition-colors"
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

function ConfirmDialog({ action, guardrailName, onConfirm, onCancel }: ConfirmDialogProps) {
  const isApprove = action === "approve";
  return (
    <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-overlay">
      <div className="bg-card rounded-xl shadow-xl p-6 max-w-sm w-full mx-4">
        <div
          className={`w-10 h-10 rounded-full flex items-center justify-center mb-4 ${
            isApprove ? "bg-success/15" : "bg-destructive/15"
          }`}
        >
          {isApprove ? (
            <CheckIcon className="h-5 w-5 text-success" />
          ) : (
            <AlertCircleIcon className="h-5 w-5 text-destructive" />
          )}
        </div>
        <h3 className="text-base font-semibold text-foreground mb-1">
          {isApprove ? "Approve Guardrail" : "Reject Guardrail"}
        </h3>
        <p className="text-sm text-muted-foreground mb-5">
          Are you sure you want to {action}{" "}
          <span className="font-medium text-foreground">&quot;{guardrailName}&quot;</span>?{" "}
          {isApprove
            ? "This will make it active and available for use."
            : "This will mark it as rejected and notify the team."}
        </p>
        <div className="flex gap-3">
          <button
            type="button"
            onClick={onCancel}
            className="flex-1 border border-border text-foreground hover:bg-muted text-sm font-medium py-2 rounded-md transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className={`flex-1 text-sm font-medium py-2 rounded-md transition-colors ${
              isApprove
                ? "bg-success text-success-foreground hover:bg-success/80"
                : "bg-destructive text-destructive-foreground hover:bg-destructive/80"
            }`}
          >
            {isApprove ? "Approve" : "Reject"}
          </button>
        </div>
      </div>
    </div>
  );
}

interface TeamGuardrailsTabProps {
  accessToken: string | null;
}

export function TeamGuardrailsTab({ accessToken }: TeamGuardrailsTabProps) {
  const { userRole } = useAuthorized();
  const isAdmin = userRole ? isProxyAdminRole(userRole) : false;
  const [guardrails, setGuardrails] = useState<TeamGuardrail[]>([]);
  const [summary, setSummary] = useState({
    total: 0,
    pending_review: 0,
    active: 0,
    rejected: 0,
  });
  const [search, setSearch] = useState("");
  const [searchDebounced] = useDebouncedValue(search, { wait: DEBOUNCE_WAIT_MS });
  const [statusFilter, setStatusFilter] = useState<"all" | GuardrailStatus>("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [expandedHeaders, setExpandedHeaders] = useState<Set<string>>(new Set());
  const [confirmAction, setConfirmAction] = useState<{
    id: string;
    action: "approve" | "reject";
  } | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitModalOpen, setIsSubmitModalOpen] = useState(false);
  const submitForm = useZodForm(submitGuardrailSchema, { defaultValues: EMPTY_SUBMIT_VALUES });
  const registerGuardrail = useRegisterGuardrail();

  const fetchSubmissions = useCallback(async () => {
    if (!accessToken) {
      setIsLoading(false);
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const statusParam =
        statusFilter === "all" ? undefined : statusFilter === "pending" ? "pending_review" : statusFilter;
      const res = await listGuardrailSubmissions(accessToken, {
        status: statusParam,
        search: searchDebounced.trim() || undefined,
      });
      setGuardrails(res.submissions.map(submissionToTeamGuardrail));
      setSummary(res.summary);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load submissions");
      setGuardrails([]);
    } finally {
      setIsLoading(false);
    }
  }, [accessToken, statusFilter, searchDebounced]);

  useEffect(() => {
    fetchSubmissions();
  }, [fetchSubmissions]);

  const handleSubmitGuardrail = submitForm.handleSubmit(async (values) => {
    const litellm_params: Record<string, unknown> = {
      ...(values.extra_litellm_params ? JSON.parse(values.extra_litellm_params) : {}),
      guardrail: "generic_guardrail_api",
      mode: values.mode,
      api_base: values.api_base,
    };
    try {
      await registerGuardrail.mutateAsync({
        team_id: values.team_id,
        guardrail_name: values.guardrail_name,
        litellm_params,
        guardrail_info: values.guardrail_info ? JSON.parse(values.guardrail_info) : undefined,
      });
      toast.success("Guardrail submitted for review");
      setIsSubmitModalOpen(false);
      submitForm.reset();
      fetchSubmissions();
    } catch {
      return;
    }
  });

  const filtered = guardrails;
  const selected = guardrails.find((g) => g.id === selectedId) ?? null;
  const totalCount = summary.total;
  const pendingCount = summary.pending_review;
  const activeCount = summary.active;
  const rejectedCount = summary.rejected;

  async function toggleForwardKey(id: string) {
    if (!accessToken) return;
    const g = guardrails.find((x) => x.id === id);
    if (!g) return;
    const newValue = !g.forwardKey;
    try {
      await updateGuardrailCall(accessToken, id, {
        litellm_params: { forward_api_key: newValue },
      });
      setGuardrails((prev) => prev.map((x) => (x.id === id ? { ...x, forwardKey: newValue } : x)));
      toast.success(newValue ? "Forward API key enabled" : "Forward API key disabled");
    } catch {
      toast.fromError("Failed to update forward API key");
    }
  }

  async function updateCustomHeaders(id: string, customHeaders: { key: string; value: string }[]) {
    if (!accessToken) return;
    const headersObj: Record<string, string> = {};
    for (const { key, value } of customHeaders) {
      if (key.trim()) headersObj[key.trim()] = value;
    }
    try {
      await updateGuardrailCall(accessToken, id, {
        litellm_params: { headers: headersObj },
      });
      setGuardrails((prev) =>
        prev.map((x) =>
          x.id === id
            ? {
                ...x,
                customHeaders: customHeaders.filter((h) => h.key.trim()),
              }
            : x,
        ),
      );
      toast.success("Static headers updated");
    } catch {
      toast.fromError("Failed to update static headers");
    }
  }

  async function updateExtraHeaders(id: string, extraHeaders: string[]) {
    if (!accessToken) return;
    try {
      await updateGuardrailCall(accessToken, id, {
        litellm_params: { extra_headers: extraHeaders },
      });
      setGuardrails((prev) => prev.map((x) => (x.id === id ? { ...x, extraHeaders } : x)));
      toast.success("Forward client headers updated");
    } catch {
      toast.fromError("Failed to update forward client headers");
    }
  }

  async function handleApprove(id: string) {
    if (!accessToken) return;
    try {
      await approveGuardrailSubmission(accessToken, id);
      setConfirmAction(null);
      if (selectedId === id) setSelectedId(null);
      await fetchSubmissions();
      toast.success("Guardrail approved");
    } catch {
      toast.fromError("Failed to approve guardrail");
    }
  }

  async function handleReject(id: string) {
    if (!accessToken) return;
    try {
      await rejectGuardrailSubmission(accessToken, id);
      setConfirmAction(null);
      if (selectedId === id) setSelectedId(null);
      await fetchSubmissions();
      toast.success("Guardrail rejected");
    } catch {
      toast.fromError("Failed to reject guardrail");
    }
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
      <div className={`flex-1 min-w-0 p-6 overflow-auto ${selected ? "border-r border-border" : ""}`}>
        <div className="grid grid-cols-4 gap-4 mb-6">
          <StatCard label="Total Submitted" value={totalCount} color="text-foreground" />
          <StatCard label="Pending Review" value={pendingCount} color="text-warning" />
          <StatCard label="Active" value={activeCount} color="text-success" />
          <StatCard label="Rejected" value={rejectedCount} color="text-destructive" />
        </div>
        <div className="flex items-center gap-3 mb-5">
          <div className="relative flex-1 max-w-xs">
            <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search guardrails..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-9 pr-4 py-2 border border-border rounded-md text-sm text-foreground placeholder:text-muted-foreground focus:outline-hidden focus:ring-1 focus:ring-ring focus:border-info"
            />
          </div>
          <select
            aria-label="Filter by status"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as typeof statusFilter)}
            className="border border-border rounded-md px-3 py-2 text-sm text-foreground focus:outline-hidden focus:ring-1 focus:ring-ring focus:border-info bg-background"
          >
            <option value="all">All Status</option>
            <option value="pending">Pending Review</option>
            <option value="active">Active</option>
            <option value="rejected">Rejected</option>
          </select>
          <button
            type="button"
            onClick={() => setIsSubmitModalOpen(true)}
            className="ml-auto flex items-center gap-2 bg-info hover:bg-info/80 text-info-foreground text-sm font-medium px-4 py-2 rounded-md transition-colors"
          >
            <PlusIcon className="h-4 w-4" />
            Add Guardrail
          </button>
        </div>
        <div className="space-y-3">
          {isLoading && <div className="text-center py-12 text-muted-foreground text-sm">Loading submissions…</div>}
          {error && <div className="text-center py-12 text-destructive text-sm">{error}</div>}
          {!isLoading && !error && filtered.length === 0 && (
            <div className="text-center py-12 text-muted-foreground text-sm">No guardrails match your filters.</div>
          )}
          {!isLoading &&
            !error &&
            filtered.map((g) => (
              <GuardrailCard
                key={g.id}
                guardrail={g}
                isSelected={selectedId === g.id}
                isHeadersExpanded={expandedHeaders.has(g.id)}
                isAdmin={isAdmin}
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
          isAdmin={isAdmin}
          onClose={() => setSelectedId(null)}
          onApprove={() => setConfirmAction({ id: selected.id, action: "approve" })}
          onReject={() => setConfirmAction({ id: selected.id, action: "reject" })}
          onToggleForwardKey={() => toggleForwardKey(selected.id)}
          onUpdateCustomHeaders={(customHeaders) => updateCustomHeaders(selected.id, customHeaders)}
          onUpdateExtraHeaders={(extraHeaders) => updateExtraHeaders(selected.id, extraHeaders)}
        />
      )}
      {confirmAction && (
        <ConfirmDialog
          action={confirmAction.action}
          guardrailName={guardrails.find((g) => g.id === confirmAction.id)?.name ?? ""}
          onConfirm={() =>
            confirmAction.action === "approve" ? handleApprove(confirmAction.id) : handleReject(confirmAction.id)
          }
          onCancel={() => setConfirmAction(null)}
        />
      )}

      <Dialog
        open={isSubmitModalOpen}
        onOpenChange={(open) => {
          if (!open) {
            setIsSubmitModalOpen(false);
            submitForm.reset();
          }
        }}
      >
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Submit Guardrail for Review</DialogTitle>
          </DialogHeader>
          <div className="rounded-md bg-info/10 border border-info/20 px-4 py-3 text-sm text-info mb-4">
            Your guardrail will be sent for admin review before it becomes active.
          </div>
          <TooltipProvider>
            <form onSubmit={handleSubmitGuardrail}>
              <FieldGroup>
                <FormField control={submitForm.control} name="team_id" label="Team">
                  {({ id, value, onChange }) => <TeamDropdown id={id} value={value} onChange={onChange} />}
                </FormField>
                <FormField control={submitForm.control} name="guardrail_name" label="Guardrail Name">
                  {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="e.g. pii-detection" />}
                </FormField>
                <FormField control={submitForm.control} name="mode" label="Mode">
                  {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                    <Select items={GUARDRAIL_MODES} value={value} onValueChange={onChange}>
                      <SelectTrigger
                        id={id}
                        aria-invalid={ariaInvalid}
                        aria-describedby={ariaDescribedBy}
                        className="w-full"
                      >
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {GUARDRAIL_MODES.map((mode) => (
                          <SelectItem key={mode.value} value={mode.value} title={mode.label}>
                            {mode.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                </FormField>
                <FormField control={submitForm.control} name="api_base" label="API Base URL">
                  {({ ref, ...field }) => (
                    <Input
                      {...field}
                      ref={ref}
                      placeholder="https://your-guardrail-api.com/v1/check"
                      className="font-mono"
                    />
                  )}
                </FormField>
                <FormField
                  control={submitForm.control}
                  name="extra_litellm_params"
                  label={labelWithHint(
                    "Additional litellm_params (optional)",
                    "JSON object merged into litellm_params. e.g. forward_api_key, headers, model, unreachable_fallback",
                  )}
                >
                  {({ ref, ...field }) => (
                    <Textarea
                      {...field}
                      ref={ref}
                      rows={3}
                      className="font-mono text-xs"
                      placeholder='{"forward_api_key": true, "headers": {"X-Custom": "value"}}'
                    />
                  )}
                </FormField>
                <FormField control={submitForm.control} name="guardrail_info" label="Guardrail Info (optional)">
                  {({ ref, ...field }) => (
                    <Textarea
                      {...field}
                      ref={ref}
                      rows={3}
                      className="font-mono text-xs"
                      placeholder='{"description": "Detects PII in requests"}'
                    />
                  )}
                </FormField>
              </FieldGroup>
            </form>
          </TooltipProvider>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setIsSubmitModalOpen(false);
                submitForm.reset();
              }}
            >
              Cancel
            </Button>
            <Button onClick={handleSubmitGuardrail}>Submit for Review</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
