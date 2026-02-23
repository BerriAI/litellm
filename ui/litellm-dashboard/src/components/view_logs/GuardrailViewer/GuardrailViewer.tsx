import React, { useState, useMemo } from "react";
import { Tooltip } from "antd";
import PresidioDetectedEntities from "./PresidioDetectedEntities";
import BedrockGuardrailDetails, {
  BedrockGuardrailResponse,
} from "@/components/view_logs/GuardrailViewer/BedrockGuardrailDetails";
import ContentFilterDetails from "./ContentFilterDetails";
import CompliancePanel from "./CompliancePanel";

// ── Interfaces ──────────────────────────────────────────────────────────────

interface RecognitionMetadata {
  recognizer_name: string;
  recognizer_identifier: string;
}

interface GuardrailEntity {
  end: number;
  score: number;
  start: number;
  entity_type: string;
  analysis_explanation: string | null;
  recognition_metadata: RecognitionMetadata;
}

interface MaskedEntityCount {
  [key: string]: number;
}

interface MatchDetail {
  type: string;
  detection_method?: string;
  action_taken?: string;
  snippet?: string;
  category?: string;
  position?: number;
}

interface GuardrailInformation {
  duration: number;
  end_time: number;
  start_time: number;
  guardrail_mode: string;
  guardrail_name: string;
  guardrail_status: string;
  guardrail_response: GuardrailEntity[] | BedrockGuardrailResponse | any;
  masked_entity_count: MaskedEntityCount;
  guardrail_provider?: string;
  guardrail_id?: string;
  policy_template?: string;
  detection_method?: string;
  confidence_score?: number;
  classification?: Record<string, any>;
  match_details?: MatchDetail[];
  patterns_checked?: number;
  alert_recipients?: string[];
  risk_score?: number;
}

interface GuardrailViewerProps {
  data: GuardrailInformation | GuardrailInformation[];
  accessToken?: string | null;
  logEntry?: {
    request_id: string;
    user?: string;
    model?: string;
    startTime?: string;
    metadata?: Record<string, any>;
  };
}

// ── Helpers ─────────────────────────────────────────────────────────────────

const PROVIDERS_WITH_CUSTOM_RENDERERS = new Set([
  "presidio",
  "bedrock",
  "litellm_content_filter",
]);

const formatMode = (mode: string): string => {
  return mode.replace(/_/g, "-").toUpperCase();
};

const formatDurationMs = (seconds: number): string => {
  const ms = Math.round(seconds * 1000);
  return `${ms}ms`;
};

const getTotalMasked = (entry: GuardrailInformation): number => {
  return Object.values(entry.masked_entity_count || {}).reduce(
    (sum, count) => sum + (typeof count === "number" ? count : 0),
    0,
  );
};

const isEntrySuccess = (entry: GuardrailInformation): boolean => {
  return (entry.guardrail_status ?? "").toLowerCase() === "success";
};

const getRiskColor = (score: number): string => {
  if (score <= 3) return "text-green-600 bg-green-50 border-green-200";
  if (score <= 6) return "text-amber-600 bg-amber-50 border-amber-200";
  return "text-red-600 bg-red-50 border-red-200";
};

const getRiskScore = (entry: GuardrailInformation): number | null => {
  if (!isEntrySuccess(entry)) return null;

  // Prefer backend-computed score
  if (entry.risk_score != null) return entry.risk_score;

  // Fallback: compute from available data
  const totalMasked = getTotalMasked(entry);
  const patternsChecked = entry.patterns_checked ?? 0;
  const confidence = entry.confidence_score ?? 0;

  if (patternsChecked === 0 && confidence === 0) return 0;

  const matchRatio = patternsChecked > 0 ? totalMasked / patternsChecked : 0;
  let score = matchRatio * 7 + confidence * 3;
  if (totalMasked > 0 && score < 2) score = 2;
  return Math.min(10, Math.round(score * 10) / 10);
};

const getDisplayName = (entry: GuardrailInformation): string => {
  return entry.policy_template || entry.guardrail_name;
};

// ── Icons (inline SVGs) ─────────────────────────────────────────────────────

const ShieldIcon = () => (
  <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
    <circle cx="20" cy="20" r="20" fill="#EEF2FF" />
    <path
      d="M20 10l8 4v6c0 5.25-3.4 10.15-8 11.5C15.4 30.15 12 25.25 12 20v-6l8-4z"
      stroke="#6366F1"
      strokeWidth="1.5"
      fill="none"
    />
    <path
      d="M16 20l3 3 5-6"
      stroke="#6366F1"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
    />
  </svg>
);

const CheckCircleIcon = ({ className }: { className?: string }) => (
  <svg width="22" height="22" viewBox="0 0 22 22" fill="none" className={className}>
    <circle cx="11" cy="11" r="10" stroke="#16A34A" strokeWidth="1.5" fill="#F0FDF4" />
    <path d="M7 11l3 3 5-6" stroke="#16A34A" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const FailCircleIcon = ({ className }: { className?: string }) => (
  <svg width="22" height="22" viewBox="0 0 22 22" fill="none" className={className}>
    <circle cx="11" cy="11" r="10" stroke="#DC2626" strokeWidth="1.5" fill="#FEF2F2" />
    <path d="M8 8l6 6M14 8l-6 6" stroke="#DC2626" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);

const PlayCircleIcon = () => (
  <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
    <circle cx="11" cy="11" r="10" stroke="#3B82F6" strokeWidth="1.5" fill="#EFF6FF" />
    <path d="M9 7.5l6 3.5-6 3.5V7.5z" fill="#3B82F6" />
  </svg>
);

const GrayDotIcon = () => (
  <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
    <circle cx="11" cy="11" r="5" fill="#9CA3AF" />
  </svg>
);

const ChevronIcon = ({ expanded }: { expanded: boolean }) => (
  <svg
    width="20"
    height="20"
    viewBox="0 0 20 20"
    fill="none"
    className={`transition-transform ${expanded ? "rotate-180" : ""}`}
  >
    <path d="M6 8l4 4 4-4" stroke="#6B7280" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const DownloadIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
    <path d="M8 2v8m0 0l-3-3m3 3l3-3M3 12h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const ExternalLinkIcon = () => (
  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="inline ml-1">
    <path d="M6 2H3a1 1 0 00-1 1v8a1 1 0 001 1h8a1 1 0 001-1V8M8 2h4m0 0v4m0-4L6.5 7.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

// ── Sub-components ──────────────────────────────────────────────────────────

const MatchDetailsTable = ({ matchDetails }: { matchDetails: MatchDetail[] }) => {
  if (!matchDetails || matchDetails.length === 0) return null;

  return (
    <div className="mt-3">
      <h5 className="text-sm font-medium mb-2 text-gray-700">Match Details ({matchDetails.length})</h5>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-gray-500">
              <th className="pb-2 pr-4 font-medium">Type</th>
              <th className="pb-2 pr-4 font-medium">Method</th>
              <th className="pb-2 pr-4 font-medium">Action</th>
              <th className="pb-2 font-medium">Detail</th>
            </tr>
          </thead>
          <tbody>
            {matchDetails.map((match, idx) => (
              <tr key={idx} className="border-b border-gray-100">
                <td className="py-2 pr-4">{match.type}</td>
                <td className="py-2 pr-4">
                  <span className="px-2 py-0.5 bg-slate-100 text-slate-700 rounded text-xs">
                    {match.detection_method ?? "-"}
                  </span>
                </td>
                <td className="py-2 pr-4">
                  <span
                    className={`px-2 py-0.5 rounded text-xs font-medium ${
                      match.action_taken === "BLOCK" ? "bg-red-100 text-red-800" : "bg-blue-50 text-blue-700"
                    }`}
                  >
                    {match.action_taken ?? "-"}
                  </span>
                </td>
                <td className="py-2 font-mono text-xs text-gray-600 break-all">
                  {match.category ? `[${match.category}] ` : ""}
                  {match.snippet ?? "-"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const GenericGuardrailResponse = ({ response }: { response: any }) => {
  const [showRaw, setShowRaw] = useState(false);
  return (
    <div className="mt-3">
      <div className="border rounded-lg overflow-hidden">
        <div
          className="flex items-center justify-between p-3 bg-gray-50 cursor-pointer hover:bg-gray-100"
          onClick={() => setShowRaw(!showRaw)}
        >
          <div className="flex items-center">
            <ChevronIcon expanded={showRaw} />
            <h5 className="font-medium text-sm ml-1">Raw Guardrail Response</h5>
          </div>
        </div>
        {showRaw && (
          <div className="p-3 border-t bg-white">
            <pre className="bg-gray-50 rounded p-3 text-xs overflow-x-auto">
              {JSON.stringify(response, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

// ── Timeline entry types ────────────────────────────────────────────────────

interface TimelineEntry {
  type: "request" | "guardrail" | "llm" | "response";
  label: string;
  offsetMs: number;
  status?: string;
  isSuccess?: boolean;
}

const RequestLifecycle = ({ entries }: { entries: GuardrailInformation[] }) => {
  const sorted = useMemo(
    () => [...entries].sort((a, b) => (a.start_time ?? 0) - (b.start_time ?? 0)),
    [entries],
  );

  const timeline = useMemo(() => {
    if (sorted.length === 0) return [];

    const baseTime = sorted[0].start_time;
    const items: TimelineEntry[] = [];

    // Request received
    items.push({ type: "request", label: "Request received", offsetMs: 0 });

    // Pre-call guardrails
    const preCalls = sorted.filter((e) => e.guardrail_mode === "pre_call");
    const postCalls = sorted.filter((e) => e.guardrail_mode === "post_call" || e.guardrail_mode === "logging_only");
    const duringCalls = sorted.filter((e) => e.guardrail_mode === "during_call");

    for (const e of preCalls) {
      const offsetMs = Math.round((e.end_time - baseTime) * 1000);
      items.push({
        type: "guardrail",
        label: `Pre-call guardrail: ${getDisplayName(e)}`,
        offsetMs,
        status: isEntrySuccess(e) ? "PASSED" : "FAILED",
        isSuccess: isEntrySuccess(e),
      });
    }

    // LLM call — infer from gap between pre-call end and post-call start
    const lastPreEnd = preCalls.length > 0 ? Math.max(...preCalls.map((e) => e.end_time)) : baseTime;
    const firstPostStart = postCalls.length > 0 ? Math.min(...postCalls.map((e) => e.start_time)) : undefined;
    const llmEndTime = firstPostStart ?? (lastPreEnd + 1);
    const llmOffsetMs = Math.round((llmEndTime - baseTime) * 1000);

    items.push({
      type: "llm",
      label: "LLM call",
      offsetMs: llmOffsetMs,
    });

    // During-call guardrails (rare)
    for (const e of duringCalls) {
      const offsetMs = Math.round((e.end_time - baseTime) * 1000);
      items.push({
        type: "guardrail",
        label: `During-call guardrail: ${getDisplayName(e)}`,
        offsetMs,
        status: isEntrySuccess(e) ? "PASSED" : "FAILED",
        isSuccess: isEntrySuccess(e),
      });
    }

    // Post-call guardrails
    for (const e of postCalls) {
      const offsetMs = Math.round((e.end_time - baseTime) * 1000);
      items.push({
        type: "guardrail",
        label: `Post-call guardrail: ${getDisplayName(e)}`,
        offsetMs,
        status: isEntrySuccess(e) ? "PASSED" : "FAILED",
        isSuccess: isEntrySuccess(e),
      });
    }

    // Response returned
    const maxEnd = Math.max(...sorted.map((e) => e.end_time));
    const responseOffsetMs = Math.round((maxEnd - baseTime) * 1000) + 1;
    items.push({ type: "response", label: "Response returned", offsetMs: responseOffsetMs });

    return items;
  }, [sorted]);

  return (
    <div>
      <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
        Request Lifecycle
      </h4>
      <div className="relative">
        {timeline.map((item, idx) => (
          <div key={idx} className="flex items-start gap-3 relative">
            {/* Vertical line */}
            <div className="flex flex-col items-center">
              <div className="flex-shrink-0">
                {item.type === "request" || item.type === "response" ? (
                  <GrayDotIcon />
                ) : item.type === "llm" ? (
                  <PlayCircleIcon />
                ) : item.isSuccess ? (
                  <CheckCircleIcon />
                ) : (
                  <FailCircleIcon />
                )}
              </div>
              {idx < timeline.length - 1 && (
                <div className="w-0.5 bg-gray-200 flex-grow" style={{ minHeight: "24px" }} />
              )}
            </div>

            {/* Content */}
            <div className="pb-4 flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span
                  className={`text-sm ${
                    item.type === "llm" ? "text-blue-600 font-medium" : "text-gray-900"
                  }`}
                >
                  {item.label}
                </span>
                {item.status && (
                  <span
                    className={`px-1.5 py-0.5 rounded text-[10px] font-bold uppercase ${
                      item.isSuccess
                        ? "bg-green-100 text-green-700"
                        : "bg-red-100 text-red-700"
                    }`}
                  >
                    {item.status}
                  </span>
                )}
                <span className="text-xs text-gray-400 font-mono ml-auto flex-shrink-0">
                  T+{item.offsetMs}ms
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// ── Evaluation Card ─────────────────────────────────────────────────────────

const EvaluationCard = ({ entry }: { entry: GuardrailInformation }) => {
  const [expanded, setExpanded] = useState(false);
  const success = isEntrySuccess(entry);
  const totalMasked = getTotalMasked(entry);
  const displayName = getDisplayName(entry);
  const durationStr = formatDurationMs(entry.duration);
  const modeStr = formatMode(entry.guardrail_mode);
  const riskScore = getRiskScore(entry);

  const guardrailProvider = entry.guardrail_provider ?? "presidio";
  const guardrailResponse = entry.guardrail_response;
  const presidioEntities = Array.isArray(guardrailResponse) ? guardrailResponse : [];
  const bedrockResponse =
    guardrailProvider === "bedrock" &&
    guardrailResponse !== null &&
    typeof guardrailResponse === "object" &&
    !Array.isArray(guardrailResponse)
      ? (guardrailResponse as BedrockGuardrailResponse)
      : undefined;

  // Match count string: "X/Y matched" or "X matched"
  const matchCountStr =
    entry.patterns_checked != null
      ? `${totalMasked}/${entry.patterns_checked} matched`
      : totalMasked > 0
        ? `${totalMasked} matched`
        : null;

  return (
    <div className="border border-gray-200 rounded-lg bg-white">
      {/* Collapsed header row */}
      <div
        className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-gray-50 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        {/* Status icon */}
        <div className="flex-shrink-0">
          {success ? <CheckCircleIcon /> : <FailCircleIcon />}
        </div>

        {/* Name + badges */}
        <div className="flex items-center gap-2 flex-wrap flex-1 min-w-0">
          <span className="font-semibold text-gray-900 text-sm truncate">{displayName}</span>

          <span className="px-2 py-0.5 border border-blue-200 bg-blue-50 text-blue-700 rounded text-[11px] font-semibold uppercase flex-shrink-0">
            {modeStr}
          </span>

          <span
            className={`px-2 py-0.5 rounded text-[11px] font-semibold uppercase flex-shrink-0 ${
              success ? "bg-green-100 text-green-700 border border-green-200" : "bg-red-100 text-red-700 border border-red-200"
            }`}
          >
            {success ? "PASSED" : "FAILED"}
          </span>

          {matchCountStr && (
            <span
              className={`px-2 py-0.5 rounded text-[11px] font-medium flex-shrink-0 ${
                totalMasked === 0 ? "bg-green-50 text-green-700 border border-green-200" : "bg-amber-50 text-amber-700 border border-amber-200"
              }`}
            >
              {matchCountStr}
            </span>
          )}

          {entry.confidence_score != null && (
            <span className="px-2 py-0.5 bg-gray-100 text-gray-600 border border-gray-200 rounded text-[11px] font-medium flex-shrink-0">
              {(entry.confidence_score * 100).toFixed(0)}% conf
            </span>
          )}

          {riskScore != null && success && (
            <Tooltip title={`Risk score: ${riskScore}/10`}>
              <span className={`px-2 py-0.5 border rounded text-[11px] font-semibold flex-shrink-0 ${getRiskColor(riskScore)}`}>
                Risk {riskScore}/10
              </span>
            </Tooltip>
          )}
        </div>

        {/* Right side: duration + method + chevron */}
        <div className="flex items-center gap-3 flex-shrink-0">
          <span className="text-sm text-gray-500 font-mono">{durationStr}</span>
          {entry.detection_method && (
            <span className="px-2 py-0.5 bg-gray-100 text-gray-600 border border-gray-200 rounded text-[11px] font-medium">
              {entry.detection_method.split(",")[0].trim()}
            </span>
          )}
          <ChevronIcon expanded={expanded} />
        </div>
      </div>

      {/* Expanded details */}
      {expanded && (
        <div className="border-t border-gray-100 px-4 py-3">

          {/* Classification details for llm-judge */}
          {entry.classification && (
            <div className="mb-3 bg-gray-50 rounded-lg p-3 space-y-1">
              <h5 className="text-sm font-medium text-gray-700 mb-2">Classification</h5>
              {entry.classification.category && (
                <div className="flex text-sm">
                  <span className="font-medium w-1/3 text-gray-500">Category:</span>
                  <span>{entry.classification.category}</span>
                </div>
              )}
              {entry.classification.article_reference && (
                <div className="flex text-sm">
                  <span className="font-medium w-1/3 text-gray-500">Reference:</span>
                  <span className="font-mono">{entry.classification.article_reference}</span>
                </div>
              )}
              {entry.classification.confidence != null && (
                <div className="flex text-sm">
                  <span className="font-medium w-1/3 text-gray-500">Confidence:</span>
                  <span>{(entry.classification.confidence * 100).toFixed(0)}%</span>
                </div>
              )}
              {entry.classification.reason && (
                <div className="flex text-sm">
                  <span className="font-medium w-1/3 text-gray-500">Reason:</span>
                  <span>{entry.classification.reason}</span>
                </div>
              )}
            </div>
          )}

          {/* Match details table */}
          {entry.match_details && entry.match_details.length > 0 && (
            <MatchDetailsTable matchDetails={entry.match_details} />
          )}

          {/* Masked entity summary */}
          {totalMasked > 0 && (
            <div className="mt-3">
              <h5 className="text-sm font-medium text-gray-700 mb-2">Masked Entities</h5>
              <div className="flex flex-wrap gap-2">
                {Object.entries(entry.masked_entity_count || {}).map(([entityType, count]) => (
                  <span key={entityType} className="px-2 py-1 bg-blue-50 text-blue-700 rounded text-xs font-medium">
                    {entityType}: {count}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Provider-specific details */}
          {guardrailProvider === "presidio" && presidioEntities.length > 0 && (
            <div className="mt-3">
              <PresidioDetectedEntities entities={presidioEntities} />
            </div>
          )}
          {guardrailProvider === "bedrock" && bedrockResponse && (
            <div className="mt-3">
              <BedrockGuardrailDetails response={bedrockResponse} />
            </div>
          )}
          {guardrailProvider === "litellm_content_filter" && guardrailResponse && (
            <div className="mt-3">
              <ContentFilterDetails response={guardrailResponse} />
            </div>
          )}
          {guardrailProvider &&
            !PROVIDERS_WITH_CUSTOM_RENDERERS.has(guardrailProvider) &&
            guardrailResponse && <GenericGuardrailResponse response={guardrailResponse} />}
        </div>
      )}
    </div>
  );
};

// ── Main Component ──────────────────────────────────────────────────────────

const GuardrailViewer = ({ data, accessToken, logEntry }: GuardrailViewerProps) => {
  const guardrailEntries = useMemo(() => {
    return Array.isArray(data)
      ? data.filter((entry): entry is GuardrailInformation => Boolean(entry))
      : data
        ? [data]
        : [];
  }, [data]);

  const passedCount = guardrailEntries.filter(isEntrySuccess).length;
  const allPassed = passedCount === guardrailEntries.length;

  const totalOverheadMs = useMemo(() => {
    return Math.round(guardrailEntries.reduce((sum, e) => sum + (e.duration ?? 0), 0) * 1000);
  }, [guardrailEntries]);

  const policyTemplates = useMemo(() => {
    return Array.from(new Set(guardrailEntries.map((e) => e.policy_template).filter(Boolean)));
  }, [guardrailEntries]);

  if (guardrailEntries.length === 0) {
    return null;
  }

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(guardrailEntries, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `guardrail-compliance-log-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm w-full max-w-full overflow-hidden mb-6">
      {/* ── Header ─────────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
        <div className="flex items-center gap-4">
          <ShieldIcon />
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              Guardrails &amp; Policy Compliance
            </h3>
            <div className="flex items-center gap-2 mt-0.5">
              <span className="text-sm text-gray-500">
                {guardrailEntries.length} guardrail{guardrailEntries.length !== 1 ? "s" : ""} evaluated
              </span>
              <span className="text-gray-300">|</span>
              <span
                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold ${
                  allPassed
                    ? "bg-green-50 text-green-700 border border-green-200"
                    : "bg-red-50 text-red-700 border border-red-200"
                }`}
              >
                {allPassed ? (
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                    <path d="M3 6l2.5 2.5L9 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                ) : null}
                {passedCount} Passed
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="text-right">
            <div className="text-sm font-medium text-gray-900">
              Total: {totalOverheadMs}ms overhead
            </div>
          </div>

          <button
            onClick={handleExport}
            className="inline-flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 transition-colors"
          >
            <DownloadIcon />
            Export Compliance Log
          </button>
        </div>
      </div>

      {/* ── Compliance Panel ──────────────────────────────────── */}
      {accessToken && logEntry && (
        <div className="px-6 py-4 border-b border-gray-100">
          <CompliancePanel accessToken={accessToken} logEntry={logEntry} />
        </div>
      )}

      {/* ── Body: two columns ──────────────────────────────────── */}
      <div className="flex">
        {/* Left column: Request Lifecycle */}
        <div className="w-[340px] flex-shrink-0 border-r border-gray-100 px-6 py-5">
          <RequestLifecycle entries={guardrailEntries} />
        </div>

        {/* Right column: Evaluation Details */}
        <div className="flex-1 px-6 py-5 min-w-0">
          <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
            Evaluation Details
          </h4>
          <div className="space-y-3">
            {guardrailEntries.map((entry, index) => (
              <EvaluationCard
                key={`${entry.guardrail_name ?? "guardrail"}-${index}`}
                entry={entry}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default GuardrailViewer;
