import React, { useState, useMemo } from "react";
import { Tooltip } from "antd";
import {
  ComplianceResponse,
} from "@/components/networking";

interface CompliancePanelProps {
  accessToken: string | null;
  logEntry: {
    request_id: string;
    user?: string;
    model?: string;
    startTime?: string;
    metadata?: Record<string, any>;
  };
}

function runComplianceChecksLocally(logEntry: CompliancePanelProps["logEntry"]): {
  euAiAct: ComplianceResponse;
  gdpr: ComplianceResponse;
} {
  const guardrails: Record<string, any>[] = logEntry.metadata?.guardrail_information ?? [];
  const hasGuardrails = guardrails.length > 0;
  const preCallGuardrails = guardrails.filter(
    (g) => g.guardrail_mode === "pre_call" || !g.guardrail_mode
  );
  const hasPreCall = preCallGuardrails.length > 0;
  const hasUser = Boolean(logEntry.user);
  const hasModel = Boolean(logEntry.model);
  const hasTimestamp = Boolean(logEntry.startTime);
  const auditComplete = hasUser && hasModel && hasTimestamp && hasGuardrails;

  const missingFields: string[] = [];
  if (!hasUser) missingFields.push("user_id");
  if (!hasModel) missingFields.push("model");
  if (!hasTimestamp) missingFields.push("timestamp");
  if (!hasGuardrails) missingFields.push("guardrail_results");

  const hasIntervention = preCallGuardrails.some((g) =>
    ["guardrail_intervened", "failed", "blocked"].includes(g.guardrail_status ?? "")
  );
  const allPassed =
    preCallGuardrails.length > 0 &&
    preCallGuardrails.every((g) => g.guardrail_status === "success");
  const dataProtected = hasIntervention || allPassed;

  const euAiAct: ComplianceResponse = {
    regulation: "EU AI Act",
    compliant: hasGuardrails && hasPreCall && auditComplete,
    checks: [
      {
        check_name: "Guardrails applied",
        article: "Art. 9",
        passed: hasGuardrails,
        detail: hasGuardrails
          ? `${guardrails.length} guardrail(s) applied`
          : "No guardrails applied",
      },
      {
        check_name: "Content screened before LLM",
        article: "Art. 5",
        passed: hasPreCall,
        detail: hasPreCall
          ? `${preCallGuardrails.length} pre-call guardrail(s) screened content`
          : "No pre-call screening applied",
      },
      {
        check_name: "Audit record complete",
        article: "Art. 12",
        passed: auditComplete,
        detail: auditComplete
          ? "All required audit fields present"
          : `Missing: ${missingFields.join(", ")}`,
      },
    ],
  };

  const gdpr: ComplianceResponse = {
    regulation: "GDPR",
    compliant: hasPreCall && dataProtected && auditComplete,
    checks: [
      {
        check_name: "Data protection applied",
        article: "Art. 32",
        passed: hasPreCall,
        detail: hasPreCall
          ? `${preCallGuardrails.length} pre-call guardrail(s) protect data`
          : "No pre-call data protection applied",
      },
      {
        check_name: "Sensitive data protected",
        article: "Art. 5(1)(c)",
        passed: dataProtected,
        detail: hasIntervention
          ? "Guardrail intervened to protect sensitive data"
          : allPassed
            ? "No sensitive data detected"
            : "No pre-call guardrails to protect sensitive data",
      },
      {
        check_name: "Audit record complete",
        article: "Art. 30",
        passed: auditComplete,
        detail: auditComplete
          ? "All required audit fields present"
          : `Missing: ${missingFields.join(", ")}`,
      },
    ],
  };

  return { euAiAct, gdpr };
}

// -- Icons --

const CheckIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
    <circle cx="8" cy="8" r="7" stroke="#16A34A" strokeWidth="1.5" fill="#F0FDF4" />
    <path d="M5 8l2 2 4-4" stroke="#16A34A" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const CrossIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
    <circle cx="8" cy="8" r="7" stroke="#DC2626" strokeWidth="1.5" fill="#FEF2F2" />
    <path d="M6 6l4 4M10 6l-4 4" stroke="#DC2626" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);

const SpinnerIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="animate-spin">
    <circle cx="8" cy="8" r="6" stroke="#D1D5DB" strokeWidth="2" />
    <path d="M8 2a6 6 0 0 1 6 6" stroke="#6366F1" strokeWidth="2" strokeLinecap="round" />
  </svg>
);

// -- Sub-components --

const ComplianceCard = ({
  title,
  data,
  loading,
  error,
}: {
  title: string;
  data: ComplianceResponse | null;
  loading: boolean;
  error: string | null;
}) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border border-gray-200 rounded-lg bg-white">
      <div
        className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-gray-50 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          {loading ? (
            <SpinnerIcon />
          ) : error ? (
            <Tooltip title={error}>
              <span className="text-gray-400 text-sm">--</span>
            </Tooltip>
          ) : data?.compliant ? (
            <CheckIcon />
          ) : (
            <CrossIcon />
          )}
          <span className="font-medium text-sm text-gray-900">{title}</span>
        </div>
        <div className="flex items-center gap-2">
          {!loading && !error && data && (
            <span
              className={`px-2 py-0.5 rounded text-[11px] font-semibold uppercase ${
                data.compliant
                  ? "bg-green-100 text-green-700 border border-green-200"
                  : "bg-red-100 text-red-700 border border-red-200"
              }`}
            >
              {data.compliant ? "COMPLIANT" : "NON-COMPLIANT"}
            </span>
          )}
          {error && (
            <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-gray-100 text-gray-500 border border-gray-200">
              UNAVAILABLE
            </span>
          )}
          <svg
            width="20"
            height="20"
            viewBox="0 0 20 20"
            fill="none"
            className={`transition-transform ${expanded ? "rotate-180" : ""}`}
          >
            <path d="M6 8l4 4 4-4" stroke="#6B7280" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      </div>

      {expanded && (
        <div className="border-t border-gray-100 px-4 py-3">
          {loading && <p className="text-sm text-gray-500">Checking compliance...</p>}
          {error && <p className="text-sm text-red-600">{error}</p>}
          {data && (
            <div className="space-y-2">
              {data.checks.map((check, idx) => (
                <div key={idx} className="flex items-start gap-2">
                  <div className="flex-shrink-0 mt-0.5">
                    {check.passed ? <CheckIcon /> : <CrossIcon />}
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-gray-900">{check.check_name}</span>
                      <span className="text-[10px] font-mono text-gray-400">{check.article}</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-0.5">{check.detail}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// -- Main Component --

const CompliancePanel: React.FC<CompliancePanelProps> = ({ accessToken, logEntry }) => {
  const { euAiAct, gdpr } = useMemo(() => runComplianceChecksLocally(logEntry), [logEntry]);

  return (
    <div>
      <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
        Regulatory Compliance
      </h4>
      <div className="space-y-3">
        <ComplianceCard
          title="EU AI Act"
          data={euAiAct}
          loading={false}
          error={null}
        />
        <ComplianceCard
          title="GDPR"
          data={gdpr}
          loading={false}
          error={null}
        />
      </div>
    </div>
  );
};

export default CompliancePanel;
