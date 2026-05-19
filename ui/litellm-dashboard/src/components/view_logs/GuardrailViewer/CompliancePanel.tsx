import React, { useState, useEffect } from "react";
import { Tooltip } from "antd";
import {
  checkEuAiActCompliance,
  checkGdprCompliance,
  ComplianceResponse,
  ComplianceCheckRequest,
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
  const [euAiActData, setEuAiActData] = useState<ComplianceResponse | null>(null);
  const [gdprData, setGdprData] = useState<ComplianceResponse | null>(null);
  const [euAiActLoading, setEuAiActLoading] = useState(false);
  const [gdprLoading, setGdprLoading] = useState(false);
  const [euAiActError, setEuAiActError] = useState<string | null>(null);
  const [gdprError, setGdprError] = useState<string | null>(null);

  useEffect(() => {
    if (!accessToken || !logEntry.request_id) return;

    const payload: ComplianceCheckRequest = {
      request_id: logEntry.request_id,
      user_id: logEntry.user,
      model: logEntry.model,
      timestamp: logEntry.startTime,
      guardrail_information: logEntry.metadata?.guardrail_information,
    };

    setEuAiActLoading(true);
    setEuAiActError(null);
    checkEuAiActCompliance(accessToken, payload)
      .then(setEuAiActData)
      .catch((err) => setEuAiActError(err.message || "Failed to check EU AI Act compliance"))
      .finally(() => setEuAiActLoading(false));

    setGdprLoading(true);
    setGdprError(null);
    checkGdprCompliance(accessToken, payload)
      .then(setGdprData)
      .catch((err) => setGdprError(err.message || "Failed to check GDPR compliance"))
      .finally(() => setGdprLoading(false));
  }, [accessToken, logEntry]);

  return (
    <div>
      <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
        Regulatory Compliance
      </h4>
      <div className="space-y-3">
        <ComplianceCard
          title="EU AI Act"
          data={euAiActData}
          loading={euAiActLoading}
          error={euAiActError}
        />
        <ComplianceCard
          title="GDPR"
          data={gdprData}
          loading={gdprLoading}
          error={gdprError}
        />
      </div>
    </div>
  );
};

export default CompliancePanel;
