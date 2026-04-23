import React, { useState, useEffect } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { CheckCircle2, ChevronDown, Loader2, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";
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
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    metadata?: Record<string, any>;
  };
}

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
    <div className="border border-border rounded-lg bg-background">
      <div
        className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-muted transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          {loading ? (
            <Loader2 className="h-4 w-4 animate-spin text-indigo-500" />
          ) : error ? (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-muted-foreground text-sm">--</span>
                </TooltipTrigger>
                <TooltipContent>{error}</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          ) : data?.compliant ? (
            <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
          ) : (
            <XCircle className="h-4 w-4 text-destructive" />
          )}
          <span className="font-medium text-sm text-foreground">{title}</span>
        </div>
        <div className="flex items-center gap-2">
          {!loading && !error && data && (
            <span
              className={cn(
                "px-2 py-0.5 rounded text-[11px] font-semibold uppercase border",
                data.compliant
                  ? "bg-emerald-100 text-emerald-700 border-emerald-200 dark:bg-emerald-950 dark:text-emerald-300 dark:border-emerald-900"
                  : "bg-red-100 text-red-700 border-red-200 dark:bg-red-950 dark:text-red-300 dark:border-red-900",
              )}
            >
              {data.compliant ? "COMPLIANT" : "NON-COMPLIANT"}
            </span>
          )}
          {error && (
            <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-muted text-muted-foreground border border-border">
              UNAVAILABLE
            </span>
          )}
          <ChevronDown
            className={cn(
              "h-4 w-4 text-muted-foreground transition-transform",
              expanded && "rotate-180",
            )}
          />
        </div>
      </div>

      {expanded && (
        <div className="border-t border-border px-4 py-3">
          {loading && (
            <p className="text-sm text-muted-foreground">
              Checking compliance...
            </p>
          )}
          {error && <p className="text-sm text-destructive">{error}</p>}
          {data && (
            <div className="space-y-2">
              {data.checks.map((check, idx) => (
                <div key={idx} className="flex items-start gap-2">
                  <div className="flex-shrink-0 mt-0.5">
                    {check.passed ? (
                      <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                    ) : (
                      <XCircle className="h-4 w-4 text-destructive" />
                    )}
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-foreground">
                        {check.check_name}
                      </span>
                      <span className="text-[10px] font-mono text-muted-foreground">
                        {check.article}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {check.detail}
                    </p>
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

const CompliancePanel: React.FC<CompliancePanelProps> = ({
  accessToken,
  logEntry,
}) => {
  const [euAiActData, setEuAiActData] = useState<ComplianceResponse | null>(
    null,
  );
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
      .catch((err) =>
        setEuAiActError(err.message || "Failed to check EU AI Act compliance"),
      )
      .finally(() => setEuAiActLoading(false));

    setGdprLoading(true);
    setGdprError(null);
    checkGdprCompliance(accessToken, payload)
      .then(setGdprData)
      .catch((err) =>
        setGdprError(err.message || "Failed to check GDPR compliance"),
      )
      .finally(() => setGdprLoading(false));
  }, [accessToken, logEntry]);

  return (
    <div>
      <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-4">
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
