import { Card, Title } from "@tremor/react";
import React from "react";

type Severity = "CRITICAL" | "WARNING" | "INFO";

interface Alert {
  id: string;
  severity: Severity;
  timestamp: string;
  agent: string;
  description: string;
}

const alerts: Alert[] = [
  {
    id: "1",
    severity: "CRITICAL",
    timestamp: "2 mins ago",
    agent: "order-processor-v3",
    description: "Hit max iterations 12 times in last hour",
  },
  {
    id: "2",
    severity: "WARNING",
    timestamp: "15 mins ago",
    agent: "support-bot-v2",
    description: "Showing 34% purpose drift from baseline",
  },
  {
    id: "3",
    severity: "CRITICAL",
    timestamp: "1 hour ago",
    agent: "data-analyst",
    description: "Abnormal data volume: uploaded 2.3GB in 10 min",
  },
  {
    id: "4",
    severity: "WARNING",
    timestamp: "2 hours ago",
    agent: "code-reviewer",
    description: "Response payload exceeded 50MB threshold",
  },
  {
    id: "5",
    severity: "INFO",
    timestamp: "3 hours ago",
    agent: "research-bot-v1",
    description: "Kill switch activated by admin@company.com",
  },
];

const severityConfig: Record<Severity, { border: string; bg: string; text: string }> = {
  CRITICAL: { border: "border-red-500", bg: "bg-red-500", text: "text-white" },
  WARNING: { border: "border-amber-500", bg: "bg-amber-500", text: "text-white" },
  INFO: { border: "border-blue-500", bg: "bg-blue-500", text: "text-white" },
};

export const ActiveAlerts: React.FC = () => {
  const criticalCount = alerts.filter((a) => a.severity === "CRITICAL").length;

  return (
    <Card className="bg-white border border-gray-200 h-full flex flex-col">
      <div className="flex items-center mb-6">
        <Title className="text-base font-semibold text-gray-900 mr-3">Active Alerts</Title>
        <span className="bg-red-500 text-white text-xs font-bold px-2 py-0.5 rounded-full">
          {criticalCount}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto pr-2 -mr-2 space-y-3">
        {alerts.map((alert) => {
          const config = severityConfig[alert.severity];
          return (
            <div
              key={alert.id}
              className={`border-l-4 ${config.border} bg-white border-y border-r border-gray-100 rounded-r-md p-3 shadow-sm`}
            >
              <div className="flex justify-between items-start mb-1.5">
                <span
                  className={`text-[10px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wider ${config.bg} ${config.text}`}
                >
                  {alert.severity}
                </span>
                <span className="text-xs text-gray-400">{alert.timestamp}</span>
              </div>
              <div className="mb-1">
                <span className="font-mono text-xs bg-gray-100 text-gray-700 px-1.5 py-0.5 rounded border border-gray-200">
                  {alert.agent}
                </span>
              </div>
              <p className="text-sm text-gray-700 leading-snug">{alert.description}</p>
            </div>
          );
        })}
      </div>
    </Card>
  );
};
