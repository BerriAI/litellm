import React from "react";
import { Badge } from "@tremor/react";
import { Tooltip } from "antd";
import {
  ArrowDownOutlined,
  ArrowUpOutlined,
  MinusOutlined,
  WarningOutlined,
} from "@ant-design/icons";

export interface AgentDriftSummary {
  status: "healthy" | "warning" | "drifting" | "no_evals";
  avgScore: number | null;
  trend: "up" | "down" | "flat";
}

function hashCode(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

export function getMockDriftSummary(agentId: string): AgentDriftSummary {
  const seed = hashCode(agentId);
  const bucket = seed % 10;

  if (bucket < 3) {
    const avg = 0.85 + (seed % 12) / 100;
    return { status: "healthy", avgScore: +avg.toFixed(2), trend: "flat" };
  }
  if (bucket < 6) {
    const avg = 0.55 + (seed % 15) / 100;
    return { status: "warning", avgScore: +avg.toFixed(2), trend: "down" };
  }
  if (bucket < 8) {
    const avg = 0.30 + (seed % 18) / 100;
    return { status: "drifting", avgScore: +avg.toFixed(2), trend: "down" };
  }
  return { status: "no_evals", avgScore: null, trend: "flat" };
}

interface DriftBadgeCellProps {
  drift: AgentDriftSummary;
  onClick: () => void;
}

export function DriftBadgeCell({ drift, onClick }: DriftBadgeCellProps) {
  if (drift.status === "no_evals") {
    return (
      <Badge size="xs" color="gray">
        No evals
      </Badge>
    );
  }

  let badgeColor = "green";
  if (drift.status === "warning") badgeColor = "yellow";
  if (drift.status === "drifting") badgeColor = "red";

  let trendIcon = null;
  if (drift.trend === "down") {
    trendIcon = <ArrowDownOutlined className="text-red-500 text-xs" />;
  } else if (drift.trend === "up") {
    trendIcon = <ArrowUpOutlined className="text-green-500 text-xs" />;
  } else {
    trendIcon = <MinusOutlined className="text-gray-400 text-xs" />;
  }

  const scoreText = drift.avgScore !== null ? drift.avgScore.toFixed(2) : "--";
  const tooltipText = "Avg eval score (1.0 = best). Click to see details.";

  return (
    <Tooltip title={tooltipText}>
      <button
        type="button"
        className="flex items-center gap-1.5 cursor-pointer bg-transparent border-none p-0"
        onClick={onClick}
      >
        <Badge size="xs" color={badgeColor as any}>
          {drift.status === "drifting" && (
            <WarningOutlined className="mr-1" />
          )}
          {scoreText}
        </Badge>
        {trendIcon}
      </button>
    </Tooltip>
  );
}
