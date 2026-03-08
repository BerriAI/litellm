import React from "react";
import { Tooltip } from "antd";

interface CooldownBadgeProps {
  status: "healthy" | "cooldown";
  remainingSeconds?: number;
  cooldownTime?: number;
  exception?: string;
  statusCode?: string;
}

export const CooldownBadge: React.FC<CooldownBadgeProps> = ({
  status,
  remainingSeconds,
  cooldownTime,
  exception,
  statusCode,
}) => {
  if (status === "healthy") {
    return (
      <div className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-green-50 text-green-600">
        <span className="w-1.5 h-1.5 rounded-full bg-green-500 mr-1.5" />
        Healthy
      </div>
    );
  }

  const tooltipContent = [
    statusCode && `Status: ${statusCode}`,
    cooldownTime && `Cooldown: ${cooldownTime}s`,
    exception && `Error: ${exception}`,
  ]
    .filter(Boolean)
    .join(" | ");

  return (
    <Tooltip title={tooltipContent}>
      <div className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-red-50 text-red-600 cursor-help">
        <span className="w-1.5 h-1.5 rounded-full bg-red-500 mr-1.5" />
        Cooldown{remainingSeconds ? ` (${Math.ceil(remainingSeconds)}s)` : ""}
      </div>
    </Tooltip>
  );
};
