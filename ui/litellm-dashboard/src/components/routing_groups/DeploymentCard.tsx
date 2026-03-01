"use client";
import React from "react";

interface DeploymentCardProps {
  provider: string;
  displayName: string;
  providerColor: string;
  label: string;          // "Priority 1" or "Weight 83%"
  avgLatencyMs: number;
  percentage: number;     // 0-100
  requestCount: number;
  width?: number;         // px, default 220
}

export default function DeploymentCard({
  provider,
  displayName,
  providerColor,
  label,
  avgLatencyMs,
  percentage,
  requestCount,
  width = 220,
}: DeploymentCardProps) {
  const initial = displayName.charAt(0).toUpperCase();
  return (
    <div
      style={{ width, borderColor: providerColor + "33" }}
      className="flex items-center gap-3 rounded-xl border bg-[#1f2937] px-4 py-3"
    >
      {/* Provider icon */}
      <div
        style={{ backgroundColor: providerColor + "33", color: providerColor }}
        className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg text-sm font-bold"
      >
        {initial}
      </div>

      {/* Name + label */}
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-semibold text-white">{displayName}</div>
        <div className="text-xs text-gray-400">
          {label} · {avgLatencyMs > 0 ? `${Math.round(avgLatencyMs)}ms avg` : "—"}
        </div>
      </div>

      {/* Percentage + req count */}
      <div className="flex flex-shrink-0 flex-col items-end">
        <span style={{ color: providerColor }} className="text-lg font-bold leading-none">
          {Math.round(percentage)}%
        </span>
        <span className="text-xs text-gray-500">{requestCount} req</span>
      </div>
    </div>
  );
}
