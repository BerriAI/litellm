"use client";
import React from "react";

interface StatsBarProps {
  requests: number;
  successRate: number;    // 0-100
  avgLatencyMs: number;
  fallbackCount: number;
}

export default function StatsBar({ requests, successRate, avgLatencyMs, fallbackCount }: StatsBarProps) {
  const stats = [
    { label: "REQUESTS", value: requests.toLocaleString() },
    { label: "SUCCESS", value: `${Math.round(successRate)}%` },
    { label: "AVG LATENCY", value: avgLatencyMs > 0 ? `${Math.round(avgLatencyMs)}ms` : "—" },
    { label: "FALLBACKS", value: fallbackCount.toLocaleString() },
  ];

  return (
    <div className="grid grid-cols-4 divide-x divide-gray-700 rounded-xl border border-gray-700 bg-[#1a1f2e] px-0">
      {stats.map(({ label, value }) => (
        <div key={label} className="flex flex-col items-start gap-1 px-6 py-4">
          <span className="text-xs font-medium uppercase tracking-widest text-gray-500">{label}</span>
          <span className="text-2xl font-bold text-white">{value}</span>
        </div>
      ))}
    </div>
  );
}
