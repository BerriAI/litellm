"use client";
import React from "react";
import AnimatedStream from "./AnimatedStream";
import DeploymentCard from "./DeploymentCard";

const PROVIDER_COLORS: Record<string, string> = {
  nebius: "#14b8a6",
  fireworks_ai: "#f97316",
  azure: "#8b5cf6",
  openai: "#10b981",
  anthropic: "#d97706",
  google: "#3b82f6",
  aws: "#ef4444",
};
const getProviderColor = (p: string) => PROVIDER_COLORS[p.toLowerCase()] ?? "#6b7280";

interface Deployment {
  deployment_id: string;
  provider: string;
  display_name: string;
  weight?: number;
  avg_latency_ms: number;
  percent_of_total: number;
  request_count: number;
}

interface WeightedRoundRobinFlowProps {
  deployments: Deployment[];
  routingGroupName: string;
}

const CARD_HEIGHT = 76;
const CARD_GAP = 28;
const CARD_WIDTH = 220;
const LEFT_PADDING = 40;
const HUB_RADIUS = 28;

export default function WeightedRoundRobinFlow({ deployments }: WeightedRoundRobinFlowProps) {
  // Sort by weight desc (highest traffic first = top position)
  const sorted = [...deployments].sort((a, b) => (b.weight ?? 0) - (a.weight ?? 0));
  const totalHeight = sorted.length * CARD_HEIGHT + (sorted.length - 1) * CARD_GAP + 80;
  const svgWidth = 600;

  const hubX = svgWidth * 0.32;
  const hubY = totalHeight / 2;

  const cardX = svgWidth - CARD_WIDTH - LEFT_PADDING;

  const blockHeight = sorted.length * CARD_HEIGHT + (sorted.length - 1) * CARD_GAP;
  const blockTop = (totalHeight - blockHeight) / 2;
  const cardCenterYs = sorted.map((_, i) => blockTop + i * (CARD_HEIGHT + CARD_GAP) + CARD_HEIGHT / 2);

  // Max thickness 14px for 100%, min 2px
  const maxPct = Math.max(...sorted.map((d) => d.percent_of_total), 1);
  const getThickness = (pct: number) => Math.max(2, Math.round((pct / maxPct) * 14));

  // Bezier path from hub to each card center
  const getPath = (targetY: number) => {
    const cp1x = hubX + 80;
    const cp2x = cardX - 60;
    return `M ${hubX} ${hubY} C ${cp1x} ${hubY} ${cp2x} ${targetY} ${cardX} ${targetY}`;
  };

  return (
    <div className="relative w-full overflow-hidden rounded-2xl bg-[#0f1117]" style={{ minHeight: totalHeight + 40 }}>
      <svg
        className="absolute inset-0"
        width="100%"
        height={totalHeight + 40}
        viewBox={`0 0 ${svgWidth} ${totalHeight + 40}`}
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Client → hub */}
        <AnimatedStream
          path={`M 20 ${hubY} L ${hubX - HUB_RADIUS} ${hubY}`}
          color="#14b8a6"
          thickness={6}
        />

        {/* Hub */}
        <circle cx={hubX} cy={hubY} r={HUB_RADIUS} fill="#1a2535" stroke="#14b8a6" strokeWidth={2} strokeOpacity={0.6} />
        <text x={hubX} y={hubY + 4} textAnchor="middle" fill="#14b8a6" fontSize={8} fontWeight="600" fontFamily="monospace">
          LITELLM
        </text>

        {/* Fan-out streams */}
        {sorted.map((dep, i) => (
          <AnimatedStream
            key={dep.deployment_id}
            path={getPath(cardCenterYs[i])}
            color={getProviderColor(dep.provider)}
            thickness={getThickness(dep.percent_of_total)}
          />
        ))}
      </svg>

      {/* "client" label */}
      <div className="absolute text-xs font-mono text-gray-500" style={{ left: 24, top: hubY + 40 - 8 }}>
        client
      </div>

      {/* Cards */}
      <div className="absolute" style={{ right: LEFT_PADDING, top: 20 }}>
        <div className="flex flex-col" style={{ gap: CARD_GAP }}>
          {sorted.map((dep) => (
            <DeploymentCard
              key={dep.deployment_id}
              provider={dep.provider}
              displayName={dep.display_name}
              providerColor={getProviderColor(dep.provider)}
              label={`Weight ${dep.weight ?? Math.round(dep.percent_of_total)}%`}
              avgLatencyMs={dep.avg_latency_ms}
              percentage={dep.percent_of_total}
              requestCount={dep.request_count}
              width={CARD_WIDTH}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
