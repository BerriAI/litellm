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
  priority?: number;
  avg_latency_ms: number;
  percent_of_total: number;
  request_count: number;
}

interface OrderedFallbackFlowProps {
  deployments: Deployment[];
  routingGroupName: string;
}

const CARD_HEIGHT = 76;
const CARD_GAP = 48;        // space between cards (for the "fail" arrow)
const CARD_WIDTH = 220;
const LEFT_PADDING = 40;
const HUB_RADIUS = 28;

export default function OrderedFallbackFlow({ deployments, routingGroupName }: OrderedFallbackFlowProps) {
  const sorted = [...deployments].sort((a, b) => (a.priority ?? 999) - (b.priority ?? 999));
  const totalHeight = sorted.length * CARD_HEIGHT + (sorted.length - 1) * CARD_GAP + 40;
  const svgWidth = 600;

  // Hub sits vertically centered, 1/3 from left
  const hubX = svgWidth * 0.32;
  const hubY = totalHeight / 2;

  // Cards start at x = svgWidth - CARD_WIDTH - LEFT_PADDING
  const cardX = svgWidth - CARD_WIDTH - LEFT_PADDING;

  // Y positions for each card (centered block)
  const blockHeight = sorted.length * CARD_HEIGHT + (sorted.length - 1) * CARD_GAP;
  const blockTop = (totalHeight - blockHeight) / 2;
  const cardYs = sorted.map((_, i) => blockTop + i * (CARD_HEIGHT + CARD_GAP));
  const cardCenterYs = cardYs.map((y) => y + CARD_HEIGHT / 2);

  // Primary stream: from hub to first card center
  const primaryPath = `M ${hubX} ${hubY} L ${cardX} ${cardCenterYs[0]}`;

  // Fallback dashed paths between cards
  const fallbackPaths = sorted.slice(0, -1).map((_, i) => {
    const x = cardX + CARD_WIDTH / 2;
    const fromY = cardYs[i] + CARD_HEIGHT;
    const toY = cardYs[i + 1];
    return { path: `M ${x} ${fromY} L ${x} ${toY}`, midY: (fromY + toY) / 2 };
  });

  return (
    <div className="relative w-full overflow-hidden rounded-2xl bg-[#0f1117]" style={{ minHeight: totalHeight + 40 }}>
      {/* SVG layer */}
      <svg
        className="absolute inset-0"
        width="100%"
        height={totalHeight + 40}
        viewBox={`0 0 ${svgWidth} ${totalHeight + 40}`}
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Client → hub stream */}
        <AnimatedStream
          path={`M 20 ${hubY} L ${hubX - HUB_RADIUS} ${hubY}`}
          color="#14b8a6"
          thickness={6}
        />

        {/* Hub circle */}
        <circle cx={hubX} cy={hubY} r={HUB_RADIUS} fill="#1a2535" stroke="#14b8a6" strokeWidth={2} strokeOpacity={0.6} />
        <text x={hubX} y={hubY + 4} textAnchor="middle" fill="#14b8a6" fontSize={8} fontWeight="600" fontFamily="monospace">
          LITELLM
        </text>

        {/* Primary stream: hub → first card */}
        <AnimatedStream path={primaryPath} color="#14b8a6" thickness={10} />

        {/* Fallback dashed arrows between cards */}
        {fallbackPaths.map(({ path, midY }, i) => (
          <g key={i}>
            <path d={path} fill="none" stroke="#ef4444" strokeWidth={1.5} strokeDasharray="5 4" strokeOpacity={0.7} />
            <text x={cardX + CARD_WIDTH / 2 + 6} y={midY + 4} fill="#ef4444" fontSize={9} fontFamily="monospace" opacity={0.8}>
              fail →
            </text>
          </g>
        ))}
      </svg>

      {/* "client" label */}
      <div className="absolute text-xs font-mono text-gray-500" style={{ left: 24, top: hubY + 40 - 8 }}>
        client
      </div>

      {/* Deployment cards */}
      <div className="absolute" style={{ right: LEFT_PADDING, top: 20 }}>
        <div className="flex flex-col" style={{ gap: CARD_GAP }}>
          {sorted.map((dep, i) => (
            <DeploymentCard
              key={dep.deployment_id}
              provider={dep.provider}
              displayName={dep.display_name}
              providerColor={getProviderColor(dep.provider)}
              label={`Priority ${dep.priority ?? i + 1}`}
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
