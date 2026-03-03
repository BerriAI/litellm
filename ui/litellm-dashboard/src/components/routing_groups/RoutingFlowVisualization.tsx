"use client";

import React from "react";

interface Deployment {
  model_id: string;
  model_name: string;
  provider: string;
  weight?: number;
  priority?: number;
}

interface RoutingFlowVisualizationProps {
  strategy: string;
  deployments: Deployment[];
}

const DEPLOY_COLORS = ["#3b82f6", "#6366f1", "#0ea5e9"];

const VerticalLine: React.FC<{ height?: number }> = ({ height = 28 }) => (
  <div className="flex justify-center" style={{ height }}>
    <div style={{ width: 2, height: "100%", backgroundColor: "#e5e7eb" }} />
  </div>
);

const StatusDot: React.FC = () => (
  <div style={{ width: 8, height: 8, borderRadius: "50%", backgroundColor: "#22c55e", flexShrink: 0 }} />
);

const FailConnector: React.FC = () => (
  <div className="flex flex-col items-center" style={{ margin: "4px 0" }}>
    <VerticalLine height={12} />
    <div className="flex items-center gap-1.5" style={{ padding: "3px 10px", backgroundColor: "#fef2f2", borderRadius: 12 }}>
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" /><path d="M15 9l-6 6M9 9l6 6" />
      </svg>
      <span style={{ fontSize: 10, fontWeight: 700, color: "#ef4444", letterSpacing: "0.04em" }}>FAIL → TRY NEXT</span>
    </div>
    <VerticalLine height={12} />
  </div>
);

const TriggerCard: React.FC<{ subtitle: string }> = ({ subtitle }) => (
  <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: "16px 20px", backgroundColor: "#fff", width: "100%", maxWidth: 340 }}>
    <div style={{ fontSize: 10, fontWeight: 700, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>TRIGGER</div>
    <div className="flex items-center gap-2">
      <div style={{ width: 28, height: 28, borderRadius: "50%", backgroundColor: "#f3f4f6", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <svg width="11" height="11" viewBox="0 0 24 24" fill="#6b7280" stroke="none"><polygon points="6,3 20,12 6,21" /></svg>
      </div>
      <div>
        <div style={{ fontSize: 15, fontWeight: 600, color: "#111827" }}>Incoming Request</div>
        <div style={{ fontSize: 12, color: "#9ca3af" }}>{subtitle}</div>
      </div>
    </div>
  </div>
);

const OutputCard: React.FC<{ subtitle: string }> = ({ subtitle }) => (
  <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: "16px 20px", backgroundColor: "#fff", width: "100%", maxWidth: 340 }}>
    <div style={{ fontSize: 10, fontWeight: 700, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>OUTPUT</div>
    <div className="flex items-center gap-2">
      <div style={{ width: 28, height: 28, borderRadius: "50%", backgroundColor: "#f0fdf4", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" /><path d="M9 12l2 2 4-4" />
        </svg>
      </div>
      <div>
        <div style={{ fontSize: 15, fontWeight: 600, color: "#111827" }}>Response</div>
        <div style={{ fontSize: 12, color: "#9ca3af" }}>{subtitle}</div>
      </div>
    </div>
  </div>
);

const ModelNodeCard: React.FC<{ label: string; name: string; subName: string }> = ({ label, name, subName }) => (
  <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: "14px 18px", backgroundColor: "#fff", width: "100%", maxWidth: 340, position: "relative" }}>
    <div style={{ position: "absolute", top: 12, right: 14 }}><StatusDot /></div>
    <div style={{ fontSize: 10, fontWeight: 700, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>{label}</div>
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2.5">
        <div style={{ width: 32, height: 32, borderRadius: "50%", backgroundColor: "#eff6ff", border: "1.5px solid #bfdbfe", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, fontSize: 13, fontWeight: 700, color: "#3b82f6" }}>
          {name.charAt(0).toUpperCase()}
        </div>
        <div>
          <div style={{ fontSize: 14, fontWeight: 600, color: "#111827" }}>{name}</div>
          <div style={{ fontSize: 12, color: "#9ca3af" }}>{subName}</div>
        </div>
      </div>
    </div>
  </div>
);

const FailoverFlow: React.FC<{ deployments: Deployment[] }> = ({ deployments }) => (
  <div className="flex flex-col items-center">
    <TriggerCard subtitle="→ first available" />
    {deployments.map((dep, i) => {
      const label = i === 0 ? "PRIMARY" : `FALLBACK-${i}`;
      const shortName = dep.model_name.split("/").pop() || dep.model_name;
      return (
        <React.Fragment key={dep.model_id + "-" + i}>
          {i === 0 ? <VerticalLine /> : <FailConnector />}
          <ModelNodeCard label={label} name={shortName} subName={dep.model_name} />
        </React.Fragment>
      );
    })}
    <VerticalLine />
    <OutputCard subtitle="first successful result" />
  </div>
);

const WeightedFlow: React.FC<{ deployments: Deployment[] }> = ({ deployments }) => {
  const totalWeight = deployments.reduce((s, d) => s + (d.weight ?? 1), 0);

  return (
    <div className="flex flex-col items-center">
      <TriggerCard subtitle="→ weighted distribution" />
      <VerticalLine />

      <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: "14px 18px", backgroundColor: "#fff", width: "100%", maxWidth: 400 }}>
        <div className="flex items-center gap-2" style={{ marginBottom: 10 }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#6b7280" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M16 3h5v5M4 20L21 3M21 16v5h-5M15 15l6 6M4 4l5 5" />
          </svg>
          <span style={{ fontSize: 10, fontWeight: 700, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em" }}>LOAD BALANCER</span>
        </div>

        <div style={{ width: "100%", height: 6, borderRadius: 3, overflow: "hidden", display: "flex", backgroundColor: "#e5e7eb", marginBottom: 12 }}>
          {deployments.map((dep, i) => {
            const pct = totalWeight > 0 ? ((dep.weight ?? 1) / totalWeight) * 100 : 0;
            return <div key={dep.model_id + "-bar-" + i} style={{ width: `${pct}%`, height: "100%", backgroundColor: DEPLOY_COLORS[i % DEPLOY_COLORS.length] }} />;
          })}
        </div>

        <div className="flex flex-col gap-2">
          {deployments.map((dep, i) => {
            const pct = totalWeight > 0 ? Math.round(((dep.weight ?? 1) / totalWeight) * 100) : 0;
            const shortName = dep.model_name.split("/").pop() || dep.model_name;
            const color = DEPLOY_COLORS[i % DEPLOY_COLORS.length];
            return (
              <div
                key={dep.model_id + "-row-" + i}
                className="flex items-center justify-between"
                style={{ padding: "8px 12px", backgroundColor: "#f9fafb", borderRadius: 8, border: "1px solid #f3f4f6" }}
              >
                <div className="flex items-center gap-2.5">
                  <div style={{ width: 28, height: 28, borderRadius: "50%", backgroundColor: color + "18", border: `1.5px solid ${color}44`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, fontSize: 11, fontWeight: 700, color }}>
                    {shortName.charAt(0).toUpperCase()}
                  </div>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: "#111827" }}>{shortName}</div>
                    <div style={{ fontSize: 10, color: "#9ca3af" }}>{dep.model_name}</div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span style={{ fontSize: 13, fontWeight: 700, color }}>{pct}%</span>
                  <div style={{ width: 8, height: 8, borderRadius: "50%", backgroundColor: "#22c55e", flexShrink: 0 }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <VerticalLine />
      <OutputCard subtitle="weighted distribution" />
    </div>
  );
};

const STRATEGY_META: Record<string, { label: string; subtitle: string; output: string; color: string }> = {
  "simple-shuffle": { label: "SHUFFLE ROUTER", subtitle: "→ random selection", output: "random deployment selected", color: "#6b7280" },
  "latency-based-routing": { label: "LATENCY ROUTER", subtitle: "→ latency-based", output: "fastest model wins", color: "#6366f1" },
  "usage-based-routing-v2": { label: "USAGE ROUTER", subtitle: "→ usage-based", output: "least busy model selected", color: "#6366f1" },
  "cost-based-routing": { label: "COST ROUTER", subtitle: "→ cost-based", output: "cheapest deployment selected", color: "#059669" },
  "least-busy": { label: "LEAST BUSY ROUTER", subtitle: "→ least busy", output: "deployment with fewest requests", color: "#6366f1" },
};

const COMPLEXITY_TIER_LABELS = [
  { tier: "SIMPLE", label: "Simple", threshold: "< 0.15" },
  { tier: "MEDIUM", label: "Medium", threshold: "0.15 – 0.35" },
  { tier: "COMPLEX", label: "Complex", threshold: "0.35 – 0.60" },
  { tier: "REASONING", label: "Reasoning", threshold: "> 0.60" },
];

const ComplexityFlow: React.FC<{ deployments: Deployment[] }> = ({ deployments }) => (
  <div className="flex flex-col items-center">
    <TriggerCard subtitle="→ auto complexity scoring" />
    <VerticalLine />

    <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: "14px 18px", backgroundColor: "#fff", width: "100%", maxWidth: 400 }}>
      <div className="flex items-center gap-2" style={{ marginBottom: 4 }}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#6b7280" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
        </svg>
        <span style={{ fontSize: 10, fontWeight: 700, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.06em" }}>AUTO ROUTER</span>
      </div>
      <div style={{ fontSize: 10, color: "#9ca3af", marginBottom: 12 }}>Rule-based scoring &middot; no API calls &middot; &lt;1ms</div>

      <div className="flex flex-col gap-2">
        {COMPLEXITY_TIER_LABELS.map((t, i) => {
          const dep = deployments[i];
          const shortName = dep ? (dep.model_name.split("/").pop() || dep.model_name) : null;
          return (
            <div key={t.tier} style={{ padding: "10px 12px", backgroundColor: "#f9fafb", borderRadius: 8, border: "1px solid #f3f4f6", borderLeft: "3px solid #d1d5db" }}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span style={{ fontSize: 12, fontWeight: 700, color: "#374151" }}>{t.label}</span>
                  <span style={{ fontSize: 10, color: "#9ca3af" }}>score {t.threshold}</span>
                </div>
              </div>
              {shortName ? (
                <div className="flex items-center gap-2 mt-1.5">
                  <div style={{ width: 22, height: 22, borderRadius: "50%", backgroundColor: "#eff6ff", border: "1.5px solid #bfdbfe", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, fontSize: 10, fontWeight: 700, color: "#3b82f6" }}>
                    {shortName.charAt(0).toUpperCase()}
                  </div>
                  <span style={{ fontSize: 12, fontWeight: 600, color: "#111827" }}>{shortName}</span>
                  <StatusDot />
                </div>
              ) : (
                <div style={{ fontSize: 11, color: "#9ca3af", fontStyle: "italic", marginTop: 4 }}>No model assigned</div>
              )}
            </div>
          );
        })}
      </div>
    </div>

    <VerticalLine />
    <OutputCard subtitle="complexity-matched model responds" />
  </div>
);

const SmartRoutingFlow: React.FC<{ deployments: Deployment[]; strategy: string }> = ({ deployments, strategy }) => {
  const meta = STRATEGY_META[strategy] ?? STRATEGY_META["simple-shuffle"];
  return (
    <div className="flex flex-col items-center">
      <TriggerCard subtitle={meta.subtitle} />
      <VerticalLine />
      <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: "14px 18px", backgroundColor: "#fff", width: "100%", maxWidth: 340 }}>
        <div style={{ fontSize: 10, fontWeight: 700, color: meta.color, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
          {meta.label}
        </div>
        <div className="flex flex-col gap-2">
          {deployments.map((dep, i) => {
            const shortName = dep.model_name.split("/").pop() || dep.model_name;
            return (
              <div key={dep.model_id + "-" + i} className="flex items-center justify-between" style={{ padding: "8px 12px", backgroundColor: "#f9fafb", borderRadius: 8, border: "1px solid #f3f4f6" }}>
                <div className="flex items-center gap-2">
                  <div style={{ width: 28, height: 28, borderRadius: "50%", backgroundColor: "#eff6ff", border: "1.5px solid #bfdbfe", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, fontSize: 11, fontWeight: 700, color: "#3b82f6" }}>
                    {shortName.charAt(0).toUpperCase()}
                  </div>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: "#111827" }}>{shortName}</div>
                    <div style={{ fontSize: 10, color: "#9ca3af" }}>{dep.model_name}</div>
                  </div>
                </div>
                <StatusDot />
              </div>
            );
          })}
        </div>
      </div>
      <VerticalLine />
      <OutputCard subtitle={meta.output} />
    </div>
  );
};

const EmptyFlow: React.FC = () => (
  <div className="flex flex-col items-center">
    <TriggerCard subtitle="→ select a strategy" />
    <VerticalLine />
    <div style={{ border: "1px dashed #d1d5db", borderRadius: 12, padding: "28px 20px", width: "100%", maxWidth: 340, textAlign: "center" }}>
      <span style={{ fontSize: 13, color: "#9ca3af" }}>Add deployments to see the flow</span>
    </div>
    <VerticalLine />
    <OutputCard subtitle="waiting for configuration" />
  </div>
);

export default function RoutingFlowVisualization({ strategy, deployments }: RoutingFlowVisualizationProps) {
  if (strategy === "complexity-router") return <ComplexityFlow deployments={deployments} />;
  if (deployments.length === 0) return <EmptyFlow />;
  switch (strategy) {
    case "priority-failover": return <FailoverFlow deployments={deployments} />;
    case "weighted": return <WeightedFlow deployments={deployments} />;
    case "simple-shuffle":
    case "latency-based-routing":
    case "usage-based-routing-v2":
    case "cost-based-routing":
    case "least-busy":
      return <SmartRoutingFlow deployments={deployments} strategy={strategy} />;
    default: return <SmartRoutingFlow deployments={deployments} strategy={strategy} />;
  }
}

export { DEPLOY_COLORS };
