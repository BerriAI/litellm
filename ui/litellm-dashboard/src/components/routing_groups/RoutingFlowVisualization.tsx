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

const DEPLOY_COLORS = ["#10b981", "#f59e0b", "#3b82f6", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"];

const VerticalLine: React.FC<{ height?: number }> = ({ height = 28 }) => (
  <div className="flex justify-center" style={{ height }}>
    <div style={{ width: 2, height: "100%", backgroundColor: "#e5e7eb" }} />
  </div>
);

const StatusDot: React.FC = () => (
  <div style={{ width: 8, height: 8, borderRadius: "50%", backgroundColor: "#22c55e", flexShrink: 0 }} />
);

const KillTag: React.FC = () => (
  <span style={{ fontSize: 10, fontWeight: 600, color: "#ef4444", backgroundColor: "#fef2f2", padding: "2px 8px", borderRadius: 4, flexShrink: 0 }}>
    kill
  </span>
);

const FailConnector: React.FC = () => (
  <div className="flex flex-col items-center" style={{ margin: "4px 0" }}>
    <VerticalLine height={12} />
    <div className="flex items-center gap-1.5" style={{ padding: "3px 10px", backgroundColor: "#fef2f2", borderRadius: 12 }}>
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <path d="M15 9l-6 6M9 9l6 6" />
      </svg>
      <span style={{ fontSize: 10, fontWeight: 700, color: "#ef4444", letterSpacing: "0.04em" }}>
        FAIL → TRY NEXT
      </span>
    </div>
    <VerticalLine height={12} />
  </div>
);

const ProviderIcon: React.FC<{ color: string; name: string }> = ({ color, name }) => (
  <div
    style={{
      width: 32,
      height: 32,
      borderRadius: "50%",
      backgroundColor: color + "18",
      border: `1.5px solid ${color}40`,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      flexShrink: 0,
      fontSize: 13,
      fontWeight: 700,
      color: color,
    }}
  >
    {name.charAt(0).toUpperCase()}
  </div>
);

const TriggerCard: React.FC<{ subtitle: string }> = ({ subtitle }) => (
  <div
    style={{
      border: "1px solid #e5e7eb",
      borderRadius: 12,
      padding: "16px 20px",
      backgroundColor: "#fff",
      width: "100%",
      maxWidth: 340,
    }}
  >
    <div style={{ fontSize: 10, fontWeight: 700, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>
      TRIGGER
    </div>
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
  <div
    style={{
      border: "1px solid #e5e7eb",
      borderRadius: 12,
      padding: "16px 20px",
      backgroundColor: "#fff",
      width: "100%",
      maxWidth: 340,
    }}
  >
    <div style={{ fontSize: 10, fontWeight: 700, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>
      OUTPUT
    </div>
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

const ModelCard: React.FC<{
  label: string;
  name: string;
  subName: string;
  color: string;
}> = ({ label, name, subName, color }) => (
  <div
    style={{
      border: "1px solid #e5e7eb",
      borderRadius: 12,
      padding: "14px 18px",
      backgroundColor: "#fff",
      width: "100%",
      maxWidth: 340,
      position: "relative",
    }}
  >
    <div style={{ position: "absolute", top: 12, right: 14 }}><StatusDot /></div>
    <div style={{ fontSize: 10, fontWeight: 700, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>
      {label}
    </div>
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2.5">
        <ProviderIcon color={color} name={name} />
        <div>
          <div style={{ fontSize: 14, fontWeight: 600, color: "#111827" }}>{name}</div>
          <div style={{ fontSize: 12, color: "#9ca3af" }}>{subName}</div>
        </div>
      </div>
      <KillTag />
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
        <React.Fragment key={dep.model_id + i}>
          {i === 0 ? <VerticalLine /> : <FailConnector />}
          <ModelCard
            label={label}
            name={shortName}
            subName={dep.model_name}
            color={DEPLOY_COLORS[i % DEPLOY_COLORS.length]}
          />
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

      {/* Load balancer split node */}
      <div
        style={{
          border: "1px solid #e5e7eb",
          borderRadius: 12,
          padding: "14px 18px",
          backgroundColor: "#fff",
          width: "100%",
          maxWidth: 340,
        }}
      >
        <div className="flex items-center gap-2" style={{ marginBottom: 10 }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#6b7280" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M16 3h5v5M4 20L21 3M21 16v5h-5M15 15l6 6M4 4l5 5" />
          </svg>
          <span style={{ fontSize: 10, fontWeight: 700, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em" }}>
            LOAD BALANCER
          </span>
        </div>
        {/* Weight bar */}
        <div style={{ width: "100%", height: 8, borderRadius: 4, overflow: "hidden", display: "flex", backgroundColor: "#f3f4f6", marginBottom: 4 }}>
          {deployments.map((dep, i) => {
            const pct = totalWeight > 0 ? ((dep.weight ?? 1) / totalWeight) * 100 : 0;
            return (
              <div key={dep.model_id + i} style={{ width: `${pct}%`, height: "100%", backgroundColor: DEPLOY_COLORS[i % DEPLOY_COLORS.length] }} />
            );
          })}
        </div>
      </div>

      {/* Fan-out lines from center */}
      <div style={{ position: "relative", width: "100%", maxWidth: 500 }}>
        {/* Vertical stem */}
        <div className="flex justify-center"><VerticalLine height={16} /></div>
        {/* Horizontal bar */}
        <div style={{ position: "relative", marginLeft: "10%", marginRight: "10%", height: 2, backgroundColor: "#e5e7eb" }} />
        {/* Branch lines + cards */}
        <div
          style={{
            display: "flex",
            gap: 12,
            width: "100%",
            paddingLeft: "2%",
            paddingRight: "2%",
          }}
        >
          {deployments.map((dep, i) => {
            const pct = totalWeight > 0 ? Math.round(((dep.weight ?? 1) / totalWeight) * 100) : 0;
            const shortName = dep.model_name.split("/").pop() || dep.model_name;
            return (
              <div key={dep.model_id + i} className="flex flex-col items-center flex-1" style={{ minWidth: 0 }}>
                <VerticalLine height={16} />
                <div
                  style={{
                    border: "1px solid #e5e7eb",
                    borderRadius: 10,
                    padding: "10px 8px",
                    backgroundColor: "#fff",
                    width: "100%",
                    textAlign: "center",
                    position: "relative",
                  }}
                >
                  <div style={{ position: "absolute", top: 6, right: 6 }}><StatusDot /></div>
                  <div style={{ fontSize: 10, fontWeight: 700, color: DEPLOY_COLORS[i % DEPLOY_COLORS.length], marginBottom: 2 }}>
                    {pct}%
                  </div>
                  <ProviderIcon color={DEPLOY_COLORS[i % DEPLOY_COLORS.length]} name={shortName} />
                  <div style={{ fontSize: 11, fontWeight: 600, color: "#111827", marginTop: 4, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={dep.model_name}>
                    {shortName}
                  </div>
                  <div style={{ fontSize: 10, color: "#9ca3af", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {dep.model_name}
                  </div>
                </div>
                <VerticalLine height={16} />
              </div>
            );
          })}
        </div>
        {/* Fan-in horizontal bar */}
        <div style={{ position: "relative", marginLeft: "10%", marginRight: "10%", height: 2, backgroundColor: "#e5e7eb" }} />
        <div className="flex justify-center"><VerticalLine height={16} /></div>
      </div>

      <OutputCard subtitle="weighted distribution" />
    </div>
  );
};

const SmartRoutingFlow: React.FC<{ deployments: Deployment[]; strategy: string }> = ({ deployments, strategy }) => {
  const isLatency = strategy === "latency-based-routing";
  return (
    <div className="flex flex-col items-center">
      <TriggerCard subtitle={isLatency ? "→ latency-based" : "→ usage-based"} />
      <VerticalLine />

      <div
        style={{
          border: "1px solid #8b5cf6",
          borderRadius: 12,
          padding: "14px 18px",
          backgroundColor: "#faf5ff",
          width: "100%",
          maxWidth: 340,
        }}
      >
        <div style={{ fontSize: 10, fontWeight: 700, color: "#8b5cf6", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
          {isLatency ? "LATENCY ROUTER" : "USAGE ROUTER"}
        </div>
        <div className="flex flex-col gap-2">
          {deployments.map((dep, i) => {
            const shortName = dep.model_name.split("/").pop() || dep.model_name;
            return (
              <div key={dep.model_id + i} className="flex items-center justify-between" style={{ padding: "6px 10px", backgroundColor: "#fff", borderRadius: 8, border: "1px solid #e5e7eb" }}>
                <div className="flex items-center gap-2">
                  <ProviderIcon color={DEPLOY_COLORS[i % DEPLOY_COLORS.length]} name={shortName} />
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
      <OutputCard subtitle={isLatency ? "fastest model wins" : "least busy model selected"} />
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
  if (deployments.length === 0) return <EmptyFlow />;
  switch (strategy) {
    case "priority-failover": return <FailoverFlow deployments={deployments} />;
    case "weighted": return <WeightedFlow deployments={deployments} />;
    case "latency-based-routing":
    case "usage-based-routing-v2": return <SmartRoutingFlow deployments={deployments} strategy={strategy} />;
    default: return <FailoverFlow deployments={deployments} />;
  }
}

export { DEPLOY_COLORS };
