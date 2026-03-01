"use client";
import React, { useState } from "react";
import { Button, InputNumber, Radio, Spin, Typography } from "antd";
import { SwapOutlined } from "@ant-design/icons";
import OrderedFallbackFlow from "./OrderedFallbackFlow";
import WeightedRoundRobinFlow from "./WeightedRoundRobinFlow";
import StatsBar from "./StatsBar";
import { routingGroupSimulateCall } from "@/components/networking";

interface Deployment {
  deployment_id: string;
  provider: string;
  display_name: string;
  priority?: number;
  weight?: number;
  avg_latency_ms: number;
  percent_of_total: number;
  request_count: number;
  success_count: number;
  failure_count: number;
}

interface LiveTesterProps {
  accessToken: string;
  routingGroupId: string;
  routingGroupName: string;
  routingStrategy: string;  // "priority-failover" | "weighted" | etc.
  initialDeployments?: Deployment[];
}

const isOrderedStrategy = (s: string) =>
  s === "priority-failover";

export default function LiveTester({
  accessToken,
  routingGroupId,
  routingGroupName,
  routingStrategy,
  initialDeployments = [],
}: LiveTesterProps) {
  // Force mode based on strategy, but allow user to toggle
  const [viewMode, setViewMode] = useState<"ordered" | "weighted">(
    isOrderedStrategy(routingStrategy) ? "ordered" : "weighted"
  );
  const [loading, setLoading] = useState(false);
  const [numRequests, setNumRequests] = useState(100);
  const [concurrency, setConcurrency] = useState(10);
  const [mockMode, setMockMode] = useState(true);

  // Live stats state
  const [deployments, setDeployments] = useState<Deployment[]>(
    initialDeployments.length > 0 ? initialDeployments : []
  );
  const [stats, setStats] = useState({
    totalRequests: 0,
    successRate: 100,
    avgLatencyMs: 0,
    fallbackCount: 0,
  });

  const isOrdered = viewMode === "ordered";

  const handleSimulate = async () => {
    if (!accessToken) return;
    setLoading(true);
    try {
      const result = await routingGroupSimulateCall(accessToken, routingGroupId, {
        num_requests: numRequests,
        concurrency,
        mock: mockMode,
      }) as Record<string, unknown>;

      const dist = (result.traffic_distribution as Deployment[]) || [];
      setDeployments(dist);
      const total = (result.total_requests as number) || 0;
      const successful = (result.successful_requests as number) || 0;
      const avgLatency = (result.avg_latency_ms as number) || 0;
      const fallbacks = (result.fallback_count as number) || 0;
      setStats({
        totalRequests: total,
        successRate: total > 0 ? (successful / total) * 100 : 100,
        avgLatencyMs: avgLatency,
        fallbackCount: fallbacks,
      });
    } catch (e) {
      console.error("Simulation error:", e);
    } finally {
      setLoading(false);
    }
  };

  const infoText = isOrdered
    ? "Traffic flows to the primary provider first. When a request fails, it cascades down the priority chain. Thicker streams = more traffic. Dashed red lines show the fallback path."
    : "Traffic is distributed across providers based on weights. Thicker streams = more traffic. Adjust weights in the routing group settings.";

  const badgeLabel = isOrdered ? "Ordered Fallback" : "Weighted Round-Robin";
  const badgeColor = isOrdered ? "#92400e" : "#134e4a";
  const badgeTextColor = isOrdered ? "#fbbf24" : "#2dd4bf";

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3">
            <Typography.Title level={3} style={{ margin: 0, color: "#f1f5f9" }}>
              Live Tester
            </Typography.Title>
            <span
              style={{ background: badgeColor, color: badgeTextColor }}
              className="rounded-md px-2.5 py-1 text-xs font-semibold"
            >
              {badgeLabel}
            </span>
          </div>
          <Typography.Text style={{ color: "#64748b" }}>
            Sending to{" "}
            <span style={{ color: "#38bdf8" }} className="font-mono">
              {routingGroupName}
            </span>
          </Typography.Text>
        </div>
        <Button
          icon={<SwapOutlined />}
          onClick={() => setViewMode(isOrdered ? "weighted" : "ordered")}
          style={{ background: "#1e293b", borderColor: "#334155", color: "#94a3b8" }}
        >
          Switch to {isOrdered ? "Weighted" : "Ordered"}
        </Button>
      </div>

      {/* Stats bar */}
      <StatsBar
        requests={stats.totalRequests}
        successRate={stats.successRate}
        avgLatencyMs={stats.avgLatencyMs}
        fallbackCount={stats.fallbackCount}
      />

      {/* Flow diagram */}
      <Spin spinning={loading}>
        {deployments.length === 0 ? (
          <div className="flex h-48 items-center justify-center rounded-2xl bg-[#0f1117] text-gray-500">
            Run a simulation to see traffic flow
          </div>
        ) : isOrdered ? (
          <OrderedFallbackFlow deployments={deployments} routingGroupName={routingGroupName} />
        ) : (
          <WeightedRoundRobinFlow deployments={deployments} routingGroupName={routingGroupName} />
        )}
      </Spin>

      {/* Info box */}
      <div className="rounded-xl border border-gray-700 bg-[#111827] px-4 py-3 text-sm text-gray-400">
        {infoText}
      </div>

      {/* Simulation controls */}
      <div className="rounded-xl border border-gray-700 bg-[#1a1f2e] p-4">
        <Typography.Text strong style={{ color: "#94a3b8", display: "block", marginBottom: 12 }}>
          Simulation Controls
        </Typography.Text>
        <div className="flex flex-wrap items-end gap-4">
          <div>
            <div className="mb-1 text-xs text-gray-500">Requests</div>
            <InputNumber
              min={1}
              max={10000}
              value={numRequests}
              onChange={(v) => setNumRequests(v ?? 100)}
              style={{ width: 100, background: "#0f172a", borderColor: "#334155", color: "#f1f5f9" }}
            />
          </div>
          <div>
            <div className="mb-1 text-xs text-gray-500">Concurrency</div>
            <InputNumber
              min={1}
              max={100}
              value={concurrency}
              onChange={(v) => setConcurrency(v ?? 10)}
              style={{ width: 90, background: "#0f172a", borderColor: "#334155", color: "#f1f5f9" }}
            />
          </div>
          <div>
            <div className="mb-1 text-xs text-gray-500">Mode</div>
            <Radio.Group
              value={mockMode ? "mock" : "real"}
              onChange={(e) => setMockMode(e.target.value === "mock")}
              buttonStyle="solid"
              size="small"
            >
              <Radio.Button value="mock">Mock</Radio.Button>
              <Radio.Button value="real">Real</Radio.Button>
            </Radio.Group>
          </div>
          <Button
            type="primary"
            loading={loading}
            onClick={handleSimulate}
            style={{ background: "#0f766e", borderColor: "#0f766e" }}
          >
            Run Simulation
          </Button>
        </div>
      </div>
    </div>
  );
}
