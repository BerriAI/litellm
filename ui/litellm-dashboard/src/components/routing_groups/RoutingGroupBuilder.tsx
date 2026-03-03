"use client";

import React, { useEffect, useMemo, useState } from "react";
import { Button, Checkbox, Input, InputNumber, Typography } from "antd";
import {
  ArrowLeftOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  BranchesOutlined,
  CloudOutlined,
  CodeOutlined,
  DeleteOutlined,
  OrderedListOutlined,
  PlusOutlined,
  PercentageOutlined,
  ThunderboltOutlined,
  DashboardOutlined,
} from "@ant-design/icons";
import { routingGroupCreateCall, routingGroupUpdateCall } from "@/components/networking";
import DeploymentSelector, { DeploymentOption } from "./DeploymentSelector";

interface Deployment {
  model_id: string;
  model_name: string;
  provider: string;
  weight?: number;
  priority?: number;
  enabled?: boolean;
}

interface RoutingGroupBuilderProps {
  accessToken: string | null;
  editTarget: Record<string, unknown> | null;
  onClose: () => void;
  onSuccess: () => void;
}

const PROVIDER_COLORS: Record<string, string> = {
  "Nebius": "#10b981",
  "Fireworks AI": "#ef4444",
  "Azure": "#3b82f6",
  "Azure OpenAI": "#3b82f6",
  "OpenAI": "#10b981",
  "Anthropic": "#f59e0b",
  "Google AI Studio": "#4285f4",
  "Groq": "#8b5cf6",
  "Deepseek": "#06b6d4",
  "Mistral AI": "#f97316",
  "Amazon Bedrock": "#f59e0b",
  "custom": "#6b7280",
};

function getProviderColor(provider: string): string {
  return PROVIDER_COLORS[provider] ?? "#6366f1";
}

function getProviderInitial(provider: string): string {
  if (provider === "-" || !provider) return "?";
  return provider.charAt(0).toUpperCase();
}

const STRATEGIES = [
  {
    value: "priority-failover",
    label: "Ordered Fallback",
    desc: "Try in priority order, skip on failure",
    icon: <OrderedListOutlined style={{ fontSize: 24 }} />,
  },
  {
    value: "weighted",
    label: "Weighted Round-Robin",
    desc: "Distribute traffic by weight %",
    icon: <PercentageOutlined style={{ fontSize: 24 }} />,
  },
  {
    value: "usage-based-routing-v2",
    label: "Usage-Based",
    desc: "Route to least-loaded deployment",
    icon: <DashboardOutlined style={{ fontSize: 24 }} />,
  },
  {
    value: "latency-based-routing",
    label: "Latency-Based",
    desc: "Route to fastest responding backend",
    icon: <ThunderboltOutlined style={{ fontSize: 24 }} />,
  },
];

export default function RoutingGroupBuilder({
  accessToken,
  editTarget,
  onClose,
  onSuccess,
}: RoutingGroupBuilderProps) {
  const [strategy, setStrategy] = useState("priority-failover");
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [logicalName, setLogicalName] = useState("");
  const [description, setDescription] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [showAddSelector, setShowAddSelector] = useState(false);
  const [selectedDeployment, setSelectedDeployment] = useState<DeploymentOption | null>(null);
  const [selectorKey, setSelectorKey] = useState(0);

  const isEditing = editTarget !== null;

  useEffect(() => {
    if (editTarget) {
      setStrategy((editTarget.routing_strategy as string) ?? "priority-failover");
      setLogicalName((editTarget.routing_group_name as string) ?? "");
      setDescription((editTarget.description as string) ?? "");
      if (Array.isArray(editTarget.deployments)) {
        setDeployments(
          (editTarget.deployments as any[]).map((d, i) => ({
            model_id: d.model_id ?? d.model_name,
            model_name: d.model_name,
            provider: d.provider ?? d.litellm_provider ?? "-",
            weight: d.weight,
            priority: d.priority ?? i + 1,
            enabled: true,
          }))
        );
      }
    }
  }, [editTarget]);

  const handleStrategyChange = (value: string) => {
    setStrategy(value);
    setDeployments((prev) =>
      prev.map((d, i) => ({
        ...d,
        weight: value === "weighted" ? (d.weight ?? 1) : undefined,
        priority: value === "priority-failover" ? i + 1 : undefined,
      }))
    );
  };

  const addDeployment = () => {
    if (!selectedDeployment) return;
    setDeployments((prev) => [
      ...prev,
      {
        model_id: selectedDeployment.model_id,
        model_name: selectedDeployment.model_name,
        provider: selectedDeployment.provider,
        weight: strategy === "weighted" ? 1 : undefined,
        priority: strategy === "priority-failover" ? prev.length + 1 : undefined,
        enabled: true,
      },
    ]);
    setSelectedDeployment(null);
    setSelectorKey((k) => k + 1);
    setShowAddSelector(false);
  };

  const removeDeployment = (index: number) => {
    setDeployments((prev) => {
      const updated = prev.filter((_, i) => i !== index);
      if (strategy === "priority-failover") {
        return updated.map((d, i) => ({ ...d, priority: i + 1 }));
      }
      return updated;
    });
  };

  const moveDeployment = (index: number, direction: "up" | "down") => {
    const newIndex = direction === "up" ? index - 1 : index + 1;
    if (newIndex < 0 || newIndex >= deployments.length) return;
    setDeployments((prev) => {
      const updated = [...prev];
      [updated[index], updated[newIndex]] = [updated[newIndex], updated[index]];
      if (strategy === "priority-failover") {
        return updated.map((d, i) => ({ ...d, priority: i + 1 }));
      }
      return updated;
    });
  };

  const updateWeight = (index: number, weight: number | null) => {
    setDeployments((prev) =>
      prev.map((d, i) => (i === index ? { ...d, weight: weight ?? 1 } : d))
    );
  };

  const toggleEnabled = (index: number) => {
    setDeployments((prev) =>
      prev.map((d, i) => (i === index ? { ...d, enabled: !d.enabled } : d))
    );
  };

  const handleSubmit = async () => {
    if (!accessToken || !logicalName.trim() || deployments.length === 0) return;
    try {
      setSubmitting(true);
      const payload: Record<string, unknown> = {
        routing_group_name: logicalName.trim(),
        description: description.trim() || undefined,
        routing_strategy: strategy,
        deployments: deployments.filter((d) => d.enabled !== false),
      };
      if (isEditing && editTarget?.routing_group_id) {
        await routingGroupUpdateCall(accessToken, editTarget.routing_group_id as string, payload);
      } else {
        await routingGroupCreateCall(accessToken, payload);
      }
      onSuccess();
    } catch (err) {
      console.error("Failed to save routing group:", err);
    } finally {
      setSubmitting(false);
    }
  };

  const excludeIds = useMemo(() => new Set(deployments.map((d) => d.model_id)), [deployments]);

  const codeSnippet = `# invoke this group:\nclient.chat.completions.create(\n  model="${logicalName || "your-model-name"}",\n  messages=[...]\n)`;

  return (
    <div className="w-full max-w-2xl mx-auto py-6 px-4">
      {/* Header */}
      <div className="mb-1">
        <button
          onClick={onClose}
          className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 mb-3 bg-transparent border-none cursor-pointer p-0"
        >
          <ArrowLeftOutlined /> Back to Routing Groups
        </button>
        <h1 className="text-xl font-bold text-gray-900 m-0">Routing Group Builder</h1>
        <p className="text-sm text-gray-500 mt-1 mb-0">
          Configure a named model group with a routing strategy
        </p>
      </div>

      {/* Section 1: Routing Strategy */}
      <div className="mt-8">
        <div className="text-xs font-semibold text-gray-500 tracking-wide mb-4">
          1 &middot; ROUTING STRATEGY
        </div>
        <div className="grid grid-cols-2 gap-3">
          {STRATEGIES.map((s) => {
            const isActive = strategy === s.value;
            return (
              <button
                key={s.value}
                onClick={() => handleStrategyChange(s.value)}
                className="text-left p-4 rounded-xl border-2 transition-all bg-white cursor-pointer"
                style={{
                  borderColor: isActive ? "#3b82f6" : "#e5e7eb",
                  boxShadow: isActive ? "0 0 0 1px #3b82f6" : "none",
                }}
              >
                <div className="mb-2 text-gray-400">{s.icon}</div>
                <div className="font-semibold text-gray-900 text-sm">{s.label}</div>
                <div className="text-xs text-gray-500 mt-0.5">{s.desc}</div>
                {isActive && (
                  <span className="inline-block mt-2 text-xs font-semibold text-blue-600 border border-blue-600 rounded px-1.5 py-0.5">
                    Active
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Section 2: Deployments */}
      <div className="mt-10">
        <div className="flex items-center justify-between mb-4">
          <div className="text-xs font-semibold text-gray-500 tracking-wide">
            2 &middot; DEPLOYMENTS
          </div>
          <button
            onClick={() => setShowAddSelector(!showAddSelector)}
            className="flex items-center gap-1 text-sm font-medium text-blue-600 hover:text-blue-700 bg-transparent border-none cursor-pointer p-0"
          >
            <PlusOutlined /> Add
          </button>
        </div>

        {/* Add selector */}
        {showAddSelector && (
          <div className="mb-4 p-3 rounded-lg border border-blue-200 bg-blue-50">
            <div className="flex items-end gap-2">
              <div className="flex-1">
                <DeploymentSelector
                  key={selectorKey}
                  value={selectedDeployment?.model_id ?? null}
                  placeholder="Search and select a model..."
                  showLabel={false}
                  excludeIds={excludeIds}
                  onChange={(dep) => setSelectedDeployment(dep)}
                />
              </div>
              <Button
                type="primary"
                size="small"
                disabled={!selectedDeployment}
                onClick={addDeployment}
              >
                Add
              </Button>
              <Button
                size="small"
                onClick={() => { setShowAddSelector(false); setSelectedDeployment(null); }}
              >
                Cancel
              </Button>
            </div>
          </div>
        )}

        {/* Pipeline visualization */}
        {deployments.length > 0 && (
          <div className="flex items-center gap-1 mb-4 flex-wrap">
            {deployments.map((dep, idx) => {
              const color = getProviderColor(dep.provider);
              return (
                <React.Fragment key={dep.model_id}>
                  {idx > 0 && (
                    <span className="text-gray-300 text-xs mx-0.5">&rarr;</span>
                  )}
                  <span
                    className="inline-flex items-center gap-1 text-xs font-medium rounded-full px-2.5 py-1 border"
                    style={{
                      borderColor: color + "44",
                      color: color,
                      backgroundColor: color + "0d",
                    }}
                  >
                    <span className="font-bold">{idx + 1}</span>
                    {dep.provider !== "-" ? dep.provider.split(" ")[0] : dep.model_name}
                  </span>
                </React.Fragment>
              );
            })}
          </div>
        )}

        {/* Deployment cards */}
        {deployments.length === 0 && !showAddSelector && (
          <div className="text-center py-8 text-gray-400 text-sm border border-dashed border-gray-200 rounded-xl">
            No deployments yet. Click <strong>+ Add</strong> to get started.
          </div>
        )}

        <div className="flex flex-col gap-2">
          {deployments.map((dep, idx) => {
            const color = getProviderColor(dep.provider);
            return (
              <div
                key={dep.model_id}
                className="flex items-center gap-3 rounded-xl border bg-white px-4 py-3 transition-all"
                style={{ borderLeft: `4px solid ${color}` }}
              >
                <Checkbox
                  checked={dep.enabled !== false}
                  onChange={() => toggleEnabled(idx)}
                />
                <span className="text-xs font-bold text-gray-400 min-w-[16px]">
                  {idx + 1}
                </span>
                {/* Provider icon */}
                <div
                  className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg text-xs font-bold"
                  style={{ backgroundColor: color + "1a", color }}
                >
                  {getProviderInitial(dep.provider)}
                </div>
                {/* Name */}
                <div className="flex-1 min-w-0">
                  <span className="text-sm font-semibold text-gray-800">
                    {dep.provider}
                  </span>
                  <span className="text-sm text-gray-500 ml-1.5">
                    / {dep.model_name}
                  </span>
                </div>
                {/* Weight (only for weighted) */}
                {strategy === "weighted" && (
                  <InputNumber
                    size="small"
                    min={1}
                    max={100}
                    value={dep.weight ?? 1}
                    onChange={(val) => updateWeight(idx, val)}
                    addonAfter="%"
                    style={{ width: 90 }}
                  />
                )}
                {/* Reorder */}
                <div className="flex flex-col gap-0">
                  <button
                    onClick={() => moveDeployment(idx, "up")}
                    disabled={idx === 0}
                    className="text-gray-300 hover:text-gray-500 disabled:opacity-30 bg-transparent border-none cursor-pointer p-0 leading-none"
                  >
                    <ArrowUpOutlined style={{ fontSize: 12 }} />
                  </button>
                  <button
                    onClick={() => moveDeployment(idx, "down")}
                    disabled={idx === deployments.length - 1}
                    className="text-gray-300 hover:text-gray-500 disabled:opacity-30 bg-transparent border-none cursor-pointer p-0 leading-none"
                  >
                    <ArrowDownOutlined style={{ fontSize: 12 }} />
                  </button>
                </div>
                {/* Delete */}
                <button
                  onClick={() => removeDeployment(idx)}
                  className="text-gray-300 hover:text-red-400 bg-transparent border-none cursor-pointer p-0"
                >
                  <DeleteOutlined style={{ fontSize: 14 }} />
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* Section 3: Logical Model Name */}
      <div className="mt-10">
        <div className="text-xs font-semibold text-gray-500 tracking-wide mb-1">
          3 &middot; LOGICAL MODEL NAME
        </div>
        <div className="text-xs text-gray-400 mb-3">
          What callers use to invoke this group
        </div>
        <div className="flex items-center gap-2 border border-gray-200 rounded-lg px-3 py-2 bg-white">
          <CodeOutlined className="text-gray-400" />
          <input
            type="text"
            value={logicalName}
            onChange={(e) => setLogicalName(e.target.value)}
            placeholder="prod-model"
            className="flex-1 border-none outline-none text-sm font-mono text-gray-800 bg-transparent"
          />
        </div>

        {/* Code snippet */}
        <div className="mt-3 rounded-lg bg-gray-900 px-4 py-3 font-mono text-xs text-gray-300 leading-relaxed overflow-x-auto">
          <pre className="m-0 whitespace-pre">{codeSnippet}</pre>
        </div>
      </div>

      {/* Save button */}
      <div className="mt-8 mb-4">
        <Button
          type="primary"
          size="large"
          block
          loading={submitting}
          disabled={!logicalName.trim() || deployments.length === 0}
          onClick={handleSubmit}
          style={{
            height: 48,
            borderRadius: 12,
            fontWeight: 600,
            fontSize: 15,
          }}
        >
          {isEditing ? "Update Routing Group" : "Save Routing Group"}
        </Button>
      </div>
    </div>
  );
}
