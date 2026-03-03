"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Button, InputNumber, message } from "antd";
import {
  ArrowLeftOutlined,
  CodeOutlined,
  DeleteOutlined,
  DollarOutlined,
  ForkOutlined,
  HolderOutlined,
  OrderedListOutlined,
  PlusOutlined,
  PercentageOutlined,
  SwapOutlined,
  ThunderboltOutlined,
  DashboardOutlined,
  NodeIndexOutlined,
} from "@ant-design/icons";
import { routingGroupCreateCall, routingGroupUpdateCall } from "@/components/networking";
import ModelSelector from "@/components/common_components/ModelSelector";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import RoutingFlowVisualization, { DEPLOY_COLORS } from "./RoutingFlowVisualization";

interface Deployment {
  model_id: string;
  model_name: string;
  provider: string;
  weight?: number;
  priority?: number;
}

interface RoutingGroupBuilderProps {
  accessToken: string | null;
  editTarget: Record<string, unknown> | null;
  onClose: () => void;
  onSuccess: () => void;
}

const STRATEGIES = [
  {
    value: "simple-shuffle",
    label: "Simple Shuffle",
    desc: "Random selection across deployments",
    icon: <SwapOutlined style={{ fontSize: 20 }} />,
  },
  {
    value: "priority-failover",
    label: "Ordered Fallback",
    desc: "Try in priority order, skip on failure",
    icon: <OrderedListOutlined style={{ fontSize: 20 }} />,
  },
  {
    value: "weighted",
    label: "Weighted Round-Robin",
    desc: "Distribute traffic by weight %",
    icon: <PercentageOutlined style={{ fontSize: 20 }} />,
  },
  {
    value: "usage-based-routing-v2",
    label: "Usage-Based",
    desc: "Route to least-loaded deployment",
    icon: <DashboardOutlined style={{ fontSize: 20 }} />,
  },
  {
    value: "latency-based-routing",
    label: "Latency-Based",
    desc: "Route to fastest responding backend",
    icon: <ThunderboltOutlined style={{ fontSize: 20 }} />,
  },
  {
    value: "cost-based-routing",
    label: "Cost-Based",
    desc: "Route to cheapest deployment",
    icon: <DollarOutlined style={{ fontSize: 20 }} />,
  },
  {
    value: "least-busy",
    label: "Least Busy",
    desc: "Route to deployment with fewest active requests",
    icon: <NodeIndexOutlined style={{ fontSize: 20 }} />,
  },
  {
    value: "complexity-router",
    label: "Auto Router",
    desc: "Route by request complexity, no API calls, <1ms",
    icon: <ForkOutlined style={{ fontSize: 20 }} />,
  },
];

const COMPLEXITY_TIERS: { key: string; label: string; desc: string; examples: string }[] = [
  { key: "SIMPLE", label: "Simple", desc: "Basic questions, greetings, simple facts", examples: '"Hello!", "What is Python?"' },
  { key: "MEDIUM", label: "Medium", desc: "Standard queries requiring explanation", examples: '"Explain REST APIs", "Debug this error"' },
  { key: "COMPLEX", label: "Complex", desc: "Technical, multi-part deep requests", examples: '"Design a microservices arch"' },
  { key: "REASONING", label: "Reasoning", desc: "Chain-of-thought, analysis", examples: '"Think step by step..."' },
];

const INITIAL_SLOT_COUNT = 3;

export default function RoutingGroupBuilder({
  accessToken,
  editTarget,
  onClose,
  onSuccess,
}: RoutingGroupBuilderProps) {
  const { accessToken: authToken } = useAuthorized();
  const effectiveToken = accessToken ?? authToken ?? "";

  const [strategy, setStrategy] = useState("simple-shuffle");
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [emptySlots, setEmptySlots] = useState(INITIAL_SLOT_COUNT);
  const [logicalName, setLogicalName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [tierModels, setTierModels] = useState<Record<string, string>>({ SIMPLE: "", MEDIUM: "", COMPLEX: "", REASONING: "" });

  const dragItem = useRef<number | null>(null);
  const dragOverItem = useRef<number | null>(null);

  const isEditing = editTarget !== null;

  useEffect(() => {
    if (editTarget) {
      setStrategy((editTarget.routing_strategy as string) ?? "priority-failover");
      setLogicalName((editTarget.routing_group_name as string) ?? "");
      if (Array.isArray(editTarget.deployments)) {
        const deps = (editTarget.deployments as any[]).map((d, i) => ({
          model_id: d.model_id ?? d.model_name,
          model_name: d.model_name,
          provider: d.provider ?? d.litellm_provider ?? "-",
          weight: d.weight,
          priority: d.priority ?? i + 1,
        }));
        setDeployments(deps);
        setEmptySlots(Math.max(0, INITIAL_SLOT_COUNT - deps.length));
      }
    }
  }, [editTarget]);

  const totalWeight = useMemo(
    () => deployments.reduce((s, d) => s + (d.weight ?? 1), 0),
    [deployments]
  );

  const isComplexityRouter = strategy === "complexity-router";

  const handleTierModelChange = (tierKey: string, model: string) => {
    const updated = { ...tierModels, [tierKey]: model };
    setTierModels(updated);
    const deps = COMPLEXITY_TIERS
      .map((t) => updated[t.key])
      .filter(Boolean)
      .map((name) => ({ model_id: name, model_name: name, provider: "-" }));
    setDeployments(deps);
  };

  const handleStrategyChange = (value: string) => {
    setStrategy(value);
    setDeployments((prev) => {
      const equal = prev.length > 0 ? Math.max(1, Math.round(100 / prev.length)) : 1;
      return prev.map((d, i) => ({
        ...d,
        weight: value === "weighted" ? equal : undefined,
        priority: value === "priority-failover" ? i + 1 : undefined,
      }));
    });
  };

  const handleSlotSelected = (slotIndex: number, modelGroup: string) => {
    if (!modelGroup || deployments.some((d) => d.model_name === modelGroup)) return;
    const count = deployments.length + 1;
    const equal = Math.max(1, Math.round(100 / count));
    const newDep: Deployment = {
      model_id: modelGroup,
      model_name: modelGroup,
      provider: "-",
      weight: strategy === "weighted" ? equal : undefined,
      priority: strategy === "priority-failover" ? count : undefined,
    };
    if (strategy === "weighted") {
      setDeployments((prev) => [...prev.map((d) => ({ ...d, weight: equal })), newDep]);
    } else {
      setDeployments((prev) => [...prev, newDep]);
    }
    setEmptySlots((prev) => Math.max(0, prev - 1));
  };

  const handleDeploymentChange = (index: number, modelGroup: string) => {
    if (!modelGroup) return;
    if (deployments.some((d, i) => i !== index && d.model_name === modelGroup)) return;
    setDeployments((prev) =>
      prev.map((d, i) =>
        i === index ? { ...d, model_id: modelGroup, model_name: modelGroup } : d
      )
    );
  };

  const addEmptySlot = () => setEmptySlots((prev) => prev + 1);

  const removeDeployment = (index: number) => {
    setDeployments((prev) => {
      const updated = prev.filter((_, i) => i !== index);
      if (strategy === "priority-failover") {
        return updated.map((d, i) => ({ ...d, priority: i + 1 }));
      }
      return updated;
    });
  };

  const updatePercent = (index: number, newPct: number | null) => {
    const pct = Math.max(1, Math.min(99, newPct ?? 1));
    setDeployments((prev) => {
      const otherTotal = prev.reduce((s, d, i) => i === index ? s : s + (d.weight ?? 1), 0);
      if (otherTotal === 0) return prev;
      const scale = (100 - pct) / otherTotal;
      return prev.map((d, i) =>
        i === index ? { ...d, weight: pct } : { ...d, weight: Math.max(1, Math.round((d.weight ?? 1) * scale)) }
      );
    });
  };

  const handleDragStart = (index: number) => { dragItem.current = index; };
  const handleDragEnter = (index: number) => { dragOverItem.current = index; };
  const handleDragEnd = () => {
    if (dragItem.current === null || dragOverItem.current === null) return;
    if (dragItem.current === dragOverItem.current) {
      dragItem.current = null;
      dragOverItem.current = null;
      return;
    }
    setDeployments((prev) => {
      const updated = [...prev];
      const [removed] = updated.splice(dragItem.current!, 1);
      updated.splice(dragOverItem.current!, 0, removed);
      if (strategy === "priority-failover") {
        return updated.map((d, i) => ({ ...d, priority: i + 1 }));
      }
      return updated;
    });
    dragItem.current = null;
    dragOverItem.current = null;
  };

  const handleSubmit = async () => {
    const token = accessToken ?? effectiveToken;
    if (!token) {
      message.error("No access token available");
      return;
    }
    if (!logicalName.trim()) {
      message.error("Please enter a logical model name");
      return;
    }
    if (deployments.length === 0) {
      message.error("Please add at least one deployment");
      return;
    }
    try {
      setSubmitting(true);
      const payload: Record<string, unknown> = {
        routing_group_name: logicalName.trim(),
        routing_strategy: strategy,
        deployments,
      };
      if (isEditing && editTarget?.routing_group_id) {
        await routingGroupUpdateCall(token, editTarget.routing_group_id as string, payload);
        message.success("Routing group updated");
      } else {
        await routingGroupCreateCall(token, payload);
        message.success("Routing group created");
      }
      onSuccess();
    } catch (err) {
      console.error("Failed to save routing group:", err);
      message.error("Failed to save: " + (err instanceof Error ? err.message : String(err)));
    } finally {
      setSubmitting(false);
    }
  };

  const codeSnippet = `# invoke this group:\nclient.chat.completions.create(\n  model="${logicalName || "your-model-name"}",\n  messages=[...]\n)`;

  return (
    <div className="w-full max-w-screen overflow-x-hidden box-border">
      <div className="flex" style={{ minHeight: "100vh" }}>
        {/* ── LEFT: Config Panel ── */}
        <div style={{ width: 480, flexShrink: 0, borderRight: "1px solid #e5e7eb", backgroundColor: "#fff", overflowY: "auto", padding: "20px 24px 48px" }}>
          <button
            onClick={onClose}
            className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 mb-3 bg-transparent border-none cursor-pointer p-0"
          >
            <ArrowLeftOutlined /> Back to Routing Groups
          </button>
          <h1 className="text-lg font-semibold m-0">Routing Group Builder</h1>
          <p className="text-xs text-gray-500 mt-1 mb-5">Configure a named model group with a routing strategy</p>

          {/* 1 · Routing Strategy */}
          <div className="mb-5">
            <div className="text-xs font-semibold text-gray-400 tracking-wide uppercase mb-3">1 &middot; Routing Strategy</div>
            <div className="grid grid-cols-2 gap-2">
              {STRATEGIES.map((s) => {
                const isActive = strategy === s.value;
                return (
                  <button
                    key={s.value}
                    onClick={() => handleStrategyChange(s.value)}
                    className="text-left p-3 rounded-lg border-2 transition-all bg-white cursor-pointer"
                    style={{ borderColor: isActive ? "#3b82f6" : "#e5e7eb", boxShadow: isActive ? "0 0 0 1px #3b82f6" : "none" }}
                  >
                    <div className="mb-1.5 text-gray-400">{s.icon}</div>
                    <div className="font-semibold text-gray-900 text-xs">{s.label}</div>
                    <div className="text-xs text-gray-500 mt-0.5 leading-tight">{s.desc}</div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* 2 · Deployments / Tier Config */}
          <div className="mb-5">
            {isComplexityRouter ? (
              <>
                <div className="text-xs font-semibold text-gray-400 tracking-wide uppercase mb-1">2 &middot; Complexity Tiers</div>
                <div className="text-xs text-gray-400 mb-3">Assign a model to each complexity tier. Simple queries go to cheaper models, complex ones to more capable models.</div>
                <div className="flex flex-col gap-2.5">
                  {COMPLEXITY_TIERS.map((t) => (
                    <div key={t.key} className="rounded-lg border border-gray-200 bg-white px-4 py-3">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-semibold text-gray-800">{t.label}</span>
                        <span className="text-xs text-gray-400">{t.examples}</span>
                      </div>
                      <div className="text-xs text-gray-400 mb-2">{t.desc}</div>
                      <ModelSelector
                        accessToken={effectiveToken}
                        value={tierModels[t.key] || undefined}
                        placeholder={`Select model for ${t.label.toLowerCase()} tier...`}
                        showLabel={false}
                        onChange={(model) => handleTierModelChange(t.key, model)}
                      />
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <>
                <div className="flex items-center justify-between mb-2">
                  <div className="text-xs font-semibold text-gray-400 tracking-wide uppercase">2 &middot; Deployments</div>
                  <button
                    onClick={addEmptySlot}
                    className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700 bg-transparent border-none cursor-pointer p-0 font-medium"
                  >
                    <PlusOutlined /> Add
                  </button>
                </div>

                {deployments.length > 0 && (
                  <div className="flex items-center flex-wrap gap-1 mb-3">
                    {deployments.map((dep, i) => {
                      const shortName = dep.model_name.split("/").pop() || dep.model_name;
                      return (
                        <React.Fragment key={dep.model_id + i}>
                          {i > 0 && <span style={{ fontSize: 10, color: "#9ca3af" }}>→</span>}
                          <span
                            style={{
                              fontSize: 11,
                              fontWeight: 600,
                              color: DEPLOY_COLORS[i % DEPLOY_COLORS.length],
                              backgroundColor: DEPLOY_COLORS[i % DEPLOY_COLORS.length] + "14",
                              border: `1px solid ${DEPLOY_COLORS[i % DEPLOY_COLORS.length]}30`,
                              padding: "2px 8px",
                              borderRadius: 6,
                              display: "inline-flex",
                              alignItems: "center",
                              gap: 4,
                            }}
                          >
                            <span style={{ fontWeight: 700, fontSize: 10 }}>{i + 1}</span>
                            {shortName}
                          </span>
                        </React.Fragment>
                      );
                    })}
                  </div>
                )}

                {strategy === "weighted" && deployments.length > 0 && (
                  <div className="mb-3">
                    <div style={{ width: "100%", height: 6, borderRadius: 3, overflow: "hidden", display: "flex", backgroundColor: "#e5e7eb" }}>
                      {deployments.map((dep, i) => {
                        const pct = totalWeight > 0 ? ((dep.weight ?? 1) / totalWeight) * 100 : 0;
                        return <div key={dep.model_id + i} style={{ width: `${pct}%`, height: "100%", backgroundColor: DEPLOY_COLORS[i % DEPLOY_COLORS.length] }} />;
                      })}
                    </div>
                    <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-1">
                      {deployments.map((dep, i) => {
                        const pct = totalWeight > 0 ? Math.round(((dep.weight ?? 1) / totalWeight) * 100) : 0;
                        const shortName = dep.model_name.split("/").pop() || dep.model_name;
                        return (
                          <div key={dep.model_id + i} className="flex items-center gap-1">
                            <div style={{ width: 6, height: 6, borderRadius: "50%", backgroundColor: DEPLOY_COLORS[i % DEPLOY_COLORS.length] }} />
                            <span className="text-xs text-gray-500">{shortName}: {pct}%</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                <div className="flex flex-col gap-2">
                  {deployments.map((dep, idx) => {
                    const pct = totalWeight > 0 ? Math.round(((dep.weight ?? 1) / totalWeight) * 100) : 0;
                    return (
                      <div
                        key={dep.model_id + idx}
                        draggable
                        onDragStart={() => handleDragStart(idx)}
                        onDragEnter={() => handleDragEnter(idx)}
                        onDragEnd={handleDragEnd}
                        onDragOver={(e) => e.preventDefault()}
                        className="rounded-lg border border-gray-200 bg-white px-3 py-2.5 cursor-grab active:cursor-grabbing hover:border-gray-300 transition-colors"
                      >
                        <div className="flex items-center gap-2">
                          <HolderOutlined className="text-gray-300 text-xs" />
                          <span className="text-xs font-bold text-gray-400 w-4 text-center">{idx + 1}</span>
                          <div className="flex-1 min-w-0" onClick={(e) => e.stopPropagation()}>
                            <ModelSelector
                              accessToken={effectiveToken}
                              value={dep.model_name}
                              placeholder="Select a model..."
                              showLabel={false}
                              onChange={(model) => handleDeploymentChange(idx, model)}
                            />
                          </div>
                          <button
                            onClick={() => removeDeployment(idx)}
                            className="text-gray-300 hover:text-red-400 bg-transparent border-none cursor-pointer p-0 flex-shrink-0"
                          >
                            <DeleteOutlined style={{ fontSize: 13 }} />
                          </button>
                        </div>
                        {strategy === "weighted" && (
                          <div className="flex items-center gap-2 mt-2 ml-6" onClick={(e) => e.stopPropagation()}>
                            <InputNumber
                              size="small"
                              min={1}
                              max={99}
                              value={pct}
                              onChange={(val) => updatePercent(idx, val)}
                              addonAfter="%"
                              style={{ width: 80 }}
                            />
                            <span className="text-xs text-gray-400">traffic</span>
                          </div>
                        )}
                      </div>
                    );
                  })}

                  {Array.from({ length: emptySlots }).map((_, slotIdx) => (
                    <div key={`slot-${slotIdx}`} className="rounded-lg border border-dashed border-gray-200 bg-gray-50 px-3 py-1.5">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold text-gray-300 w-4 text-center">{deployments.length + slotIdx + 1}</span>
                        <div className="flex-1">
                          <ModelSelector
                            accessToken={effectiveToken}
                            placeholder="Select a model..."
                            showLabel={false}
                            onChange={(model) => handleSlotSelected(slotIdx, model)}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>

          {/* 3 · Logical Model Name */}
          <div className="mb-5">
            <div className="text-xs font-semibold text-gray-400 tracking-wide uppercase mb-1">3 &middot; Logical Model Name</div>
            <div className="text-xs text-gray-400 mb-2">What callers use to invoke this group</div>
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
            <div className="mt-2 rounded-lg bg-gray-900 px-4 py-3 font-mono text-xs text-gray-300 leading-relaxed overflow-x-auto">
              <pre className="m-0 whitespace-pre">{codeSnippet}</pre>
            </div>
          </div>

          <Button
            type="primary"
            size="large"
            block
            loading={submitting}
            disabled={!logicalName.trim() || deployments.length === 0}
            onClick={handleSubmit}
            style={{ height: 44, borderRadius: 8, fontWeight: 600, fontSize: 14 }}
          >
            {isEditing ? "Update Routing Group" : "Save Routing Group"}
          </Button>
        </div>

        {/* ── RIGHT: Light Flow Preview ── */}
        <div style={{ flex: 1, minWidth: 0, backgroundColor: "#f3f4f6", display: "flex", flexDirection: "column" }}>
          {/* Header */}
          <div
            className="flex items-center justify-between"
            style={{ padding: "10px 24px", borderBottom: "1px solid #e5e7eb", backgroundColor: "#fff", flexShrink: 0 }}
          >
            <span style={{ fontSize: 14, fontWeight: 600, color: "#374151" }}>Flow Preview</span>
            <div className="flex items-center gap-2">
              <div className="flex" style={{ border: "1px solid #e5e7eb", borderRadius: 6, overflow: "hidden" }}>
                {["1x", "2x", "5x"].map((label) => (
                  <button
                    key={label}
                    className="text-xs px-2.5 py-1 bg-white hover:bg-gray-50 border-none cursor-pointer text-gray-500"
                    style={{ borderRight: label !== "5x" ? "1px solid #e5e7eb" : "none" }}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <button className="flex items-center gap-1 text-xs text-gray-500 px-2.5 py-1 border border-gray-200 rounded-md bg-white hover:bg-gray-50 cursor-pointer">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M1 4v6h6M23 20v-6h-6" /><path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15" />
                </svg>
                Reset
              </button>
              <button
                className="flex items-center gap-1 text-xs text-white px-3 py-1.5 border-none rounded-md cursor-pointer font-semibold"
                style={{ backgroundColor: "#111827" }}
              >
                <svg width="10" height="10" viewBox="0 0 24 24" fill="#fff" stroke="none"><polygon points="6,3 20,12 6,21" /></svg>
                Run
              </button>
            </div>
          </div>

          {/* Flow canvas */}
          <div style={{ flex: 1, overflowY: "auto", padding: "24px", display: "flex", justifyContent: "center" }}>
            <div style={{ maxWidth: 520, width: "100%" }}>
              <RoutingFlowVisualization strategy={strategy} deployments={deployments} />
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
